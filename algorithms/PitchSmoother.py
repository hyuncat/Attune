"""Pitch tracking, production candidate support, and final voicing."""

from __future__ import annotations

import copy
import time
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy.signal import get_window
from tqdm import tqdm

from algorithms.Config import Config, PYIN_GLOBAL_VOLUME_PERCENTILE
from app_logic.user.ds.PitchData import Pitch

if TYPE_CHECKING:
    from app_logic.user.ds.Recording import Recording


FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


try:
    from numba import njit
except ImportError:  # pragma: no cover -- librosa currently installs numba
    def njit(*_args, **_kwargs):
        def decorate(function):
            return function

        return decorate


@njit(cache=True)
def _viterbi_0110(
    log_probability: FloatArray,
    log_transition: FloatArray,
    log_initial: FloatArray,
) -> npt.NDArray[np.uint16]:
    """Librosa 0.11's Viterbi recurrence, retained for exact pYIN parity."""
    frame_count, state_count = log_probability.shape
    states = np.zeros(frame_count, dtype=np.uint16)
    if frame_count == 0:
        return states
    pointers = np.zeros((frame_count, state_count), dtype=np.uint16)

    previous = log_probability[0] + log_initial
    current = np.empty(state_count, dtype=np.float64)
    for frame in range(1, frame_count):
        transition_out = previous + log_transition.T
        for destination in range(state_count):
            pointers[frame, destination] = np.argmax(
                transition_out[destination]
            )
            current[destination] = (
                log_probability[frame, destination]
                + transition_out[
                    destination,
                    pointers[frame, destination],
                ]
            )
        previous, current = current, previous

    states[-1] = np.argmax(previous)
    for frame in range(frame_count - 2, -1, -1):
        states[frame] = pointers[frame + 1, states[frame + 1]]
    return states


class PitchSmoother:
    """Production pitch tracking; joint and pitch_only modes support benchmarks."""

    METHOD_VERSION = 'pitch_only_confidence_gap_50ms_support_v2'
    RESOLUTION = 0.1
    MAX_TRANSITION_RATE = 35.92
    SWITCH_PROB = 0.01

    def __init__(
        self,
        recording: Recording | None = None,
        config: Config | None = None,
        *,
        mode: str = "production",
        max_gap_seconds: float | None = None,
        confidence_emissions: bool | None = None,
        resolution: float = RESOLUTION,
        max_transition_rate: float = MAX_TRANSITION_RATE,
        switch_prob: float = SWITCH_PROB,
    ) -> None:
        if recording is None and config is None:
            raise ValueError("PitchSmoother requires a recording or config")
        if mode not in ("production", "pitch_only", "joint"):
            raise ValueError(f"Unknown pitch smoothing mode: {mode}")
        self.mode = mode
        self.max_gap_seconds = float(
            (0.050 if mode == "production" else 0.0)
            if max_gap_seconds is None else max_gap_seconds
        )
        self.confidence_emissions = (
            mode == "production" if confidence_emissions is None
            else bool(confidence_emissions)
        )
        if not np.isfinite(self.max_gap_seconds) or self.max_gap_seconds < 0:
            raise ValueError("max_gap_seconds must be finite and nonnegative")
        self.recording = recording
        self.resolution = float(resolution)
        self.max_transition_rate = float(max_transition_rate)
        self.switch_prob = float(switch_prob)
        self.update_config(config if config is not None else recording.config)

    def update_config(self, config: Config) -> None:
        self.config = config
        self.fmin = float(config.fmin)
        self.fmax = float(config.fmax)
        self.sr = int(config.sr)
        self.hop_length = int(config.h1)
        self.n_bins_per_semitone = int(np.ceil(1.0 / self.resolution))
        self.n_pitch_bins = (
            int(np.floor(
                12
                * self.n_bins_per_semitone
                * np.log2(self.fmax / self.fmin)
            ))
            + 1
        )
        self.bin_freqs = self.fmin * 2.0 ** (
            np.arange(self.n_pitch_bins, dtype=np.float64)
            / (12 * self.n_bins_per_semitone)
        )
        self.bin_midis = np.asarray(
            [config.freq_to_midi(freq) for freq in self.bin_freqs],
            dtype=np.float64,
        )
        self.midi_min = float(self.bin_midis[0])
        self.midi_max = float(self.bin_midis[-1])
        self.transition = self._transition_matrix()
        state_count = self.transition.shape[0]
        self.initial = np.ones(state_count, dtype=np.float64)
        self.initial /= state_count

    def smooth(
        self, pitches: list[Pitch | None],
        show_progress: bool = False, verbose: bool = False,
    ) -> list[Pitch | None]:
        tracked = self._smooth_pitch_path(pitches, show_progress, verbose)
        if self.mode == "joint":
            return tracked
        if self.mode == "pitch_only":
            return VoicingSmoother(config=self.config).smooth(tracked)
        # Unsupported bins may connect the path, but cannot become measured pitches.
        for pitch in tracked:
            if pitch is None:
                continue
            if not self._has_candidate_support(pitch):
                pitch.value = -1.0
                # Prevent the final voicing controller from reviving this frame.
                pitch.unvoiced_prob = 1.0
        return tracked

    def smooth_to_arrays(
        self,
        pitches: list[Pitch | None],
    ) -> tuple[FloatArray, FloatArray, BoolArray]:
        states = self.decode(pitches)
        times = np.asarray([
            float(pitch.time) if pitch is not None else np.nan
            for pitch in pitches
        ])
        midi = self.bin_midis[states % self.n_pitch_bins]
        voiced = states < self.n_pitch_bins
        return times, np.asarray(midi), np.asarray(voiced)

    def smooth_pitch_data(self, pitch_data):
        from app_logic.user.ds.PitchData import PitchData

        output = PitchData(config=pitch_data.config)
        output.load(self.smooth(list(pitch_data.data)))
        output.t_origin = pitch_data.t_origin
        return output

    def decode(self, pitches: list[Pitch | None]) -> npt.NDArray[np.uint16]:
        if self.mode == "joint":
            return self._decode_joint(pitches)
        # States encode unvoiced frames as n_pitch_bins + pitch_bin.
        states = np.full(len(pitches), self.n_pitch_bins, dtype=np.uint16)
        voiced = VoicingSmoother(config=self.config).decode(pitches)
        tracking = self._tracking_mask(voiced)
        edges = np.diff(np.r_[False, tracking, False].astype(np.int8))
        tiny = np.finfo(np.float64).tiny
        log_transition = np.log(self.transition + tiny)
        log_initial = np.log(self.initial + tiny)
        for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            observations, _ = self.observation_probabilities(pitches[start:stop])
            # Gap frames retain their time steps without favoring a pitch.
            observations[:, ~voiced[start:stop]] = 1.0 / self.n_pitch_bins
            states[start:stop] = _viterbi_0110(
                np.log(observations.T + tiny), log_transition, log_initial
            )
        states[tracking & ~voiced] += self.n_pitch_bins
        return states

    def observation_probabilities(
        self, pitches: list[Pitch | None],
    ) -> tuple[FloatArray, FloatArray]:
        if self.mode == "joint":
            return self._joint_observation_probabilities(pitches)
        observations = np.zeros((self.n_pitch_bins, len(pitches)))
        for frame, pitch in enumerate(pitches):
            if pitch is None:
                continue
            for midi, probability in pitch.candidate_pitches:
                observations[self._midi_to_bin(midi), frame] = probability
        mass = observations.sum(axis=0)
        np.divide(observations, mass, out=observations, where=mass[None, :] > 0)
        observations[:, mass == 0] = 1.0 / self.n_pitch_bins
        if self.confidence_emissions:
            confidence = np.asarray([
                np.clip(1.0 - pitch.unvoiced_prob, 0.0, 1.0)
                if pitch is not None else 0.0
                for pitch in pitches
            ])
            observations *= confidence[None, :]
            observations += (1.0 - confidence[None, :]) / self.n_pitch_bins
        return observations, mass

    def _smooth_pitch_path(
        self,
        pitches: list[Pitch | None],
        show_progress: bool = False,
        verbose: bool = False,
    ) -> list[Pitch | None]:
        if verbose:
            print("Tracking pitch... ", end="", flush=True)
        started = time.perf_counter()
        states = self.decode(pitches)
        output: list[Pitch | None] = []
        frames = zip(pitches, states)
        if show_progress:
            frames = tqdm(
                frames,
                total=len(pitches),
                desc="Tracking pitch",
                leave=False,
                mininterval=0.25,
            )
        for pitch, state in frames:
            if pitch is None:
                output.append(None)
                continue
            smoothed = copy.copy(pitch)
            smoothed.candidate_pitches = list(pitch.candidate_pitches)
            pitch_bin = int(state % self.n_pitch_bins)
            # Keep the hidden pitch bin; final voicing is a separate stage.
            smoothed.value = float(self.bin_midis[pitch_bin])
            smoothed.unvoiced_prob = float(pitch.unvoiced_prob)
            output.append(smoothed)
        if verbose:
            print(f"Done! Took {time.perf_counter() - started:.2f} sec.")
        return output

    def _tracking_mask(self, voiced: BoolArray) -> BoolArray:
        tracking = voiced.copy()
        edges = np.diff(np.r_[False, ~voiced, False].astype(np.int8))
        for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if (start > 0 and stop < len(voiced)
                    and (stop - start) * self.hop_length / self.sr
                    <= self.max_gap_seconds):
                tracking[start:stop] = True
        return tracking

    def _has_candidate_support(self, pitch: Pitch) -> bool:
        decoded_bin = self._midi_to_bin(pitch.value)
        return any(
            np.isfinite(probability) and probability > 0
            and self._midi_to_bin(midi) == decoded_bin
            for midi, probability in pitch.candidate_pitches
        )

    def _midi_to_bin(self, midi: float) -> int:
        index = int(np.round(
            (float(midi) - float(self.bin_midis[0]))
            * self.n_bins_per_semitone
        ))
        return int(np.clip(index, 0, self.n_pitch_bins - 1))

    def _transition_matrix(self) -> np.ndarray:
        pitch_transition = self._pitch_transition_matrix()
        if self.mode != "joint":
            return pitch_transition
        voicing_transition = np.asarray([
            [1.0 - self.switch_prob, self.switch_prob],
            [self.switch_prob, 1.0 - self.switch_prob],
        ])
        return np.kron(voicing_transition, pitch_transition)

    def _pitch_transition_matrix(self) -> np.ndarray:
        max_semitones_per_frame = round(
            self.max_transition_rate
            * 12
            * self.hop_length
            / self.sr
        )
        width = (
            max_semitones_per_frame * self.n_bins_per_semitone + 1
        )
        if width > self.n_pitch_bins:
            raise ValueError(
                f"transition width {width} exceeds pitch-bin count "
                f"{self.n_pitch_bins}"
            )

        pitch_transition = np.zeros(
            (self.n_pitch_bins, self.n_pitch_bins),
            dtype=np.float64,
        )
        window = get_window("triangle", width, fftbins=False)
        left_padding = (self.n_pitch_bins - width) // 2
        padded = np.pad(
            window,
            (left_padding, self.n_pitch_bins - width - left_padding),
        )
        for source in range(self.n_pitch_bins):
            row = np.roll(
                padded,
                self.n_pitch_bins // 2 + source + 1,
            )
            row[min(
                self.n_pitch_bins,
                source + width // 2 + 1,
            ):] = 0
            row[:max(0, source - width // 2)] = 0
            pitch_transition[source] = row
        pitch_transition /= pitch_transition.sum(axis=1, keepdims=True)

        return pitch_transition

    def _decode_joint(self, pitches: list[Pitch | None]) -> npt.NDArray[np.uint16]:
        if not pitches:
            return np.empty(0, dtype=np.uint16)
        observations, _ = self.observation_probabilities(pitches)
        tiny = np.finfo(observations.dtype).tiny
        return _viterbi_0110(
            np.log(observations.T + tiny),
            np.log(self.transition + tiny),
            np.log(self.initial + tiny),
        )

    def _joint_observation_probabilities(
        self,
        pitches: list[Pitch | None],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rebuild the joint HMM emissions from canonical pYIN observations."""
        observations = np.zeros(
            (2 * self.n_pitch_bins, len(pitches)),
            dtype=np.float64,
        )
        for frame, pitch in enumerate(pitches):
            if pitch is None:
                continue
            pitch.ensure_compatible(self.config)
            for midi, probability in pitch.candidate_pitches:
                observations[self._midi_to_bin(midi), frame] = float(probability)

        voiced_probability = np.clip(
            np.sum(observations[:self.n_pitch_bins], axis=0),
            0,
            1,
        )
        observations[self.n_pitch_bins:, :] = (
            1.0 - voiced_probability
        ) / self.n_pitch_bins
        return observations, voiced_probability


class VoicingSmoother:
    """Apply the promoted whole-track pYIN-confidence and p95 RMS controller."""

    def __init__(
        self,
        recording: Recording | None = None,
        config: Config | None = None,
    ) -> None:
        if recording is None and config is None:
            raise ValueError("VoicingSmoother requires a recording or config")
        self.recording = recording
        self.update_config(config if config is not None else recording.config)

    def update_config(self, config: Config) -> None:
        self.config = config

    def smooth(
        self,
        pitches: list[Pitch | None],
        show_progress: bool = False,
        verbose: bool = False,
    ) -> list[Pitch | None]:
        if verbose:
            print("Applying post-hoc voicing controller... ", end="", flush=True)
        started = time.perf_counter()
        voiced = self.decode(pitches)
        output: list[Pitch | None] = []
        frames = pitches
        if show_progress:
            frames = tqdm(
                frames,
                total=len(pitches),
                desc="Applying post-hoc voicing controller",
                leave=False,
                mininterval=0.25,
            )
        for pitch, is_voiced in zip(frames, voiced):
            if pitch is None:
                output.append(None)
                continue
            gated = copy.copy(pitch)
            if not is_voiced:
                gated.value = -1.0
                gated.unvoiced_prob = 1.0
            else:
                gated.unvoiced_prob = 0.0
            output.append(gated)
        if verbose:
            print(f"Done! Took {time.perf_counter() - started:.2f} sec.")
        return output

    def decode(self, pitches: list[Pitch | None]) -> BoolArray:
        """Return the global controller's voiced mask."""
        if not pitches:
            return np.empty(0, dtype=bool)
        volumes = np.asarray([
            max(0.0, float(pitch.volume))
            for pitch in pitches
            if pitch is not None
        ])
        threshold = self.volume_floor(volumes)
        eligible = np.asarray([
            bool(
                pitch is not None
                and pitch.candidate_pitches
                and max(0.0, float(pitch.volume)) >= threshold
            )
            for pitch in pitches
        ])
        allowed_by_confidence = np.asarray([
            bool(
                pitch is not None
                and float(pitch.unvoiced_prob)
                < float(self.config.posthoc_unv_thresh)
            )
            for pitch in pitches
        ])
        return allowed_by_confidence & eligible

    def volume_floor(self, volumes: npt.ArrayLike) -> float:
        """Return ``posthoc_min_volume * p95(RMS)`` for the recording."""
        values = np.asarray(volumes, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return 0.0
        reference = float(
            np.percentile(values, PYIN_GLOBAL_VOLUME_PERCENTILE)
        )
        return reference * max(0.0, float(self.config.posthoc_min_volume))
