"""Local librosa 0.11.0-compatible pYIN competitor implementation.

This module contains the complete split pYIN implementation used by the
single ``pyin`` benchmark row. The detector ports the observation frontend into
Attune's ``Pitch`` data model.  It intentionally performs no Attune preprocessing, prominent-peak
selection, range padding, or volume gating.  Those changes belong in later
ablations; this class is the unchanged reference implementation.

Portions derived from librosa, Copyright (c) 2013--2023, librosa development
team, under the ISC License:
https://github.com/librosa/librosa/blob/0.11.0/librosa/core/pitch.py
https://github.com/librosa/librosa/blob/0.11.0/librosa/sequence.py
https://github.com/librosa/librosa/blob/0.11.0/LICENSE.md
"""

from __future__ import annotations
from benchmarks.modules.pitch.sweeps.DecoupledVoicing import VoicingFeatures
from benchmarks.modules.pitch.sweeps.DecoupledVoicing import (
    PraatInspiredVoicingParameters,
)
from benchmarks.modules.pitch.sweeps.DecoupledVoicing import PraatInspiredVoicingDecoder
from app_logic.midi.ScoreData import ScoreData
from app_logic.NoteData import NoteData
from algorithms.Config import Config
from scipy.stats import boltzmann
from scipy.signal import find_peaks
from scipy import fft as scipy_fft
import copy
import numpy as np
from typing import Any
from pathlib import Path
from dataclasses import replace
import copy
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, TYPE_CHECKING
import numpy as np
import numpy.typing as npt
from scipy import fft as scipy_fft
from scipy.signal import get_window
from scipy.stats import beta as beta_distribution
from scipy.stats import boltzmann
from tqdm import tqdm
from algorithms.Config import Config
from app_logic.user.ds.PitchData import Pitch
from benchmarks.modules.pitch.PitchCache import PitchCache
import benchmarks.modules.pitch.PitchDetectorBase as _api_PitchDetectorBase
from benchmarks.modules.pitch.competitors.Attune import AttuneRealtime

if TYPE_CHECKING:
    from app_logic.user.ds.Recording import Recording
try:
    from numba import njit
except ImportError:

    class _NumbaFallback:

        @staticmethod
        def decorate(*_args, **_kwargs):

            def decorate(function):
                return function

            return decorate

    njit = _NumbaFallback.decorate
FloatArray = npt.NDArray[np.float64]


class PYINPitchDetector:
    """Produce the frame-local observations used by librosa 0.11.0 pYIN.

    ``detect_pitches`` returns one :class:`Pitch` per librosa frame. Candidate
    pitches are the non-zero voiced HMM bins and ``unvoiced_prob`` is one minus
    their total mass. :class:`PYINPitchSmoother` turns
    those observations into the final joint pitch/voicing path.
    """

    REFERENCE_SR = 22050
    REFERENCE_FRAME_LENGTH = 2048
    N_THRESHOLDS = 100
    BETA_PARAMETERS = (2.0, 18.0)
    BOLTZMANN_PARAMETER = 2.0
    RESOLUTION = 0.1
    NO_TROUGH_PROB = 0.01

    def __init__(
        self,
        recording: Recording | None = None,
        config: Config | None = None,
        *,
        frame_length: int | None = None,
        n_thresholds: int = N_THRESHOLDS,
        beta_parameters: tuple[float, float] = BETA_PARAMETERS,
        boltzmann_parameter: float = BOLTZMANN_PARAMETER,
        resolution: float = RESOLUTION,
        no_trough_prob: float = NO_TROUGH_PROB,
        center: bool = True,
        pad_mode: str | Callable = "constant",
    ) -> None:
        if recording is None and config is None:
            raise ValueError("PYINPitchDetector requires a recording or config")
        self.recording = recording
        self._frame_length_override = (
            None if frame_length is None else int(frame_length)
        )
        self.n_thresholds = int(n_thresholds)
        self.beta_parameters = tuple((float(value) for value in beta_parameters))
        self.boltzmann_parameter = float(boltzmann_parameter)
        self.resolution = float(resolution)
        self.no_trough_prob = float(no_trough_prob)
        self.center = bool(center)
        self.pad_mode = pad_mode
        self.load_config(config if config is not None else recording.config)

    def load_config(self, config: Config) -> None:
        """Rebuild the exact librosa period and pitch-bin geometry."""
        self.config = config
        self.SR = int(config.sr)
        self.frame_length = (
            int(round(self.REFERENCE_FRAME_LENGTH * self.SR / self.REFERENCE_SR))
            if self._frame_length_override is None
            else self._frame_length_override
        )
        self.HOP_SIZE = int(config.h1)
        self.FRAME_SIZE = self.frame_length
        self.INTEGRATION_SIZE = self.frame_length
        self.fmin = float(config.fmin)
        self.fmax = float(config.fmax)
        self._check_parameters()
        self.min_period = int(np.floor(self.SR / self.fmax))
        self.max_period = min(int(np.ceil(self.SR / self.fmin)), self.frame_length - 1)
        self.n_bins_per_semitone = int(np.ceil(1.0 / self.resolution))
        self.n_pitch_bins = (
            int(
                np.floor(12 * self.n_bins_per_semitone * np.log2(self.fmax / self.fmin))
            )
            + 1
        )
        self.bin_freqs = self.fmin * 2.0 ** (
            np.arange(self.n_pitch_bins, dtype=np.float64)
            / (12 * self.n_bins_per_semitone)
        )
        self.bin_midis = np.asarray(
            [self.config.freq_to_midi(freq) for freq in self.bin_freqs],
            dtype=np.float64,
        )
        self.thresholds = np.linspace(0.0, 1.0, self.n_thresholds + 1)
        beta_cdf = beta_distribution.cdf(
            self.thresholds, self.beta_parameters[0], self.beta_parameters[1]
        )
        self.beta_probs = np.diff(beta_cdf)

    def re_init(self, config: Config | None = None) -> None:
        if config is not None:
            self.load_config(config)

    def _check_parameters(self) -> None:
        if self.fmax > self.SR / 2:
            raise ValueError(
                f"fmax={self.fmax:.3f} cannot exceed Nyquist frequency {self.SR / 2}"
            )
        if self.fmin >= self.fmax:
            raise ValueError(
                f"fmin={self.fmin:.3f} must be less than fmax={self.fmax:.3f}"
            )
        if self.fmin <= 0:
            raise ValueError(f"fmin={self.fmin:.3f} must be strictly positive")
        if self.SR / self.fmin >= self.frame_length - 1:
            feasible_fmin = self.SR / (self.frame_length - 1)
            feasible_frame_length = int(np.ceil(self.SR / self.fmin) + 1)
            raise ValueError(
                f"fmin={self.fmin:.3f} is too small for frame_length={self.frame_length} and sr={self.SR}. Either increase to fmin={feasible_fmin:.3f} or frame_length={feasible_frame_length}"
            )
        if self.SR / self.fmin >= self.frame_length // 2:
            optimal_fmin = self.SR / (self.frame_length / 2)
            optimal_frame_length = int(np.ceil(self.SR / self.fmin) * 2 + 1)
            warnings.warn(
                f"With fmin={self.fmin:.3f}, sr={self.SR} and frame_length={self.frame_length}, less than two periods of fmin fit into the frame, which can cause inaccurate pitch detection. Consider increasing to fmin={optimal_fmin:.3f} or frame_length={optimal_frame_length}.",
                stacklevel=3,
            )

    @staticmethod
    def _frame_audio(
        audio: npt.ArrayLike,
        frame_length: int,
        hop_length: int,
        *,
        center: bool,
        pad_mode: str | Callable,
    ) -> np.ndarray:
        samples = np.asarray(audio)
        if samples.ndim != 1:
            raise ValueError("PYINPitchDetector currently supports mono audio only")
        if not np.issubdtype(samples.dtype, np.floating):
            raise ValueError("audio data must be floating-point")
        if not np.all(np.isfinite(samples)):
            raise ValueError("audio data must be finite")
        if center:
            samples = np.pad(
                samples, (frame_length // 2, frame_length // 2), mode=pad_mode
            )
        if samples.size < frame_length:
            raise ValueError(
                f"input is too short ({samples.size}) for frame_length={frame_length}"
            )
        return np.lib.stride_tricks.sliding_window_view(samples, frame_length)[
            ::hop_length
        ].T

    @staticmethod
    def _cumulative_mean_normalized_difference(
        frames: np.ndarray, min_period: int, max_period: int
    ) -> np.ndarray:
        """Port of librosa 0.11.0's YIN equation-8 implementation."""
        frame_length = frames.shape[-2]
        fft_size = scipy_fft.next_fast_len(2 * frame_length - 1, real=True)
        spectrum = scipy_fft.rfft(frames, n=fft_size, axis=-2)
        power_spectrum = spectrum.real**2 + spectrum.imag**2
        acf_frames = scipy_fft.irfft(power_spectrum, n=fft_size, axis=-2)[
            ..., : max_period + 1, :
        ]
        yin_frames = np.square(frames)
        np.cumsum(yin_frames, out=yin_frames, axis=-2)
        periods = slice(1, max_period + 1)
        yin_frames[..., 0, :] = 0
        yin_frames[..., periods, :] = (
            2 * (acf_frames[..., 0:1, :] - acf_frames[..., periods, :])
            - yin_frames[..., : periods.stop - 1, :]
        )
        yin_numerator = yin_frames[..., min_period : max_period + 1, :]
        period_range = np.arange(1, max_period + 1).reshape(
            (1,) * (yin_frames.ndim - 2) + (max_period, 1)
        )
        cumulative_mean = np.cumsum(yin_frames[..., periods, :], axis=-2) / period_range
        yin_denominator = cumulative_mean[..., min_period - 1 : max_period, :]
        tiny = np.finfo(yin_denominator.dtype).tiny
        return yin_numerator / (yin_denominator + tiny)

    @staticmethod
    def _parabolic_interpolation(values: np.ndarray) -> np.ndarray:
        """Return librosa's bounded, piecewise-parabolic bin shifts."""
        shifts = np.zeros_like(values)
        if values.shape[-2] <= 2:
            return shifts
        previous = values[..., :-2, :]
        current = values[..., 1:-1, :]
        following = values[..., 2:, :]
        a = following + previous - 2 * current
        b = (following - previous) / 2
        interior = shifts[..., 1:-1, :]
        np.divide(-b, a, out=interior, where=np.abs(b) < np.abs(a))
        return shifts

    @staticmethod
    def _localmin(values: np.ndarray) -> npt.NDArray[np.bool_]:
        """Match ``librosa.util.localmin(..., axis=-2)`` edge semantics."""
        local = np.zeros_like(values, dtype=bool)
        if values.shape[-2] == 1:
            return local
        local[..., 1:-1, :] = (values[..., 1:-1, :] < values[..., :-2, :]) & (
            values[..., 1:-1, :] <= values[..., 2:, :]
        )
        local[..., -1, :] = values[..., -1, :] < values[..., -2, :]
        return local

    def _pyin_probabilities(
        self, yin_frames: np.ndarray, parabolic_shifts: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Port of librosa 0.11.0's private ``__pyin_helper``."""
        yin_probs = np.zeros_like(yin_frames)
        for frame_index, yin_frame in enumerate(yin_frames.T):
            is_trough = self._localmin(yin_frame[:, np.newaxis])[:, 0]
            is_trough[0] = yin_frame[0] < yin_frame[1]
            (trough_index,) = np.nonzero(is_trough)
            if trough_index.size == 0:
                continue
            trough_heights = yin_frame[trough_index]
            trough_thresholds = np.less.outer(trough_heights, self.thresholds[1:])
            trough_positions = np.cumsum(trough_thresholds, axis=0) - 1
            n_troughs = np.count_nonzero(trough_thresholds, axis=0)
            trough_prior = boltzmann.pmf(
                trough_positions, self.boltzmann_parameter, n_troughs
            )
            trough_prior[~trough_thresholds] = 0
            probabilities = trough_prior.dot(self.beta_probs)
            global_minimum = int(np.argmin(trough_heights))
            thresholds_below_minimum = np.count_nonzero(
                ~trough_thresholds[global_minimum, :]
            )
            probabilities[global_minimum] += self.no_trough_prob * np.sum(
                self.beta_probs[:thresholds_below_minimum]
            )
            yin_probs[trough_index, frame_index] = probabilities
        yin_period, frame_index = np.nonzero(yin_probs)
        period_candidates = self.min_period + yin_period
        period_candidates = (
            period_candidates + parabolic_shifts[yin_period, frame_index]
        )
        f0_candidates = self.SR / period_candidates
        bin_index = 12 * self.n_bins_per_semitone * np.log2(f0_candidates / self.fmin)
        bin_index = np.clip(np.round(bin_index), 0, self.n_pitch_bins).astype(int)
        observations = np.zeros(
            (2 * self.n_pitch_bins, yin_frames.shape[1]), dtype=np.float64
        )
        observations[bin_index, frame_index] = yin_probs[yin_period, frame_index]
        voiced_probability = np.clip(
            np.sum(observations[: self.n_pitch_bins], axis=0), 0, 1
        )
        observations[self.n_pitch_bins :, :] = (
            1.0 - voiced_probability
        ) / self.n_pitch_bins
        return (observations[: self.n_pitch_bins], voiced_probability)

    def probabilities(
        self, audio: npt.ArrayLike, *, center: bool | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(voiced-bin observations, voiced probability)``."""
        frames = self._frame_audio(
            audio,
            self.frame_length,
            self.HOP_SIZE,
            center=self.center if center is None else bool(center),
            pad_mode=self.pad_mode,
        )
        yin_frames = self._cumulative_mean_normalized_difference(
            frames, self.min_period, self.max_period
        )
        parabolic_shifts = self._parabolic_interpolation(yin_frames)
        return self._pyin_probabilities(yin_frames, parabolic_shifts)

    def detect_pitches(
        self,
        audio: npt.ArrayLike,
        show_progress: bool = False,
        progress_desc: str = "Detecting pitches",
        verbose: bool = False,
    ) -> list[Pitch]:
        """Convert exact pYIN observations into Attune ``Pitch`` frames."""
        started = time.perf_counter()
        observations, voiced_probability = self.probabilities(audio)
        frame_count = observations.shape[1]
        frames = range(frame_count)
        if show_progress:
            frames = tqdm(
                frames,
                total=frame_count,
                desc=progress_desc,
                leave=False,
                mininterval=0.25,
            )
        pitches: list[Pitch] = []
        for frame_index in frames:
            nonzero = np.flatnonzero(observations[:, frame_index])
            candidates = [
                (
                    float(self.bin_midis[pitch_bin]),
                    float(observations[pitch_bin, frame_index]),
                )
                for pitch_bin in nonzero
            ]
            candidates.sort(key=lambda candidate: candidate[1], reverse=True)
            pitches.append(
                Pitch(
                    time=frame_index * self.HOP_SIZE / self.SR,
                    candidates=candidates,
                    volume=0.0,
                    unvoiced_prob=1.0 - float(voiced_probability[frame_index]),
                    live_distance=None,
                    config=self.config,
                )
            )
        if verbose:
            print(
                f"[PYINPitchDetector] done: {len(pitches)} pitch frame(s) in {time.perf_counter() - started:.2f}s",
                flush=True,
            )
        return pitches

    def detect_pitch(
        self, audio_frame: npt.ArrayLike, start_time: float | None = None
    ) -> Pitch:
        """Run the frontend on one unpadded analysis frame."""
        samples = np.asarray(audio_frame)
        if samples.size != self.frame_length:
            raise ValueError(
                f"PYINPitchDetector frame must contain {self.frame_length} samples; received {samples.size}"
            )
        observations, voiced_probability = self.probabilities(samples, center=False)
        nonzero = np.flatnonzero(observations[:, 0])
        candidates = [
            (float(self.bin_midis[index]), float(observations[index, 0]))
            for index in nonzero
        ]
        candidates.sort(key=lambda candidate: candidate[1], reverse=True)
        return Pitch(
            time=0.0 if start_time is None else float(start_time),
            candidates=candidates,
            volume=0.0,
            unvoiced_prob=1.0 - float(voiced_probability[0]),
            live_distance=None,
            config=self.config,
        )


class PYINPitchSmoother:
    """Decode ``PYINPitchDetector`` observations exactly as librosa 0.11.0."""

    RESOLUTION = 0.1
    MAX_TRANSITION_RATE = 35.92
    SWITCH_PROB = 0.01

    def __init__(
        self,
        recording: Recording | None = None,
        config: Config | None = None,
        *,
        resolution: float = RESOLUTION,
        max_transition_rate: float = MAX_TRANSITION_RATE,
        switch_prob: float = SWITCH_PROB,
    ) -> None:
        if recording is None and config is None:
            raise ValueError("PYINPitchSmoother requires a recording or config")
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
            int(
                np.floor(12 * self.n_bins_per_semitone * np.log2(self.fmax / self.fmin))
            )
            + 1
        )
        self.bin_freqs = self.fmin * 2.0 ** (
            np.arange(self.n_pitch_bins, dtype=np.float64)
            / (12 * self.n_bins_per_semitone)
        )
        self.bin_midis = np.asarray(
            [config.freq_to_midi(freq) for freq in self.bin_freqs], dtype=np.float64
        )
        self.transition = self._transition_matrix()
        self.initial = np.ones(2 * self.n_pitch_bins, dtype=np.float64)
        self.initial /= 2 * self.n_pitch_bins

    def _transition_matrix(self) -> np.ndarray:
        max_semitones_per_frame = round(
            self.max_transition_rate * 12 * self.hop_length / self.sr
        )
        width = max_semitones_per_frame * self.n_bins_per_semitone + 1
        if width > self.n_pitch_bins:
            raise ValueError(
                f"transition width {width} exceeds pitch-bin count {self.n_pitch_bins}"
            )
        pitch_transition = np.zeros(
            (self.n_pitch_bins, self.n_pitch_bins), dtype=np.float64
        )
        window = get_window("triangle", width, fftbins=False)
        left_padding = (self.n_pitch_bins - width) // 2
        padded = np.pad(
            window, (left_padding, self.n_pitch_bins - width - left_padding)
        )
        for source in range(self.n_pitch_bins):
            row = np.roll(padded, self.n_pitch_bins // 2 + source + 1)
            row[min(self.n_pitch_bins, source + width // 2 + 1) :] = 0
            row[: max(0, source - width // 2)] = 0
            pitch_transition[source] = row
        pitch_transition /= pitch_transition.sum(axis=1, keepdims=True)
        voicing_transition = np.asarray(
            [
                [1.0 - self.switch_prob, self.switch_prob],
                [self.switch_prob, 1.0 - self.switch_prob],
            ]
        )
        return np.kron(voicing_transition, pitch_transition)

    def _midi_to_bin(self, midi: float) -> int:
        index = int(
            np.round(
                (float(midi) - float(self.bin_midis[0])) * self.n_bins_per_semitone
            )
        )
        return int(np.clip(index, 0, self.n_pitch_bins - 1))

    def observation_probabilities(
        self, pitches: list[Pitch | None]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rebuild librosa's ``[voiced bins, unvoiced bins]`` emissions."""
        observations = np.zeros((2 * self.n_pitch_bins, len(pitches)), dtype=np.float64)
        for frame, pitch in enumerate(pitches):
            if pitch is None:
                continue
            for midi, probability in pitch.candidate_pitches:
                observations[self._midi_to_bin(midi), frame] = float(probability)
        voiced_probability = np.clip(
            np.sum(observations[: self.n_pitch_bins], axis=0), 0, 1
        )
        observations[self.n_pitch_bins :, :] = (
            1.0 - voiced_probability
        ) / self.n_pitch_bins
        return (observations, voiced_probability)

    def decode(self, pitches: list[Pitch | None]) -> npt.NDArray[np.uint16]:
        if not pitches:
            return np.empty(0, dtype=np.uint16)
        observations, _ = self.observation_probabilities(pitches)
        tiny = np.finfo(observations.dtype).tiny
        return PYIN._viterbi_0110(
            np.log(observations.T + tiny),
            np.log(self.transition + tiny),
            np.log(self.initial + tiny),
        )

    def smooth_to_arrays(
        self, pitches: list[Pitch | None]
    ) -> tuple[FloatArray, FloatArray, npt.NDArray[np.bool_]]:
        states = self.decode(pitches)
        times = np.asarray(
            [float(pitch.time) if pitch is not None else np.nan for pitch in pitches]
        )
        midi = self.bin_midis[states % self.n_pitch_bins]
        voiced = states < self.n_pitch_bins
        return (times, np.asarray(midi), np.asarray(voiced))

    def smooth(
        self,
        pitches: list[Pitch | None],
        show_progress: bool = False,
        verbose: bool = False,
    ) -> list[Pitch | None]:
        """Return copied frames carrying the exact decoded pYIN path."""
        started = time.perf_counter()
        states = self.decode(pitches)
        output: list[Pitch | None] = []
        frames = zip(pitches, states)
        if show_progress:
            frames = tqdm(
                frames,
                total=len(pitches),
                desc="Smoothing pitches",
                leave=False,
                mininterval=0.25,
            )
        for pitch, state in frames:
            if pitch is None:
                output.append(None)
                continue
            smoothed = copy.copy(pitch)
            pitch_bin = int(state % self.n_pitch_bins)
            if state < self.n_pitch_bins:
                smoothed.value = float(self.bin_midis[pitch_bin])
                smoothed.unvoiced_prob = 0.0
            else:
                smoothed.value = -1
                smoothed.unvoiced_prob = 1.0
            output.append(smoothed)
        if verbose:
            print(
                f"[PYINPitchSmoother] done: {len(output)} frame(s) in {time.perf_counter() - started:.2f}s",
                flush=True,
            )
        return output

    def smooth_pitch_data(self, pitch_data):
        from app_logic.user.ds.PitchData import PitchData

        output = PitchData(config=pitch_data.config)
        output.load(self.smooth(list(pitch_data.data)))
        output.t_origin = pitch_data.t_origin
        return output


class PYIN(AttuneRealtime):

    @staticmethod
    @njit(cache=True)
    def _viterbi_0110(
        log_probability: FloatArray, log_transition: FloatArray, log_initial: FloatArray
    ) -> npt.NDArray[np.uint16]:
        """Port of librosa 0.11.0's private Viterbi recurrence."""
        frame_count, state_count = log_probability.shape
        states = np.zeros(frame_count, dtype=np.uint16)
        values = np.zeros((frame_count, state_count), dtype=np.float64)
        pointers = np.zeros((frame_count, state_count), dtype=np.uint16)
        values[0] = log_probability[0] + log_initial
        for frame in range(1, frame_count):
            transition_out = values[frame - 1] + log_transition.T
            for destination in range(state_count):
                pointers[frame, destination] = np.argmax(transition_out[destination])
                values[frame, destination] = (
                    log_probability[frame, destination]
                    + transition_out[destination, pointers[frame, destination]]
                )
        states[-1] = np.argmax(values[-1])
        for frame in range(frame_count - 2, -1, -1):
            states[frame] = pointers[frame + 1, states[frame + 1]]
        return states

    "The complete local librosa 0.11.0 pYIN benchmark implementation."
    name = "pyin"
    description = "Local librosa-compatible pYIN at Attune's 44.1 kHz rate"
    input_sr = 44100
    smooth: ClassVar[bool] = True
    FRAME_LENGTH = 4096
    CACHE_TAG = "pyin_local_librosa_0p11p0_44100_4096_128_v2"
    DEFAULT_CONFIG = Config(
        sr=44100,
        w1=FRAME_LENGTH,
        h1=128,
        fmin=196.0,
        fmax=3000.0,
        tuning=440.0,
        unv_thresh=0.5,
        min_volume=0.0,
    )

    def config_for(self, fmin: float, fmax: float, **overrides: Any) -> Config:
        """Use Attune's rate and librosa's equivalent physical frame grid."""
        overrides = {
            "sr": self.DEFAULT_CONFIG.sr,
            "w1": self.FRAME_LENGTH,
            "h1": self.DEFAULT_CONFIG.h1,
            **overrides,
        }
        return super().config_for(fmin, fmax, **overrides)

    @classmethod
    def variant_cache_path(cls, path: Path | str) -> Path:
        source = Path(path)
        marker = ".pitch.pkl.xz"
        if source.name.endswith(marker):
            prefix = source.name[: -len(marker)]
            name = f"{prefix}.{cls.CACHE_TAG}{marker}"
        else:
            name = f"{source.name}.{cls.CACHE_TAG}"
        return source.with_name(name)

    def cache(self, example: PitchExample) -> PitchCache:
        return PitchCache(self.variant_cache_path(example.stage_cache_path))

    def has_cache(self, dataset: Any, track: Any) -> bool:
        cache_path = self.variant_cache_path(dataset.pitch_cache_path(track))
        return PitchCache(cache_path).has_current_timing(self.stage, self.COMPUTE_CLOCK)

    def recording_for(self, config: Config, score_notes=None, score_data=None):
        recording = super().recording_for(
            config, score_notes=score_notes, score_data=score_data
        )
        recording.pitch_detector = PYINPitchDetector(recording=recording)
        recording.pitch_smoother = PYINPitchSmoother(recording=recording)
        recording.voicing_smoother = _PassthroughSmoother(recording.config)
        return recording

    @staticmethod
    def _frequency_for_pitch(pitch: Pitch, config: Config) -> float:
        if pitch.value == -1 or pitch.unvoiced_prob >= config.unv_thresh:
            return 0.0
        bins_per_semitone = int(np.ceil(1.0 / PYINPitchSmoother.RESOLUTION))
        base_midi = float(config.freq_to_midi(config.fmin))
        pitch_bin = int(np.round((float(pitch.value) - base_midi) * bins_per_semitone))
        return float(config.fmin) * 2.0 ** (pitch_bin / (12 * bins_per_semitone))

    @classmethod
    def melody(cls, pitch_data, config: Config):
        """Map decoded bins to Hz with librosa's exact frequency formula."""
        times: list[float] = []
        frequencies: list[float] = []
        for pitch in pitch_data.data:
            if pitch is None:
                continue
            times.append(float(pitch.time))
            frequencies.append(cls._frequency_for_pitch(pitch, config))
        return (
            np.asarray(times, dtype=np.float64),
            np.asarray(frequencies, dtype=np.float64),
        )

    def predict_frame(
        self, audio, sr: int, fmin: float, fmax: float, hop_length: int
    ) -> float:
        """Use the reference one-frame joint-state decision, without an extra gate.

        With uniform initialization and one frame, Viterbi selects the largest
        observation probability. Unvoiced mass is spread over the pitch bins;
        it is not compared against total voiced mass at a 0.5 cutoff.
        """
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size != self.FRAME_LENGTH:
            raise ValueError(
                f"pYIN frame must contain {self.FRAME_LENGTH} samples; received {samples.size}"
            )
        config = self.config_for(
            fmin, fmax, sr=int(sr), h1=int(hop_length), w1=self.FRAME_LENGTH
        )
        detector = PYINPitchDetector(
            config=config, frame_length=self.FRAME_LENGTH, center=False
        )
        observations, voiced_probability = detector.probabilities(samples)
        pitch_bin = int(np.argmax(observations[:, 0]))
        unvoiced_bin_probability = (
            1.0 - float(voiced_probability[0])
        ) / detector.n_pitch_bins
        if observations[pitch_bin, 0] < unvoiced_bin_probability:
            return 0.0
        return float(detector.bin_freqs[pitch_bin])


class _PassthroughSmoother:
    """Keep Attune's two-hook adapter after pYIN jointly decodes voicing."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def update_config(self, config: Config) -> None:
        self.config = config

    @staticmethod
    def smooth(pitches, **_kwargs):
        return pitches


class PYINAdditionDetector(PYINPitchDetector):
    """Faithful pYIN frontend with optional prominent ACF peak picking.

    Centered frame RMS is retained for the volume-gate variants, but it does
    not influence pYIN's observations or joint HMM.
    """

    def __init__(self, *, prominence: bool, **kwargs: Any) -> None:
        self.use_prominence = bool(prominence)
        self._last_volumes = np.empty(0, dtype=np.float64)
        super().__init__(**kwargs)

    @staticmethod
    def _autocorrelation_frames(frames: np.ndarray, max_period: int) -> np.ndarray:
        frame_length = frames.shape[-2]
        fft_size = scipy_fft.next_fast_len(2 * frame_length - 1, real=True)
        spectrum = scipy_fft.rfft(frames, n=fft_size, axis=-2)
        power = spectrum.real**2 + spectrum.imag**2
        return scipy_fft.irfft(power, n=fft_size, axis=-2)[..., : max_period + 1, :]

    def _prominent_peak_indices(self, acf_frame: np.ndarray) -> np.ndarray:
        """Return in-range ACF peaks on Attune's five-pass prominence rule."""
        base_prominence = abs(
            (float(np.max(acf_frame)) - float(np.min(acf_frame))) / 2.0
        )
        for pass_index in range(5):
            threshold = base_prominence * (1.0 - pass_index / 5.0)
            peaks, _ = find_peaks(acf_frame, prominence=threshold)
            peaks = peaks[(peaks >= self.min_period) & (peaks <= self.max_period)]
            if peaks.size:
                return peaks - self.min_period
        region = acf_frame[self.min_period : self.max_period + 1]
        return np.asarray([int(np.argmax(region))], dtype=int)

    def _prominent_pyin_probabilities(
        self,
        yin_frames: np.ndarray,
        parabolic_shifts: np.ndarray,
        acf_frames: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reference pYIN probabilities with only its candidate set changed."""
        yin_probs = np.zeros_like(yin_frames)
        for frame_index, yin_frame in enumerate(yin_frames.T):
            trough_index = self._prominent_peak_indices(acf_frames[:, frame_index])
            trough_heights = yin_frame[trough_index]
            trough_thresholds = np.less.outer(trough_heights, self.thresholds[1:])
            trough_positions = np.cumsum(trough_thresholds, axis=0) - 1
            n_troughs = np.count_nonzero(trough_thresholds, axis=0)
            trough_prior = boltzmann.pmf(
                trough_positions, self.boltzmann_parameter, n_troughs
            )
            trough_prior[~trough_thresholds] = 0
            probabilities = trough_prior.dot(self.beta_probs)
            global_minimum = int(np.argmin(trough_heights))
            thresholds_below_minimum = np.count_nonzero(
                ~trough_thresholds[global_minimum, :]
            )
            probabilities[global_minimum] += self.no_trough_prob * np.sum(
                self.beta_probs[:thresholds_below_minimum]
            )
            yin_probs[trough_index, frame_index] = probabilities
        yin_period, frame_index = np.nonzero(yin_probs)
        period_candidates = self.min_period + yin_period
        period_candidates = (
            period_candidates + parabolic_shifts[yin_period, frame_index]
        )
        f0_candidates = self.SR / period_candidates
        bin_index = 12 * self.n_bins_per_semitone * np.log2(f0_candidates / self.fmin)
        bin_index = np.clip(np.round(bin_index), 0, self.n_pitch_bins).astype(int)
        observations = np.zeros(
            (2 * self.n_pitch_bins, yin_frames.shape[1]), dtype=np.float64
        )
        observations[bin_index, frame_index] = yin_probs[yin_period, frame_index]
        voiced_probability = np.clip(
            np.sum(observations[: self.n_pitch_bins], axis=0), 0, 1
        )
        observations[self.n_pitch_bins :, :] = (
            1.0 - voiced_probability
        ) / self.n_pitch_bins
        return (observations[: self.n_pitch_bins], voiced_probability)

    def probabilities(
        self, audio: Any, *, center: bool | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        frames = self._frame_audio(
            audio,
            self.frame_length,
            self.HOP_SIZE,
            center=self.center if center is None else bool(center),
            pad_mode=self.pad_mode,
        )
        centered = frames - np.mean(frames, axis=-2, keepdims=True)
        self._last_volumes = np.sqrt(np.mean(np.square(centered), axis=-2)).reshape(-1)
        yin_frames = self._cumulative_mean_normalized_difference(
            frames, self.min_period, self.max_period
        )
        parabolic_shifts = self._parabolic_interpolation(yin_frames)
        if not self.use_prominence:
            return self._pyin_probabilities(yin_frames, parabolic_shifts)
        acf_frames = self._autocorrelation_frames(frames, self.max_period)
        return self._prominent_pyin_probabilities(
            yin_frames, parabolic_shifts, acf_frames
        )

    def detect_pitches(self, audio: Any, **kwargs: Any):
        pitches = super().detect_pitches(audio, **kwargs)
        if len(pitches) != len(self._last_volumes):
            raise AssertionError("pYIN volume grid drifted from its pitch grid")
        for pitch, volume in zip(pitches, self._last_volumes):
            pitch.volume = float(volume)
        return pitches

    def detect_pitch(self, audio_frame: Any, start_time: float | None = None):
        pitch = super().detect_pitch(audio_frame, start_time=start_time)
        if self._last_volumes.size != 1:
            raise AssertionError("single-frame pYIN did not produce one RMS value")
        pitch.volume = float(self._last_volumes[0])
        return pitch


class PYINGlobalVolumeGateSmoother(PYINPitchSmoother):
    """Apply a whole-track RMS gate after the unchanged joint pYIN HMM."""

    def __init__(
        self, *, relative_floor: float, ceiling_percentile: float, **kwargs: Any
    ) -> None:
        self.relative_floor = float(relative_floor)
        self.ceiling_percentile = float(ceiling_percentile)
        super().__init__(**kwargs)

    def volume_threshold(self, pitches: list[Any]) -> tuple[float, float]:
        volumes = np.asarray(
            [max(0.0, float(pitch.volume)) for pitch in pitches if pitch is not None],
            dtype=np.float64,
        )
        ceiling = (
            float(np.percentile(volumes, self.ceiling_percentile))
            if volumes.size
            else 0.0
        )
        return (ceiling * self.relative_floor, ceiling)

    def smooth(self, pitches: list[Any], **kwargs: Any):
        smoothed = super().smooth(pitches, **kwargs)
        threshold, _ = self.volume_threshold(pitches)
        if threshold <= 0.0:
            return smoothed
        for pitch in smoothed:
            if pitch is not None and float(pitch.volume) < threshold:
                pitch.value = -1
                pitch.unvoiced_prob = 1.0
        return smoothed


class PYINPraatStyleVoicingSmoother(PYINPitchSmoother):
    """Use one global Praat-shaped V/U path over pYIN and volume evidence.

    The ordinary pYIN joint HMM still supplies the pitch-bin path.  Its voiced
    state is replaced by a separate whole-track path whose local evidence is
    aggregate pYIN periodicity and whose hard silence eligibility is the same
    p95-relative RMS floor used by :class:`PYINGlobalVolumeGateSmoother`.
    This isolates the value of integrating those two voicing cues globally.
    """

    def __init__(
        self,
        *,
        relative_floor: float,
        ceiling_percentile: float,
        unvoiced_threshold: float,
        switch_cost: float,
        **kwargs: Any,
    ) -> None:
        self.relative_floor = float(relative_floor)
        self.ceiling_percentile = float(ceiling_percentile)
        self.unvoiced_threshold = float(unvoiced_threshold)
        self.switch_cost = float(switch_cost)
        super().__init__(**kwargs)

    def _controller_mask(self, pitches: list[Any]) -> np.ndarray:
        features = VoicingFeatures.from_pitches(pitches)
        present_volumes = features.volumes[features.present]
        ceiling = (
            float(np.percentile(present_volumes, self.ceiling_percentile))
            if present_volumes.size
            else 0.0
        )
        controlled_features = replace(
            features, volumes=np.minimum(features.volumes, ceiling)
        )
        decoder = PraatInspiredVoicingDecoder(
            PraatInspiredVoicingParameters(
                candidate_strength_threshold=1.0 - self.unvoiced_threshold,
                silence_threshold=self.relative_floor,
                absolute_volume_floor_dbfs=None,
                voiced_unvoiced_cost=self.switch_cost,
            ),
            time_step_seconds=self.hop_length / self.sr,
        )
        return decoder.decode(controlled_features)

    def smooth(
        self, pitches: list[Any], show_progress: bool = False, verbose: bool = False
    ) -> list[Any]:
        states = self.decode(pitches)
        voiced = self._controller_mask(pitches)
        output: list[Any] = []
        for pitch, state, is_voiced in zip(pitches, states, voiced):
            if pitch is None:
                output.append(None)
                continue
            smoothed = copy.copy(pitch)
            pitch_bin = int(state % self.n_pitch_bins)
            if is_voiced:
                smoothed.value = float(self.bin_midis[pitch_bin])
                smoothed.unvoiced_prob = 0.0
            else:
                smoothed.value = -1
                smoothed.unvoiced_prob = 1.0
            output.append(smoothed)
        return output


class PYINAblationAdapter(PYIN):
    """Keep the reference pYIN adapter/HMM and add only selected features."""

    def __init__(
        self,
        variant: Variant,
        run_root: Path,
        *,
        volume_floor_ratio: float,
        volume_ceiling_percentile: float,
        unvoiced_threshold: float,
        praat_switch_cost: float,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.run_root = Path(run_root)
        self.volume_floor_ratio = float(volume_floor_ratio)
        self.volume_ceiling_percentile = float(volume_ceiling_percentile)
        self.unvoiced_threshold = float(unvoiced_threshold)
        self.praat_switch_cost = float(praat_switch_cost)

    def recording_for(
        self,
        config: Config,
        score_notes: NoteData | None = None,
        score_data: ScoreData | None = None,
    ):
        recording = super().recording_for(config, score_notes, score_data)
        if self.variant.prominence or self.variant.volume_gate:
            recording.pitch_detector = PYINAdditionDetector(
                recording=recording, prominence=self.variant.prominence
            )
        if self.variant.praat_controller:
            recording.pitch_smoother = PYINPraatStyleVoicingSmoother(
                recording=recording,
                relative_floor=self.volume_floor_ratio,
                ceiling_percentile=self.volume_ceiling_percentile,
                unvoiced_threshold=self.unvoiced_threshold,
                switch_cost=self.praat_switch_cost,
            )
        elif self.variant.volume_gate:
            recording.pitch_smoother = PYINGlobalVolumeGateSmoother(
                recording=recording,
                relative_floor=self.volume_floor_ratio,
                ceiling_percentile=self.volume_ceiling_percentile,
            )
        return recording

    def cache(self, example: PitchExample) -> PitchCache:
        from benchmarks.modules.pitch.PitchCache import PitchCache

        condition = (
            example.degradation.cache_tag
            if example.degradation is not None
            else "clean"
        )
        path = (
            self.run_root
            / "pitch_data"
            / self.variant.name
            / condition
            / f"{example.dataset}__{example.safe_id}.pitch.pkl.xz"
        )
        return PitchCache(path)
