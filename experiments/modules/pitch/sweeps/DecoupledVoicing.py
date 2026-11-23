"""Archived pitch/voicing-decoupling sweep implementation.

This module preserves the parameter-search machinery that selected Attune's
current split pitch and voicing smoothers. It remains benchmark-only:

* :class:`BandedPitchDecoder` runs one pitch-only Viterbi pass conditional on
  the frame being voiced.  Its observations depend on candidate *ratios*, not
  aggregate candidate mass, unvoiced probability, or volume.
* :class:`TwoStateVoicingDecoder` then makes the voiced/unvoiced decision from
    pYIN periodicity and explicit relative/absolute volume floors.  A parameter
  sweep is O(T) per setting and never reruns the pitch HMM.
* :class:`PraatInspiredVoicingDecoder` applies Praat's broad path-finding
  methodology to those same cached pYIN frames: candidate strength competes
  with a voicing threshold, quiet frames are forbidden, and state changes pay
  a global voiced/unvoiced transition cost.  It does not invoke Praat or
  compute another autocorrelation.

The application owns its concise production copies in
``algorithms/PitchSmoother.py``; these configurable versions exist only to
reproduce or extend the sweep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from algorithms.Config import Config, guarded_pyin_frequency
from app_logic.user.ds.PitchData import Pitch


FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


try:
    from numba import njit
except ImportError:  # pragma: no cover - librosa currently installs numba

    def njit(*_args, **_kwargs):
        def decorate(function):
            return function

        return decorate


@njit(cache=True)
def _decode_banded_path(
    log_observations: FloatArray,
    max_jump_bins: int,
) -> npt.NDArray[np.int32]:
    """Viterbi over a triangular band without constructing a dense HMM.

    The production transition is row-normalized, so edge source states have a
    slightly different normalization constant.  Keeping that detail here
    makes the pitch transition identical to PitchSmoother's triangular block
    while reducing work from O(T*M^2) to O(T*M*K).
    """
    frame_count, bin_count = log_observations.shape
    if frame_count == 0 or bin_count == 0:
        return np.empty(0, dtype=np.int32)

    row_norm = np.empty(bin_count, dtype=np.float64)
    for source in range(bin_count):
        total = 0.0
        destination_lo = max(0, source - max_jump_bins)
        destination_hi = min(bin_count, source + max_jump_bins + 1)
        for destination in range(destination_lo, destination_hi):
            total += max_jump_bins + 1 - abs(destination - source)
        row_norm[source] = total

    backpointers = np.empty((frame_count, bin_count), dtype=np.int32)
    backpointers[0, :] = -1
    previous = log_observations[0].copy() - np.log(float(bin_count))
    current = np.empty(bin_count, dtype=np.float64)

    for frame in range(1, frame_count):
        for destination in range(bin_count):
            source_lo = max(0, destination - max_jump_bins)
            source_hi = min(bin_count, destination + max_jump_bins + 1)
            best_score = -np.inf
            best_source = source_lo
            for source in range(source_lo, source_hi):
                weight = max_jump_bins + 1 - abs(destination - source)
                score = previous[source] + np.log(weight / row_norm[source])
                if score > best_score:
                    best_score = score
                    best_source = source
            current[destination] = best_score + log_observations[frame, destination]
            backpointers[frame, destination] = best_source
        swap = previous
        previous = current
        current = swap

    states = np.empty(frame_count, dtype=np.int32)
    states[-1] = int(np.argmax(previous))
    for frame in range(frame_count - 1, 0, -1):
        states[frame - 1] = backpointers[frame, states[frame]]
    return states


@njit(cache=True)
def _decode_two_state_path(
    voiced_emission: FloatArray,
    onset_probability: float,
    offset_probability: float,
) -> BoolArray:
    """Two-state Viterbi inner loop, compiled for large parameter sweeps."""
    frame_count = voiced_emission.size
    voiced = np.zeros(frame_count, dtype=np.bool_)
    if frame_count == 0:
        return voiced

    log_uu = np.log(1.0 - onset_probability)
    log_uv = np.log(onset_probability)
    log_vu = np.log(offset_probability)
    log_vv = np.log(1.0 - offset_probability)
    previous_unvoiced = 0.0
    previous_voiced = -np.inf
    backpointers = np.zeros((frame_count, 2), dtype=np.int8)

    for frame in range(frame_count):
        from_u_to_u = previous_unvoiced + log_uu
        from_v_to_u = previous_voiced + log_vu
        if from_u_to_u >= from_v_to_u:
            current_unvoiced = from_u_to_u
            backpointers[frame, 0] = 0
        else:
            current_unvoiced = from_v_to_u
            backpointers[frame, 0] = 1

        from_u_to_v = previous_unvoiced + log_uv
        from_v_to_v = previous_voiced + log_vv
        if from_u_to_v >= from_v_to_v:
            current_voiced = from_u_to_v + voiced_emission[frame]
            backpointers[frame, 1] = 0
        else:
            current_voiced = from_v_to_v + voiced_emission[frame]
            backpointers[frame, 1] = 1
        previous_unvoiced = current_unvoiced
        previous_voiced = current_voiced

    state = 0 if previous_unvoiced >= previous_voiced else 1
    for frame in range(frame_count - 1, -1, -1):
        voiced[frame] = state == 1
        state = int(backpointers[frame, state])
    return voiced


@njit(cache=True)
def _decode_switch_cost_path(
    voiced_advantage: FloatArray,
    switch_cost: float,
) -> BoolArray:
    """Two-state maximum-score path with an additive state-change cost.

    ``voiced_advantage`` is the local score of voiced relative to unvoiced.
    This deliberately mirrors the shape of Praat's path problem rather than
    treating the two states as a Markov chain with fitted probabilities.
    """
    frame_count = voiced_advantage.size
    voiced = np.zeros(frame_count, dtype=np.bool_)
    if frame_count == 0:
        return voiced

    previous_unvoiced = 0.0
    previous_voiced = voiced_advantage[0]
    backpointers = np.zeros((frame_count, 2), dtype=np.int8)
    backpointers[0, :] = -1

    for frame in range(1, frame_count):
        stay_unvoiced = previous_unvoiced
        leave_voiced = previous_voiced - switch_cost
        if stay_unvoiced >= leave_voiced:
            current_unvoiced = stay_unvoiced
            backpointers[frame, 0] = 0
        else:
            current_unvoiced = leave_voiced
            backpointers[frame, 0] = 1

        enter_voiced = previous_unvoiced - switch_cost
        stay_voiced = previous_voiced
        if enter_voiced >= stay_voiced:
            current_voiced = enter_voiced + voiced_advantage[frame]
            backpointers[frame, 1] = 0
        else:
            current_voiced = stay_voiced + voiced_advantage[frame]
            backpointers[frame, 1] = 1
        previous_unvoiced = current_unvoiced
        previous_voiced = current_voiced

    state = 0 if previous_unvoiced >= previous_voiced else 1
    for frame in range(frame_count - 1, -1, -1):
        voiced[frame] = state == 1
        if frame > 0:
            state = int(backpointers[frame, state])
    return voiced


@dataclass(frozen=True)
class PitchPath:
    """A pitch estimate for every input frame, independent of voicing."""

    times: FloatArray
    midi: FloatArray
    present: BoolArray


class BandedPitchDecoder:
    """Pitch-only pYIN candidate decoder with a sparse triangular transition."""

    def __init__(
        self,
        config: Config,
        *,
        resolution_semitones: float = 0.1,
        max_jump_bins: int = 9,
        observation_floor: float = 1e-12,
    ) -> None:
        if resolution_semitones <= 0.0:
            raise ValueError("resolution_semitones must be positive")
        if max_jump_bins < 1:
            raise ValueError("max_jump_bins must be positive")
        if not 0.0 < observation_floor < 1.0:
            raise ValueError("observation_floor must be between zero and one")

        self.config = config
        self.resolution = float(resolution_semitones)
        self.max_jump_bins = int(max_jump_bins)
        self.observation_floor = float(observation_floor)

        guarded_fmin = guarded_pyin_frequency(config.fmin, lower=True)
        guarded_fmax = guarded_pyin_frequency(config.fmax, lower=False)
        midi_lo = config.freq_to_midi(guarded_fmin)
        midi_hi = config.freq_to_midi(guarded_fmax)
        self.midi_min = np.floor(midi_lo / self.resolution) * self.resolution
        midi_max = np.ceil(midi_hi / self.resolution) * self.resolution
        self.n_bins = int(round((midi_max - self.midi_min) / self.resolution)) + 1
        self.bin_midis = self.midi_min + self.resolution * np.arange(self.n_bins)

    def _midi_to_bin(self, midi: float) -> int | None:
        index = int(round((float(midi) - self.midi_min) / self.resolution))
        return index if 0 <= index < self.n_bins else None

    def observation_logprobs(self, pitches: list[Pitch | None]) -> FloatArray:
        """Candidate likelihood conditional on voicing.

        Normalizing each frame's candidate mass is the key decoupling step:
        total candidate mass remains available to the voicing decoder, but it
        cannot alter this pitch path.  Candidate-free frames receive a uniform
        observation so the transition carries pitch memory through rests.
        """
        observations = np.full(
            (len(pitches), self.n_bins),
            self.observation_floor,
            dtype=np.float64,
        )
        for frame, pitch in enumerate(pitches):
            if pitch is None:
                observations[frame, :] = 1.0 / self.n_bins
                continue
            mass = 0.0
            for midi, probability in pitch.candidate_pitches:
                pitch_bin = self._midi_to_bin(midi)
                probability = max(0.0, float(probability))
                if pitch_bin is None or probability == 0.0:
                    continue
                observations[frame, pitch_bin] += probability
                mass += probability
            if mass == 0.0:
                observations[frame, :] = 1.0 / self.n_bins
            else:
                observations[frame, :] /= observations[frame, :].sum()
        return np.log(observations)

    def decode(self, pitches: list[Pitch | None]) -> PitchPath:
        if not pitches:
            return PitchPath(
                times=np.empty(0, dtype=np.float64),
                midi=np.empty(0, dtype=np.float64),
                present=np.empty(0, dtype=bool),
            )
        states = _decode_banded_path(
            self.observation_logprobs(pitches),
            self.max_jump_bins,
        )
        times = np.asarray(
            [float(pitch.time) if pitch is not None else np.nan for pitch in pitches],
            dtype=np.float64,
        )
        return PitchPath(
            times=times,
            midi=np.asarray(self.bin_midis[states], dtype=np.float64),
            present=np.asarray([pitch is not None for pitch in pitches], dtype=bool),
        )


@dataclass(frozen=True)
class VoicingParameters:
    """Independent pYIN periodicity, gain, and temporal voicing controls."""

    max_unvoiced_prob: float = 0.85
    relative_volume_floor: float = 0.02
    absolute_volume_floor_dbfs: float | None = None
    onset_probability: float = 0.01
    offset_probability: float = 0.01

    def __post_init__(self) -> None:
        if not 0.0 < self.max_unvoiced_prob < 1.0:
            raise ValueError("max_unvoiced_prob must be between zero and one")
        if self.relative_volume_floor < 0.0:
            raise ValueError("relative_volume_floor cannot be negative")
        if not 0.0 < self.onset_probability < 1.0:
            raise ValueError("onset_probability must be between zero and one")
        if not 0.0 < self.offset_probability < 1.0:
            raise ValueError("offset_probability must be between zero and one")


@dataclass(frozen=True)
class VoicingFeatures:
    """Frame evidence extracted once and reused across a parameter sweep."""

    periodicity: FloatArray
    volumes: FloatArray
    has_candidates: BoolArray
    present: BoolArray

    @classmethod
    def from_pitches(cls, pitches: list[Pitch | None]) -> "VoicingFeatures":
        periodicity = np.zeros(len(pitches), dtype=np.float64)
        volumes = np.zeros(len(pitches), dtype=np.float64)
        has_candidates = np.zeros(len(pitches), dtype=bool)
        present = np.zeros(len(pitches), dtype=bool)
        for frame, pitch in enumerate(pitches):
            if pitch is None:
                continue
            present[frame] = True
            periodicity[frame] = np.clip(
                1.0 - float(pitch.unvoiced_prob),
                0.0,
                1.0,
            )
            volumes[frame] = max(0.0, float(pitch.volume))
            has_candidates[frame] = bool(pitch.candidate_pitches)
        return cls(periodicity, volumes, has_candidates, present)


class TwoStateVoicingDecoder:
    """O(T) Viterbi voicing based on pYIN periodicity and hard gain floors."""

    def __init__(self, parameters: VoicingParameters | None = None) -> None:
        self.parameters = parameters or VoicingParameters()

    def volume_floor(self, features: VoicingFeatures) -> float:
        relative = (
            float(np.max(features.volumes, initial=0.0))
            * self.parameters.relative_volume_floor
        )
        absolute = (
            0.0
            if self.parameters.absolute_volume_floor_dbfs is None
            else 10.0 ** (float(self.parameters.absolute_volume_floor_dbfs) / 20.0)
        )
        return max(relative, absolute)

    def eligible(self, features: VoicingFeatures) -> BoolArray:
        minimum_periodicity = 1.0 - self.parameters.max_unvoiced_prob
        return (
            features.present
            & features.has_candidates
            & (features.periodicity >= minimum_periodicity)
            & (features.volumes >= self.volume_floor(features))
        )

    def decode(self, features: VoicingFeatures) -> BoolArray:
        frame_count = features.periodicity.size
        if frame_count == 0:
            return np.empty(0, dtype=bool)
        if not (
            features.volumes.size
            == features.has_candidates.size
            == features.present.size
            == frame_count
        ):
            raise ValueError("voicing feature arrays must have equal lengths")

        params = self.parameters
        eligible = self.eligible(features)
        epsilon = np.finfo(np.float64).eps
        periodicity = np.clip(features.periodicity, epsilon, 1.0 - epsilon)
        threshold = np.clip(
            1.0 - params.max_unvoiced_prob,
            epsilon,
            1.0 - epsilon,
        )
        # Center the emission log odds on the explicit periodicity floor.  A
        # frame exactly at the floor is neutral; stronger periodicity supports
        # voiced and weaker periodicity is already forbidden by the hard gate.
        voiced_emission = (
            np.log(periodicity)
            - np.log1p(-periodicity)
            - np.log(threshold)
            + np.log1p(-threshold)
        )
        voiced_emission[~eligible] = -np.inf

        voiced = _decode_two_state_path(
            voiced_emission,
            params.onset_probability,
            params.offset_probability,
        )
        # This assertion protects future emission changes from accidentally
        # turning a hard user-requested floor into a soft preference.
        voiced &= eligible
        return voiced


@dataclass(frozen=True)
class PraatInspiredVoicingParameters:
    """Praat-shaped costs calibrated to Attune's cached pYIN evidence.

    Praat's raw-autocorrelation candidate strength and Attune's pYIN candidate
    mass are not numerically interchangeable.  ``candidate_strength_threshold``
    is therefore intentionally calibrated on pYIN's aggregate candidate mass,
    which is already stored as ``1 - unvoiced_prob``.  As in Praat,
    ``voiced_unvoiced_cost`` is specified on a 10-ms reference grid and scaled
    to the actual frame step by the decoder.
    """

    candidate_strength_threshold: float = 0.25
    silence_threshold: float = 0.01
    absolute_volume_floor_dbfs: float | None = -54.0
    voiced_unvoiced_cost: float = 0.02

    def __post_init__(self) -> None:
        if not 0.0 < self.candidate_strength_threshold < 1.0:
            raise ValueError(
                "candidate_strength_threshold must be between zero and one"
            )
        if self.silence_threshold < 0.0:
            raise ValueError("silence_threshold cannot be negative")
        if self.voiced_unvoiced_cost < 0.0:
            raise ValueError("voiced_unvoiced_cost cannot be negative")


class PraatInspiredVoicingDecoder:
    """Global V/U path over cached pYIN strength and centered-RMS volume.

    The local voiced advantage is pYIN's autocorrelation-derived aggregate
    candidate mass minus a calibrated threshold.  As in Praat's methodology,
    a global path pays an additive cost whenever it changes between voiced and
    unvoiced.  ``silence_threshold`` is a hard fraction of the track's peak
    centered-RMS volume, combined with an optional absolute dBFS floor.

    No waveform or autocorrelation enters this class; all inputs come from the
    raw pYIN stage cache.
    """

    def __init__(
        self,
        parameters: PraatInspiredVoicingParameters | None = None,
        *,
        time_step_seconds: float = 0.01,
    ) -> None:
        self.parameters = parameters or PraatInspiredVoicingParameters()
        if time_step_seconds <= 0.0:
            raise ValueError("time_step_seconds must be positive")
        self.time_step_seconds = float(time_step_seconds)

    def volume_floor(self, features: VoicingFeatures) -> float:
        relative = (
            float(np.max(features.volumes, initial=0.0))
            * self.parameters.silence_threshold
        )
        absolute = (
            0.0
            if self.parameters.absolute_volume_floor_dbfs is None
            else 10.0 ** (float(self.parameters.absolute_volume_floor_dbfs) / 20.0)
        )
        return max(relative, absolute)

    def eligible(self, features: VoicingFeatures) -> BoolArray:
        return (
            features.present
            & features.has_candidates
            & (features.volumes >= self.volume_floor(features))
        )

    def decode(self, features: VoicingFeatures) -> BoolArray:
        frame_count = features.periodicity.size
        if frame_count == 0:
            return np.empty(0, dtype=bool)
        if not (
            features.volumes.size
            == features.has_candidates.size
            == features.present.size
            == frame_count
        ):
            raise ValueError("voicing feature arrays must have equal lengths")

        eligible = self.eligible(features)
        voiced_advantage = (
            features.periodicity - self.parameters.candidate_strength_threshold
        ).astype(np.float64, copy=True)
        voiced_advantage[~eligible] = -np.inf
        voiced = _decode_switch_cost_path(
            voiced_advantage,
            # Praat defines this cost on a 10-ms reference grid and corrects
            # it for the actual time step. Attune's 128/44100-s hop is much
            # denser, so omitting this factor would change the temporal prior.
            self.parameters.voiced_unvoiced_cost * 0.01 / self.time_step_seconds,
        )
        voiced &= eligible
        return voiced


def nearest_voicing_mask(
    target_times: npt.ArrayLike,
    source_times: npt.ArrayLike,
    source_freqs: npt.ArrayLike,
) -> BoolArray:
    """Map a detector's voiced frames onto another grid without edge hold.

    Frames outside half a source hop are unvoiced.  This avoids extending
    Praat's first or last decision into analysis padding that Praat never saw.
    """
    target = np.asarray(target_times, dtype=np.float64).reshape(-1)
    times = np.asarray(source_times, dtype=np.float64).reshape(-1)
    freqs = np.asarray(source_freqs, dtype=np.float64).reshape(-1)
    if times.size != freqs.size:
        raise ValueError("source times and frequencies must have equal lengths")
    result = np.zeros(target.size, dtype=bool)
    if times.size == 0 or target.size == 0:
        return result
    if times.size == 1:
        result[np.isclose(target, times[0])] = freqs[0] > 0.0
        return result

    insertion = np.searchsorted(times, target, side="left")
    right = np.clip(insertion, 1, times.size - 1)
    left = right - 1
    nearest = np.where(
        np.abs(target - times[left]) <= np.abs(times[right] - target),
        left,
        right,
    )
    differences = np.diff(times)
    differences = differences[np.isfinite(differences) & (differences > 0.0)]
    half_hop = 0.5 * (float(np.median(differences)) if differences.size else 0.0)
    covered = (target >= times[0] - half_hop) & (target <= times[-1] + half_hop)
    result[covered] = freqs[nearest[covered]] > 0.0
    return result


def voiced_frequencies(
    path: PitchPath,
    voiced: npt.ArrayLike,
    config: Config,
) -> tuple[FloatArray, FloatArray]:
    """Convert a decoupled path and mask to mir_eval's 0-Hz convention."""
    mask = np.asarray(voiced, dtype=bool).reshape(-1)
    if mask.size != path.midi.size:
        raise ValueError("pitch path and voicing mask must have equal lengths")
    keep = path.present & np.isfinite(path.times)
    frequencies = np.zeros(int(np.count_nonzero(keep)), dtype=np.float64)
    kept_midi = path.midi[keep]
    candidate_frequencies = np.asarray(
        config.midi_to_freq(kept_midi),
        dtype=np.float64,
    )
    # The decoder uses the same +/- two-semitone guard as pYIN internally, but
    # benchmark output must still obey the score's exact requested range.  This
    # mirrors PitchDetectorBase.constrain_estimate_to_range and prevents the
    # experiment from undoing the range-gate correction in Attune's adapter.
    kept_voiced = (
        mask[keep]
        & (candidate_frequencies >= float(config.fmin))
        & (candidate_frequencies <= float(config.fmax))
    )
    frequencies[kept_voiced] = candidate_frequencies[kept_voiced]
    return path.times[keep], frequencies
