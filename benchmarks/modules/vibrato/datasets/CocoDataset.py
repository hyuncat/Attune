"""Build shared vibrato cases by injecting curves into CocoChorales MIDI."""
from __future__ import annotations
import json
import math
import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import pretty_midi
import soundfile as sf
from algorithms.Config import PYIN_RANGE_PADDING_SEMITONES, guarded_pyin_frequency
from benchmarks.modules.pitch.competitors.Attune import AttuneRealtime
from benchmarks.modules.pitch.datasets.CocoChorales import CocoChorales
from benchmarks.modules.vibrato.VibratoDetectorBase import VibratoExample
from benchmarks.modules.vibrato.datasets.CocoRenderer import DEFAULT_SOUNDFONTS_ROOT, CocoRenderer
from benchmarks.modules.vibrato.competitors.Yang import Yang
DEFAULT_PROFILES = ('constant', 'accelerating', 'decelerating', 'widening', 'narrowing', 'none')
INJECTION_RANGES = ('native', 'yang')
AUTOMATIC_YIN_WINDOW_MINIMUM = 1024
AUTOMATIC_YIN_WINDOW_PERIODS = 4.0
AUTOMATIC_YIN_WINDOW_PADDING_SEMITONES = PYIN_RANGE_PADDING_SEMITONES
PROFILE_PARAMETER_SAMPLER_VERSION = 'seeded_bounded_uniform_v1'
NATIVE_RATE_RANGE_HZ = (3.0, 10.0)
NATIVE_AMPLITUDE_RANGE_SEMITONES = (0.15, 1.0)
CHANGING_RATE_SPAN_RANGE_HZ = (1.0, 3.0)
CHANGING_AMPLITUDE_SPAN_RANGE_SEMITONES = (0.15, 0.5)

class CocoDataset:
    """Shared end-to-end CocoChorales vibrato corpus builder."""

    @staticmethod
    def automatic_yin_window_size(score_fmin_hz: float, *, sample_rate: int=44100, padding_semitones: float=AUTOMATIC_YIN_WINDOW_PADDING_SEMITONES, periods: float=AUTOMATIC_YIN_WINDOW_PERIODS, minimum: int=AUTOMATIC_YIN_WINDOW_MINIMUM) -> int:
        """Return a power-of-two YIN window for the guarded score minimum."""
        if not np.isfinite(score_fmin_hz) or score_fmin_hz <= 0.0:
            raise ValueError('score_fmin_hz must be a positive finite frequency')
        if sample_rate <= 0:
            raise ValueError('sample_rate must be positive')
        if not np.isfinite(padding_semitones) or padding_semitones < 0.0:
            raise ValueError('padding_semitones must be finite and non-negative')
        if not np.isfinite(periods) or periods <= 0.0:
            raise ValueError('periods must be positive and finite')
        if minimum <= 0:
            raise ValueError('minimum must be positive')
        target_fmin_hz = float(score_fmin_hz) * 2.0 ** (-float(padding_semitones) / 12.0)
        required = int(math.ceil(float(periods) * float(sample_rate) / target_fmin_hz))
        window = int(minimum)
        while window < required:
            window *= 2
        return window
    adaptive_yin_window_size = automatic_yin_window_size

    @dataclass(frozen=True)
    class VibratoProfileParameters:
        """Scalar anchors used to construct one injected profile."""
        rate_start_hz: float
        rate_peak_hz: float
        rate_end_hz: float
        amplitude_start_semitones: float
        amplitude_end_semitones: float

    @dataclass(frozen=True)
    class InjectedNoteTruth:
        """One MIDI note and the exact vibrato envelope assigned to it."""
        case_id: str
        scenario: str
        note_index: int
        pitch_midi: int
        start: float
        end: float
        bend_times: np.ndarray
        rate_hz: np.ndarray
        amplitude_semitones: np.ndarray
        offset_semitones: np.ndarray
        parameter_sampler: str
        parameter_seed: int
        profile_parameters: VibratoProfileParameters

        @property
        def duration(self) -> float:
            return self.end - self.start

        def curves_at(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            times = np.asarray(times, dtype=np.float64)
            rate = np.interp(times, self.bend_times, self.rate_hz)
            amplitude = np.interp(times, self.bend_times, self.amplitude_semitones)
            return (rate, amplitude)

    @classmethod
    def profile_curves(cls, profile: str, normalized_time: np.ndarray, *, injection_range: str='native', parameters: VibratoProfileParameters | None=None) -> tuple[np.ndarray, np.ndarray]:
        """Return instantaneous rate (Hz) and one-sided amplitude (semitones).

        ``native`` retains each profile's defining curve. ``yang`` constrains every
        positive frame to AVA's final decision-tree support (4--9 Hz and at least
        0.10-semitone one-sided extent); the ``none`` control remains zero.
        """
        if injection_range not in INJECTION_RANGES:
            raise ValueError(f'unknown injection range {injection_range!r}; expected one of {', '.join(INJECTION_RANGES)}')
        u = np.clip(np.asarray(normalized_time, dtype=np.float64), 0.0, 1.0)
        if profile == 'none':
            return (np.zeros_like(u), np.zeros_like(u))
        if parameters is None:
            defaults = {'constant': cls.VibratoProfileParameters(6.0, 6.0, 6.0, 0.5, 0.5), 'accelerating': cls.VibratoProfileParameters(4.0, 8.0, 8.0, 0.5, 0.5), 'decelerating': cls.VibratoProfileParameters(8.0, 8.0, 4.0, 0.5, 0.5), 'widening': cls.VibratoProfileParameters(6.0, 6.0, 6.0, 0.15, 1.0), 'narrowing': cls.VibratoProfileParameters(6.0, 6.0, 6.0, 1.0, 0.15)}
            try:
                parameters = defaults[profile]
            except KeyError as error:
                raise ValueError(f'unknown vibrato profile: {profile}') from error
        if profile in {'constant', 'accelerating', 'decelerating'}:
            rate = parameters.rate_start_hz + (parameters.rate_end_hz - parameters.rate_start_hz) * u
        elif profile in {'widening', 'narrowing'}:
            rate = np.full_like(u, parameters.rate_start_hz)
        else:
            raise ValueError(f'unknown vibrato profile: {profile}')
        amplitude = parameters.amplitude_start_semitones + (parameters.amplitude_end_semitones - parameters.amplitude_start_semitones) * u
        if injection_range == 'yang':
            rate = np.clip(rate, *Yang.DECISION_RATE_LIMITS_HZ)
            amplitude = np.maximum(amplitude, Yang.DECISION_MIN_EXTENT_SEMITONES)
        return (rate, amplitude)

    @classmethod
    def sample_profile_parameters(cls, profile: str, *, seed: int, injection_range: str='native') -> VibratoProfileParameters:
        """Draw reproducible profile anchors from the declared product support.

        Constant quantities use one uniform draw over the complete range. Changing
        quantities draw a human-plausible span and then a valid lower anchor, so a
        changing case cannot collapse into an almost-constant or extreme sweep.
        """
        if injection_range not in INJECTION_RANGES:
            raise ValueError(f'unknown injection range {injection_range!r}; expected one of {', '.join(INJECTION_RANGES)}')
        if profile not in DEFAULT_PROFILES:
            raise ValueError(f'unknown vibrato profile: {profile}')
        if profile == 'none':
            return cls.VibratoProfileParameters(0.0, 0.0, 0.0, 0.0, 0.0)
        rate_min, rate_max = NATIVE_RATE_RANGE_HZ
        if injection_range == 'yang':
            rate_min = max(rate_min, Yang.DECISION_RATE_LIMITS_HZ[0])
            rate_max = min(rate_max, Yang.DECISION_RATE_LIMITS_HZ[1])
        amplitude_min, amplitude_max = NATIVE_AMPLITUDE_RANGE_SEMITONES
        if injection_range == 'yang':
            amplitude_min = max(amplitude_min, Yang.DECISION_MIN_EXTENT_SEMITONES)
        rng = random.Random(int(seed))
        fixed_rate = rng.uniform(rate_min, rate_max)
        fixed_amplitude = rng.uniform(amplitude_min, amplitude_max)
        rate_span = rng.uniform(*CHANGING_RATE_SPAN_RANGE_HZ)
        low_rate = rng.uniform(rate_min, rate_max - rate_span)
        high_rate = low_rate + rate_span
        amplitude_span = rng.uniform(*CHANGING_AMPLITUDE_SPAN_RANGE_SEMITONES)
        low_amplitude = rng.uniform(amplitude_min, amplitude_max - amplitude_span)
        high_amplitude = low_amplitude + amplitude_span
        if profile == 'constant':
            return cls.VibratoProfileParameters(fixed_rate, fixed_rate, fixed_rate, fixed_amplitude, fixed_amplitude)
        if profile == 'accelerating':
            return cls.VibratoProfileParameters(low_rate, high_rate, high_rate, fixed_amplitude, fixed_amplitude)
        if profile == 'decelerating':
            return cls.VibratoProfileParameters(high_rate, high_rate, low_rate, fixed_amplitude, fixed_amplitude)
        if profile == 'widening':
            return cls.VibratoProfileParameters(fixed_rate, fixed_rate, fixed_rate, low_amplitude, high_amplitude)
        if profile == 'narrowing':
            return cls.VibratoProfileParameters(fixed_rate, fixed_rate, fixed_rate, high_amplitude, low_amplitude)
        raise ValueError(f'unknown vibrato profile: {profile}')

    @staticmethod
    def _integrated_phase(rate_hz: np.ndarray, times: np.ndarray) -> np.ndarray:
        phase = np.zeros(len(times), dtype=np.float64)
        if len(times) > 1:
            dt = np.diff(times)
            phase[1:] = 2.0 * np.pi * np.cumsum(0.5 * (rate_hz[:-1] + rate_hz[1:]) * dt)
        return phase

    @staticmethod
    def _set_pitch_bend_range(instrument: pretty_midi.Instrument, semitones: int) -> None:
        """Set MIDI RPN 0 (pitch-bend sensitivity) for deterministic rendering."""
        retained = [cc for cc in instrument.control_changes if cc.number not in (6, 38, 100, 101)]
        rpn = [pretty_midi.ControlChange(101, 0, 0.0), pretty_midi.ControlChange(100, 0, 0.0), pretty_midi.ControlChange(6, int(semitones), 0.0), pretty_midi.ControlChange(38, 0, 0.0), pretty_midi.ControlChange(101, 127, 0.0), pretty_midi.ControlChange(100, 127, 0.0)]
        instrument.control_changes = sorted(retained + rpn, key=lambda event: event.time)

    @classmethod
    def inject_midi(cls, source_midi: str | Path, output_midi: str | Path, *, profiles: Sequence[str]=DEFAULT_PROFILES, min_note_seconds: float=0.75, max_notes: int | None=None, bend_sample_hz: float=100.0, bend_range_semitones: int=2, write_pitch_bend_rpn: bool=True, injection_range: str='native', seed: int=0, case_prefix: str | None=None) -> list[InjectedNoteTruth]:
        """Inject one selected profile into each eligible note of a MIDI stem."""
        if min_note_seconds <= 0.0:
            raise ValueError('min_note_seconds must be positive')
        if bend_sample_hz <= 0.0:
            raise ValueError('bend_sample_hz must be positive')
        if bend_range_semitones <= 0:
            raise ValueError('bend_range_semitones must be positive')
        profiles = tuple(profiles)
        if not profiles:
            raise ValueError('at least one profile is required')
        for name in profiles:
            cls.profile_curves(name, np.array([0.0]), injection_range=injection_range)
        source_midi = Path(source_midi)
        output_midi = Path(output_midi)
        midi = pretty_midi.PrettyMIDI(str(source_midi))
        instruments = [instrument for instrument in midi.instruments if not instrument.is_drum]
        if not instruments:
            raise ValueError(f'no pitched instrument in {source_midi}')
        instrument = max(instruments, key=lambda item: len(item.notes))
        notes = sorted(instrument.notes, key=lambda note: (note.start, note.end, note.pitch))
        eligible = [(index, note) for index, note in enumerate(notes) if note.end - note.start >= min_note_seconds]
        if max_notes is None:
            max_notes = len(profiles)
        count = min(int(max_notes), len(profiles), len(eligible))
        if count <= 0:
            raise ValueError(f'no notes at least {min_note_seconds:g} s long in {source_midi}')
        selected_positions = np.linspace(0, len(eligible) - 1, count).round().astype(int)
        selected = [eligible[position] for position in selected_positions]
        prefix = case_prefix or source_midi.stem
        shift = CocoChorales.seed_for(seed, PROFILE_PARAMETER_SAMPLER_VERSION, prefix, 'profile_order') % len(profiles)
        profile_order = list(profiles[shift:] + profiles[:shift])
        instrument.pitch_bends = []
        if write_pitch_bend_rpn:
            cls._set_pitch_bend_range(instrument, bend_range_semitones)
        truths: list[CocoDataset.InjectedNoteTruth] = []
        sample_step = 1.0 / bend_sample_hz
        for (note_index, note), profile in zip(selected, profile_order, strict=False):
            duration = float(note.end - note.start)
            local_times = np.arange(0.0, duration, sample_step, dtype=np.float64)
            if len(local_times) < 2:
                continue
            times = note.start + local_times
            normalized = local_times / duration
            parameter_seed = CocoChorales.seed_for(seed, PROFILE_PARAMETER_SAMPLER_VERSION, prefix, note_index, profile)
            parameters = cls.sample_profile_parameters(profile, seed=parameter_seed, injection_range=injection_range)
            rate_hz, amplitude = cls.profile_curves(profile, normalized, injection_range=injection_range, parameters=parameters)
            phase = cls._integrated_phase(rate_hz, times)
            offset = amplitude * np.sin(phase)
            bends = np.rint(offset / bend_range_semitones * 8192.0)
            bends = np.clip(bends, -8192, 8191).astype(int)
            instrument.pitch_bends.extend((pretty_midi.PitchBend(int(value), float(time)) for value, time in zip(bends, times, strict=False)))
            instrument.pitch_bends.append(pretty_midi.PitchBend(0, float(note.end)))
            truths.append(cls.InjectedNoteTruth(case_id=f'{prefix}__note{note_index:03d}__{profile}', scenario=profile, note_index=note_index, pitch_midi=int(note.pitch), start=float(note.start), end=float(note.end), bend_times=times, rate_hz=rate_hz, amplitude_semitones=amplitude, offset_semitones=offset, parameter_sampler=PROFILE_PARAMETER_SAMPLER_VERSION, parameter_seed=parameter_seed, profile_parameters=parameters))
        instrument.pitch_bends.sort(key=lambda event: event.time)
        output_midi.parent.mkdir(parents=True, exist_ok=True)
        midi.write(str(output_midi))
        return truths

    @staticmethod
    def _write_truth_artifacts(truths: Sequence[InjectedNoteTruth], destination: str | Path) -> tuple[Path, Path]:
        """Write inspectable metadata JSON and exact sampled curves NPZ."""
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        metadata_path = destination / 'ground_truth.json'
        curves_path = destination / 'ground_truth_curves.npz'
        metadata: list[dict[str, Any]] = []
        arrays: dict[str, np.ndarray] = {}
        for index, truth in enumerate(truths):
            key = f'note_{index:03d}'
            row = asdict(truth)
            for field in ('bend_times', 'rate_hz', 'amplitude_semitones', 'offset_semitones'):
                row.pop(field)
            row['array_key'] = key
            metadata.append(row)
            arrays[f'{key}__times'] = truth.bend_times
            arrays[f'{key}__rate_hz'] = truth.rate_hz
            arrays[f'{key}__amplitude_semitones'] = truth.amplitude_semitones
            arrays[f'{key}__offset_semitones'] = truth.offset_semitones
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
        np.savez_compressed(curves_path, **arrays)
        return (metadata_path, curves_path)

    @staticmethod
    def _voiced_sample_mask(sample_count: int, sample_rate: int, truths: Sequence[InjectedNoteTruth]) -> np.ndarray:
        mask = np.zeros(sample_count, dtype=bool)
        for truth in truths:
            lo = max(0, int(math.floor(truth.start * sample_rate)))
            hi = min(sample_count, int(math.ceil(truth.end * sample_rate)))
            mask[lo:hi] = True
        return mask

    @staticmethod
    def _load_noise_pool(noise_file: str | Path | None, sample_rate: int) -> np.ndarray | None:
        if noise_file is None:
            return None
        import librosa
        noise, _ = librosa.load(str(noise_file), sr=sample_rate, mono=True)
        noise = np.asarray(noise, dtype=np.float64)
        noise -= np.mean(noise)
        if not np.any(noise):
            raise ValueError(f'noise file is silent: {noise_file}')
        return noise

    @staticmethod
    def add_noise(samples: np.ndarray, *, snr_db: float, voiced_mask: np.ndarray, rng: np.random.Generator, noise_pool: np.ndarray | None=None) -> np.ndarray:
        """The voiced-power SNR model used by ``noisy_pitch_detection.py``."""
        if not math.isfinite(snr_db):
            return np.asarray(samples, dtype=np.float32)
        samples = np.asarray(samples, dtype=np.float64)
        signal = samples[voiced_mask] if np.any(voiced_mask) else samples
        signal_power = float(np.mean(signal ** 2))
        if signal_power <= 0.0:
            return np.asarray(samples, dtype=np.float32)
        if noise_pool is None:
            noise = rng.standard_normal(len(samples))
        else:
            repeats = int(np.ceil(len(samples) / len(noise_pool)))
            start = int(rng.integers(0, len(noise_pool)))
            noise = np.roll(np.tile(noise_pool, repeats), -start)[:len(samples)]
        noise = noise - np.mean(noise)
        noise_power = float(np.mean(noise ** 2))
        if noise_power <= 0.0:
            return np.asarray(samples, dtype=np.float32)
        scale = math.sqrt(signal_power / 10.0 ** (snr_db / 10.0) / noise_power)
        return np.asarray(samples + scale * noise, dtype=np.float32)

    @staticmethod
    def _pitch_stage_contour_diagnostics(commanded_pitch_midi: np.ndarray, observed_pitch_midi: np.ndarray) -> dict[str, float]:
        """Measure direct pitch-contour damage for any injected trajectory."""
        commanded = np.asarray(commanded_pitch_midi, dtype=np.float64)
        observed = np.asarray(observed_pitch_midi, dtype=np.float64)
        expected = np.isfinite(commanded)
        if not expected.any():
            return {}
        voiced = expected & np.isfinite(observed)
        output = {'dropout_fraction': 1.0 - float(np.sum(voiced)) / float(np.sum(expected)), 'octave_error_fraction': float(np.sum(np.abs(observed[voiced] - commanded[voiced]) >= 6.0)) / float(np.sum(expected)) if voiced.any() else 0.0}
        if not voiced.any():
            return output
        errors_cents = 100.0 * (observed[voiced] - commanded[voiced])
        output.update({'contour_bias_cents': float(np.mean(errors_cents)), 'contour_mae_cents': float(np.mean(np.abs(errors_cents))), 'contour_rmse_cents': float(np.sqrt(np.mean(errors_cents ** 2)))})
        return output

    @classmethod
    def _constant_pitch_stage_diagnostics(cls, times: np.ndarray, commanded_pitch_midi: np.ndarray, observed_pitch_midi: np.ndarray, center_midi: np.ndarray, rate_hz: np.ndarray) -> dict[str, float]:
        """Measure contour distortion before the final vibrato model."""
        times = np.asarray(times, dtype=np.float64)
        commanded = np.asarray(commanded_pitch_midi, dtype=np.float64)
        observed = np.asarray(observed_pitch_midi, dtype=np.float64)
        centers = np.asarray(center_midi, dtype=np.float64)
        rates = np.asarray(rate_hz, dtype=np.float64)
        expected = np.isfinite(commanded) & np.isfinite(centers) & (rates > 0.0)
        if not expected.any():
            return {}
        voiced = expected & np.isfinite(observed)
        output = cls._pitch_stage_contour_diagnostics(commanded, observed)
        if np.sum(voiced) < 8:
            return output
        true_rate = float(np.median(rates[expected]))
        local_times = times[voiced]
        local_times = local_times - float(np.mean(local_times))

        def sinusoid_fit(values: np.ndarray, frequency: float):
            angle = 2.0 * np.pi * frequency * local_times
            design = np.column_stack((np.ones(len(local_times)), local_times, np.sin(angle), np.cos(angle)))
            coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
            error = values - design @ coefficients
            amplitude = float(np.hypot(coefficients[2], coefficients[3]))
            phase = float(np.arctan2(coefficients[3], coefficients[2]))
            return (float(np.dot(error, error)), amplitude, phase, float(coefficients[0]))
        observed_values = observed[voiced]
        commanded_values = commanded[voiced]
        _, commanded_amplitude, commanded_phase, _ = sinusoid_fit(commanded_values, true_rate)
        _, observed_amplitude, observed_phase, observed_center = sinusoid_fit(observed_values, true_rate)
        scan = np.linspace(max(0.5, true_rate - 2.0), true_rate + 2.0, 161)
        fitted_rate = min(scan, key=lambda frequency: sinusoid_fit(observed_values, float(frequency))[0])
        phase_error = (observed_phase - commanded_phase + np.pi) % (2.0 * np.pi) - np.pi
        truth_center = float(np.median(centers[expected]))
        output.update({'amplitude_gain': observed_amplitude / commanded_amplitude if commanded_amplitude > np.finfo(float).eps else np.nan, 'amplitude_error_semitones': observed_amplitude - commanded_amplitude, 'rate_error_hz': abs(float(fitted_rate) - true_rate), 'phase_error_degrees': abs(float(np.degrees(phase_error))), 'center_bias_cents': 100.0 * (observed_center - truth_center), 'center_error_cents': 100.0 * abs(observed_center - truth_center)})
        return output

    @classmethod
    def _examples_from_pitch_data(cls, truths: Sequence[InjectedNoteTruth], pitch_data, config, *, raw_pitch_data=None, split: str, metadata: dict[str, Any], case_suffix: str='') -> list[VibratoExample]:
        """Expose one full detected stem and score each injected note separately."""
        examples: list[VibratoExample] = []
        pitches = pitch_data.data[:pitch_data.frames_available()]
        if len(pitches) < 2:
            return examples
        times = np.asarray([pitch.time if pitch is not None else pitch_data._frame_time(index) for index, pitch in enumerate(pitches)], dtype=np.float64)
        values = np.asarray([pitch.value if pitch is not None and pitch.value != -1 and (pitch.unvoiced_prob < config.unv_thresh) else np.nan for pitch in pitches], dtype=np.float64)
        transition_mask = np.asarray([bool(getattr(pitch, 'is_transition', False)) if pitch is not None else False for pitch in pitches], dtype=np.bool_)
        raw_pitches = raw_pitch_data.data[:raw_pitch_data.frames_available()] if raw_pitch_data is not None else pitches
        raw_values = np.full(len(times), np.nan, dtype=np.float64)
        for index, pitch in enumerate(raw_pitches[:len(times)]):
            if pitch is not None and pitch.value != -1 and (pitch.unvoiced_prob < config.unv_thresh):
                raw_values[index] = pitch.value
        center_midi = np.full(len(times), np.nan, dtype=np.float64)
        commanded_pitch_midi = np.full(len(times), np.nan, dtype=np.float64)
        rate_hz = np.zeros(len(times), dtype=np.float64)
        width_cents = np.zeros(len(times), dtype=np.float64)
        is_vibrato = np.zeros(len(times), dtype=np.bool_)
        target_masks: dict[str, np.ndarray] = {}
        for truth in truths:
            target = (times >= truth.start) & (times < truth.end)
            if int(np.sum(target)) < 2:
                continue
            target_rate, target_amplitude = truth.curves_at(times[target])
            center_midi[target] = float(truth.pitch_midi)
            commanded_pitch_midi[target] = np.interp(times[target], truth.bend_times, float(truth.pitch_midi) + truth.offset_semitones)
            rate_hz[target] = target_rate
            width_cents[target] = 200.0 * target_amplitude
            is_vibrato[target] = (target_rate > 0.0) & (target_amplitude > 0.0)
            target_masks[truth.case_id] = target
        analysis_group = str(metadata.get('analysis_group', f'{metadata.get('track', 'track')}__{metadata.get('stem', 'stem')}__{metadata.get('snr', 'condition')}'))
        analysis_note_bounds = metadata.get('analysis_note_bounds')
        if analysis_note_bounds is None:
            analysis_note_bounds = [(truth.start, truth.end, float(truth.pitch_midi)) for truth in truths]
        for truth in truths:
            target = target_masks.get(truth.case_id)
            if target is None:
                continue
            stage_diagnostics: dict[str, float] = {}
            for stage, contour in (('raw_pitch', raw_values), ('smoothed_pitch', values), ('vibrato_input_pitch', np.where(transition_mask, np.nan, values))):
                diagnostics = cls._pitch_stage_contour_diagnostics(commanded_pitch_midi[target], contour[target])
                if truth.scenario == 'constant':
                    diagnostics.update(cls._constant_pitch_stage_diagnostics(times[target], commanded_pitch_midi[target], contour[target], center_midi[target], rate_hz[target]))
                stage_diagnostics.update({f'{stage}_{key}': value for key, value in diagnostics.items()})
            examples.append(VibratoExample(case_id=f'{truth.case_id}{case_suffix}', scenario=truth.scenario, split=split, times=times, pitch_midi=values, center_midi=center_midi, rate_hz=rate_hz, width_cents=width_cents, is_vibrato=is_vibrato, commanded_pitch_midi=commanded_pitch_midi, raw_pitch_midi=raw_values, transition_mask=transition_mask, evaluation_mask=target, audio_path=str(metadata['audio_path']) if metadata.get('audio_path') is not None else None, metadata={**metadata, 'family': 'cocochorales_midi_injected', 'analysis_group': analysis_group, 'analysis_note_bounds': analysis_note_bounds, 'continuous_context': True, 'note_index': truth.note_index, 'note_pitch_midi': truth.pitch_midi, 'note_duration': truth.duration, 'target_start_time': truth.start, 'target_end_time': truth.end, 'parameter_sampler': truth.parameter_sampler, 'parameter_seed': truth.parameter_seed, 'sampled_rate_start_hz': truth.profile_parameters.rate_start_hz, 'sampled_rate_peak_hz': truth.profile_parameters.rate_peak_hz, 'sampled_rate_end_hz': truth.profile_parameters.rate_end_hz, 'sampled_amplitude_start_semitones': truth.profile_parameters.amplitude_start_semitones, 'sampled_amplitude_end_semitones': truth.profile_parameters.amplitude_end_semitones, 'smoothed_pitch_transition_frames': int(np.sum(transition_mask[target])), 'smoothed_pitch_transition_fraction': float(np.mean(transition_mask[target])), **stage_diagnostics}))
        return examples

    @staticmethod
    def _midi_note_bounds(path: str | Path) -> list[tuple[float, float, float]]:
        """Return every note boundary supplied to note-aware estimators."""
        midi = pretty_midi.PrettyMIDI(str(path))
        instruments = [item for item in midi.instruments if not item.is_drum]
        if not instruments:
            return []
        instrument = max(instruments, key=lambda item: len(item.notes))
        return [(float(note.start), float(note.end), float(note.pitch)) for note in sorted(instrument.notes, key=lambda item: (item.start, item.end, item.pitch)) if note.end > note.start]

    @staticmethod
    def _annotation_pitch_range(note_bounds: Sequence[tuple[float, float, float]], truths: Sequence[InjectedNoteTruth]) -> tuple[float, float]:
        """Return the exact rendered pitch range encoded by MIDI annotations."""
        annotated_midis = [float(pitch) for _, _, pitch in note_bounds]
        for truth in truths:
            offsets = np.asarray(truth.offset_semitones, dtype=np.float64)
            offsets = offsets[np.isfinite(offsets)]
            if offsets.size:
                annotated_midis.extend((float(truth.pitch_midi + offsets.min()), float(truth.pitch_midi + offsets.max())))
        if not annotated_midis:
            raise ValueError('injected MIDI has no annotated pitched notes')
        return AttuneRealtime.range_from_midi(annotated_midis)

    @staticmethod
    def _local_midi_path(benchmarker: CocoChorales, record: CocoChorales.Stem) -> Path | None:
        candidates = [benchmarker.materialized_midi_path(record), benchmarker.root / 'main_dataset' / record.split / record.track / 'stems_midi' / f'{record.stem}.mid']
        return next((candidate for candidate in candidates if candidate.exists()), None)

    @staticmethod
    def _snr_label(snr_db: float) -> str:
        return 'clean' if not math.isfinite(snr_db) else f'snr{snr_db:g}db'

    @dataclass(frozen=True)
    class _CocoVibratoStemJob:
        index: int
        record: CocoChorales.Stem
        coco_root: str
        output_root: str
        profiles: tuple[str, ...]
        snrs_db: tuple[float, ...]
        noise_file: str | None
        min_note_seconds: float
        notes_per_stem: int
        bend_sample_hz: float
        injection_range: str
        seed: int
        smooth_pitch: bool
        adaptive_yin_window: bool
        fixed_yin_window_size: int | None
        sfizz_render: str
        soundfonts_root: str
        sfizz_render_sha256: str
        renderer_identity: str
        force: bool
        force_pitch: bool

    @staticmethod
    def _build_stem(job: _CocoVibratoStemJob) -> tuple[int, str, list[VibratoExample], str | None]:
        """Render and pitch-detect one stem inside a spawned worker."""
        record = job.record
        coco = CocoChorales(root=job.coco_root)
        pitch_bench = AttuneRealtime()
        renderer = CocoRenderer(executable=job.sfizz_render, soundfonts_root=job.soundfonts_root, executable_sha256=job.sfizz_render_sha256)
        patch = renderer.patch_for(record.instrument)
        renderer_manifest = renderer.manifest_for(record.instrument)
        if renderer_manifest['identity'] != job.renderer_identity:
            raise RuntimeError(f'SFZ renderer identity changed for {record.instrument}; restart the benchmark so stale audio cannot be reused')
        source_midi = CocoDataset._local_midi_path(coco, record)
        if source_midi is None:
            return (job.index, record.track_id, [], 'materialized MIDI was not found')
        injection_config_id = CocoChorales.seed_for(job.seed, PROFILE_PARAMETER_SAMPLER_VERSION, ','.join(job.profiles), job.notes_per_stem, job.min_note_seconds, job.bend_sample_hz, job.injection_range, job.renderer_identity)
        stem_root = Path(job.output_root) / f'{PROFILE_PARAMETER_SAMPLER_VERSION}__seed{job.seed}__cfg{injection_config_id:016x}' / record.track_id
        stem_dir = stem_root if job.injection_range == 'native' else stem_root / f'range_{job.injection_range}'
        injected_midi = stem_dir / f'{record.stem}__vibrato.mid'
        try:
            truths = CocoDataset.inject_midi(source_midi, injected_midi, profiles=job.profiles, min_note_seconds=job.min_note_seconds, max_notes=job.notes_per_stem, bend_sample_hz=job.bend_sample_hz, bend_range_semitones=patch.pitch_bend_range_semitones, write_pitch_bend_rpn=False, injection_range=job.injection_range, seed=job.seed, case_prefix=record.track_id)
        except ValueError as error:
            if str(error).startswith('no notes at least'):
                return (job.index, record.track_id, [], str(error))
            raise
        renderer.prepare_midi(injected_midi, injected_midi, instrument=record.instrument)
        analysis_note_bounds = CocoDataset._midi_note_bounds(injected_midi)
        CocoDataset._write_truth_artifacts(truths, stem_dir)
        clean_wav = renderer.render(injected_midi, instrument=record.instrument, out_dir=stem_dir, sample_rate=pitch_bench.DEFAULT_CONFIG.sr, force=job.force)
        samples, sample_rate = sf.read(clean_wav, always_2d=True)
        samples = np.asarray(samples.mean(axis=1), dtype=np.float32)
        if int(sample_rate) != pitch_bench.DEFAULT_CONFIG.sr:
            raise ValueError(f'unexpected render rate {sample_rate}; expected {pitch_bench.DEFAULT_CONFIG.sr}')
        if not np.isfinite(samples).all() or not np.any(samples):
            raise ValueError(f'sfizz rendered invalid or silent audio: {clean_wav}')
        voiced_mask = CocoDataset._voiced_sample_mask(len(samples), int(sample_rate), truths)
        noise_pool = CocoDataset._load_noise_pool(job.noise_file, pitch_bench.DEFAULT_CONFIG.sr)
        fmin, fmax = CocoDataset._annotation_pitch_range(analysis_note_bounds, truths)
        score_fmin, _ = AttuneRealtime.range_from_midi((pitch for _, _, pitch in analysis_note_bounds))
        yin_window_padding_semitones: float | None = None
        yin_window_target_fmin_hz: float | None = None
        yin_window_policy = 'fixed_production_default'
        yin_window_periods: float | None = None
        yin_window_minimum: int | None = None
        if job.adaptive_yin_window:
            yin_window_padding_semitones = AUTOMATIC_YIN_WINDOW_PADDING_SEMITONES
            yin_window_target_fmin_hz = guarded_pyin_frequency(score_fmin, lower=True)
            yin_window_size = CocoDataset.automatic_yin_window_size(score_fmin, sample_rate=pitch_bench.DEFAULT_CONFIG.sr, padding_semitones=yin_window_padding_semitones)
            config = pitch_bench.config_for(fmin, fmax, w1=yin_window_size)
            cache_variant = f'scorepad{yin_window_padding_semitones:g}_w{yin_window_size}'
            yin_window_policy = 'four_guarded_periods_power_of_two'
            yin_window_periods = AUTOMATIC_YIN_WINDOW_PERIODS
            yin_window_minimum = AUTOMATIC_YIN_WINDOW_MINIMUM
        elif job.fixed_yin_window_size is not None:
            yin_window_padding_semitones = PYIN_RANGE_PADDING_SEMITONES
            yin_window_target_fmin_hz = guarded_pyin_frequency(score_fmin, lower=True)
            required_samples = int(math.ceil(pitch_bench.DEFAULT_CONFIG.sr / yin_window_target_fmin_hz))
            yin_window_size = int(job.fixed_yin_window_size)
            if yin_window_size < required_samples:
                return (job.index, record.track_id, [], f'fixed YIN window {yin_window_size} cannot cover one period ({required_samples} samples) at guarded fmin {yin_window_target_fmin_hz:.3f} Hz')
            config = pitch_bench.config_for(fmin, fmax, w1=yin_window_size)
            cache_variant = f'fixed_w{yin_window_size}'
            yin_window_policy = 'fixed_guard_validated'
        else:
            config = pitch_bench.config_for(fmin, fmax)
            cache_variant = None
        examples: list[VibratoExample] = []
        for snr_index, snr_db in enumerate(job.snrs_db):
            label = CocoDataset._snr_label(float(snr_db))
            if math.isfinite(float(snr_db)):
                noisy_wav = stem_dir / f'{injected_midi.stem}__{label}.wav'
                if job.force or not noisy_wav.exists():
                    degraded = CocoDataset.add_noise(samples, snr_db=float(snr_db), voiced_mask=voiced_mask, rng=np.random.default_rng(CocoChorales.seed_for(job.seed, record.track_id, label, 'additive_noise')), noise_pool=noise_pool)
                    sf.write(noisy_wav, degraded, int(sample_rate))
                audio_path = noisy_wav
            else:
                audio_path = clean_wav
            recording = pitch_bench.recording_for(config)
            recording.audio_data = coco.load_resampled_audio(audio_path, config.sr)
            cache_name = f'{label}__{cache_variant}.pitch.pkl.xz' if cache_variant is not None else f'{label}.pitch.pkl.xz'
            cache_path = stem_dir / 'pitch_data' / cache_name
            pitch_bench.load_or_detect_pitches(recording, cache_path=cache_path, smooth=job.smooth_pitch, use_cache=not (job.force or job.force_pitch), write_cache=True)
            raw_pitch_data = recording.pitch_data
            if job.smooth_pitch:
                try:
                    raw_pitch_data, _ = pitch_bench.load_pitch_data(cache_path, config, smooth=False)
                except ValueError:
                    pitch_bench.load_or_detect_pitches(recording, cache_path=cache_path, smooth=False, use_cache=True, write_cache=True)
                    raw_pitch_data = recording.pitch_data
                    recording.pitch_data, _ = pitch_bench.load_pitch_data(cache_path, config, smooth=True)
            examples.extend(CocoDataset._examples_from_pitch_data(truths, recording.pitch_data, config, raw_pitch_data=raw_pitch_data, split=record.split, metadata={'track': record.track, 'stem': record.stem, 'instrument': record.instrument, 'ensemble': record.ensemble, 'snr': label, 'snr_db': float(snr_db), 'track_selection_policy': CocoChorales.BALANCED_SELECTION_POLICY, 'profile_parameter_sampler': PROFILE_PARAMETER_SAMPLER_VERSION, 'profile_parameter_global_seed': job.seed, 'pitch_stage': 'attune_adhoc' if job.smooth_pitch else 'attune_realtime', 'pitch_smoother': 'original' if job.smooth_pitch else 'none', 'pitch_range_source': 'full_midi_notes_plus_injected_bend_annotations', 'pitch_fmin_hz': float(config.fmin), 'pitch_fmax_hz': float(config.fmax), 'pyin_range_padding_semitones': PYIN_RANGE_PADDING_SEMITONES, 'pyin_autocorrelation_fmin_hz': float(recording.pitch_detector.guarded_fmin), 'pyin_autocorrelation_fmax_hz': float(recording.pitch_detector.guarded_fmax), 'hmm_range_padding_semitones': PYIN_RANGE_PADDING_SEMITONES, 'hmm_state_fmin_hz': float(config.midi_to_freq(recording.pitch_smoother.midi_min)), 'hmm_state_fmax_hz': float(config.midi_to_freq(recording.pitch_smoother.midi_max)), 'yin_integration_size': int(config.w1), 'yin_window_policy': yin_window_policy, 'yin_window_periods': yin_window_periods, 'yin_window_minimum': yin_window_minimum, 'yin_window_padding_semitones': yin_window_padding_semitones, 'yin_window_score_fmin_hz': float(score_fmin) if job.adaptive_yin_window or job.fixed_yin_window_size is not None else None, 'yin_window_target_fmin_hz': yin_window_target_fmin_hz, 'injection_range': job.injection_range, 'source_midi': str(source_midi), 'injected_midi': str(injected_midi), 'audio_path': str(audio_path), 'analysis_note_bounds': analysis_note_bounds, 'synthesis_renderer': renderer_manifest}, case_suffix=f'__{label}'))
        return (job.index, record.track_id, examples, None)

    @classmethod
    def build(cls, *, coco_root: str | Path | None=None, output_root: str | Path, split: str='test', max_stems: int | None=1, notes_per_stem: int=6, profiles: Sequence[str]=DEFAULT_PROFILES, snrs_db: Sequence[float]=(math.inf,), noise_file: str | Path | None=None, min_note_seconds: float=0.75, bend_sample_hz: float=100.0, injection_range: str='native', seed: int=0, smooth_pitch: bool=True, adaptive_yin_window: bool=False, fixed_yin_window_size: int | None=None, sfizz_render: str | Path | None=None, soundfonts_root: str | Path=DEFAULT_SOUNDFONTS_ROOT, force: bool=False, force_pitch: bool=False, workers: int=1) -> list[VibratoExample]:
        """Build an end-to-end CocoChorales audio-to-vibrato corpus."""
        if max_stems is not None and max_stems <= 0:
            raise ValueError('max_stems must be positive or None')
        if notes_per_stem <= 0:
            raise ValueError('notes_per_stem must be positive')
        if workers <= 0:
            raise ValueError('workers must be positive')
        if adaptive_yin_window and fixed_yin_window_size is not None:
            raise ValueError('adaptive_yin_window and fixed_yin_window_size are mutually exclusive')
        if fixed_yin_window_size is not None and fixed_yin_window_size <= 0:
            raise ValueError('fixed_yin_window_size must be positive')
        if injection_range not in INJECTION_RANGES:
            raise ValueError(f'unknown injection range {injection_range!r}; expected one of {', '.join(INJECTION_RANGES)}')
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        coco = CocoChorales(root=coco_root)
        records = coco.select_records(split=split, max_tracks=max_stems, seed=seed, balanced=True)
        coco.materialize_records(records)
        renderer = CocoRenderer(executable=sfizz_render, soundfonts_root=soundfonts_root)
        instruments = {record.instrument for record in records}
        renderer.validate(instruments)
        renderer_manifests = {instrument: renderer.manifest_for(instrument) for instrument in instruments}
        jobs = [cls._CocoVibratoStemJob(index=index, record=record, coco_root=str(coco.root), output_root=str(output_root), profiles=tuple(profiles), snrs_db=tuple((float(value) for value in snrs_db)), noise_file=str(noise_file) if noise_file is not None else None, min_note_seconds=float(min_note_seconds), notes_per_stem=int(notes_per_stem), bend_sample_hz=float(bend_sample_hz), injection_range=injection_range, seed=int(seed), smooth_pitch=bool(smooth_pitch), adaptive_yin_window=bool(adaptive_yin_window), fixed_yin_window_size=int(fixed_yin_window_size) if fixed_yin_window_size is not None else None, sfizz_render=str(renderer.executable), soundfonts_root=str(renderer.soundfonts_root), sfizz_render_sha256=renderer.executable_sha256, renderer_identity=str(renderer_manifests[record.instrument]['identity']), force=bool(force), force_pitch=bool(force_pitch)) for index, record in enumerate(records)]
        if not jobs:
            raise RuntimeError('no CocoChorales stems were selected')
        results: dict[int, list[VibratoExample]] = {}
        skipped: list[tuple[str, str]] = []
        progress_width = 0

        def retain(result: tuple[int, str, list[VibratoExample], str | None]) -> None:
            index, track_id, stem_examples, reason = result
            results[index] = stem_examples
            if reason is not None:
                skipped.append((track_id, reason))

        def show_progress(completed: int, track_id: str) -> None:
            nonlocal progress_width
            message = f'[{completed:>4}/{len(jobs)}] coco-pitch: {track_id}'
            progress_width = max(progress_width, len(message))
            print(f'\r{message:<{progress_width}}', end='', flush=True)
        if workers == 1 or len(jobs) == 1:
            for completed, job in enumerate(jobs, start=1):
                retain(cls._build_stem(job))
                show_progress(completed, job.record.track_id)
        else:
            context = multiprocessing.get_context('spawn')
            with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=context) as executor:
                pending = {executor.submit(cls._build_stem, job): job for job in jobs}
                for completed, future in enumerate(as_completed(pending), start=1):
                    job = pending[future]
                    retain(future.result())
                    show_progress(completed, job.record.track_id)
        if progress_width:
            print()
        examples = [example for index in sorted(results) for example in results[index]]
        for track_id, reason in skipped:
            print(f'[skip] {track_id}: {reason}', flush=True)
        if not examples:
            raise RuntimeError('no CocoChorales vibrato cases were built; check materialized MIDI and the minimum note duration')
        return examples
