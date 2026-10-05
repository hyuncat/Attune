"""F0-free template detector after Driedger et al. (ISMIR 2016)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import grey_dilation
from scipy.signal import fftconvolve

from benchmarks.modules.vibrato.VibratoDetectorBase import (
    VibratoDetectorBase,
    VibratoEstimate,
    VibratoExample,
)


DRIEDGER_FRAME_RATE = 150.0
DRIEDGER_BINS_PER_SEMITONE = 10
DRIEDGER_TEMPLATE_SECONDS = 0.4
DRIEDGER_SALIENCE_THRESHOLD = 0.55
DRIEDGER_GLOBAL_MINIMUM_HZ = 196.0
DRIEDGER_GLOBAL_MAXIMUM_HZ = 3_000.0
DRIEDGER_SCORE_LOWER_PADDING_SEMITONES = 2.0
DRIEDGER_SCORE_UPPER_PADDING_SEMITONES = 24.0
DRIEDGER_DETECTION_RATES_HZ = np.arange(5.0, 7.0 + 0.25, 0.5)
DRIEDGER_DETECTION_EXTENTS_CENTS = np.arange(50.0, 100.0 + 5.0, 10.0)
DRIEDGER_ANALYSIS_RATES_HZ = np.arange(4.0, 11.0 + 0.25, 0.5)
DRIEDGER_ANALYSIS_EXTENTS_CENTS = np.arange(30.0, 210.0 + 5.0, 10.0)
# Benchmark-support ablation: span Attune's complete 3--10 Hz product range
# and 10--200 cent full peak-to-peak width range. Driedger's template extent
# is one-sided, so the latter is represented as e=5--100 cents. Retain every
# extent in the published detection bank and its 10-cent spacing; the 5-cent
# endpoint adds coverage at Attune's minimum reportable width.
DRIEDGER_BENCHMARK_RATES_HZ = np.arange(3.0, 10.0 + 0.25, 0.5)
DRIEDGER_BENCHMARK_EXTENTS_CENTS = np.concatenate(
    (
        np.array([5.0], dtype=np.float64),
        np.arange(10.0, 100.0 + 5.0, 10.0),
    )
)


class Driedger(VibratoDetectorBase):
    """Driedger et al.'s F0-free spectrogram-template method."""

    @dataclass(frozen=True)
    class Template:
        rate_hz: float
        extent_cents: float
        values: np.ndarray
        positive_support: np.ndarray

    @dataclass(frozen=True)
    class Spectrogram:
        binary: np.ndarray
        times: np.ndarray
        minimum_midi: float
        bins_per_semitone: int

    @staticmethod
    def _stft_size(sample_rate: int, preferred: int = 2_048) -> int:
        # The paper does not publish this value. Keep the frozen 2048-sample
        # adapter geometry across sample rates; frequency reassignment supplies
        # the finer log-frequency localization.
        del sample_rate
        return int(preferred)

    @staticmethod
    def _score_padded_frequency_range(
        example: VibratoExample,
        *,
        fallback_minimum_hz: float = DRIEDGER_GLOBAL_MINIMUM_HZ,
        fallback_maximum_hz: float = DRIEDGER_GLOBAL_MAXIMUM_HZ,
    ) -> tuple[float, float, str]:
        """Return score-padded carrier bounds or the fixed corpus fallback."""

        metadata = example.metadata
        has_score_source = any(
            metadata.get(key) for key in ("source_midi", "score_path", "score_filepath")
        )
        raw_bounds = metadata.get("analysis_note_bounds")
        if has_score_source and isinstance(raw_bounds, (list, tuple)):
            score_pitches: list[float] = []
            for bounds in raw_bounds:
                if not isinstance(bounds, (list, tuple)) or len(bounds) != 3:
                    continue
                try:
                    midi = float(bounds[2])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(midi):
                    score_pitches.append(midi)
            if score_pitches:
                minimum_midi = (
                    min(score_pitches) - DRIEDGER_SCORE_LOWER_PADDING_SEMITONES
                )
                maximum_midi = (
                    max(score_pitches) + DRIEDGER_SCORE_UPPER_PADDING_SEMITONES
                )
                minimum_hz = 440.0 * 2.0 ** ((minimum_midi - 69.0) / 12.0)
                maximum_hz = 440.0 * 2.0 ** ((maximum_midi - 69.0) / 12.0)
                return float(minimum_hz), float(maximum_hz), "score_padded"
        return (
            float(fallback_minimum_hz),
            float(fallback_maximum_hz),
            "global_fallback",
        )

    @classmethod
    def _reassigned_log_spectrogram(
        cls,
        samples: np.ndarray,
        sample_rate: int,
        *,
        frame_rate: float = DRIEDGER_FRAME_RATE,
        bins_per_semitone: int = DRIEDGER_BINS_PER_SEMITONE,
        minimum_hz: float = 27.5,
        maximum_hz: float = 7_040.0,
        top_fraction: float = 0.10,
        reassignment_power_floor: float = 0.05,
    ) -> Driedger.Spectrogram:
        """Build the paper's frame-wise top-decile binary log-frequency image."""

        import librosa

        values = np.asarray(samples, dtype=np.float64).reshape(-1)
        hop = max(1, int(round(sample_rate / frame_rate)))
        n_fft = cls._stft_size(sample_rate)
        # librosa computes reassignment as a ratio of derivative and base STFTs.
        # Exact-zero base bins legitimately create divide/invalid intermediates;
        # fill_nan plus the explicit finite/power mask below discards them.
        with np.errstate(divide="ignore", invalid="ignore"):
            reassigned_frequencies, _, magnitude = librosa.reassigned_spectrogram(
                y=values,
                sr=sample_rate,
                n_fft=n_fft,
                win_length=n_fft,
                hop_length=hop,
                center=True,
                reassign_frequencies=True,
                reassign_times=True,
                ref_power=1e-8,
                fill_nan=True,
                clip=True,
            )
        maximum_hz = min(float(maximum_hz), 0.5 * sample_rate)
        minimum_hz = max(float(minimum_hz), sample_rate / n_fft)
        minimum_midi = float(69.0 + 12.0 * np.log2(minimum_hz / 440.0))
        maximum_midi = float(69.0 + 12.0 * np.log2(maximum_hz / 440.0))
        log_bins = max(
            1,
            int(np.ceil((maximum_midi - minimum_midi) * bins_per_semitone)) + 1,
        )
        nominal_frames = magnitude.shape[1]
        log_magnitude = np.zeros((log_bins, nominal_frames), dtype=np.float32)

        frequencies = np.asarray(reassigned_frequencies, dtype=np.float64).reshape(-1)
        strengths = np.asarray(magnitude, dtype=np.float64).reshape(-1)
        nominal_time_indices = np.tile(
            np.arange(nominal_frames, dtype=int),
            magnitude.shape[0],
        )
        frame_reference = np.tile(
            np.max(magnitude, axis=0),
            magnitude.shape[0],
        )
        valid = (
            np.isfinite(frequencies)
            & np.isfinite(strengths)
            & (frequencies >= minimum_hz)
            & (frequencies <= maximum_hz)
            # Frequency reassignment is unstable in near-zero cells. A relative
            # per-frame floor is standard reassignment hygiene; the paper does not
            # state the value, so it remains adapter behavior.
            & (strengths >= reassignment_power_floor * frame_reference)
        )
        midi = 69.0 + 12.0 * np.log2(frequencies[valid] / 440.0)
        pitch_indices = np.rint((midi - minimum_midi) * bins_per_semitone).astype(int)
        # The source uses phase-vocoder frequency reassignment on a regular time
        # grid. Keep each cell in its nominal STFT frame rather than applying a
        # second time-reassignment step, which would puncture continuous traces.
        time_indices = nominal_time_indices[valid]
        valid_indices = (
            (pitch_indices >= 0)
            & (pitch_indices < log_bins)
            & (time_indices >= 0)
            & (time_indices < nominal_frames)
        )
        np.maximum.at(
            log_magnitude,
            (pitch_indices[valid_indices], time_indices[valid_indices]),
            strengths[valid][valid_indices].astype(np.float32),
        )

        keep = max(1, int(np.ceil(top_fraction * log_bins)))
        threshold_index = max(0, log_bins - keep)
        thresholds = np.partition(log_magnitude, threshold_index, axis=0)[
            threshold_index
        ]
        binary = (log_magnitude >= thresholds[None, :]) & (log_magnitude > 0.0)
        times = np.arange(nominal_frames, dtype=np.float64) / frame_rate
        return cls.Spectrogram(
            binary=binary.astype(np.float32),
            times=times,
            minimum_midi=minimum_midi,
            bins_per_semitone=bins_per_semitone,
        )

    @classmethod
    def build_template(
        cls,
        rate_hz: float,
        extent_cents: float,
        *,
        frame_rate: float = DRIEDGER_FRAME_RATE,
        bins_per_semitone: int = DRIEDGER_BINS_PER_SEMITONE,
        duration_seconds: float = DRIEDGER_TEMPLATE_SECONDS,
        positive_radius_bins: float = 0.5,
        negative_inner_bins: float = 7.0,
        negative_outer_bins: float = 9.0,
    ) -> Driedger.Template:
        frame_count = int(round(duration_seconds * frame_rate)) + 1
        times = np.arange(frame_count, dtype=np.float64) / frame_rate
        # Equation 2 and Figure 3 define e as the one-sided sinusoidal deviation:
        # e=50 cents produces a trace spanning -50 to +50 cents.
        offsets = (
            extent_cents
            / 100.0
            * bins_per_semitone
            * np.sin(2.0 * np.pi * rate_hz * times)
        )
        vertical_radius = int(np.ceil(np.max(np.abs(offsets)) + negative_outer_bins))
        vertical = np.arange(-vertical_radius, vertical_radius + 1, dtype=np.float64)
        distance = np.abs(vertical[:, None] - offsets[None, :])
        positive = distance <= positive_radius_bins
        negative = (distance >= negative_inner_bins) & (distance <= negative_outer_bins)
        values = np.zeros(distance.shape, dtype=np.float64)
        if np.any(positive):
            values[positive] = 1.0 / float(np.sum(positive))
        if np.any(negative):
            values[negative] = -1.0 / float(np.sum(negative))
        return cls.Template(
            rate_hz=float(rate_hz),
            extent_cents=float(extent_cents),
            values=values,
            positive_support=positive,
        )

    @staticmethod
    def _template_salience(
        binary_spectrogram: np.ndarray,
        template: Driedger.Template,
    ) -> np.ndarray:
        """Compute S_T using one FFT correlation and support-mask dilation."""

        correlation = fftconvolve(
            binary_spectrogram,
            template.values[::-1, ::-1],
            mode="same",
        )
        return grey_dilation(
            correlation,
            footprint=template.positive_support,
            mode="constant",
            cval=-np.inf,
        )

    @staticmethod
    def _template_frame_salience(
        binary_spectrogram: np.ndarray,
        template: Driedger.Template,
    ) -> np.ndarray:
        """Compute Equation 5's frequency maximum without its full 2-D image."""

        correlation = fftconvolve(
            binary_spectrogram,
            template.values[::-1, ::-1],
            mode="same",
        )
        frequency_bins, frame_count = correlation.shape
        unrestricted_max = np.max(correlation, axis=0)
        unrestricted_argmax = np.argmax(correlation, axis=0)
        center = np.asarray(template.positive_support.shape, dtype=int) // 2
        offsets = np.argwhere(template.positive_support) - center[None, :]

        # For an offset df, only correlation rows whose shifted output remains
        # inside the finite spectrogram are eligible. Usually the unrestricted
        # maximum is already eligible; recompute only the affected edge frames.
        maxima_by_frequency_offset: dict[int, np.ndarray] = {}
        frame_salience = np.full(frame_count, -np.inf, dtype=np.float64)
        for frequency_offset_raw, time_offset_raw in offsets:
            frequency_offset = int(frequency_offset_raw)
            time_offset = int(time_offset_raw)
            shifted_frequency_max = maxima_by_frequency_offset.get(frequency_offset)
            if shifted_frequency_max is None:
                lower = max(0, -frequency_offset)
                upper = min(frequency_bins, frequency_bins - frequency_offset)
                if lower >= upper:
                    shifted_frequency_max = np.full(
                        frame_count,
                        -np.inf,
                        dtype=np.float64,
                    )
                else:
                    unrestricted_is_valid = (unrestricted_argmax >= lower) & (
                        unrestricted_argmax < upper
                    )
                    if np.all(unrestricted_is_valid):
                        shifted_frequency_max = unrestricted_max
                    else:
                        shifted_frequency_max = unrestricted_max.copy()
                        invalid = ~unrestricted_is_valid
                        shifted_frequency_max[invalid] = np.max(
                            correlation[lower:upper, :][:, invalid],
                            axis=0,
                        )
                maxima_by_frequency_offset[frequency_offset] = shifted_frequency_max

            # scipy.ndimage.grey_dilation places an input sample at output index
            # ``input + footprint_offset``. Apply that time shift directly.
            if time_offset >= 0:
                if time_offset < frame_count:
                    frame_salience[time_offset:] = np.maximum(
                        frame_salience[time_offset:],
                        shifted_frequency_max[: frame_count - time_offset],
                    )
            elif -time_offset < frame_count:
                frame_salience[: frame_count + time_offset] = np.maximum(
                    frame_salience[: frame_count + time_offset],
                    shifted_frequency_max[-time_offset:],
                )
        return frame_salience

    name = "driedger"
    description = (
        "Driedger et al. F0-free reassigned-spectrogram templates "
        "(independent reproduction)"
    )
    requires = {"audio"}
    scores_center = False

    def __init__(
        self,
        *,
        template_mode: str = "detection",
        salience_threshold: float = DRIEDGER_SALIENCE_THRESHOLD,
        minimum_hz: float = DRIEDGER_GLOBAL_MINIMUM_HZ,
        maximum_hz: float = DRIEDGER_GLOBAL_MAXIMUM_HZ,
        reassignment_power_floor: float = 0.05,
        positive_radius_bins: float = 0.5,
        negative_inner_bins: float = 7.0,
        negative_outer_bins: float = 9.0,
    ) -> None:
        if template_mode not in {"detection", "analysis", "benchmark_range"}:
            raise ValueError(
                "template_mode must be 'detection', 'analysis', or " "'benchmark_range'"
            )
        self.template_mode = template_mode
        if template_mode == "benchmark_range":
            self.name = "driedger_benchmark_range"
            self.description = (
                "Driedger et al. F0-free templates with Attune's complete "
                "3--10 Hz / 10--200 cent benchmark range (ablation)"
            )
        self.salience_threshold = float(salience_threshold)
        self.minimum_hz = float(minimum_hz)
        self.maximum_hz = float(maximum_hz)
        self.reassignment_power_floor = float(reassignment_power_floor)
        self.positive_radius_bins = float(positive_radius_bins)
        self.negative_inner_bins = float(negative_inner_bins)
        self.negative_outer_bins = float(negative_outer_bins)
        self._prepared_path: str | None = None
        self._samples: np.ndarray | None = None
        self._sample_rate: int | None = None
        if template_mode == "detection":
            rates = DRIEDGER_DETECTION_RATES_HZ
            extents = DRIEDGER_DETECTION_EXTENTS_CENTS
        elif template_mode == "analysis":
            rates = DRIEDGER_ANALYSIS_RATES_HZ
            extents = DRIEDGER_ANALYSIS_EXTENTS_CENTS
        else:
            rates = DRIEDGER_BENCHMARK_RATES_HZ
            extents = DRIEDGER_BENCHMARK_EXTENTS_CENTS
        self.template_rates_hz = tuple(float(value) for value in rates)
        self.template_extents_cents = tuple(float(value) for value in extents)
        self.templates = tuple(
            self.build_template(
                rate,
                extent,
                positive_radius_bins=self.positive_radius_bins,
                negative_inner_bins=self.negative_inner_bins,
                negative_outer_bins=self.negative_outer_bins,
            )
            for rate in self.template_rates_hz
            for extent in self.template_extents_cents
        )

    def prepare(self, example: VibratoExample) -> None:
        """Decode once outside benchmark timing for one analysis group."""

        if example.audio_path is None:
            raise ValueError("Driedger estimator requires an audio_path")
        resolved = str(Path(example.audio_path).resolve())
        if resolved == self._prepared_path and self._samples is not None:
            return
        samples, sample_rate = sf.read(resolved, always_2d=True, dtype="float64")
        self._samples = np.mean(samples, axis=1)
        self._sample_rate = int(sample_rate)
        self._prepared_path = resolved

    def estimate(self, example: VibratoExample) -> VibratoEstimate:
        resolved = (
            str(Path(example.audio_path).resolve())
            if example.audio_path is not None
            else None
        )
        if resolved != self._prepared_path or self._samples is None:
            # Direct callers remain supported; VibratoBenchmarker invokes
            # prepare() before starting the method timer.
            self.prepare(example)
        assert self._samples is not None and self._sample_rate is not None
        minimum_hz, maximum_hz, frequency_range_source = (
            self._score_padded_frequency_range(
                example,
                fallback_minimum_hz=self.minimum_hz,
                fallback_maximum_hz=self.maximum_hz,
            )
        )
        spectrogram = self._reassigned_log_spectrogram(
            self._samples,
            self._sample_rate,
            minimum_hz=minimum_hz,
            maximum_hz=maximum_hz,
            reassignment_power_floor=self.reassignment_power_floor,
        )
        frame_count = len(spectrogram.times)
        best_salience = np.full(frame_count, -np.inf, dtype=np.float64)
        best_rates = np.zeros(frame_count, dtype=np.float64)
        best_widths = np.zeros(frame_count, dtype=np.float64)
        for template in self.templates:
            frame_salience = self._template_frame_salience(
                spectrogram.binary,
                template,
            )
            improve = frame_salience > best_salience
            best_salience[improve] = frame_salience[improve]
            best_rates[improve] = template.rate_hz
            # Driedger's e is one-sided; VibratoEstimate stores full
            # peak-to-peak width.
            best_widths[improve] = 2.0 * template.extent_cents

        nearest, covered = self.nearest_frame_map(example.times, spectrogram.times)
        rates = np.where(covered, best_rates[nearest], 0.0)
        widths = np.where(covered, best_widths[nearest], 0.0)
        quality = np.where(covered, best_salience[nearest], 0.0)
        detected = covered & (quality >= self.salience_threshold)
        return VibratoEstimate(
            rates,
            widths,
            detected,
            quality=quality,
            metadata={
                "source": "Driedger, Balke, Ewert, and Müller, ISMIR 2016",
                "port": "independent reproduction",
                "adapter_provenance": str(
                    Path(__file__).with_name("driedger_adapter_provenance.json")
                ),
                "template_mode": self.template_mode,
                "template_count": len(self.templates),
                "published_detection_configuration": (
                    self.template_mode == "detection"
                ),
                "benchmark_range_configuration": (
                    self.template_mode == "benchmark_range"
                ),
                "template_rates_hz": self.template_rates_hz,
                "template_extents_cents_one_sided": (self.template_extents_cents),
                "template_width_range_cents_peak_to_peak": (
                    2.0 * min(self.template_extents_cents),
                    2.0 * max(self.template_extents_cents),
                ),
                "salience_threshold": self.salience_threshold,
                "analysis_frame_rate": DRIEDGER_FRAME_RATE,
                "bins_per_semitone": DRIEDGER_BINS_PER_SEMITONE,
                "minimum_hz_adapter": minimum_hz,
                "maximum_hz_adapter": maximum_hz,
                "global_fallback_minimum_hz_adapter": self.minimum_hz,
                "global_fallback_maximum_hz_adapter": self.maximum_hz,
                "positive_radius_bins_adapter": self.positive_radius_bins,
                "negative_band_bins_adapter": (
                    self.negative_inner_bins,
                    self.negative_outer_bins,
                ),
                "reassignment_power_floor_adapter": self.reassignment_power_floor,
                "frequency_maximum_adapter": "whole configured log-frequency axis",
                "frequency_range_provenance": (
                    "Attune score range padded -2/+24 semitones; fixed global "
                    "fallback when no score is available"
                ),
                "frequency_range_source": frequency_range_source,
                "salience_reduction": (
                    "edge-exact per-frame reduction of Equation 5 dilation"
                ),
                "decoded_audio_path": self._prepared_path,
                "decoded_sample_rate": self._sample_rate,
                "native_frame_times": spectrogram.times,
                "native_salience": best_salience,
            },
        )
