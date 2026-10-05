"""Local-extrema detector after Rossignol's 2000 thesis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from algorithms.Config import Config
from benchmarks.modules.vibrato.VibratoDetectorBase import (
    VibratoDetectorBase,
    VibratoEstimate,
    VibratoExample,
)


ROSSIGNOL_F0_FRAME_RATE_HZ = 100.0
ROSSIGNOL_PORTION_SECONDS = 0.35
ROSSIGNOL_MINIMUM_PB = 0.02
ATTUNE_RATE_LIMITS_HZ = (
    float(Config.vib2_min_rate_hz),
    float(Config.vib2_max_rate_hz),
)
ATTUNE_MINIMUM_WIDTH_CENTS = float(Config.vib2_min_width_cents)


class Rossignol(VibratoDetectorBase):
    @dataclass(frozen=True)
    class ExtremaRun:
        extrema: tuple[InterpolatedExtremum, ...]
        rate_hz: np.ndarray
        width_cents: np.ndarray
        center_midi: np.ndarray
        max_interval_cv: float
        min_interval_cv: float
        mfreq: float

    @staticmethod
    def _coefficient_of_variation(values: np.ndarray) -> float:
        mean = float(np.mean(values)) if len(values) else 0.0
        return float(np.std(values) / mean) if mean > 0.0 else np.inf

    @staticmethod
    def _midi_to_hz(values: np.ndarray) -> np.ndarray:
        return 440.0 * np.power(2.0, (values - 69.0) / 12.0)

    @staticmethod
    def _hz_to_midi(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            return 69.0 + 12.0 * np.log2(values / 440.0)

    @staticmethod
    def _hz_span_to_cents(upper: np.ndarray, lower: np.ndarray) -> np.ndarray:
        upper = np.asarray(upper, dtype=np.float64)
        lower = np.asarray(lower, dtype=np.float64)
        valid = (
            np.isfinite(upper) & np.isfinite(lower) & (upper > lower) & (lower > 0.0)
        )
        output = np.zeros(len(upper), dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            output[valid] = 1200.0 * np.log2(upper[valid] / lower[valid])
        return output

    @classmethod
    def measure_extrema_run(
        cls,
        extrema: (
            list[VibratoDetectorBase.Extremum]
            | tuple[VibratoDetectorBase.Extremum, ...]
        ),
    ) -> Rossignol.ExtremaRun:
        """Measure adjacent half-cycles using Prame's conventions."""

        points = tuple(extrema)
        rates: list[float] = []
        widths: list[float] = []
        centers: list[float] = []
        relative_extents: list[float] = []
        for left, right in zip(points, points[1:]):
            half_period = right.time - left.time
            rates.append(0.5 / half_period if half_period > 0.0 else 0.0)
            widths.append(100.0 * abs(right.value - left.value))
            centers.append(0.5 * (right.value + left.value))
            frequencies = cls._midi_to_hz(np.asarray([left.value, right.value]))
            high, low = float(np.max(frequencies)), float(np.min(frequencies))
            relative_extents.append((high - low) / max(0.5 * (high + low), 1e-12))

        maxima = np.asarray([point.time for point in points if point.kind > 0])
        minima = np.asarray([point.time for point in points if point.kind < 0])
        return cls.ExtremaRun(
            extrema=points,
            rate_hz=np.asarray(rates, dtype=np.float64),
            width_cents=np.asarray(widths, dtype=np.float64),
            center_midi=np.asarray(centers, dtype=np.float64),
            max_interval_cv=cls._coefficient_of_variation(np.diff(maxima)),
            min_interval_cv=cls._coefficient_of_variation(np.diff(minima)),
            mfreq=float(np.mean(relative_extents)) if relative_extents else 0.0,
        )

    @staticmethod
    def _envelope_trajectory(
        points: np.ndarray,
        values: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        """Linearly interpolate Rossignol's upper or lower envelope."""

        if len(points) == 0:
            return np.full(len(times), np.nan, dtype=np.float64)
        if len(points) == 1:
            return np.full(len(times), float(values[0]), dtype=np.float64)
        return np.interp(times, points, values)

    @staticmethod
    def _portion_counts(
        event_times: np.ndarray,
        times: np.ndarray,
        portion_seconds: float,
    ) -> np.ndarray:
        """Number of events inside the portion centred on every frame."""

        half = 0.5 * portion_seconds
        right = np.searchsorted(event_times, times + half, side="right")
        left = np.searchsorted(event_times, times - half, side="left")
        return (right - left).astype(np.float64)

    @staticmethod
    def _portion_mean(
        values: np.ndarray,
        times: np.ndarray,
        portion_seconds: float,
    ) -> np.ndarray:
        """Mean of a per-frame quantity over the portion centred on each frame."""

        finite = np.isfinite(values)
        filled = np.where(finite, values, 0.0)
        totals = np.concatenate(([0.0], np.cumsum(filled)))
        counts = np.concatenate(([0.0], np.cumsum(finite.astype(np.float64))))
        half = 0.5 * portion_seconds
        right = np.searchsorted(times, times + half, side="right")
        left = np.searchsorted(times, times - half, side="left")
        span = counts[right] - counts[left]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(span > 0, (totals[right] - totals[left]) / span, np.nan)

    @staticmethod
    def _mean_of_two(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Thesis 13.3.2's arithmetic mean of max- and min-derived rates."""

        supported = np.isfinite(left) & np.isfinite(right)
        output = np.full(len(left), np.nan, dtype=np.float64)
        output[supported] = 0.5 * (left[supported] + right[supported])
        return output

    @staticmethod
    def _portion_rate(
        event_times: np.ndarray,
        times: np.ndarray,
        portion_seconds: float,
    ) -> np.ndarray:
        """Thesis 13.3.2: mean of 1/(P_{j+1} - P_j) over the portion's intervals."""

        if len(event_times) < 2:
            return np.full(len(times), np.nan, dtype=np.float64)
        intervals = np.diff(event_times)
        rates = np.where(intervals > 0.0, 1.0 / np.maximum(intervals, 1e-12), np.nan)
        finite = np.isfinite(rates)
        totals = np.concatenate(([0.0], np.cumsum(np.where(finite, rates, 0.0))))
        counts = np.concatenate(([0.0], np.cumsum(finite.astype(np.float64))))
        half = 0.5 * portion_seconds
        # An interval belongs to a portion only when both of its bounding extrema
        # are in that portion.  Selecting by midpoint would admit measurements
        # whose extrema lie outside the analysis window.
        first_event = np.searchsorted(event_times, times - half, side="left")
        event_stop = np.searchsorted(event_times, times + half, side="right")
        left = np.minimum(first_event, len(rates))
        right = np.minimum(np.maximum(event_stop - 1, 0), len(rates))
        span = counts[right] - counts[left]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(span > 0, (totals[right] - totals[left]) / span, np.nan)

    @staticmethod
    def _analysis_note_bounds(
        example: VibratoExample,
    ) -> tuple[list[tuple[float, float, float]], str]:
        """Return validated half-open note bounds for note-local analysis."""

        raw_bounds = example.metadata.get("analysis_note_bounds")
        source = "metadata.analysis_note_bounds"
        if not isinstance(raw_bounds, (list, tuple)) or not raw_bounds:
            start, end = example.scored_time_bounds
            center = np.asarray(example.center_midi, dtype=np.float64)
            representative = center[example.score_mask & np.isfinite(center)]
            midi = float(np.median(representative)) if representative.size else 60.0
            raw_bounds = [(start, end, midi)]
            source = "scored_time_bounds_fallback"

        bounds_out: list[tuple[float, float, float]] = []
        for bounds in raw_bounds:
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 3:
                raise ValueError(
                    "analysis_note_bounds entries must be (start, end, midi) triples"
                )
            start, end, midi = map(float, bounds)
            if not np.all(np.isfinite((start, end, midi))) or end <= start:
                raise ValueError(f"invalid analysis note bounds: {bounds!r}")
            bounds_out.append((start, end, midi))
        return sorted(bounds_out), source

    """Independent reproduction of Rossignol thesis section 13.3."""

    name = "rossignol"
    description = (
        "Rossignol's note-local 0.35 s minima/maxima F0-contour method "
        "with thesis pb decision"
    )
    requires = {"pitch"}
    scores_center = False

    def __init__(
        self,
        *,
        portion_seconds: float = ROSSIGNOL_PORTION_SECONDS,
        minimum_pb: float = ROSSIGNOL_MINIMUM_PB,
        minimum_rate_hz: float = ATTUNE_RATE_LIMITS_HZ[0],
        maximum_rate_hz: float = ATTUNE_RATE_LIMITS_HZ[1],
        minimum_width_cents: float = ATTUNE_MINIMUM_WIDTH_CENTS,
    ) -> None:
        self.portion_seconds = float(portion_seconds)
        self.minimum_pb = float(minimum_pb)
        self.minimum_rate_hz = float(minimum_rate_hz)
        self.maximum_rate_hz = float(maximum_rate_hz)
        self.minimum_width_cents = float(minimum_width_cents)
        if not np.isfinite(self.portion_seconds) or self.portion_seconds <= 0.0:
            raise ValueError("portion_seconds must be positive and finite")
        if not np.isfinite(self.minimum_pb) or self.minimum_pb < 0.0:
            raise ValueError("minimum_pb must be finite and non-negative")
        if (
            not np.isfinite(self.minimum_rate_hz)
            or not np.isfinite(self.maximum_rate_hz)
            or self.minimum_rate_hz < 0.0
            or self.maximum_rate_hz < self.minimum_rate_hz
        ):
            raise ValueError("invalid rate acceptance limits")
        if not np.isfinite(self.minimum_width_cents) or self.minimum_width_cents < 0.0:
            raise ValueError("minimum_width_cents must be finite and non-negative")

    @property
    def rate_limits_hz(self) -> tuple[float, float]:
        return self.minimum_rate_hz, self.maximum_rate_hz

    def _meets_note_gate(
        self,
        rates_hz: np.ndarray,
        widths_cents: np.ndarray,
    ) -> bool:
        """Mirror ``VibratoDetector._meets_detection_floor`` for one note."""

        supported = np.isfinite(rates_hz) & np.isfinite(widths_cents)
        if not supported.any():
            return False
        median_rate_hz = float(np.median(rates_hz[supported]))
        median_width_cents = float(np.median(widths_cents[supported]))
        return (
            median_rate_hz >= self.minimum_rate_hz
            and median_rate_hz <= self.maximum_rate_hz
            and median_width_cents >= self.minimum_width_cents
        )

    def estimate(self, example: VibratoExample) -> VibratoEstimate:
        times = np.asarray(example.times, dtype=np.float64)
        pitch_midi = np.asarray(example.pitch_midi, dtype=np.float64)
        n = len(times)
        rates = np.zeros(n, dtype=np.float64)
        widths = np.zeros(n, dtype=np.float64)
        centers = np.full(n, np.nan, dtype=np.float64)
        detected = np.zeros(n, dtype=bool)
        qualities = np.zeros(n, dtype=np.float64)
        note_bounds, note_bounds_source = self._analysis_note_bounds(example)

        total_extrema = 0
        filled_unvoiced_frames = 0
        analyzed_frames = 0
        complete_portion_frames = 0
        accepted_notes = 0
        accepted_frames = 0
        detected_distfreq_hz: list[np.ndarray] = []
        for start, end, _midi in note_bounds:
            lo = int(np.searchsorted(times, start, side="left"))
            hi = int(np.searchsorted(times, end, side="left"))
            lo = max(0, min(n, lo))
            hi = max(lo, min(n, hi))
            if hi - lo < 3:
                continue

            note_times = times[lo:hi]
            filled_midi, voiced = self.fill_unvoiced(pitch_midi[lo:hi])
            analyzed_frames += hi - lo
            filled_unvoiced_frames += int(np.sum(~voiced))
            if not np.any(voiced):
                continue

            # Rossignol operates on f0 in Hz.  The supplied MIDI contour is
            # converted only after note-local interpolation; there is no
            # Savitzky--Golay or other method-owned smoothing here.  Section
            # 13.1 fixes the f0-contour sampling frequency at 100 Hz, so each
            # note is independently resampled to that grid before extrema are
            # selected.  The regular-grid conversion is interpolation, not a
            # smoothing filter.
            analysis_count = max(
                1,
                int(np.ceil((end - start) * ROSSIGNOL_F0_FRAME_RATE_HZ - 1e-12)),
            )
            analysis_times = (
                start
                + np.arange(analysis_count, dtype=np.float64)
                / ROSSIGNOL_F0_FRAME_RATE_HZ
            )
            analysis_times = analysis_times[analysis_times < end]
            if len(analysis_times) < 3:
                continue
            analysis_midi = np.interp(analysis_times, note_times, filled_midi)
            filled_hz = self._midi_to_hz(analysis_midi)
            extrema = self.interpolated_extrema(
                filled_hz,
                analysis_times,
                prominence=None,
                same_kind_distance=1,
            )
            total_extrema += len(extrema)
            if len(extrema) < 4:
                continue

            extrema_times = np.asarray(
                [point.time for point in extrema], dtype=np.float64
            )
            extrema_kinds = np.asarray([point.kind for point in extrema])
            extrema_hz = np.asarray(
                [point.value for point in extrema], dtype=np.float64
            )
            maxima = extrema_kinds > 0
            minima = extrema_kinds < 0
            max_times = extrema_times[maxima]
            min_times = extrema_times[minima]
            if len(max_times) < 2 or len(min_times) < 2:
                continue

            maxinterp_hz = self._envelope_trajectory(
                max_times, extrema_hz[maxima], analysis_times
            )
            mininterp_hz = self._envelope_trajectory(
                min_times, extrema_hz[minima], analysis_times
            )
            center_hz = 0.5 * (maxinterp_hz + mininterp_hz)
            extent_hz = np.maximum(maxinterp_hz - mininterp_hz, 0.0)
            width_cents = self._hz_span_to_cents(maxinterp_hz, mininterp_hz)
            center_midi = self._hz_to_midi(center_hz)
            mfreq_frame = extent_hz / np.maximum(center_hz, 1e-12)

            count_max = self._portion_counts(
                max_times, analysis_times, self.portion_seconds
            )
            count_min = self._portion_counts(
                min_times, analysis_times, self.portion_seconds
            )
            mfreq_local = self._portion_mean(
                mfreq_frame, analysis_times, self.portion_seconds
            )
            pb = (
                np.exp(-((count_max - 2.0) ** 2) / 3.0)
                * np.exp(-((count_min - 2.0) ** 2) / 3.0)
                * np.nan_to_num(mfreq_local, nan=0.0)
            )
            rate_max = self._portion_rate(
                max_times, analysis_times, self.portion_seconds
            )
            rate_min = self._portion_rate(
                min_times, analysis_times, self.portion_seconds
            )
            analysis_rates = self._mean_of_two(rate_max, rate_min)

            # Thesis 13.1 says analyzed portions must not overlap note
            # transitions.  Requiring the complete centered portion to fit
            # inside the note preserves the published 0.35 s geometry.
            half = 0.5 * self.portion_seconds
            analysis_complete_portion = (analysis_times - start >= half - 1e-12) & (
                end - analysis_times >= half - 1e-12
            )
            analysis_candidate = (
                analysis_complete_portion
                & (pb >= self.minimum_pb)
                & np.isfinite(analysis_rates)
                & np.isfinite(width_cents)
            )
            if not np.any(analysis_candidate):
                continue

            # Nearest-neighbour adaptation changes only the output grid.  All
            # extrema, windows, envelopes, and decisions above remain on the
            # thesis's 100 Hz analysis grid and within this one note.
            right = np.searchsorted(analysis_times, note_times, side="left")
            right = np.clip(right, 0, len(analysis_times) - 1)
            left = np.maximum(right - 1, 0)
            use_left = np.abs(note_times - analysis_times[left]) <= np.abs(
                analysis_times[right] - note_times
            )
            nearest = np.where(use_left, left, right)
            complete_portion = (note_times - start >= half - 1e-12) & (
                end - note_times >= half - 1e-12
            )
            complete_portion_frames += int(np.sum(complete_portion))
            candidate = analysis_candidate[nearest] & complete_portion
            note_rates = analysis_rates[nearest]
            note_widths = width_cents[nearest]
            note_centers = center_midi[nearest]
            note_pb = pb[nearest]
            note_extent_hz = extent_hz[nearest]
            if not np.any(candidate):
                continue
            if not self._meets_note_gate(note_rates[candidate], note_widths[candidate]):
                continue

            accepted_notes += 1
            accepted_frames += int(np.sum(candidate))
            destination = slice(lo, hi)
            rates[destination] = np.where(candidate, note_rates, rates[destination])
            widths[destination] = np.where(candidate, note_widths, widths[destination])
            centers[destination] = np.where(
                candidate, note_centers, centers[destination]
            )
            qualities[destination] = np.where(
                candidate, note_pb, qualities[destination]
            )
            detected[destination] |= candidate
            detected_distfreq_hz.append(note_extent_hz[candidate])

        mean_mfreq_hz = (
            float(np.mean(np.concatenate(detected_distfreq_hz)))
            if detected_distfreq_hz
            else 0.0
        )
        return VibratoEstimate(
            rates,
            widths,
            detected,
            centers,
            qualities,
            metadata=self._metadata(
                note_bounds=note_bounds,
                note_bounds_source=note_bounds_source,
                total_extrema=total_extrema,
                filled_unvoiced_frames=filled_unvoiced_frames,
                analyzed_frames=analyzed_frames,
                complete_portion_frames=complete_portion_frames,
                accepted_notes=accepted_notes,
                accepted_frames=accepted_frames,
                mean_mfreq_hz=mean_mfreq_hz,
            ),
        )

    def _metadata(
        self,
        *,
        note_bounds: list[tuple[float, float, float]],
        note_bounds_source: str,
        total_extrema: int,
        filled_unvoiced_frames: int,
        analyzed_frames: int,
        complete_portion_frames: int,
        accepted_notes: int,
        accepted_frames: int,
        mean_mfreq_hz: float,
    ) -> dict:
        return {
            "source": "Rossignol thesis (HAL tel-00010732), sections 13.1, 13.3, and 16",
            "port": "independent pYIN-fronted reproduction",
            "contour_front_end": "HMM-smoothed pYIN supplied by Attune",
            "method_owned_smoothing": "none",
            "analysis_frame_rate_hz": ROSSIGNOL_F0_FRAME_RATE_HZ,
            "decision_scope": "local_within_note",
            "decision_statistics": (
                "pb = exp(-(NBmax-2)^2/3) exp(-(NBmin-2)^2/3) Mfreq"
            ),
            "portion_seconds": self.portion_seconds,
            "minimum_pb": self.minimum_pb,
            "minimum_pb_source": "Rossignol thesis chapter 16",
            "envelope_domain": "Hz",
            "envelope_interpolation": "linear",
            "extrema_interpolation": "three-point parabolic",
            "extrema_prominence": None,
            "unvoiced_adapter": "linear interpolation independently within each note",
            "note_boundary_policy": (
                "extrema, interpolation, envelopes, and complete centered portions "
                "are confined to half-open note bounds"
            ),
            "note_bounds_source": note_bounds_source,
            "analysis_note_bounds": note_bounds,
            "attune_note_gate": "median rate and full peak-to-peak width",
            "rate_limits_hz": self.rate_limits_hz,
            "minimum_width_cents": self.minimum_width_cents,
            "mean_mfreq_hz": mean_mfreq_hz,
            "filled_unvoiced_frames": filled_unvoiced_frames,
            "analyzed_note_frames": analyzed_frames,
            "complete_portion_frames": complete_portion_frames,
            "interpolated_extrema": total_extrema,
            "accepted_notes": accepted_notes,
            "accepted_frames": accepted_frames,
        }
