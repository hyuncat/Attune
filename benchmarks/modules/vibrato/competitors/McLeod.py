"""Single-sine Prony detector ported from McLeod's Tartini implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from benchmarks.modules.vibrato.VibratoDetectorBase import (
    VibratoDetectorBase,
    VibratoEstimate,
    VibratoExample,
)


TARTINI_REFERENCE_SAMPLE_RATE = 44_100
TARTINI_REFERENCE_HOP = 1_024
TARTINI_PRONY_WINDOW_SECONDS = 0.4
TARTINI_PRONY_GAP_SAMPLES = 2
TARTINI_PRONY_MAX_MEAN_SQUARED_ERROR = 1.0
TARTINI_MATRIX_EPSILON = 1e-6
TARTINI_SOURCE_COMMIT = "54e4dbaea051dbf8c32acf8f634479d094219014"


class McLeod(VibratoDetectorBase):
    """McLeod/Tartini's released sliding single-sine Prony baseline."""

    @dataclass(frozen=True)
    class PronyFit:
        amplitude_semitones: float
        phase_radians: float
        omega_radians_per_sample: float
        center_midi: float
        mean_squared_error: float

        def rate_hz(self, frame_rate: float) -> float:
            return float(self.omega_radians_per_sample * frame_rate / (2.0 * np.pi))

    name = "mcleod"
    scores_center = False
    requires = {"pitch"}
    description = (
        "McLeod/Tartini 0.4 s sliding single-sine Prony estimator "
        "(source-faithful benchmark port)"
    )

    @staticmethod
    def _normal_equation_solve(
        design: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray | None:
        gram = design.T @ design
        determinant = float(np.linalg.det(gram))
        if not np.isfinite(determinant) or abs(determinant) < TARTINI_MATRIX_EPSILON:
            return None
        try:
            coefficients = np.linalg.solve(gram, design.T @ target)
        except np.linalg.LinAlgError:
            return None
        return coefficients if np.all(np.isfinite(coefficients)) else None

    @classmethod
    def fit_single_sine(
        cls,
        samples: np.ndarray,
        *,
        gap_samples: int = TARTINI_PRONY_GAP_SAMPLES,
    ) -> McLeod.PronyFit | None:
        values = np.asarray(samples, dtype=np.float64).reshape(-1)
        gap = int(gap_samples)
        if gap < 1 or len(values) <= 2 * gap or not np.all(np.isfinite(values)):
            return None

        shifted_sum = values[: -2 * gap] + values[2 * gap :]
        middle = values[gap:-gap]
        frequency_design = np.column_stack((np.ones(len(middle)), middle))
        frequency_coefficients = cls._normal_equation_solve(
            frequency_design, shifted_sum
        )
        if frequency_coefficients is None:
            return None

        cosine = float(frequency_coefficients[1]) / 2.0
        if cosine < -1.0 or cosine > 1.0:
            return None
        omega = float(np.arccos(cosine) / gap)
        if not np.isfinite(omega):
            return None

        indices = np.arange(len(values), dtype=np.float64)
        amplitude_design = np.column_stack(
            (
                np.ones(len(values)),
                np.cos(indices * omega),
                np.sin(indices * omega),
            )
        )
        coefficients = cls._normal_equation_solve(amplitude_design, values)
        if coefficients is None:
            return None
        center, cosine_coefficient, sine_coefficient = map(float, coefficients)
        amplitude = float(np.hypot(cosine_coefficient, sine_coefficient))
        phase = float(np.pi / 2.0 - np.arctan2(sine_coefficient, cosine_coefficient))
        prediction = amplitude * np.sin(indices * omega + phase) + center
        error = float(np.mean((prediction - values) ** 2))
        if not np.isfinite(error):
            return None
        return cls.PronyFit(amplitude, phase, omega, center, error)

    def __init__(
        self,
        *,
        reference_sample_rate: int = TARTINI_REFERENCE_SAMPLE_RATE,
        reference_hop: int = TARTINI_REFERENCE_HOP,
        window_seconds: float = TARTINI_PRONY_WINDOW_SECONDS,
        gap_samples: int = TARTINI_PRONY_GAP_SAMPLES,
        maximum_mean_squared_error: float = (TARTINI_PRONY_MAX_MEAN_SQUARED_ERROR),
    ) -> None:
        if reference_sample_rate <= 0 or reference_hop <= 0:
            raise ValueError("the Tartini reference grid must be positive")
        if window_seconds <= 0.0:
            raise ValueError("the Prony window must be positive")
        if gap_samples < 1:
            raise ValueError("the Prony delay must be at least one sample")
        if maximum_mean_squared_error < 0.0:
            raise ValueError("the Prony error threshold cannot be negative")
        self.reference_sample_rate = int(reference_sample_rate)
        self.reference_hop = int(reference_hop)
        self.window_seconds = float(window_seconds)
        self.gap_samples = int(gap_samples)
        self.maximum_mean_squared_error = float(maximum_mean_squared_error)

    @property
    def analysis_frame_rate(self) -> float:
        return self.reference_sample_rate / self.reference_hop

    def estimate(self, example: VibratoExample) -> VibratoEstimate:
        input_pitch = np.asarray(example.pitch_midi, dtype=np.float64)
        voiced = np.isfinite(input_pitch)
        n = len(input_pitch)
        zeros = np.zeros(n, dtype=np.float64)
        if not np.any(voiced):
            return VibratoEstimate(zeros, zeros.copy(), zeros.astype(bool))

        # Tartini's analysis track is numeric at every chunk.  A shared pYIN
        # contour can contain NaNs, so the adapter fills them before resampling;
        # this is exposed in metadata and should be ablated separately.
        input_indices = np.arange(n, dtype=np.float64)
        filled = np.interp(input_indices, input_indices[voiced], input_pitch[voiced])
        frame_rate = self.analysis_frame_rate
        native_count = (
            int(np.floor((example.times[-1] - example.times[0]) * frame_rate + 1e-9))
            + 1
        )
        native_times = example.times[0] + np.arange(native_count) / frame_rate
        native_pitch = np.interp(native_times, example.times, filled)

        window_size = int(np.ceil(self.window_seconds * frame_rate))
        if window_size <= 2 * self.gap_samples or native_count <= window_size:
            return VibratoEstimate(
                zeros,
                zeros.copy(),
                zeros.astype(bool),
                metadata={
                    "source": "McLeod thesis Chapter 9 / Tartini prony.cpp port",
                    "source_commit": TARTINI_SOURCE_COMMIT,
                    "analysis_frame_rate": frame_rate,
                    "window_samples": window_size,
                    "gap_samples": self.gap_samples,
                    "filled_unvoiced_frames": int(np.sum(~voiced)),
                },
            )

        native_rates = np.zeros(native_count, dtype=np.float64)
        native_widths = np.zeros(native_count, dtype=np.float64)
        native_centers = native_pitch.copy()
        native_errors = np.full(native_count, np.nan, dtype=np.float64)
        evaluated = np.zeros(native_count, dtype=bool)

        # This reproduces Channel::doPronyFit: at chunk C the input is
        # [C-window, C), and the result is stored at C-window/2.
        for chunk in range(window_size, native_count):
            start = chunk - window_size
            center_index = chunk - window_size // 2
            fit = self.fit_single_sine(
                native_pitch[start:chunk],
                gap_samples=self.gap_samples,
            )
            evaluated[center_index] = True
            if fit is None:
                continue
            native_errors[center_index] = fit.mean_squared_error
            if fit.mean_squared_error >= self.maximum_mean_squared_error:
                continue

            native_rates[center_index] = fit.rate_hz(frame_rate)
            # Tartini reports speed for a successful fit but withholds width
            # until at least one fitted cycle spans the complete window.
            if fit.omega_radians_per_sample * window_size < 2.0 * np.pi:
                continue
            native_widths[center_index] = 200.0 * fit.amplitude_semitones
            native_centers[center_index] = fit.center_midi

        native_indices = np.flatnonzero(evaluated)
        if not len(native_indices):
            return VibratoEstimate(zeros, zeros.copy(), zeros.astype(bool))
        output_times = native_times[native_indices]
        nearest = np.abs(example.times[:, None] - output_times[None, :]).argmin(axis=1)
        covered = (example.times >= output_times[0]) & (
            example.times <= output_times[-1]
        )
        mapped_indices = native_indices[nearest]
        rates = np.where(covered, native_rates[mapped_indices], 0.0)
        widths = np.where(covered, native_widths[mapped_indices], 0.0)
        centers = np.where(covered, native_centers[mapped_indices], np.nan)
        detected = covered & (rates > 0.0) & (widths > 0.0)
        return VibratoEstimate(
            rates,
            widths,
            detected,
            centers,
            metadata={
                "source": "McLeod thesis Chapter 9 / Tartini prony.cpp port",
                "source_commit": TARTINI_SOURCE_COMMIT,
                "analysis_frame_rate": frame_rate,
                "window_samples": window_size,
                "gap_samples": self.gap_samples,
                "maximum_mean_squared_error": self.maximum_mean_squared_error,
                "filled_unvoiced_frames": int(np.sum(~voiced)),
                "native_times": native_times,
                "native_rates_hz": native_rates,
                "native_widths_cents": native_widths,
                "native_errors": native_errors,
                "native_evaluated": evaluated,
            },
        )
