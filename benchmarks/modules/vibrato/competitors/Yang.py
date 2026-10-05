"""Python port of Yang/AVA's released Filter Diagonalisation Method."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import linalg
from scipy.io import loadmat

from benchmarks.modules.vibrato.VibratoDetectorBase import (
    VibratoDetectorBase,
    VibratoEstimate,
    VibratoExample,
)


AVA_SOURCE_COMMIT = "77e4dfe8affa6014112bdf23769b2a97dde0d2b7"
AVA_DECISION_RATE_LIMITS_HZ = (4.0, 9.0)
AVA_DECISION_MIN_EXTENT_SEMITONES = 0.10


class Yang(VibratoDetectorBase):
    """Shared AVA FDM implementation used by Yang's decision variants."""

    @dataclass(frozen=True)
    class FrameTrack:
        times: np.ndarray
        rate_hz: np.ndarray
        extent_semitones: np.ndarray
        decision_detected: np.ndarray
        detected: np.ndarray

    SOURCE_COMMIT = AVA_SOURCE_COMMIT
    DECISION_RATE_LIMITS_HZ = AVA_DECISION_RATE_LIMITS_HZ
    DECISION_MIN_EXTENT_SEMITONES = AVA_DECISION_MIN_EXTENT_SEMITONES

    @staticmethod
    def matlab_smooth(values: np.ndarray, span: int = 10) -> np.ndarray:
        """MATLAB ``smooth(y, span)`` moving average used by AVA.

        MATLAB reduces an even span to the next lower odd integer. Its endpoints
        use progressively larger odd windows rather than padding the signal.
        """
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        n = len(values)
        if n == 0:
            return values.copy()
        span = min(max(1, int(np.floor(span))), n)
        span = span - 1 + span % 2
        if span <= 1:
            return values.copy()

        half = (span - 1) // 2
        smoothed = np.empty(n, dtype=np.float64)
        for index in range(half):
            width = 2 * index + 1
            smoothed[index] = np.mean(values[:width])
            smoothed[n - index - 1] = np.mean(values[n - width :])
        kernel = np.ones(span, dtype=np.float64) / span
        smoothed[half : n - half] = np.convolve(values, kernel, mode="valid")
        return smoothed

    @staticmethod
    def _filter_matrices(
        samples: np.ndarray,
        z: np.ndarray,
        order_m: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Equation (25) in AVA's ``frameFDM3.m`` for p = 1, 2, 3."""
        component_count = len(z)
        matrices: list[np.ndarray] = []
        for p_index in range(3):
            fp = np.empty(component_count, dtype=np.complex128)
            gp = np.zeros(component_count, dtype=np.complex128)
            for j0 in range(component_count):
                fp[j0] = samples[p_index]
                for m in range(1, order_m + 1):
                    fp[j0] += z[j0] ** (-m) * samples[m + p_index]
                    gp[j0] += z[j0] ** (-(m - 1)) * samples[m + order_m + p_index]

            matrix = np.eye(component_count, dtype=np.complex128)
            for j0 in range(component_count):
                matrix[j0, j0] = samples[p_index]
                for m in range(1, 2 * order_m + 1):
                    matrix[j0, j0] += (
                        (order_m - abs(order_m - m) + 1)
                        * samples[m + p_index]
                        * z[j0] ** (-m)
                    )
                for j1 in range(j0 + 1, component_count):
                    denominator = z[j0] - z[j1]
                    matrix[j0, j1] = (
                        z[j0] * fp[j1]
                        - z[j1] * fp[j0]
                        - z[j0] ** (-order_m) * gp[j1]
                        + z[j1] ** (-order_m) * gp[j0]
                    ) / denominator
                    # AVA assigns the same value, not the complex conjugate.
                    matrix[j1, j0] = matrix[j0, j1]
            matrices.append(matrix)
        return matrices[0], matrices[1], matrices[2]

    @classmethod
    def frame_fdm3(
        cls,
        samples: np.ndarray,
        sample_rate: float,
        minimum_hz: float = 2.0,
        maximum_hz: float = 20.0,
        iterations: int = 4,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Port of AVA ``frameFDM3`` returning complex frequency and amplitude.

        Empty arrays represent the MATLAB failure branch (which returns scalar
        zeros). Numerical failures are handled at this frame boundary so one
        singular frame does not abort a complete benchmark track.
        """
        samples = np.asarray(samples, dtype=np.complex128).reshape(-1)
        if (
            len(samples) < 8
            or not np.all(np.isfinite(samples))
            or not np.isfinite(sample_rate)
            or sample_rate <= 0.0
            or minimum_hz >= maximum_hz
            or iterations < 1
        ):
            return np.empty(0, dtype=np.complex128), np.empty(0, dtype=np.complex128)

        n_order = len(samples) - 1
        tau = 1.0 / float(sample_rate)
        omega_min = 2.0 * np.pi * minimum_hz
        omega_max = 2.0 * np.pi * maximum_hz
        initial_count = max(
            2,
            int(np.ceil(n_order * tau / (4.0 * np.pi) * (omega_max - omega_min))),
        )
        phase = tau * np.linspace(omega_min, omega_max, initial_count)
        # MATLAB arrays retain their allocated tail when J shrinks. The amplitude
        # stage still iterates over J0, so preserve that storage behavior.
        z_storage = np.exp(-1j * phase).astype(np.complex128)
        component_count = initial_count
        order_m = int(np.floor((n_order - 1) / 2.0) - 1)
        if order_m < 1 or 2 * order_m + 2 >= len(samples):
            return np.empty(0, dtype=np.complex128), np.empty(0, dtype=np.complex128)

        error_scale = 1e4
        final_frequencies = np.empty(0, dtype=np.complex128)
        final_vectors = np.empty((0, 0), dtype=np.complex128)

        for iteration in range(iterations):
            if component_count < 1:
                return np.empty(0, dtype=np.complex128), np.empty(
                    0, dtype=np.complex128
                )
            z = z_storage[:component_count]
            try:
                # The released iteration can temporarily place a basis root at
                # zero or merge two roots. Its MATLAB arithmetic produces Inf/NaN
                # candidates which the following rejection branch discards. NumPy
                # warns for those same intermediates, so keep the source behavior
                # without flooding benchmark stderr.
                with np.errstate(
                    divide="ignore",
                    invalid="ignore",
                    over="ignore",
                    under="ignore",
                ):
                    y0, y1, y2 = cls._filter_matrices(samples, z, order_m)
                y1 = np.where(np.isnan(y1) | np.isinf(y1), 1.0 + 0.0j, y1)
                y0 = np.where(np.isnan(y0) | np.isinf(y0), 0.0 + 0.0j, y0)

                eigenvalues, eigenvectors = linalg.eig(
                    y1,
                    y0,
                    check_finite=False,
                )
                eigenvalue_matrix = np.diag(eigenvalues)
                # MATLAB uses non-conjugating transpose here (VY.'), then applies
                # a second eigendecomposition to normalize the eigenvectors.
                normalization_matrix = eigenvectors.T @ y0 @ eigenvectors
                norm_eigenvalues, norm_eigenvectors = linalg.eig(
                    normalization_matrix,
                    check_finite=False,
                )
                with np.errstate(all="ignore"):
                    correction = (
                        norm_eigenvectors
                        @ np.diag(norm_eigenvalues**-0.5)
                        @ linalg.inv(norm_eigenvectors, check_finite=False)
                    )
                eigenvectors = eigenvectors @ correction
            except (ValueError, FloatingPointError, linalg.LinAlgError):
                return np.empty(0, dtype=np.complex128), np.empty(
                    0, dtype=np.complex128
                )

            vector_norms = np.linalg.norm(eigenvectors, axis=0)
            with np.errstate(all="ignore"):
                residual_norms = np.asarray(
                    [
                        np.linalg.norm(
                            (y2 - eigenvalue**2 * y0) @ eigenvectors[:, index]
                        )
                        for index, eigenvalue in enumerate(eigenvalues)
                    ]
                )
            if (
                not np.all(np.isfinite(vector_norms))
                or not np.all(np.isfinite(residual_norms))
                or len(vector_norms) == 0
            ):
                return np.empty(0, dtype=np.complex128), np.empty(
                    0, dtype=np.complex128
                )
            accepted = (vector_norms < error_scale * np.min(vector_norms)) & (
                residual_norms < error_scale * np.min(residual_norms)
            )
            if not np.any(accepted):
                return np.empty(0, dtype=np.complex128), np.empty(
                    0, dtype=np.complex128
                )

            accepted_values = eigenvalues[accepted]
            with np.errstate(all="ignore"):
                final_frequencies = 1j / (2.0 * np.pi * tau) * np.log(accepted_values)
            final_vectors = eigenvectors[:, accepted]
            component_count = len(accepted_values)
            if iteration < iterations - 1:
                z_storage[:component_count] = accepted_values

        # Equation (27), including AVA's iterative-grid behavior.
        resonant_z = np.exp(-1j * tau * (2.0 * np.pi * final_frequencies))
        ud = np.empty((initial_count, component_count), dtype=np.complex128)
        for j in range(initial_count):
            for component in range(component_count):
                if abs(z_storage[j] - resonant_z[component]) < 1e-12:
                    value = samples[0]
                    for m in range(1, 2 * order_m + 1):
                        value += (
                            (order_m - abs(order_m - m) + 1)
                            * samples[m]
                            * z_storage[j] ** (-m)
                        )
                else:
                    value = samples[0] * (z_storage[j] - resonant_z[component])
                    for m in range(1, order_m + 1):
                        value += samples[m] * (
                            z_storage[j] * resonant_z[component] ** (-m)
                            - resonant_z[component] * z_storage[j] ** (-m)
                        )
                        value += samples[m + order_m] * (
                            resonant_z[component] ** (-order_m)
                            * z_storage[j] ** (-(m - 1))
                            - z_storage[j] ** (-order_m)
                            * resonant_z[component] ** (-(m - 1))
                        )
                    value /= z_storage[j] - resonant_z[component]
                ud[j, component] = value

        amplitudes = np.empty(component_count, dtype=np.complex128)
        # After the final rejection pass MATLAB sums j=1:J, even if the input
        # basis to that pass had more rows. Preserve that indexing literally.
        vectors_for_amplitude = final_vectors[:component_count, :component_count]
        for component in range(component_count):
            value = np.sum(
                vectors_for_amplitude[:, component] * ud[:component_count, component]
            )
            amplitudes[component] = value**2 / (order_m + 1) ** 2
        return final_frequencies, amplitudes

    @staticmethod
    def delete_outlier(detected: np.ndarray) -> np.ndarray:
        """Port AVA's two sequential ``deleteOutlier.m`` cleanup passes."""
        detected = np.asarray(detected, dtype=bool).reshape(-1)
        # Port the two sequential, in-place cleanup passes literally. First remove
        # isolated positive frames; then fill isolated zero frames.
        if len(detected) < 2:
            return detected.copy()
        cleaned = detected.copy()
        for index in range(len(cleaned)):
            if index == 0:
                if cleaned[index] and not cleaned[index + 1]:
                    cleaned[index] = False
            elif index == len(cleaned) - 1:
                if cleaned[index] and not cleaned[index - 1]:
                    cleaned[index] = False
            elif not cleaned[index - 1] and not cleaned[index + 1]:
                cleaned[index] = False
        for index in range(len(cleaned)):
            if index == 0:
                if not cleaned[index] and cleaned[index + 1]:
                    cleaned[index] = True
            elif index == len(cleaned) - 1:
                if not cleaned[index] and cleaned[index - 1]:
                    cleaned[index] = True
            elif cleaned[index - 1] and cleaned[index + 1]:
                cleaned[index] = True
        return cleaned

    @classmethod
    def decision_tree(
        cls,
        rate_hz: np.ndarray,
        extent_semitones: np.ndarray,
        rate_limits_hz: tuple[float, float] = AVA_DECISION_RATE_LIMITS_HZ,
        minimum_extent_semitones: float = AVA_DECISION_MIN_EXTENT_SEMITONES,
    ) -> np.ndarray:
        """AVA ``DecisionTree.m`` followed by ``deleteOutlier.m``."""
        rate_hz = np.asarray(rate_hz, dtype=np.float64)
        extent_semitones = np.asarray(extent_semitones, dtype=np.float64)
        detected = (
            (rate_hz >= rate_limits_hz[0])
            & (rate_hz <= rate_limits_hz[1])
            & (extent_semitones >= minimum_extent_semitones)
        )
        return cls.delete_outlier(detected)

    @staticmethod
    def candidate_mask(
        decision_detected: np.ndarray,
        times: np.ndarray,
        *,
        frame_criterion: int = 5,
        duration_threshold_seconds: float = 0.25,
    ) -> np.ndarray:
        """Group and prune AVA decision frames into vibrato candidates."""
        decision_detected = np.asarray(decision_detected, dtype=bool).reshape(-1)
        times = np.asarray(times, dtype=np.float64).reshape(-1)
        if len(decision_detected) != len(times):
            raise ValueError("decision and time arrays must have the same length")
        output = np.zeros(len(decision_detected), dtype=bool)
        indices = np.flatnonzero(decision_detected)
        if len(indices) == 0:
            return output

        gaps = np.flatnonzero(np.diff(indices) > 1)
        run_start_positions = np.r_[0, gaps + 1]
        run_end_positions = np.r_[gaps, len(indices) - 1]
        starts = indices[run_start_positions]
        ends = indices[run_end_positions]
        keep = (ends - starts) >= frame_criterion
        keep &= (times[ends] - times[starts]) >= duration_threshold_seconds
        for start_frame, end_frame in zip(starts[keep], ends[keep]):
            output[start_frame : end_frame + 1] = True
        return output

    @classmethod
    def analyze_pitch_track(
        cls,
        midi_pitch: np.ndarray,
        times: np.ndarray,
        *,
        minimum_hz: float = 2.0,
        maximum_hz: float = 20.0,
        iterations: int = 4,
        window_seconds: float = 0.125,
        hop_fraction: float = 0.25,
        rate_limits_hz: tuple[float, float] = (4.0, 9.0),
        minimum_extent_semitones: float = 0.10,
    ) -> Yang.FrameTrack:
        """Run AVA's frame-wise FDM analysis on a MIDI pitch contour."""
        midi_pitch = np.asarray(midi_pitch, dtype=np.float64).reshape(-1)
        times = np.asarray(times, dtype=np.float64).reshape(-1)
        if len(midi_pitch) != len(times) or len(times) < 2:
            raise ValueError("pitch and time arrays must have the same length >= 2")
        sample_rate = 1.0 / float(times[1] - times[0])
        if not np.isfinite(sample_rate) or sample_rate <= 0.0:
            raise ValueError("the first two AVA pitch times must increase")

        smoothed = cls.matlab_smooth(midi_pitch, span=10)
        window_length = int(np.floor(window_seconds * sample_rate))
        step = int(np.floor(window_length * hop_fraction))
        if window_length < 8 or step < 1 or len(smoothed) <= window_length:
            empty_float = np.empty(0, dtype=np.float64)
            return cls.FrameTrack(
                empty_float,
                empty_float.copy(),
                empty_float.copy(),
                np.empty(0, dtype=bool),
                np.empty(0, dtype=bool),
            )

        rates: list[float] = []
        extents: list[float] = []
        pin = 0
        pend = len(smoothed) - window_length
        while pin < pend:
            frame = smoothed[pin : pin + window_length]
            frame = frame - np.mean(frame)
            frequencies, amplitudes = cls.frame_fdm3(
                frame,
                sample_rate,
                minimum_hz,
                maximum_hz,
                iterations,
            )
            amplitude_magnitudes = np.abs(amplitudes)
            usable = (
                np.isfinite(np.real(frequencies))
                & np.isfinite(np.imag(frequencies))
                & np.isfinite(amplitude_magnitudes)
                & (np.real(frequencies) > 0.0)
                & (np.real(frequencies) >= minimum_hz)
                & (np.real(frequencies) <= maximum_hz)
            )
            if not np.any(usable):
                rates.append(np.nan)
                extents.append(np.nan)
            else:
                candidate_frequencies = frequencies[usable]
                candidate_extents = 2.0 * amplitude_magnitudes[usable]
                largest = int(np.argmax(candidate_extents))
                rates.append(float(np.real(candidate_frequencies[largest])))
                extents.append(float(candidate_extents[largest]))
            pin += step

        rates_array = np.asarray(rates, dtype=np.float64)
        extents_array = np.asarray(extents, dtype=np.float64)
        frame_times = (
            times[0]
            + window_length / (2.0 * sample_rate)
            + np.arange(len(rates_array), dtype=np.float64) * step / sample_rate
        )
        decision_detected = cls.decision_tree(
            rates_array,
            extents_array,
            rate_limits_hz,
            minimum_extent_semitones,
        )
        detected = cls.candidate_mask(decision_detected, frame_times)
        return cls.FrameTrack(
            frame_times,
            rates_array,
            extents_array,
            decision_detected,
            detected,
        )


class YangDT(Yang):
    """Yang/AVA FDM with the released decision tree."""

    name = "yang_dt"
    description = "Yang/AVA FDM with the released decision tree"

    def __init__(
        self,
        *,
        window_sec: float = 0.125,
        hop_fraction: float = 0.25,
        filter_min_hz: float = 2.0,
        filter_max_hz: float = 20.0,
        decision_min_hz: float = Yang.DECISION_RATE_LIMITS_HZ[0],
        decision_max_hz: float = Yang.DECISION_RATE_LIMITS_HZ[1],
        min_extent_semitones: float = Yang.DECISION_MIN_EXTENT_SEMITONES,
        iterations: int = 4,
    ) -> None:
        self.window_sec = float(window_sec)
        self.hop_fraction = float(hop_fraction)
        self.filter_min_hz = float(filter_min_hz)
        self.filter_max_hz = float(filter_max_hz)
        self.decision_min_hz = float(decision_min_hz)
        self.decision_max_hz = float(decision_max_hz)
        self.min_extent_semitones = float(min_extent_semitones)
        self.iterations = int(iterations)

    def _track(
        self,
        example: VibratoExample,
    ) -> tuple[Yang.FrameTrack, np.ndarray]:
        raw_values = np.asarray(example.pitch_midi, dtype=np.float64)
        voiced = np.isfinite(raw_values)
        track = self.analyze_pitch_track(
            np.where(voiced, raw_values, 0.0),
            example.times,
            minimum_hz=self.filter_min_hz,
            maximum_hz=self.filter_max_hz,
            iterations=self.iterations,
            window_seconds=self.window_sec,
            hop_fraction=self.hop_fraction,
            rate_limits_hz=(self.decision_min_hz, self.decision_max_hz),
            minimum_extent_semitones=self.min_extent_semitones,
        )
        return track, voiced

    def estimate(self, example: VibratoExample) -> VibratoEstimate:
        track, voiced = self._track(example)
        if len(track.times) == 0:
            zeros = np.zeros(len(example.times), dtype=np.float64)
            return VibratoEstimate(zeros, zeros.copy(), zeros.astype(bool))
        return self._adapt_track(
            example,
            track,
            voiced,
            decision_detected=track.decision_detected,
            detected=track.detected,
            decision_rule="decision_tree",
        )

    @staticmethod
    def _adapt_track(
        example: VibratoExample,
        track: Yang.FrameTrack,
        voiced: np.ndarray,
        *,
        decision_detected: np.ndarray,
        detected: np.ndarray,
        decision_rule: str,
        extra_metadata: dict | None = None,
    ) -> VibratoEstimate:
        nearest = np.abs(example.times[:, None] - track.times[None, :]).argmin(axis=1)
        covered = (example.times >= track.times[0]) & (example.times <= track.times[-1])
        rates = np.where(
            np.isfinite(track.rate_hz[nearest]), track.rate_hz[nearest], 0.0
        )
        widths = np.where(
            np.isfinite(track.extent_semitones[nearest]),
            200.0 * track.extent_semitones[nearest],
            0.0,
        )
        output_detected = detected[nearest].copy()
        supported = covered & voiced
        output_detected &= supported
        rates[~supported] = 0.0
        widths[~supported] = 0.0
        metadata = {
            "source": "AVA frameFDM3/vibratoDetectFunc Python port",
            "decision_rule": decision_rule,
            "frame_times": track.times,
            "frame_rates_hz": track.rate_hz,
            "frame_extents_semitones": track.extent_semitones,
            "frame_decision_detected": decision_detected,
            "frame_detected": detected,
            "unvoiced_adapter": "AVA freqToMidi MIDI-zero sentinel",
            "unvoiced_frames": int(np.sum(~voiced)),
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        return VibratoEstimate(rates, widths, output_detected, metadata=metadata)


class YangBR(YangDT):
    """Yang/AVA FDM with the released Bayes-rule KDE classifier."""

    name = "yang_br"
    description = "Yang/AVA FDM with the released Bayes-rule KDE classifier"
    DEFAULT_MODEL_PATH = (
        Path(__file__).resolve().parent
        / "data"
        / "yang"
        / "ava_br_kde_models_plain.mat"
    )
    PROBABILITY_THRESHOLD = 0.25

    @dataclass(frozen=True)
    class Kernel:
        samples: np.ndarray
        frequency: np.ndarray
        bandwidth: float

        def pdf(self, values: np.ndarray, *, chunk_size: int = 256) -> np.ndarray:
            values = np.asarray(values, dtype=np.float64)
            flat = values.reshape(-1)
            density = np.full(len(flat), np.nan, dtype=np.float64)
            positions = np.flatnonzero(~np.isnan(flat))
            normalizer = (
                np.sqrt(2.0 * np.pi) * self.bandwidth * float(np.sum(self.frequency))
            )
            for start in range(0, len(positions), chunk_size):
                selected = positions[start : start + chunk_size]
                with np.errstate(over="ignore", invalid="ignore"):
                    standardized = (
                        flat[selected, None] - self.samples[None, :]
                    ) / self.bandwidth
                    kernels = np.exp(-0.5 * standardized * standardized)
                density[selected] = (kernels @ self.frequency) / normalizer
            return density.reshape(values.shape)

    @dataclass(frozen=True)
    class Decision:
        rate_posterior: np.ndarray
        extent_posterior: np.ndarray
        probability: np.ndarray
        raw_detected: np.ndarray
        decision_detected: np.ndarray

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model_path = Path(model_path or self.DEFAULT_MODEL_PATH)
        self._models: dict[str, YangBR.Kernel] | None = None

    @classmethod
    def _kernel(cls, models: dict, variable: str, path: Path) -> Kernel:
        try:
            plain = models[variable]
        except (KeyError, TypeError) as error:
            raise ValueError(f"{path} does not contain brModels.{variable}") from error
        if (
            str(plain["kernel"]).lower() != "normal"
            or str(plain["support"]).lower() != "unbounded"
            or bool(plain["is_truncated"])
        ):
            raise ValueError(f"unsupported AVA KDE definition for {variable}")
        samples = np.asarray(plain["samples"], dtype=np.float64).reshape(-1)
        frequency = np.asarray(plain["frequency"], dtype=np.float64).reshape(-1)
        censored = np.asarray(plain["censored"], dtype=bool).reshape(-1)
        bandwidth = float(plain["bandwidth"])
        if (
            len(samples) == 0
            or len(frequency) != len(samples)
            or len(censored) != len(samples)
            or np.any(censored)
            or not np.all(np.isfinite(samples))
            or not np.all(np.isfinite(frequency))
            or np.any(frequency < 0.0)
            or np.sum(frequency) <= 0.0
            or not np.isfinite(bandwidth)
            or bandwidth <= 0.0
        ):
            raise ValueError(f"invalid AVA KDE parameters for {variable}")
        samples.setflags(write=False)
        frequency.setflags(write=False)
        return cls.Kernel(samples, frequency, bandwidth)

    def _load_models(self) -> dict[str, Kernel]:
        if self._models is not None:
            return self._models
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"AVA Bayes-rule model export was not found: {self.model_path}"
            )
        payload = loadmat(self.model_path, simplify_cells=True)
        try:
            models = payload["brModels"]
        except KeyError as error:
            raise ValueError(f"{self.model_path} does not contain brModels") from error
        self._models = {
            name: self._kernel(models, name, self.model_path)
            for name in ("pdVR", "pdNR", "pdVA", "pdNA")
        }
        return self._models

    @staticmethod
    def _posterior(vibrato_pdf: np.ndarray, nonvibrato_pdf: np.ndarray) -> np.ndarray:
        denominator = vibrato_pdf + nonvibrato_pdf
        posterior = np.full_like(denominator, np.nan, dtype=np.float64)
        np.divide(
            vibrato_pdf,
            denominator,
            out=posterior,
            where=denominator != 0.0,
        )
        return posterior

    def _evaluate_bayes(
        self,
        rate_hz: np.ndarray,
        extent_semitones: np.ndarray,
    ) -> Decision:
        if rate_hz.shape != extent_semitones.shape or rate_hz.ndim != 1:
            raise ValueError("AVA rate and extent must be matching 1-D arrays")
        models = self._load_models()
        rate_posterior = self._posterior(
            models["pdVR"].pdf(rate_hz), models["pdNR"].pdf(rate_hz)
        )
        extent_posterior = self._posterior(
            models["pdVA"].pdf(extent_semitones),
            models["pdNA"].pdf(extent_semitones),
        )
        probability = rate_posterior * extent_posterior
        raw_detected = probability >= self.PROBABILITY_THRESHOLD
        return self.Decision(
            rate_posterior,
            extent_posterior,
            probability,
            raw_detected,
            self.delete_outlier(raw_detected),
        )

    def estimate(self, example: VibratoExample) -> VibratoEstimate:
        track, voiced = self._track(example)
        if len(track.times) == 0:
            zeros = np.zeros(len(example.times), dtype=np.float64)
            return VibratoEstimate(zeros, zeros.copy(), zeros.astype(bool))
        bayes = self._evaluate_bayes(track.rate_hz, track.extent_semitones)
        detected = self.candidate_mask(bayes.decision_detected, track.times)
        return self._adapt_track(
            example,
            track,
            voiced,
            decision_detected=bayes.decision_detected,
            detected=detected,
            decision_rule="bayes_rule",
            extra_metadata={
                "frame_br_rate_posterior": bayes.rate_posterior,
                "frame_br_extent_posterior": bayes.extent_posterior,
                "frame_br_probability": bayes.probability,
                "frame_br_raw_detected": bayes.raw_detected,
            },
        )
