"""Whole-note and live vibrato estimation.

- detect, extend: completed notes and incremental pitch frames
- run, notify, stop: background worker lifecycle
- _note_frame_spans, _analysis_spans: analysis boundaries
- _live_initial_parameters, _remember_live_fit: causal fit state
- _fit_curves, _fit_joint_model: rate, width, center, and confidence
- _seed_rate_and_phase, _solve_linear_model: numerical fitting
"""

import threading
from dataclasses import dataclass
from math import ceil, floor

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import least_squares

from algorithms.Config import Config
from app_logic.user.ds.VibratoData import VibratoData


@dataclass
class _CurveFit:
    rates: np.ndarray
    widths: np.ndarray
    qualities: np.ndarray
    centers: np.ndarray
    phase_parameters: np.ndarray | None = None


@dataclass
class _LiveFitState:
    phase_parameters: np.ndarray
    source_lo: int
    source_hi: int
    last_global_source_hi: int


class VibratoDetector:
    """Fit a smooth sinusoid to each note or trailing live window."""

    ALT_PROM_FRAC = 0.3
    # Zero rate makes the amplitude and center splines indistinguishable.
    RATE_FLOOR_HZ = 1e-6
    LIVE_GLOBAL_REACQUIRE_SEC = 2.0

    def __init__(self, recording=None, config: Config = None):
        if not recording and not config:
            raise ValueError("Must provide either a recording or a config to initialize the VibratoDetector.")
        self.recording = recording
        self.config = config if config else recording.config

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._wake = threading.Event()

        self._live_fit_state: _LiveFitState | None = None

    def update_config(self, config: Config):
        self.config = config
        self._live_fit_state = None

    def detect(self, pitch_data, note_data=None) -> VibratoData:
        """Fit completed notes; return rate (Hz), peak-to-peak width (cents), and center (MIDI)."""
        vd = VibratoData(config=self.config)
        vd.t_origin = pitch_data.t_origin
        available = pitch_data.frames_available()
        if available <= 0:
            return vd

        first_written = next(
            (i for i, pitch in enumerate(pitch_data.data[:available])
             if pitch is not None),
            available,
        )
        if first_written >= available:
            return vd
        last_written = next(
            (i for i in range(available - 1, first_written - 1, -1)
             if pitch_data.data[i] is not None),
            first_written,
        )
        vd.source_first_index = first_written
        stride = vd.stride
        first_grid = ceil(first_written / stride)
        last_grid = last_written // stride

        if first_grid:
            vd.write(first_grid - 1, np.nan, np.nan, np.nan)
        for i in range(first_grid, last_grid + 1):
            vd.write(i, 0.0, 0.0, 0.0)

        note_aware = note_data is not None and bool(
            getattr(note_data, "times", None)
        )
        regions: list[tuple[int, int]] = []
        if note_aware:
            regions = self._note_frame_spans(pitch_data, note_data, available)
        else:
            regions = [(first_written, last_written + 1)]

        frame_rate = self.config.sr / self.config.h1
        for lo, hi in regions:

            spans = (
                [(self._note_values(pitch_data, lo, hi), lo, hi)]
                if note_aware
                else self._analysis_spans(pitch_data, lo, hi)
            )
            for vals, source_lo, source_hi in spans:
                confidence = (
                    self._onset_confidence(len(vals), frame_rate)
                    if note_aware else None
                )
                fit = self._fit_curves(
                    vals,
                    frame_rate,
                    causal=False,
                    confidence=confidence,
                )
                for source_i in range(source_lo, source_hi):
                    if source_i % stride:
                        continue
                    grid_i = source_i // stride
                    local_i = source_i - source_lo
                    vd.write(
                        grid_i,
                        fit.rates[local_i],
                        fit.widths[local_i],
                        fit.qualities[local_i],
                        fit.centers[local_i],
                    )
        return vd

    def extend(self, vibrato_data: VibratoData, pitch_data,
               finalize: bool = False, note_data=None) -> None:
        """Update live frames, or replace the result with a completed-note pass."""
        if finalize or note_data is not None:
            fresh = self.detect(pitch_data, note_data=note_data)
            self._replace_data(vibrato_data, fresh)
            return

        cfg = self.config
        frame_rate = cfg.sr / cfg.h1
        history = max(8, int(round(cfg.vib2_live_sec * frame_rate)))
        available = pitch_data.frames_available()
        if available <= 0:
            return
        if vibrato_data.computed_until == 0:
            vibrato_data.t_origin = pitch_data.t_origin

        first_written = vibrato_data.source_first_index
        if first_written is None:
            first_written = next(
                (j for j, pitch in enumerate(pitch_data.data[:available])
                 if pitch is not None),
                available,
            )
            if first_written < available:
                vibrato_data.source_first_index = first_written
        if first_written >= available:
            return

        last_written = next(
            (j for j in range(available - 1, first_written - 1, -1)
             if pitch_data.data[j] is not None),
            first_written,
        )
        stride = vibrato_data.stride
        first_grid = ceil(first_written / stride)
        i = max(vibrato_data.computed_until, first_grid)
        if i == first_grid and first_grid:
            vibrato_data.write(first_grid - 1, np.nan, np.nan, np.nan)

        last_grid = last_written // stride
        analysis_step = max(
            1,
            int(round(frame_rate / float(cfg.vib2_live_analysis_hz))),
        )
        force_tail = self._stop.is_set()
        pending = last_grid - i + 1
        if pending <= 0 or (not force_tail and pending < analysis_step):
            return

        # Skip stale anchors under load; interpolate their output frames below.
        anchors = [last_grid]

        for anchor in anchors:
            center = anchor * stride
            lo = max(first_written, center - history + 1)
            segment = self._live_fit_segment(
                pitch_data,
                lo,
                center + 1,
                center,
            )
            if segment is None:
                self._live_fit_state = None
                current = np.array([0.0, 0.0, 0.0, np.nan])
            else:
                vals, source_lo, source_hi = segment
                initial = self._live_initial_parameters(
                    source_lo,
                    source_hi,
                    frame_rate,
                )
                fit = self._fit_curves(
                    vals,
                    frame_rate,
                    causal=True,
                    initial_phase_parameters=initial,
                )
                used_global_seed = initial is None

                if initial is not None and not self._usable_live_fit(fit):
                    fit = self._fit_curves(
                        vals,
                        frame_rate,
                        causal=True,
                    )
                    used_global_seed = True
                self._remember_live_fit(
                    fit,
                    source_lo,
                    source_hi,
                    used_global_seed=used_global_seed,
                )
                current = np.array([
                    fit.rates[-1],
                    fit.widths[-1],
                    fit.qualities[-1],
                    fit.centers[-1],
                ])

            if i > first_grid:
                with vibrato_data.lock:
                    previous = np.array([
                        vibrato_data.rates[i - 1],
                        vibrato_data.widths[i - 1],
                        vibrato_data.qualities[i - 1],
                        vibrato_data.centers[i - 1],
                    ], dtype=np.float64)
            else:
                previous = current.copy()
            previous = np.where(np.isfinite(previous), previous, current)
            span = anchor - i + 1
            for offset, grid_i in enumerate(range(i, anchor + 1), start=1):
                fraction = offset / span
                value = previous + fraction * (current - previous)
                vibrato_data.write(grid_i, *value)
            i = anchor + 1

    def run(self):
        """Start the worker after stopping any previous take."""
        self._live_fit_state = None
        self.stop()
        self._stop.clear()
        self._wake.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def notify(self):
        self._wake.set()

    def stop(self):
        """Stop the worker after its final flush."""
        if self._thread and self._thread.is_alive():
            self._stop.set()
            self._wake.set()
            self._thread.join()
        self._thread = None
        self._live_fit_state = None

    def _run(self):
        """Process pitch updates and flush the final tail before stopping."""
        rec = self.recording
        while not self._stop.is_set():
            self._wake.wait(0.1)
            self._wake.clear()
            self.extend(rec.vibrato_data, rec.pitch_data)
        self.extend(rec.vibrato_data, rec.pitch_data)

    def _note_frame_spans(self, pitch_data, note_data, available: int):
        """Convert note times to half-open frame-center bounds containing voiced pitch."""
        cfg = self.config
        regions = []
        notes = note_data.read(i=0, j=len(note_data.times), clean=True)
        for note in notes:
            pos0 = (
                ((note.start_time - pitch_data.t_origin) * cfg.sr - 0.5 * cfg.w1)
                / cfg.h1
            )
            pos1 = (
                ((note.end_time - pitch_data.t_origin) * cfg.sr - 0.5 * cfg.w1)
                / cfg.h1
            )
            lo = max(0, int(ceil(pos0)))
            hi = min(available, int(floor(pos1)) + 1)
            if lo < hi and any(
                pitch_data.is_voiced_pitch(pitch)
                for pitch in pitch_data.data[lo:hi]
            ):
                regions.append((lo, hi))
        return regions

    @staticmethod
    def _note_values(pitch_data, lo: int, hi: int) -> np.ndarray:
        """Keep missing and transition frames as NaN within the final note boundaries."""
        vals = np.full(hi - lo, np.nan, dtype=np.float64)
        for j, pitch in enumerate(pitch_data.data[lo:hi]):
            if pitch_data.is_voiced_pitch(pitch, include_transitions=False):
                vals[j] = pitch.value
        return vals

    def _analysis_spans(self, pitch_data, lo: int, hi: int):
        """Yield provisional/live spans before final note boundaries exist."""
        if lo >= hi:
            return
        vals = self._note_values(pitch_data, lo, hi)

        voiced = np.flatnonzero(np.isfinite(vals))
        if not len(voiced):
            return
        max_gap = max(
            0,
            int(round(
                self.config.vib_max_gap_sec
                * self.config.sr / self.config.h1
            )),
        )
        start = previous = int(voiced[0])
        for current_raw in voiced[1:]:
            current = int(current_raw)
            if current - previous - 1 > max_gap:
                yield vals[start:previous + 1], lo + start, lo + previous + 1
                start = current
            previous = current
        yield vals[start:previous + 1], lo + start, lo + previous + 1

    def _live_fit_segment(self, pitch_data, lo: int, hi: int,
                          center: int):
        """Return the last voiced span causally reachable from ``center``."""
        last = None
        for span in self._analysis_spans(pitch_data, lo, hi):
            last = span
        if last is None:
            return None
        _, _, source_hi = last
        max_gap = max(
            0,
            int(round(
                self.config.vib_max_gap_sec
                * self.config.sr / self.config.h1
            )),
        )
        if center - (source_hi - 1) > max_gap:
            return None
        return last

    def _onset_confidence(self, frames: int, frame_rate: float) -> np.ndarray:
        """Taper onset weights without discarding frames or exceeding the duration cap."""
        confidence = np.ones(max(0, int(frames)), dtype=np.float64)
        if frames <= 0:
            return confidence
        seconds = float(self.config.vib2_onset_taper_seconds)
        max_fraction = float(self.config.vib2_onset_taper_max_fraction)
        floor_confidence = float(self.config.vib2_onset_taper_floor)
        if not np.isfinite(seconds) or seconds < 0.0:
            raise ValueError("vib2_onset_taper_seconds must be finite and non-negative")
        if not np.isfinite(max_fraction) or not 0.0 <= max_fraction < 1.0:
            raise ValueError("vib2_onset_taper_max_fraction must be in [0, 1)")
        if (
            not np.isfinite(floor_confidence)
            or not 0.0 < floor_confidence <= 1.0
        ):
            raise ValueError("vib2_onset_taper_floor must be in (0, 1]")
        if seconds == 0.0 or max_fraction == 0.0 or floor_confidence == 1.0:
            return confidence
        if not np.isfinite(frame_rate) or frame_rate <= 0.0:
            raise ValueError("frame_rate must be finite and positive")
        requested = max(0, int(round(seconds * frame_rate)))
        fraction_cap = max(0, int(np.floor(max_fraction * frames)))
        taper_frames = min(requested, fraction_cap)
        if taper_frames <= 0:
            return confidence
        progress = np.arange(taper_frames, dtype=np.float64) / taper_frames
        raised_cosine = 0.5 * (1.0 - np.cos(np.pi * progress))
        confidence[:taper_frames] = (
            floor_confidence + (1.0 - floor_confidence) * raised_cosine
        )
        return confidence

    @staticmethod
    def _replace_data(target: VibratoData, source: VibratoData) -> None:
        """Replace a caller-owned drop-in result without replacing its lock."""
        with target.lock, source.lock:
            target.rates = source.rates.copy()
            target.widths = source.widths.copy()
            target.qualities = source.qualities.copy()
            target.centers = source.centers.copy()
            target.computed_until = source.computed_until
            target.source_first_index = source.source_first_index
            target.t_origin = source.t_origin

    @staticmethod
    def _usable_live_fit(fit: _CurveFit) -> bool:
        return (
            fit.phase_parameters is not None
            and len(fit.rates) > 0
            and np.isfinite(fit.rates[-1])
            and fit.rates[-1] > 0.0
        )

    def _remember_live_fit(
            self,
            fit: _CurveFit,
            source_lo: int,
            source_hi: int,
            *,
            used_global_seed: bool,
    ) -> None:
        if not self._usable_live_fit(fit):
            self._live_fit_state = None
            return
        previous = self._live_fit_state
        last_global = (
            source_hi
            if used_global_seed or previous is None
            else previous.last_global_source_hi
        )
        self._live_fit_state = _LiveFitState(
            np.asarray(fit.phase_parameters, dtype=np.float64).copy(),
            int(source_lo),
            int(source_hi),
            int(last_global),
        )

    def _live_initial_parameters(
            self,
            source_lo: int,
            source_hi: int,
            frame_rate: float,
    ) -> np.ndarray | None:
        state = self._live_fit_state
        if state is None:
            return None
        if (
            source_lo >= state.source_hi
            or source_hi <= state.source_lo
            or source_hi <= source_lo
        ):
            return None
        since_global = (source_hi - state.last_global_source_hi) / frame_rate
        if since_global >= self.LIVE_GLOBAL_REACQUIRE_SEC:
            return None
        return self._shift_live_parameters(
            state.phase_parameters,
            state.source_lo,
            state.source_hi,
            source_lo,
            source_hi,
            frame_rate,
        )

    @staticmethod
    def _shift_live_parameters(
            parameters: np.ndarray,
            old_lo: int,
            old_hi: int,
            new_lo: int,
            new_hi: int,
            frame_rate: float,
    ) -> np.ndarray | None:
        """Shift the quadratic rate model; hold its endpoint rate beyond the old window."""
        parameters = np.asarray(parameters, dtype=np.float64)
        old_span = old_hi - old_lo - 1
        new_span = new_hi - new_lo - 1
        if (
            parameters.shape != (4,)
            or not np.isfinite(parameters).all()
            or old_span <= 0
            or new_span <= 0
            or not np.isfinite(frame_rate)
            or frame_rate <= 0.0
        ):
            return None

        def rate_at(u):
            u = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
            return (
                parameters[1] * (1.0 - u) ** 2
                + 2.0 * parameters[2] * u * (1.0 - u)
                + parameters[3] * u ** 2
            )

        new_positions = np.array([
            float(new_lo),
            0.5 * (new_lo + new_hi - 1),
            float(new_hi - 1),
        ])
        old_u = (new_positions - old_lo) / old_span
        sampled_rates = rate_at(old_u)
        rate0 = float(sampled_rates[0])
        rate2 = float(sampled_rates[2])

        rate1 = float(
            2.0 * sampled_rates[1] - 0.5 * (rate0 + rate2)
        )

        start_u = float(np.clip((new_lo - old_lo) / old_span, 0.0, 1.0))
        integrated_basis = np.array([
            start_u - start_u ** 2 + start_u ** 3 / 3.0,
            start_u ** 2 - 2.0 * start_u ** 3 / 3.0,
            start_u ** 3 / 3.0,
        ])
        old_duration = old_span / frame_rate
        phase = float(
            parameters[0]
            + 2.0 * np.pi * old_duration
            * np.dot(integrated_basis, parameters[1:])
        )
        phase = (phase + np.pi) % (2.0 * np.pi) - np.pi
        shifted = np.array([phase, rate0, rate1, rate2], dtype=np.float64)
        return shifted if np.isfinite(shifted).all() else None

    def _fit_curves(self, vals: np.ndarray, frame_rate: float,
                    causal: bool,
                    confidence: np.ndarray | None = None,
                    initial_phase_parameters: np.ndarray | None = None,
                    ) -> _CurveFit:
        """Fit phase/speed, center, and width jointly on original pitches."""
        vals = np.asarray(vals, dtype=np.float64)
        n = len(vals)
        voiced = np.isfinite(vals)
        zeros = np.zeros(n, dtype=np.float64)
        if not n:
            return _CurveFit(zeros, zeros.copy(), zeros.copy(), zeros.copy())

        indices = np.arange(n, dtype=np.float64)
        if voiced.any():
            filled = np.interp(indices, indices[voiced], vals[voiced])
        else:
            return _CurveFit(
                zeros,
                zeros.copy(),
                zeros.copy(),
                np.full(n, np.nan),
            )
        if confidence is None:
            confidence = np.ones(n, dtype=np.float64)
        else:
            confidence = np.asarray(confidence, dtype=np.float64)
            if confidence.shape != (n,):
                raise ValueError("confidence must match the fitted pitch segment")
            if not np.isfinite(confidence).all() or np.any(confidence <= 0.0):
                raise ValueError("every fit confidence must be positive and finite")
            confidence = np.minimum(confidence, 1.0)
        fallback_center = np.full(n, float(np.median(filled[voiced])))

        times = indices / frame_rate
        model = self._fit_joint_model(
            filled,
            times,
            voiced,
            causal=causal,
            confidence=confidence,
            initial_phase_parameters=initial_phase_parameters,
        )
        if model is None:
            return _CurveFit(zeros, zeros.copy(), zeros.copy(), fallback_center)
        phase, rates, centers, widths, prediction, parameters = model

        residual = filled - centers
        real_residual = residual[voiced]
        real_confidence = confidence[voiced]
        variance = float(np.dot(real_confidence, real_residual ** 2))
        if variance <= np.finfo(float).eps * max(1.0, float(real_confidence.sum())):
            return _CurveFit(
                zeros, zeros.copy(), zeros.copy(), centers, parameters,
            )
        error = real_residual - prediction[voiced]
        squared_error_ratio = float(np.dot(real_confidence, error ** 2)) / variance
        rms_ratio = np.sqrt(max(0.0, squared_error_ratio))
        if rms_ratio >= float(self.config.vib2_max_rms_ratio):
            return _CurveFit(
                zeros, zeros.copy(), zeros.copy(), centers, parameters,
            )
        quality = 1.0 - squared_error_ratio

        amplitude_midi = widths / 200.0
        weighted_error = np.sqrt(real_confidence) * error
        noise = 1.4826 * float(np.median(
            np.abs(weighted_error - np.median(weighted_error))
        ))
        local_amp = np.maximum(amplitude_midi, np.finfo(float).eps)
        local_evidence = local_amp ** 2 / (local_amp ** 2 + noise ** 2)
        qualities = np.clip(quality * local_evidence, 0.0, 1.0)
        if not causal:
            rates, widths, qualities = self._stabilize_offline_edges(
                rates,
                widths,
                qualities,
                float(np.median(rates)),
                frame_rate,
                hold_values=bool(self.config.vib2_hold_edge_values),
            )
        if not self._meets_detection_floor(rates, widths):
            return _CurveFit(
                zeros,
                zeros.copy(),
                zeros.copy(),
                fallback_center,
                parameters,
            )
        return _CurveFit(rates, widths, qualities, centers, parameters)

    def _fit_joint_model(self, values: np.ndarray, times: np.ndarray,
                         voiced: np.ndarray, causal: bool,
                         confidence: np.ndarray | None = None,
                         initial_phase_parameters: np.ndarray | None = None):
        """Filter-free variable projection of phase, center, and amplitude."""
        duration = float(times[-1] - times[0])
        if duration <= 0.0:
            return None
        fit_lower = max(
            self.RATE_FLOOR_HZ,
            float(self.config.vib2_fit_rate_min_hz),
        )
        fit_upper = min(
            float(self.config.vib2_fit_rate_max_hz),
            self._max_resolvable_rate_hz(self.config.sr / self.config.h1),
        )
        if not np.isfinite(fit_upper) or fit_lower >= fit_upper:
            return None
        basis, _, _ = self._spline_basis(times, causal=causal)
        scan_bins = (
            self.config.vib2_live_rate_scan_bins
            if causal else self.config.vib2_rate_scan_bins
        )
        confidence = (
            np.ones(len(values), dtype=np.float64)
            if confidence is None
            else np.asarray(confidence, dtype=np.float64)
        )
        rate_basis = self._rate_basis(times)
        initial = None
        if initial_phase_parameters is not None:
            candidate = np.asarray(
                initial_phase_parameters,
                dtype=np.float64,
            )
            if candidate.shape == (4,) and np.isfinite(candidate).all():
                initial = candidate.copy()
                initial[0] = (initial[0] + np.pi) % (2.0 * np.pi) - np.pi
        if initial is None:
            seed = self._seed_rate_and_phase(
                values,
                times,
                basis,
                voiced,
                scan_bins=scan_bins,
                confidence=confidence,
            )
            if seed is None:
                return None
            rate0, phase0, amplitude0 = seed
            initial = self._initial_phase_parameters(
                values,
                times,
                rate0,
                phase0,
                amplitude0,
                fit_upper,
            )
        initial[1:] = np.clip(initial[1:], fit_lower, fit_upper)
        phase_parameter_jacobian = self._phase_parameter_jacobian(
            times,
            rate_basis,
        )

        cached_parameters = None
        cached_evaluation = None

        def evaluate(parameters: np.ndarray):
            nonlocal cached_parameters, cached_evaluation
            if (
                cached_parameters is None
                or not np.array_equal(parameters, cached_parameters)
            ):
                cached_parameters = np.asarray(
                    parameters,
                    dtype=np.float64,
                ).copy()
                cached_evaluation = self._variable_projection_residual_jacobian(
                    cached_parameters,
                    values,
                    times,
                    rate_basis,
                    basis,
                    voiced,
                    confidence,
                    phase_parameter_jacobian,
                )
            return cached_evaluation

        def objective(parameters: np.ndarray) -> np.ndarray:
            return evaluate(parameters)[1]

        def jacobian(parameters: np.ndarray) -> np.ndarray:
            return evaluate(parameters)[2]

        lower = np.array([
            -4.0 * np.pi,
            *([fit_lower] * 3),
        ])
        upper = np.array([
            4.0 * np.pi,
            *([fit_upper] * 3),
        ])
        try:
            optimized = least_squares(
                objective,
                initial,
                jac=jacobian,
                bounds=(lower, upper),
                max_nfev=(
                    min(12, int(self.config.vib2_max_nfev))
                    if causal else int(self.config.vib2_max_nfev)
                ),
                ftol=1e-5,
                xtol=1e-5,
                gtol=1e-5,
            )
            parameters = optimized.x if np.isfinite(optimized.x).all() else initial
        except (ValueError, np.linalg.LinAlgError):
            parameters = initial

        phase, rates = self._phase_from_parameters(parameters, times, rate_basis)
        try:
            coefficients = evaluate(parameters)[0]
        except (ValueError, np.linalg.LinAlgError):
            coefficients, _, _ = self._solve_linear_model(
                values,
                phase,
                basis,
                voiced,
                confidence=confidence,
            )
        m = basis.shape[1]
        centers = basis @ coefficients[:m]
        amplitude = basis @ coefficients[m:]
        prediction = amplitude * np.sin(phase)
        widths = 200.0 * np.abs(amplitude)
        if not all(np.isfinite(a).all() for a in (phase, rates, centers, widths, prediction)):
            return None
        amplitude_max = float(self.config.vib2_fit_amplitude_max_semitones)
        if (
            not np.isfinite(amplitude_max)
            or amplitude_max < 0.0
            or np.median(np.abs(amplitude)) > amplitude_max
        ):
            return None
        return phase, rates, centers, widths, prediction, parameters.copy()

    def _meets_detection_floor(
        self,
        rates_hz: np.ndarray,
        widths_cents: np.ndarray,
    ) -> bool:
        """Whether one fitted note is large and fast enough to be vibrato."""
        supported = np.isfinite(rates_hz) & np.isfinite(widths_cents)
        if not supported.any():
            return False
        median_rate_hz = float(np.median(rates_hz[supported]))
        median_width_cents = float(np.median(widths_cents[supported]))
        return (
            median_rate_hz >= max(0.0, float(self.config.vib2_min_rate_hz))
            and median_rate_hz
            <= max(0.0, float(self.config.vib2_max_rate_hz))
            and median_width_cents
            >= max(0.0, float(self.config.vib2_min_width_cents))
        )

    @staticmethod
    def _stabilize_offline_edges(rates: np.ndarray, widths: np.ndarray,
                                 qualities: np.ndarray,
                                 representative_rate: float,
                                 frame_rate: float,
                                 *,
                                 hold_values: bool = True):
        """Taper edge confidence and optionally hold the nearest interior rate and width."""
        n = len(rates)
        if n < 5 or representative_rate <= 0.0:
            return rates, widths, qualities
        edge = min(
            int(round(2.0 * frame_rate / representative_rate)),
            (n - 1) // 4,
        )
        if edge <= 0 or n <= 2 * edge:
            return rates, widths, qualities
        rates = rates.copy()
        widths = widths.copy()
        qualities = qualities.copy()
        if hold_values:
            rates[:edge] = rates[edge]
            widths[:edge] = widths[edge]
            rates[-edge:] = rates[-edge - 1]
            widths[-edge:] = widths[-edge - 1]
        ramp = np.linspace(0.25, 1.0, edge, endpoint=False)
        qualities[:edge] *= ramp
        qualities[-edge:] *= ramp[::-1]
        return rates, widths, qualities

    def _seed_rate_and_phase(self, values: np.ndarray, times: np.ndarray,
                             basis: np.ndarray, voiced: np.ndarray,
                             scan_bins: int,
                             confidence: np.ndarray | None = None):
        """Project out the smooth center, then solve all candidate sine/cosine pairs."""
        duration = float(times[-1] - times[0])
        if duration <= 0.0:
            return None
        lower = max(
            self.RATE_FLOOR_HZ,
            float(self.config.vib2_fit_rate_min_hz),
        )
        frame_rate = self.config.sr / self.config.h1
        search_upper = min(
            float(self.config.vib2_fit_rate_max_hz),
            self._max_resolvable_rate_hz(frame_rate),
        )
        if not np.isfinite(search_upper) or lower >= search_upper:
            return None

        frequencies = np.linspace(
            lower,
            search_upper,
            max(16, int(scan_bins)),
        )
        m = basis.shape[1]
        amplitude_min = max(
            0.0,
            float(
                self.config.vib2_seed_candidate_amplitude_min_semitones
            ),
        )
        amplitude_max = float(
            self.config.vib2_seed_candidate_amplitude_max_semitones
        )
        if not np.isfinite(amplitude_max) or amplitude_max < amplitude_min:
            return None
        second = self._second_difference(m)
        weights = (
            np.ones(int(np.sum(voiced)), dtype=np.float64)
            if confidence is None
            else np.sqrt(np.asarray(confidence, dtype=np.float64)[voiced])
        )
        center_penalty = (
            np.sqrt(max(0.0, float(self.config.vib2_center_smoothness)))
            * second
        )
        center_design = np.vstack((
            basis[voiced] * weights[:, None],
            center_penalty,
        ))
        target = np.concatenate((
            values[voiced] * weights,
            np.zeros(len(center_penalty)),
        ))

        q, r = np.linalg.qr(center_design, mode="reduced")
        diagonal = np.abs(np.diag(r))
        tolerance = (
            np.finfo(np.float64).eps
            * max(center_design.shape)
            * (float(np.max(diagonal)) if len(diagonal) else 0.0)
        )
        full_rank = (
            center_design.shape[0] >= center_design.shape[1]
            and len(diagonal) == center_design.shape[1]
            and bool(np.all(diagonal > tolerance))
        )
        if not full_rank:
            return self._seed_rate_and_phase_fallback(
                values,
                times,
                basis,
                voiced,
                frequencies,
                weights,
                center_penalty,
                target,
                amplitude_min,
                amplitude_max,
            )

        center_residual = target - q @ (q.T @ target)
        voiced_times = times[voiced]
        angles = 2.0 * np.pi * frequencies[:, None] * voiced_times[None, :]
        carriers = np.zeros(
            (len(frequencies), len(target), 2),
            dtype=np.float64,
        )
        carriers[:, :len(weights), 0] = np.cos(angles) * weights
        carriers[:, :len(weights), 1] = np.sin(angles) * weights
        projected = carriers - np.einsum(
            "rm,fmk->frk",
            q,
            np.einsum("rm,frk->fmk", q, carriers, optimize=True),
            optimize=True,
        )

        gram = np.einsum("fri,frj->fij", projected, projected, optimize=True)
        rhs = np.einsum("fri,r->fi", projected, center_residual, optimize=True)
        g00 = gram[:, 0, 0]
        g01 = gram[:, 0, 1]
        g11 = gram[:, 1, 1]
        determinant = g00 * g11 - g01 * g01
        determinant_scale = np.maximum(g00 * g11, 1.0)
        solvable = (
            np.isfinite(determinant)
            & (np.abs(determinant) > np.finfo(np.float64).eps * determinant_scale)
        )
        coefficients = np.full((len(frequencies), 2), np.nan, dtype=np.float64)
        coefficients[solvable, 0] = (
            rhs[solvable, 0] * g11[solvable]
            - rhs[solvable, 1] * g01[solvable]
        ) / determinant[solvable]
        coefficients[solvable, 1] = (
            g00[solvable] * rhs[solvable, 1]
            - g01[solvable] * rhs[solvable, 0]
        ) / determinant[solvable]

        amplitudes = np.hypot(coefficients[:, 0], coefficients[:, 1])
        admitted = (
            solvable
            & np.isfinite(amplitudes)
            & (amplitudes >= amplitude_min)
            & (amplitudes <= amplitude_max)
        )
        residuals = center_residual[None, :] - np.einsum(
            "fri,fi->fr",
            projected,
            coefficients,
            optimize=True,
        )
        scores = np.einsum("fr,fr->f", residuals, residuals, optimize=True)
        scores[~admitted] = np.inf
        best_index = int(np.argmin(scores))
        if not np.isfinite(scores[best_index]):
            return None

        rate0 = float(frequencies[best_index])
        cosine_coefficient = float(coefficients[best_index, 0])
        sine_coefficient = float(coefficients[best_index, 1])
        amplitude = float(amplitudes[best_index])

        phase0 = float(np.arctan2(cosine_coefficient, sine_coefficient))
        seed_amplitude_min = max(
            0.0,
            float(self.config.vib2_seed_amplitude_min_semitones),
        )
        seed_amplitude_max = float(
            self.config.vib2_seed_amplitude_max_semitones
        )
        if (
            not np.isfinite(seed_amplitude_max)
            or seed_amplitude_max < seed_amplitude_min
        ):
            return None
        initial_amplitude = float(np.clip(
            amplitude,
            seed_amplitude_min,
            seed_amplitude_max,
        ))
        return rate0, phase0, initial_amplitude

    def _seed_rate_and_phase_fallback(
            self,
            values: np.ndarray,
            times: np.ndarray,
            basis: np.ndarray,
            voiced: np.ndarray,
            frequencies: np.ndarray,
            weights: np.ndarray,
            center_penalty: np.ndarray,
            target: np.ndarray,
            amplitude_min: float,
            amplitude_max: float,
    ):
        """Independent SVD seed scan for a rank-deficient center basis."""
        m = basis.shape[1]
        penalty = np.zeros(
            (len(center_penalty), m + 2),
            dtype=np.float64,
        )
        penalty[:, :m] = center_penalty
        best = None
        for rate in frequencies:
            angle = 2.0 * np.pi * rate * times
            design = np.column_stack((basis, np.cos(angle), np.sin(angle)))
            design_fit = np.vstack((
                design[voiced] * weights[:, None],
                penalty,
            ))
            coefficients, *_ = np.linalg.lstsq(
                design_fit,
                target,
                rcond=None,
            )
            amplitude = float(np.hypot(coefficients[m], coefficients[m + 1]))
            if not amplitude_min <= amplitude <= amplitude_max:
                continue
            residual = target - design_fit @ coefficients
            score = float(np.dot(residual, residual))
            if best is None or score < best[0]:
                best = (
                    score,
                    float(rate),
                    float(coefficients[m]),
                    float(coefficients[m + 1]),
                    amplitude,
                )
        if best is None:
            return None
        _, rate0, cosine_coefficient, sine_coefficient, amplitude = best
        phase0 = float(np.arctan2(cosine_coefficient, sine_coefficient))
        seed_amplitude_min = max(
            0.0,
            float(self.config.vib2_seed_amplitude_min_semitones),
        )
        seed_amplitude_max = float(
            self.config.vib2_seed_amplitude_max_semitones
        )
        if (
            not np.isfinite(seed_amplitude_max)
            or seed_amplitude_max < seed_amplitude_min
        ):
            return None
        return (
            rate0,
            phase0,
            float(np.clip(
                amplitude,
                seed_amplitude_min,
                seed_amplitude_max,
            )),
        )

    def _initial_phase_parameters(self, values: np.ndarray,
                                  times: np.ndarray, rate0: float,
                                  phase0: float,
                                  amplitude0: float,
                                  rate_upper: float) -> np.ndarray:
        """Use raw extrema to initialize changing speed when well supported."""
        constant = np.array([phase0, rate0, rate0, rate0], dtype=np.float64)
        extrema = self._swing_extrema(values, amplitude0)

        if len(extrema) < 6:
            return constant
        duration = float(times[-1] - times[0])
        if duration <= 0.0:
            return constant
        u = np.array(
            [(times[i] - times[0]) / duration for i, _ in extrema],
            dtype=np.float64,
        )
        first_phase = np.pi / 2.0 if extrema[0][1] > 0 else 3.0 * np.pi / 2.0
        phases = first_phase + np.pi * np.arange(len(extrema), dtype=np.float64)
        integrated_basis = np.column_stack((
            u - u ** 2 + u ** 3 / 3.0,
            u ** 2 - 2.0 * u ** 3 / 3.0,
            u ** 3 / 3.0,
        ))
        design = np.column_stack((
            np.ones(len(u)),
            2.0 * np.pi * duration * integrated_basis,
        ))
        parameters, *_ = np.linalg.lstsq(design, phases, rcond=None)
        if (
            not np.isfinite(parameters).all()
            or np.any(parameters[1:] < self.RATE_FLOOR_HZ)
            or np.any(parameters[1:] > rate_upper)
        ):
            return constant
        parameters[0] = (parameters[0] + np.pi) % (2.0 * np.pi) - np.pi
        return parameters

    @classmethod
    def _swing_extrema(cls, values: np.ndarray,
                       amplitude: float) -> list[tuple[int, int]]:
        """Prominent confirmed extrema as ``(index, +1 max / -1 min)``."""
        if amplitude <= 0.0 or len(values) < 2:
            return []
        threshold = cls.ALT_PROM_FRAC * amplitude
        if threshold <= np.finfo(float).eps:
            return []
        anchor = extreme = float(values[0])
        extreme_index = 0
        direction = 0
        extrema: list[tuple[int, int]] = []
        for i, raw in enumerate(values[1:], start=1):
            value = float(raw)
            if direction == 0:
                if value - anchor >= threshold:
                    direction = 1
                    extreme, extreme_index = value, i
                elif anchor - value >= threshold:
                    direction = -1
                    extreme, extreme_index = value, i
                continue
            if direction > 0:
                if value > extreme:
                    extreme, extreme_index = value, i
                elif extreme - value >= threshold:
                    extrema.append((extreme_index, 1))
                    direction = -1
                    extreme, extreme_index = value, i
            else:
                if value < extreme:
                    extreme, extreme_index = value, i
                elif value - extreme >= threshold:
                    extrema.append((extreme_index, -1))
                    direction = 1
                    extreme, extreme_index = value, i
        return extrema

    def _solve_linear_model(self, values: np.ndarray, phase: np.ndarray,
                            basis: np.ndarray, voiced: np.ndarray,
                            confidence: np.ndarray | None = None):
        """Exact center/amplitude solve for one proposed nonlinear phase."""
        sine = np.sin(phase)
        design = np.hstack((basis, basis * sine[:, None]))
        m = basis.shape[1]
        second = self._second_difference(m)
        penalty = np.zeros((2 * len(second), 2 * m), dtype=np.float64)
        if len(second):
            penalty[:len(second), :m] = (
                np.sqrt(max(0.0, float(self.config.vib2_center_smoothness)))
                * second
            )
            penalty[len(second):, m:] = (
                np.sqrt(max(0.0, float(self.config.vib2_width_smoothness)))
                * second
            )
        weights = (
            np.ones(int(np.sum(voiced)), dtype=np.float64)
            if confidence is None
            else np.sqrt(np.asarray(confidence, dtype=np.float64)[voiced])
        )
        design_fit = np.vstack((
            design[voiced] * weights[:, None],
            penalty,
        ))
        target = np.concatenate((
            values[voiced] * weights,
            np.zeros(len(penalty)),
        ))
        coefficients, *_ = np.linalg.lstsq(design_fit, target, rcond=None)
        return coefficients, design, penalty

    def _variable_projection_residual_jacobian(
            self,
            parameters: np.ndarray,
            values: np.ndarray,
            times: np.ndarray,
            rate_basis: np.ndarray,
            basis: np.ndarray,
            voiced: np.ndarray,
            confidence: np.ndarray,
            phase_parameter_jacobian: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve linear coefficients and differentiate their optimum, including the transpose term."""
        phase, _ = self._phase_from_parameters(parameters, times, rate_basis)
        phase_jacobian = (
            self._phase_parameter_jacobian(times, rate_basis)
            if phase_parameter_jacobian is None
            else phase_parameter_jacobian
        )
        sine = np.sin(phase)
        cosine = np.cos(phase)
        design = np.hstack((basis, basis * sine[:, None]))
        m = basis.shape[1]
        second = self._second_difference(m)
        penalty = np.zeros((2 * len(second), 2 * m), dtype=np.float64)
        if len(second):
            penalty[:len(second), :m] = (
                np.sqrt(max(0.0, float(self.config.vib2_center_smoothness)))
                * second
            )
            penalty[len(second):, m:] = (
                np.sqrt(max(0.0, float(self.config.vib2_width_smoothness)))
                * second
            )

        weights = np.sqrt(np.asarray(confidence, dtype=np.float64)[voiced])
        voiced_count = int(np.sum(voiced))
        design_fit = np.vstack((
            design[voiced] * weights[:, None],
            penalty,
        ))
        target = np.concatenate((
            values[voiced] * weights,
            np.zeros(len(penalty)),
        ))

        use_qr = design_fit.shape[0] >= design_fit.shape[1]
        if use_qr:
            q, r = np.linalg.qr(design_fit, mode="reduced")
            diagonal = np.abs(np.diag(r))
            tolerance = (
                np.finfo(np.float64).eps
                * max(design_fit.shape)
                * (float(np.max(diagonal)) if len(diagonal) else 0.0)
            )
            use_qr = (
                len(diagonal) == design_fit.shape[1]
                and bool(np.all(diagonal > tolerance))
            )
        if use_qr:
            coefficients = np.linalg.solve(r, q.T @ target)

            def project(vector):
                return q @ (q.T @ vector)

            def dual_project(vector):
                return q @ np.linalg.solve(r.T, vector)
        else:
            pseudoinverse = np.linalg.pinv(design_fit)
            coefficients = pseudoinverse @ target

            def project(vector):
                return design_fit @ (pseudoinverse @ vector)

            def dual_project(vector):
                return pseudoinverse.T @ vector

        projection_residual = target - design_fit @ coefficients

        # Penalty rows use +P*c; convert the augmented residual without changing its norm.
        row_sign = np.ones(len(target), dtype=np.float64)
        row_sign[voiced_count:] = -1.0
        residual = row_sign * projection_residual
        jacobian = np.empty((len(target), 4), dtype=np.float64)
        amplitude = basis @ coefficients[m:]
        for parameter_i in range(4):
            carrier_derivative = (
                cosine * phase_jacobian[:, parameter_i]
            )
            design_derivative_times_coefficients = np.concatenate((
                weights
                * amplitude[voiced]
                * carrier_derivative[voiced],
                np.zeros(len(penalty)),
            ))
            normal_derivative = np.zeros(2 * m, dtype=np.float64)
            derivative_amplitude_columns = (
                basis[voiced]
                * (weights * carrier_derivative[voiced])[:, None]
            )
            normal_derivative[m:] = (
                derivative_amplitude_columns.T
                @ projection_residual[:voiced_count]
            )
            derivative = -(
                design_derivative_times_coefficients
                - project(design_derivative_times_coefficients)
            ) - dual_project(normal_derivative)
            jacobian[:, parameter_i] = row_sign * derivative

        smoothness = np.sqrt(max(
            0.0,
            float(self.config.vib2_phase_smoothness),
        ))
        curvature = parameters[1] - 2.0 * parameters[2] + parameters[3]
        residual = np.concatenate((residual, [smoothness * curvature]))
        curvature_jacobian = np.array([
            0.0,
            smoothness,
            -2.0 * smoothness,
            smoothness,
        ])
        jacobian = np.vstack((jacobian, curvature_jacobian))
        return coefficients, residual, jacobian

    @staticmethod
    def _rate_basis(times: np.ndarray) -> np.ndarray:
        """Quadratic Bezier basis for a positive, smooth speed curve."""
        duration = float(times[-1] - times[0])
        u = np.zeros_like(times) if duration <= 0.0 else (times - times[0]) / duration
        return np.column_stack(((1.0 - u) ** 2, 2.0 * u * (1.0 - u), u ** 2))

    @staticmethod
    def _phase_from_parameters(parameters: np.ndarray, times: np.ndarray,
                               rate_basis: np.ndarray):
        rates = rate_basis @ parameters[1:]
        phase = parameters[0] + 2.0 * np.pi * np.concatenate((
            [0.0],
            np.cumsum(0.5 * (rates[1:] + rates[:-1]) * np.diff(times)),
        ))
        return phase, rates

    @staticmethod
    def _phase_parameter_jacobian(
            times: np.ndarray,
            rate_basis: np.ndarray,
    ) -> np.ndarray:
        """Exact d(phase)/d(initial phase, three rate controls)."""
        jacobian = np.zeros((len(times), 4), dtype=np.float64)
        if not len(times):
            return jacobian
        jacobian[:, 0] = 1.0
        if len(times) == 1:
            return jacobian
        dt = np.diff(times)
        integrated_rate_basis = np.vstack((
            np.zeros((1, rate_basis.shape[1]), dtype=np.float64),
            np.cumsum(
                0.5 * (rate_basis[1:] + rate_basis[:-1]) * dt[:, None],
                axis=0,
            ),
        ))
        jacobian[:, 1:] = 2.0 * np.pi * integrated_rate_basis
        return jacobian

    def _spline_basis(self, times: np.ndarray, causal: bool = False):
        """Cubic B-spline basis spanning the whole segment/note."""
        relative = times - times[0]
        duration = float(relative[-1])
        degree = min(3, len(times) - 1)
        spacing = max(
            1.0 / (self.config.sr / self.config.h1),
            float(self.config.vib2_curve_sec),
        )

        # A single live cubic stabilizes the endpoint; offline fits allow interior knots.
        internal = (
            np.empty(0)
            if causal
            else np.arange(spacing, duration - 1e-12, spacing)
        )
        knots = np.concatenate((
            np.repeat(0.0, degree + 1),
            internal,
            np.repeat(duration, degree + 1),
        ))
        basis = BSpline.design_matrix(
            relative,
            knots,
            degree,
            extrapolate=True,
        ).toarray()
        return basis, knots, degree

    @staticmethod
    def _second_difference(columns: int) -> np.ndarray:
        return (
            np.diff(np.eye(columns), n=2, axis=0)
            if columns >= 3 else np.empty((0, columns))
        )

    @staticmethod
    def _max_resolvable_rate_hz(frame_rate: float) -> float:
        """Largest rate below the pitch-frame Nyquist frequency."""
        if not np.isfinite(frame_rate) or frame_rate <= 0.0:
            return np.nan
        return float(np.nextafter(0.5 * frame_rate, 0.0))
