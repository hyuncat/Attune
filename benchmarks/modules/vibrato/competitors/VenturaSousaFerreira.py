"""Long-window spectral detector after Ventura, Sousa, and Ferreira."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from benchmarks.modules.vibrato.VibratoDetectorBase import (
    VibratoDetectorBase,
    VibratoEstimate,
    VibratoExample,
)
from benchmarks.modules.vibrato.competitors.Rossignol import Rossignol


VSF_FRAME_RATE = 22_050.0 / 512.0
VSF_WINDOW_SAMPLES = 64
VSF_HOP_SAMPLES = 6
VSF_SEARCH_BINS = (6, 12)
VSF_LIFTER_BINS = 4
VSF_DECISION_DB = 3.8


class VenturaSousaFerreira(VibratoDetectorBase):
    """VSF's long-window frame-wise STFT/cepstral-envelope estimator."""

    @dataclass(frozen=True)
    class FrameTrack:
        times: np.ndarray
        rate_hz: np.ndarray
        width_cents: np.ndarray
        detected: np.ndarray
        peak_above_noise_db: np.ndarray
        peak_above_envelope_db: np.ndarray

    name = "ventura_sousa_ferreira"
    description = (
        "Ventura/Sousa/Ferreira long-window STFT-of-f0 " "(independent reproduction)"
    )
    requires = {"pitch"}
    scores_center = False

    @staticmethod
    def _least_squares_frequency(
        samples: np.ndarray,
        frame_rate: float,
        peak_bin: int,
    ) -> float:
        indices = np.arange(len(samples), dtype=np.float64)
        bin_hz = frame_rate / len(samples)

        def error(rate: float) -> float:
            phase = 2.0 * np.pi * rate * indices / frame_rate
            design = np.column_stack((np.sin(phase), np.cos(phase)))
            coefficients, _, _, _ = np.linalg.lstsq(design, samples, rcond=None)
            residual = samples - design @ coefficients
            return float(residual @ residual)

        result = minimize_scalar(
            error,
            bounds=(
                max(0.0, (peak_bin - 1.0) * bin_hz),
                (peak_bin + 1.0) * bin_hz,
            ),
            method="bounded",
            options={"xatol": 1e-10},
        )
        return float(result.x if result.success else peak_bin * bin_hz)

    @staticmethod
    def _cepstral_envelope(magnitude: np.ndarray, keep: int) -> np.ndarray:
        log_magnitude = np.log(np.maximum(magnitude, np.finfo(float).tiny))
        cepstrum = np.real(np.fft.ifft(log_magnitude))
        liftered = np.zeros_like(cepstrum)
        liftered[:keep] = cepstrum[:keep]
        if keep > 1:
            liftered[-(keep - 1) :] = cepstrum[-(keep - 1) :]
        return np.exp(np.real(np.fft.fft(liftered)))

    @classmethod
    def _analyze(
        cls,
        pitch_midi: np.ndarray,
        times: np.ndarray,
        *,
        frame_rate: float = VSF_FRAME_RATE,
    ) -> VenturaSousaFerreira.FrameTrack:
        pitch_midi = np.asarray(pitch_midi, dtype=np.float64)
        times = np.asarray(times, dtype=np.float64)
        native_count = int(np.floor((times[-1] - times[0]) * frame_rate)) + 1
        native_times = times[0] + np.arange(native_count) / frame_rate
        native_pitch = np.interp(native_times, times, pitch_midi)
        starts = np.arange(
            0,
            max(0, native_count - VSF_WINDOW_SAMPLES + 1),
            VSF_HOP_SAMPLES,
            dtype=int,
        )
        count = len(starts)
        frame_times = (
            native_times[0] + (starts + 0.5 * (VSF_WINDOW_SAMPLES - 1)) / frame_rate
        )
        rates = np.zeros(count, dtype=np.float64)
        widths = np.zeros(count, dtype=np.float64)
        detected = np.zeros(count, dtype=bool)
        above_noise = np.full(count, -np.inf, dtype=np.float64)
        above_envelope = np.full(count, -np.inf, dtype=np.float64)
        for output_index, start in enumerate(starts):
            segment = native_pitch[start : start + VSF_WINDOW_SAMPLES]
            centered = segment - float(np.mean(segment))
            magnitude = np.abs(np.fft.fft(centered))
            envelope = cls._cepstral_envelope(magnitude, VSF_LIFTER_BINS)
            bins = np.arange(VSF_SEARCH_BINS[0], VSF_SEARCH_BINS[1] + 1)
            peak = int(bins[np.argmax(magnitude[bins])])
            peak_value = float(magnitude[peak])
            noise_floor = float(np.median(magnitude[1 : VSF_WINDOW_SAMPLES // 2]))
            envelope_floor = float(envelope[peak])
            above_noise[output_index] = 20.0 * np.log10(
                max(peak_value, 1e-12) / max(noise_floor, 1e-12)
            )
            above_envelope[output_index] = 20.0 * np.log10(
                max(peak_value, 1e-12) / max(envelope_floor, 1e-12)
            )
            rates[output_index] = cls._least_squares_frequency(
                centered, frame_rate, peak
            )
            local_times = native_times[start : start + VSF_WINDOW_SAMPLES]
            extrema = cls.interpolated_extrema(
                segment,
                local_times,
                same_kind_distance=max(1, int(frame_rate / 8.0)),
            )
            if len(extrema) >= 2:
                measurement = Rossignol.measure_extrema_run(extrema)
                widths[output_index] = (
                    float(np.mean(measurement.width_cents))
                    if len(measurement.width_cents)
                    else 0.0
                )
            detected[output_index] = (
                above_noise[output_index] >= VSF_DECISION_DB
                and above_envelope[output_index] >= VSF_DECISION_DB
            )
        return cls.FrameTrack(
            frame_times,
            rates,
            widths,
            detected,
            above_noise,
            above_envelope,
        )

    def estimate(self, example: VibratoExample) -> VibratoEstimate:
        filled, voiced = self.fill_unvoiced(example.pitch_midi)
        zeros = np.zeros(len(filled), dtype=np.float64)
        track = self._analyze(filled, example.times)
        if not len(track.times):
            return VibratoEstimate(zeros, zeros.copy(), zeros.astype(bool))
        nearest, covered = self.nearest_frame_map(example.times, track.times)
        supported = covered & voiced
        rates = np.where(supported, track.rate_hz[nearest], 0.0)
        widths = np.where(supported, track.width_cents[nearest], 0.0)
        detected = supported & track.detected[nearest]
        quality = np.where(
            supported,
            np.minimum(
                track.peak_above_noise_db[nearest],
                track.peak_above_envelope_db[nearest],
            ),
            0.0,
        )
        return VibratoEstimate(
            rates,
            widths,
            detected,
            quality=quality,
            metadata={
                "source": "Ventura, Sousa, and Ferreira, ISCCSP 2012, section 3",
                "port": "independent reproduction",
                "analysis_frame_rate": VSF_FRAME_RATE,
                "window_samples": VSF_WINDOW_SAMPLES,
                "window_seconds": VSF_WINDOW_SAMPLES / VSF_FRAME_RATE,
                "hop_samples": VSF_HOP_SAMPLES,
                "search_bins": VSF_SEARCH_BINS,
                "lifter_bins": VSF_LIFTER_BINS,
                "decision_db": VSF_DECISION_DB,
                "frequency_interpolator_adapter": (
                    "bounded single-sinusoid least-squares"
                ),
                "noise_floor_adapter": "median positive-frequency magnitude",
                "filled_unvoiced_frames": int(np.sum(~voiced)),
                "native_frame_times": track.times,
                "native_rates_hz": track.rate_hz,
                "native_widths_cents": track.width_cents,
            },
        )
