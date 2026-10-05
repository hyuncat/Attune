"""STFT-of-f0 detector after Herrera and Bonada (DAFx-98)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, windows

from benchmarks.modules.vibrato.VibratoDetectorBase import (
    VibratoDetectorBase,
    VibratoEstimate,
    VibratoExample,
)
from benchmarks.modules.vibrato.competitors.Yang import Yang


HB_REFERENCE_FRAME_RATE = 345.0


class HerreraBonada(VibratoDetectorBase):
    """Herrera/Bonada frame-wise STFT-of-f0 vibrato estimator."""

    @dataclass(frozen=True)
    class FrameTrack:
        times: np.ndarray
        rate_hz: np.ndarray
        extent_semitones: np.ndarray
        detected: np.ndarray
        peak_bin: np.ndarray

    requires = {"pitch"}
    scores_center = False

    @staticmethod
    def _parabolic_spectral_peak(
        magnitudes: np.ndarray,
        index: int,
        bin_hz: float,
    ) -> tuple[float, float]:
        if index <= 0 or index >= len(magnitudes) - 1:
            return float(index * bin_hz), float(magnitudes[index])
        left, center, right = map(float, magnitudes[index - 1 : index + 2])
        denominator = left - 2.0 * center + right
        offset = (
            0.5 * (left - right) / denominator
            if abs(denominator) > np.finfo(float).eps
            else 0.0
        )
        offset = float(np.clip(offset, -1.0, 1.0))
        height = center - 0.25 * (left - right) * offset
        return float((index + offset) * bin_hz), float(max(height, 0.0))

    @classmethod
    def _analyze(
        cls,
        pitch_midi: np.ndarray,
        times: np.ndarray,
        *,
        window_seconds: float,
        hop_fraction: float,
        rate_limits_hz: tuple[float, float] = Yang.DECISION_RATE_LIMITS_HZ,
        minimum_extent_semitones: float = Yang.DECISION_MIN_EXTENT_SEMITONES,
    ) -> HerreraBonada.FrameTrack:
        pitch_midi = np.asarray(pitch_midi, dtype=np.float64)
        times = np.asarray(times, dtype=np.float64)
        frame_rate = 1.0 / float(np.median(np.diff(times)))
        window_size = max(8, int(round(window_seconds * frame_rate)))
        hop = max(1, int(round(window_size * hop_fraction)))
        if len(pitch_midi) < window_size:
            empty = np.empty(0, dtype=np.float64)
            return cls.FrameTrack(
                empty, empty.copy(), empty.copy(), empty.astype(bool), empty
            )
        centered = pitch_midi - float(np.mean(pitch_midi))
        taper = windows.hamming(window_size, sym=False)
        starts = np.arange(0, len(centered) - window_size + 1, hop, dtype=int)
        frame_times = times[0] + (starts + 0.5 * (window_size - 1)) / frame_rate
        rates = np.zeros(len(starts), dtype=np.float64)
        extents = np.zeros(len(starts), dtype=np.float64)
        detected = np.zeros(len(starts), dtype=bool)
        peak_bins = np.zeros(len(starts), dtype=np.float64)
        frequencies = np.fft.rfftfreq(window_size, d=1.0 / frame_rate)
        eligible = np.flatnonzero(
            (frequencies >= max(0.5, rate_limits_hz[0] - 2.0))
            & (frequencies <= rate_limits_hz[1] + 2.0)
        )
        for output_index, start in enumerate(starts):
            magnitude = np.abs(
                np.fft.rfft(centered[start : start + window_size] * taper)
            )
            if not len(eligible):
                continue
            candidates, _ = find_peaks(magnitude)
            candidates = np.intersect1d(candidates, eligible, assume_unique=False)
            peak_pool = candidates if len(candidates) else eligible
            peak = int(peak_pool[np.argmax(magnitude[peak_pool])])
            if len(candidates):
                rate, peak_height = cls._parabolic_spectral_peak(
                    magnitude, peak, frame_rate / window_size
                )
            else:
                rate = peak * frame_rate / window_size
                peak_height = float(magnitude[peak])
            extent = 2.0 * peak_height / max(float(np.sum(taper)), 1e-12)
            rates[output_index] = rate
            extents[output_index] = extent
            peak_bins[output_index] = peak
            detected[output_index] = (
                rate_limits_hz[0] <= rate <= rate_limits_hz[1]
                and extent >= minimum_extent_semitones
            )
        return cls.FrameTrack(frame_times, rates, extents, detected, peak_bins)

    def __init__(self, *, window_mode: str = "native") -> None:
        if window_mode not in {"native", "yang_comparison"}:
            raise ValueError("window_mode must be 'native' or 'yang_comparison'")
        self.window_mode = window_mode
        if window_mode == "native":
            self.name = "herrera_bonada"
            self.description = (
                "Herrera/Bonada native-window STFT-of-f0 " "(independent reproduction)"
            )
            self.window_seconds = 128.0 / 345.0
            self.hop_fraction = 0.5
        else:
            self.name = "herrera_bonada_yang_window"
            self.description = (
                "Herrera/Bonada at Yang's short-window comparison geometry "
                "(independent reproduction)"
            )
            self.window_seconds = 0.125
            self.hop_fraction = 0.25

    def estimate(self, example: VibratoExample) -> VibratoEstimate:
        filled, voiced = self.fill_unvoiced(example.pitch_midi)
        zeros = np.zeros(len(filled), dtype=np.float64)
        # Production/Coco/Yang contours already lie at 344.53 Hz and are used
        # unchanged, as the source geometry permits. The optional synthetic
        # tier defaults to 100 Hz; resample that diagnostic input so "128
        # points" and Yang's 0.125 s window do not become different methods.
        resampled = not np.isclose(
            example.frame_rate,
            HB_REFERENCE_FRAME_RATE,
            rtol=0.01,
        )
        if resampled:
            count = (
                int(
                    np.floor(
                        (example.times[-1] - example.times[0]) * HB_REFERENCE_FRAME_RATE
                    )
                )
                + 1
            )
            analysis_times = (
                example.times[0]
                + np.arange(count, dtype=np.float64) / HB_REFERENCE_FRAME_RATE
            )
            analysis_pitch = np.interp(analysis_times, example.times, filled)
        else:
            analysis_times = example.times
            analysis_pitch = filled
        track = self._analyze(
            analysis_pitch,
            analysis_times,
            window_seconds=self.window_seconds,
            hop_fraction=self.hop_fraction,
        )
        if not len(track.times):
            return VibratoEstimate(zeros, zeros.copy(), zeros.astype(bool))
        nearest, covered = self.nearest_frame_map(example.times, track.times)
        supported = covered & voiced
        rates = np.where(supported, track.rate_hz[nearest], 0.0)
        widths = np.where(supported, 200.0 * track.extent_semitones[nearest], 0.0)
        detected = supported & track.detected[nearest]
        return VibratoEstimate(
            rates,
            widths,
            detected,
            quality=np.where(supported, track.extent_semitones[nearest], 0.0),
            metadata={
                "source": "Herrera and Bonada, DAFx-98, section 2",
                "port": "independent reproduction",
                "window_mode": self.window_mode,
                "window_seconds": self.window_seconds,
                "hop_fraction": self.hop_fraction,
                "window_function": "periodic Hamming",
                "analysis_frame_rate": (
                    HB_REFERENCE_FRAME_RATE if resampled else example.frame_rate
                ),
                "resampled_input_adapter": resampled,
                "extent_scaling_adapter": "2*peak/sum(window)",
                "decision_rate_limits_hz": Yang.DECISION_RATE_LIMITS_HZ,
                "decision_minimum_extent_semitones": (
                    Yang.DECISION_MIN_EXTENT_SEMITONES
                ),
                "filled_unvoiced_frames": int(np.sum(~voiced)),
                "native_frame_times": track.times,
                "native_rates_hz": track.rate_hz,
                "native_extents_semitones": track.extent_semitones,
            },
        )
