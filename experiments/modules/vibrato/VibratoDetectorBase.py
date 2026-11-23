from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
from scipy.signal import find_peaks


FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class VibratoExample:
    """One continuous pitch contour plus the frames scored for this case."""

    case_id: str
    scenario: str
    split: str
    times: FloatArray
    pitch_midi: FloatArray
    center_midi: FloatArray
    rate_hz: FloatArray
    width_cents: FloatArray
    is_vibrato: BoolArray
    commanded_pitch_midi: FloatArray | None = None
    raw_pitch_midi: FloatArray | None = None
    transition_mask: BoolArray | None = None
    evaluation_mask: BoolArray | None = None
    audio_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arrays = (
            self.times,
            self.pitch_midi,
            self.center_midi,
            self.rate_hz,
            self.width_cents,
            self.is_vibrato,
        )
        if self.evaluation_mask is not None:
            arrays += (self.evaluation_mask,)
        if self.commanded_pitch_midi is not None:
            arrays += (self.commanded_pitch_midi,)
        if self.raw_pitch_midi is not None:
            arrays += (self.raw_pitch_midi,)
        if self.transition_mask is not None:
            arrays += (self.transition_mask,)
        lengths = {len(array) for array in arrays}
        if lengths != {len(self.times)} or len(self.times) < 2:
            raise ValueError("all example arrays must share a length of at least two")
        if not np.all(np.diff(self.times) > 0):
            raise ValueError("example times must be strictly increasing")
        steps = np.diff(self.times)
        if not np.allclose(steps, np.median(steps), rtol=0.01, atol=1e-9):
            raise ValueError("example times must lie on a regular frame grid")
        if not np.any(self.score_mask):
            raise ValueError("an example must score at least one frame")

    @property
    def duration(self) -> float:
        dt = float(np.median(np.diff(self.times)))
        return float(self.times[-1] - self.times[0] + dt)

    @property
    def frame_rate(self) -> float:
        return float(1.0 / np.median(np.diff(self.times)))

    @property
    def score_mask(self) -> BoolArray:
        if self.evaluation_mask is None:
            return np.ones(len(self.times), dtype=np.bool_)
        return np.asarray(self.evaluation_mask, dtype=np.bool_)

    @property
    def scored_duration(self) -> float:
        return float(np.sum(self.score_mask) / self.frame_rate)

    @property
    def scored_time_bounds(self) -> tuple[float, float]:
        indices = np.flatnonzero(self.score_mask)
        return (
            float(self.times[indices[0]]),
            float(self.times[indices[-1]] + 1.0 / self.frame_rate),
        )

    @property
    def has_vibrato(self) -> bool:
        return bool(np.any(np.asarray(self.is_vibrato, dtype=bool) & self.score_mask))


@dataclass(frozen=True)
class VibratoEstimate:
    """A method's estimates, sampled on the input example's time grid."""

    rate_hz: FloatArray
    width_cents: FloatArray
    detected: BoolArray
    center_midi: FloatArray | None = None
    quality: FloatArray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate_for(self, example: VibratoExample) -> "VibratoEstimate":
        n = len(example.times)
        required = (self.rate_hz, self.width_cents, self.detected)
        if any(len(array) != n for array in required):
            raise ValueError(
                f"estimate arrays must have {n} frames for {example.case_id}"
            )
        if self.center_midi is not None and len(self.center_midi) != n:
            raise ValueError("center estimate must match the example grid")
        if self.quality is not None and len(self.quality) != n:
            raise ValueError("quality estimate must match the example grid")
        return self


class VibratoDetectorBase(ABC):
    """Common API and shared grid adapters for every benchmark detector."""

    name: str
    description: str
    requires: set[str] = {"pitch"}
    scores_center = False

    @dataclass(frozen=True)
    class Extremum:
        index: int
        time: float
        value: float
        kind: int

    @abstractmethod
    def estimate(self, example: VibratoExample) -> VibratoEstimate:
        """Estimate vibrato on the example's frame grid."""

    @staticmethod
    def fill_unvoiced(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(values, dtype=np.float64)
        voiced = np.isfinite(values)
        if not np.any(voiced):
            return np.zeros_like(values), voiced
        indices = np.arange(len(values), dtype=np.float64)
        return np.interp(indices, indices[voiced], values[voiced]), voiced

    @classmethod
    def _parabolic_extremum(
        cls,
        values: np.ndarray,
        times: np.ndarray,
        index: int,
        kind: int,
    ) -> Extremum:
        if index <= 0 or index >= len(values) - 1:
            return cls.Extremum(index, float(times[index]), float(values[index]), kind)
        left, center, right = map(float, values[index - 1 : index + 2])
        denominator = left - 2.0 * center + right
        offset = (
            0.5 * (left - right) / denominator
            if abs(denominator) > np.finfo(float).eps
            else 0.0
        )
        offset = float(np.clip(offset, -1.0, 1.0))
        step = float(np.median(np.diff(times)))
        value = center - 0.25 * (left - right) * offset
        return cls.Extremum(
            index,
            float(times[index] + offset * step),
            float(value),
            kind,
        )

    @classmethod
    def interpolated_extrema(
        cls,
        values: np.ndarray,
        times: np.ndarray,
        *,
        prominence: float | None = None,
        same_kind_distance: int = 1,
    ) -> list[Extremum]:
        values = np.asarray(values, dtype=np.float64)
        times = np.asarray(times, dtype=np.float64)
        peak_options: dict[str, float | int] = {
            "distance": max(1, int(same_kind_distance)),
        }
        if prominence is not None:
            peak_options["prominence"] = float(prominence)
        maxima, _ = find_peaks(values, **peak_options)
        minima, _ = find_peaks(-values, **peak_options)
        candidates = sorted(
            [(int(index), 1) for index in maxima]
            + [(int(index), -1) for index in minima]
        )
        alternating: list[tuple[int, int]] = []
        for index, kind in candidates:
            if alternating and alternating[-1][1] == kind:
                previous = alternating[-1][0]
                more_extreme = (
                    values[index] > values[previous]
                    if kind > 0
                    else values[index] < values[previous]
                )
                if more_extreme:
                    alternating[-1] = (index, kind)
            else:
                alternating.append((index, kind))
        return [
            cls._parabolic_extremum(values, times, index, kind)
            for index, kind in alternating
        ]

    @staticmethod
    def nearest_frame_map(
        target_times: np.ndarray,
        source_times: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        target_times = np.asarray(target_times, dtype=np.float64)
        source_times = np.asarray(source_times, dtype=np.float64)
        if len(source_times) == 0:
            return (
                np.zeros(len(target_times), dtype=int),
                np.zeros(len(target_times), dtype=bool),
            )
        right = np.searchsorted(source_times, target_times, side="left")
        right = np.clip(right, 0, len(source_times) - 1)
        left = np.maximum(right - 1, 0)
        choose_left = np.abs(target_times - source_times[left]) <= np.abs(
            source_times[right] - target_times
        )
        nearest = np.where(choose_left, left, right)
        covered = (target_times >= source_times[0]) & (target_times <= source_times[-1])
        return nearest.astype(int), covered


class CallableDetector(VibratoDetectorBase):
    """Small adapter for notebook experiments and tests."""

    def __init__(
        self,
        name: str,
        callback: Callable[[VibratoExample], VibratoEstimate],
        description: str = "External vibrato detector",
        requires: set[str] | None = None,
    ) -> None:
        self.name = name
        self.callback = callback
        self.description = description
        self.requires = set(requires or {"pitch"})

    def estimate(self, example: VibratoExample) -> VibratoEstimate:
        return self.callback(example).validate_for(example)
