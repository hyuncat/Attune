from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.modules.vibrato.VibratoDetectorBase import VibratoExample


Curve = Callable[[np.ndarray], np.ndarray]


class SyntheticDataset:
    """Generated contour corpus and its CSV serialization."""

    @dataclass(frozen=True)
    class Template:
        name: str
        duration: float
        has_vibrato: bool
        center: Curve
        rate: Curve
        width: Curve
        family: str

    @staticmethod
    def _constant(value: float) -> Curve:
        return lambda u: np.full_like(u, float(value), dtype=np.float64)

    @classmethod
    def _templates(cls) -> tuple[SyntheticDataset.Template, ...]:
        constant = cls._constant
        flat = constant(0.0)
        return (
            cls.Template(
                "constant", 2.0, True, flat, constant(5.5), constant(80.0), "stationary"
            ),
            cls.Template(
                "narrow", 2.2, True, flat, constant(6.2), constant(24.0), "stationary"
            ),
            cls.Template(
                "accelerating",
                3.0,
                True,
                flat,
                lambda u: 3.5 + 5.0 * u,
                constant(70.0),
                "rate_change",
            ),
            cls.Template(
                "decelerating",
                3.0,
                True,
                flat,
                lambda u: 8.5 - 5.0 * u,
                constant(70.0),
                "rate_change",
            ),
            cls.Template(
                "widening",
                3.0,
                True,
                flat,
                constant(5.4),
                lambda u: 18.0 + 102.0 * u,
                "width_change",
            ),
            cls.Template(
                "narrowing",
                3.0,
                True,
                flat,
                constant(5.4),
                lambda u: 120.0 - 102.0 * u,
                "width_change",
            ),
            cls.Template(
                "portamento_vibrato",
                3.0,
                True,
                lambda u: 2.0 * u,
                constant(5.2),
                constant(72.0),
                "moving_center",
            ),
            cls.Template(
                "short_vibrato",
                0.35,
                True,
                flat,
                constant(6.0),
                constant(82.0),
                "short",
            ),
            cls.Template("straight", 2.0, False, flat, flat, flat, "negative"),
            cls.Template(
                "portamento", 2.2, False, lambda u: 2.5 * u, flat, flat, "negative"
            ),
            cls.Template(
                "pitch_bend",
                2.2,
                False,
                lambda u: 1.4 * np.sin(np.pi * u),
                flat,
                flat,
                "negative",
            ),
            cls.Template(
                "center_wander",
                3.0,
                False,
                lambda u: 0.35 * np.sin(1.5 * np.pi * u),
                flat,
                flat,
                "negative",
            ),
            cls.Template(
                "pitch_step",
                2.0,
                False,
                lambda u: np.where(u < 0.5, 0.0, 0.8),
                flat,
                flat,
                "negative",
            ),
        )

    @staticmethod
    def _phase_from_rate(
        times: np.ndarray, rate_hz: np.ndarray, phase0: float
    ) -> np.ndarray:
        phase = np.empty_like(times)
        phase[0] = phase0
        if len(times) > 1:
            dt = np.diff(times)
            phase[1:] = phase0 + 2.0 * np.pi * np.cumsum(
                0.5 * (rate_hz[1:] + rate_hz[:-1]) * dt
            )
        return phase

    @staticmethod
    def _corrupt_pitch(
        pitch: np.ndarray,
        rng: np.random.Generator,
        *,
        noise_cents: float,
        dropout_probability: float,
        outlier_probability: float,
        outlier_scale_cents: float,
    ) -> np.ndarray:
        output = np.asarray(pitch, dtype=np.float64).copy()
        sigma = noise_cents / 100.0
        if sigma:
            white = rng.normal(0.0, sigma, len(output))
            output += np.convolve(
                white,
                np.array([0.2, 0.6, 0.2]),
                mode="same",
            )
        if outlier_probability:
            outliers = rng.random(len(output)) < outlier_probability
            output[outliers] += rng.normal(
                0.0,
                outlier_scale_cents / 100.0,
                int(outliers.sum()),
            )
        if dropout_probability:
            missing = rng.random(len(output)) < dropout_probability
            missing[1:] |= missing[:-1] & (rng.random(len(output) - 1) < 0.35)
            output[missing] = np.nan
        return output

    @classmethod
    def build(
        cls,
        *,
        replicates: int = 3,
        seed: int = 0,
        frame_rate: float = 100.0,
        noise_cents: float = 3.0,
        dropout_probability: float = 0.01,
        outlier_probability: float = 0.004,
        outlier_scale_cents: float = 35.0,
        context_seconds: float = 0.5,
    ) -> list[VibratoExample]:
        """Build shared multi-note contours with per-case scoring masks."""
        if replicates < 1:
            raise ValueError("replicates must be positive")
        if frame_rate <= 40.0:
            raise ValueError(
                "frame_rate must exceed 40 Hz to resolve the 20 Hz FDM band"
            )
        if noise_cents < 0.0 or outlier_scale_cents < 0.0:
            raise ValueError("noise and outlier scales cannot be negative")
        if not 0.0 <= dropout_probability < 1.0:
            raise ValueError("dropout_probability must be in [0, 1)")
        if not 0.0 <= outlier_probability < 1.0:
            raise ValueError("outlier_probability must be in [0, 1)")
        if context_seconds <= 0.0:
            raise ValueError("context_seconds must be positive")

        root_rng = np.random.default_rng(seed)
        examples: list[VibratoExample] = []
        guard_count = max(2, int(round(context_seconds * frame_rate)))
        for replicate in range(replicates):
            pitch_parts: list[np.ndarray] = []
            center_parts: list[np.ndarray] = []
            rate_parts: list[np.ndarray] = []
            width_parts: list[np.ndarray] = []
            vibrato_parts: list[np.ndarray] = []
            targets: list[dict[str, object]] = []
            cursor = 0

            for template in cls._templates():
                case_seed = int(root_rng.integers(0, np.iinfo(np.int32).max))
                rng = np.random.default_rng(case_seed)
                duration_scale = float(rng.uniform(0.92, 1.08))
                duration = template.duration * duration_scale
                count = max(8, int(round(duration * frame_rate)))
                times = np.arange(count, dtype=np.float64) / frame_rate
                u = np.linspace(0.0, 1.0, count, dtype=np.float64)
                base_pitch = float(rng.uniform(55.0, 76.0))
                center = base_pitch + template.center(u)
                rate = template.rate(u)
                width = template.width(u)
                phase = cls._phase_from_rate(
                    times, rate, float(rng.uniform(-np.pi, np.pi))
                )
                clean_pitch = center + (width / 200.0) * np.sin(phase)

                before_center = np.full(guard_count, center[0], dtype=np.float64)
                pitch_parts.append(
                    cls._corrupt_pitch(
                        before_center,
                        rng,
                        noise_cents=noise_cents,
                        dropout_probability=dropout_probability,
                        outlier_probability=outlier_probability,
                        outlier_scale_cents=outlier_scale_cents,
                    )
                )
                center_parts.append(before_center)
                rate_parts.append(np.zeros(guard_count, dtype=np.float64))
                width_parts.append(np.zeros(guard_count, dtype=np.float64))
                vibrato_parts.append(np.zeros(guard_count, dtype=np.bool_))
                cursor += guard_count

                target_start = cursor
                pitch_parts.append(
                    cls._corrupt_pitch(
                        clean_pitch,
                        rng,
                        noise_cents=noise_cents,
                        dropout_probability=dropout_probability,
                        outlier_probability=outlier_probability,
                        outlier_scale_cents=outlier_scale_cents,
                    )
                )
                center_parts.append(center.astype(np.float64))
                rate_parts.append(rate.astype(np.float64))
                width_parts.append(width.astype(np.float64))
                vibrato_parts.append(
                    np.full(count, template.has_vibrato, dtype=np.bool_)
                )
                cursor += count
                target_end = cursor

                after_center = np.full(guard_count, center[-1], dtype=np.float64)
                pitch_parts.append(
                    cls._corrupt_pitch(
                        after_center,
                        rng,
                        noise_cents=noise_cents,
                        dropout_probability=dropout_probability,
                        outlier_probability=outlier_probability,
                        outlier_scale_cents=outlier_scale_cents,
                    )
                )
                center_parts.append(after_center)
                rate_parts.append(np.zeros(guard_count, dtype=np.float64))
                width_parts.append(np.zeros(guard_count, dtype=np.float64))
                vibrato_parts.append(np.zeros(guard_count, dtype=np.bool_))
                cursor += guard_count
                targets.append(
                    {
                        "template": template,
                        "case_seed": case_seed,
                        "start": target_start,
                        "end": target_end,
                    }
                )

            pitch_track = np.concatenate(pitch_parts)
            center_track = np.concatenate(center_parts)
            rate_track = np.concatenate(rate_parts)
            width_track = np.concatenate(width_parts)
            vibrato_track = np.concatenate(vibrato_parts)
            track_times = np.arange(len(pitch_track), dtype=np.float64) / frame_rate
            analysis_note_bounds = [
                (
                    int(target["start"]) / frame_rate,
                    int(target["end"]) / frame_rate,
                    float(
                        np.median(
                            center_track[int(target["start"]) : int(target["end"])]
                        )
                    ),
                )
                for target in targets
            ]

            for target in targets:
                template = target["template"]
                target_start = int(target["start"])
                target_end = int(target["end"])
                evaluation_mask = np.zeros(len(pitch_track), dtype=np.bool_)
                evaluation_mask[target_start:target_end] = True
                examples.append(
                    VibratoExample(
                        case_id=f"{template.name}__r{replicate}",
                        scenario=template.name,
                        split="synthetic",
                        times=track_times,
                        pitch_midi=pitch_track,
                        center_midi=center_track,
                        rate_hz=rate_track,
                        width_cents=width_track,
                        is_vibrato=vibrato_track,
                        evaluation_mask=evaluation_mask,
                        metadata={
                            "family": template.family,
                            "replicate": replicate,
                            "seed": int(target["case_seed"]),
                            "analysis_group": f"synthetic_r{replicate}",
                            "analysis_note_bounds": analysis_note_bounds,
                            "continuous_context": True,
                            "context_seconds": context_seconds,
                            "target_start_time": target_start / frame_rate,
                            "target_end_time": target_end / frame_rate,
                            "noise_sigma_cents": noise_cents,
                            "dropout_probability": dropout_probability,
                            "outlier_probability": outlier_probability,
                            "outlier_scale_cents": outlier_scale_cents,
                        },
                    )
                )
        return examples

    @staticmethod
    def load_csv(path: str | Path) -> list[VibratoExample]:
        """Load externally annotated pitch contours from one tidy CSV."""
        frame = pd.read_csv(path)
        required = {"case_id", "time", "pitch_midi"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"annotated CSV is missing columns: {sorted(missing)}")

        examples: list[VibratoExample] = []
        for case_id, group in frame.groupby("case_id", sort=True):
            group = group.sort_values("time")
            n = len(group)

            def values(name: str, default: float) -> np.ndarray:
                if name not in group:
                    return np.full(n, default, dtype=np.float64)
                return group[name].to_numpy(dtype=np.float64)

            def boolean_values(name: str, default: bool) -> np.ndarray:
                if name not in group:
                    return np.full(n, default, dtype=np.bool_)
                labels = group[name]
                if pd.api.types.is_bool_dtype(labels) or pd.api.types.is_numeric_dtype(
                    labels
                ):
                    return labels.fillna(default).astype(bool).to_numpy()
                normalized = labels.fillna("").astype(str).str.strip().str.lower()
                valid = normalized.isin({"true", "false", "1", "0"})
                if not bool(valid.all()):
                    invalid = sorted(normalized[~valid].unique())
                    raise ValueError(f"invalid {name} labels: {invalid}")
                return normalized.isin({"true", "1"}).to_numpy()

            rate = values("rate_hz", 0.0)
            width = values("width_cents", 0.0)
            if "is_vibrato" in group:
                is_vibrato = boolean_values("is_vibrato", False)
            else:
                is_vibrato = (rate > 0.0) & (width > 0.0)
            evaluation_mask = boolean_values("evaluation_mask", True)
            center = values("center_midi", np.nan)
            commanded_pitch = (
                values("commanded_pitch_midi", np.nan)
                if "commanded_pitch_midi" in group
                else None
            )
            raw_pitch = (
                values("raw_pitch_midi", np.nan) if "raw_pitch_midi" in group else None
            )
            scenario = (
                str(group["scenario"].iloc[0]) if "scenario" in group else str(case_id)
            )
            split = str(group["split"].iloc[0]) if "split" in group else "external"
            metadata: dict[str, object] = {}
            audio_path = None
            if "audio_path" in group and pd.notna(group["audio_path"].iloc[0]):
                audio_path = str(group["audio_path"].iloc[0])
            if "analysis_group" in group and pd.notna(group["analysis_group"].iloc[0]):
                metadata["analysis_group"] = str(group["analysis_group"].iloc[0])
            if "continuous_context" in group:
                metadata["continuous_context"] = bool(
                    boolean_values("continuous_context", False)[0]
                )
            if "analysis_note_bounds" in group and pd.notna(
                group["analysis_note_bounds"].iloc[0]
            ):
                raw_bounds = json.loads(str(group["analysis_note_bounds"].iloc[0]))
                if not isinstance(raw_bounds, list):
                    raise ValueError("analysis_note_bounds must encode a JSON list")
                metadata["analysis_note_bounds"] = raw_bounds
            examples.append(
                VibratoExample(
                    case_id=str(case_id),
                    scenario=scenario,
                    split=split,
                    times=group["time"].to_numpy(dtype=np.float64),
                    pitch_midi=group["pitch_midi"].to_numpy(dtype=np.float64),
                    center_midi=center,
                    rate_hz=rate,
                    width_cents=width,
                    is_vibrato=is_vibrato.astype(np.bool_),
                    commanded_pitch_midi=commanded_pitch,
                    raw_pitch_midi=raw_pitch,
                    evaluation_mask=evaluation_mask,
                    audio_path=audio_path,
                    metadata=metadata,
                )
            )
        return examples

    @staticmethod
    def write_csv(examples: Iterable[VibratoExample], path: str | Path) -> Path:
        """Write examples in the same tidy format accepted by ``load_annotated_csv``."""
        rows: list[pd.DataFrame] = []
        for example in examples:
            note_bounds = example.metadata.get("analysis_note_bounds")
            rows.append(
                pd.DataFrame(
                    {
                        "case_id": example.case_id,
                        "scenario": example.scenario,
                        "split": example.split,
                        "time": example.times,
                        "pitch_midi": example.pitch_midi,
                        "center_midi": example.center_midi,
                        "rate_hz": example.rate_hz,
                        "width_cents": example.width_cents,
                        "is_vibrato": example.is_vibrato.astype(int),
                        "commanded_pitch_midi": (
                            example.commanded_pitch_midi
                            if example.commanded_pitch_midi is not None
                            else np.full(len(example.times), np.nan)
                        ),
                        "raw_pitch_midi": (
                            example.raw_pitch_midi
                            if example.raw_pitch_midi is not None
                            else np.full(len(example.times), np.nan)
                        ),
                        "evaluation_mask": example.score_mask.astype(int),
                        "audio_path": example.audio_path or "",
                        "analysis_group": example.metadata.get(
                            "analysis_group", example.case_id
                        ),
                        "continuous_context": bool(
                            example.metadata.get("continuous_context", False)
                        ),
                        "analysis_note_bounds": (
                            json.dumps(note_bounds) if note_bounds is not None else ""
                        ),
                    }
                )
            )
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(rows, ignore_index=True).to_csv(destination, index=False)
        return destination
