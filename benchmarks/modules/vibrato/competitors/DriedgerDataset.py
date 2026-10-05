"""Loader and acceptance metrics for Driedger et al.'s released dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from benchmarks.modules.vibrato.competitors.Driedger import (
    DRIEDGER_FRAME_RATE,
    Driedger,
)
from benchmarks.modules.vibrato.VibratoDetectorBase import VibratoExample


DRIEDGER_CONDITIONS = ("0dB", "-5dB", "-10dB")
DRIEDGER_PUBLISHED_MEAN_F1 = {
    "0dB": 0.80,
    "-5dB": 0.77,
    "-10dB": 0.76,
}


class DriedgerDataset:
    """Paper-release corpus used only to validate the Driedger port."""

    CONDITIONS = DRIEDGER_CONDITIONS
    PUBLISHED_MEAN_F1 = DRIEDGER_PUBLISHED_MEAN_F1

    @staticmethod
    def _reference_intervals(path: Path) -> np.ndarray:
        if not path.is_file() or path.stat().st_size == 0:
            return np.empty((0, 2), dtype=np.float64)
        rows = np.loadtxt(path, delimiter=",", ndmin=2)
        if rows.shape[1] < 3:
            raise ValueError(f"expected start,label,duration rows in {path}")
        starts = rows[:, 0]
        durations = rows[:, 2]
        if np.any(~np.isfinite(starts)) or np.any(durations <= 0.0):
            raise ValueError(f"invalid vibrato interval in {path}")
        return np.column_stack((starts, starts + durations))

    @classmethod
    def load(
        cls,
        dataset_root: str | Path,
        *,
        conditions: tuple[str, ...] = DRIEDGER_CONDITIONS,
        frame_rate: float = DRIEDGER_FRAME_RATE,
    ) -> list[VibratoExample]:
        """Load the nine items x three mixes used in the paper's Table 1."""

        root = Path(dataset_root)
        unknown = set(conditions) - set(DRIEDGER_CONDITIONS)
        if unknown:
            raise ValueError(f"unknown Driedger conditions: {sorted(unknown)}")
        examples: list[VibratoExample] = []
        for audio_path in sorted(root.glob("*_excerpt_mix_0dB.wav")):
            base = audio_path.name.removesuffix("_mix_0dB.wav")
            annotation_path = root / f"{base}_vibrato.csv"
            intervals = cls._reference_intervals(annotation_path)
            for condition in conditions:
                condition_path = root / f"{base}_mix_{condition}.wav"
                if not condition_path.is_file():
                    raise FileNotFoundError(condition_path)
                info = sf.info(condition_path)
                duration = float(info.frames / info.samplerate)
                times = (
                    np.arange(
                        max(2, int(np.floor(duration * frame_rate))),
                        dtype=np.float64,
                    )
                    / frame_rate
                )
                truth = np.zeros(len(times), dtype=bool)
                for start, end in intervals:
                    truth |= (times >= start) & (times < end)
                zeros = np.zeros(len(times), dtype=np.float64)
                examples.append(
                    VibratoExample(
                        case_id=f"{base}__{condition}",
                        scenario=condition,
                        split="driedger_release",
                        times=times,
                        pitch_midi=np.full(len(times), np.nan, dtype=np.float64),
                        center_midi=np.full(len(times), np.nan, dtype=np.float64),
                        rate_hz=zeros.copy(),
                        width_cents=zeros.copy(),
                        is_vibrato=truth,
                        audio_path=str(condition_path),
                        metadata={
                            "family": "driedger_release_detection_acceptance",
                            "item": base,
                            "condition": condition,
                            "annotation_path": str(annotation_path),
                            "annotation_intervals": len(intervals),
                        },
                    )
                )
        if len(examples) != 9 * len(conditions):
            raise ValueError(
                f"expected {9 * len(conditions)} released cases, found {len(examples)}"
            )
        return examples

    @classmethod
    def evaluate(
        cls,
        dataset_root: str | Path,
        *,
        estimator: Driedger | None = None,
        conditions: tuple[str, ...] = DRIEDGER_CONDITIONS,
        progress=None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return per-item F1 and macro means comparable to Table 1."""

        from benchmarks.modules.vibrato.VibratoBenchmarker import VibratoBenchmarker

        examples = cls.load(
            dataset_root,
            conditions=conditions,
        )
        method = estimator or Driedger(template_mode="detection")
        raw = VibratoBenchmarker().run(
            examples,
            [method],
            progress=progress,
            strict=True,
        )
        item_rows: list[dict[str, object]] = []
        for _, row in raw.iterrows():
            tp = int(row["_frame_tp"])
            fp = int(row["_frame_fp"])
            fn = int(row["_frame_fn"])
            # Equation 7's epsilon convention makes the correctly empty Sleigh
            # Ride result perfect rather than undefined/zero.
            f1 = 1.0 if tp == fp == fn == 0 else float(row["frame_f1"])
            item_rows.append(
                {
                    "item": row["meta_item"],
                    "condition": row["meta_condition"],
                    "frame_f1": f1,
                    "frame_precision": (
                        1.0 if tp == fp == 0 else float(row["frame_precision"])
                    ),
                    "frame_recall": (
                        1.0 if tp == fn == 0 else float(row["frame_recall"])
                    ),
                    "compute_seconds": float(row["compute_seconds"]),
                    "audio_seconds": float(row["audio_seconds"]),
                }
            )
        items = pd.DataFrame(item_rows)
        means = (
            items.groupby("condition", sort=False)
            .agg(
                mean_frame_f1=("frame_f1", "mean"),
                compute_seconds=("compute_seconds", "sum"),
                audio_seconds=("audio_seconds", "sum"),
            )
            .reset_index()
        )
        means["published_mean_f1"] = means["condition"].map(DRIEDGER_PUBLISHED_MEAN_F1)
        means["absolute_gap"] = np.abs(
            means["mean_frame_f1"] - means["published_mean_f1"]
        )
        means["within_0_05"] = means["absolute_gap"] <= 0.05
        means["audio_per_compute"] = means["audio_seconds"] / means["compute_seconds"]
        return items, means
