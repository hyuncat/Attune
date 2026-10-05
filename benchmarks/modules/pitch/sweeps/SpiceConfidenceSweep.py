"""Parallel worker for ``benchmarks/notebooks/sweeps/spice.ipynb``.

The notebook displays the confidence sweep using the existing gated RPA metric.

Run from the repository root:
    python -m benchmarks.modules.pitch.sweeps.SpiceConfidenceSweep
    python -m benchmarks.modules.pitch.sweeps.SpiceConfidenceSweep --datasets urmp coco bach10 --workers 4

Each worker loads one model and runs inference once per uncached track. Raw
outputs are cached ONLY for cheap threshold rescoring; every reported metric
still uses zero Hz for rejected frames and the benchmark's reference range.
No production defaults or ordinary competitor caches are changed. Winners are
exploratory results on the selected sample, not held-out performance claims.
"""

from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
from benchmarks.modules.pitch.competitors.Spice import Spice
from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase
from benchmarks.paths import RESULTS_ROOT
import mir_eval
import numpy as np
import pandas as pd

DEFAULT_THRESHOLDS = (
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.85,
    0.9,
    0.925,
    0.95,
    0.975,
)
METRICS = [
    "Raw Pitch Accuracy",
    "Overall Accuracy",
    "Voicing Recall",
    "Voicing False Alarm",
    "Raw Chroma Accuracy",
]
_detector = None


def score_thresholds(example, times, hz, confidence, thresholds):
    """Use exactly the ordinary confidence gate, range gate and mir_eval call."""
    rows = []
    for threshold in thresholds:
        gated = Spice.voiced_freqs(hz, confidence >= threshold)
        estimate = Spice.constrain_estimate_to_range(
            PitchDetectorBase.PitchEstimate.build(times, gated, 0.0),
            example.fmin,
            example.fmax,
        )
        rows.append(
            {
                "dataset": example.dataset,
                "track_id": example.track_id,
                "instrument": example.metadata.get("instrument", "unknown"),
                "threshold": threshold,
                **mir_eval.melody.evaluate(
                    example.ref_times, example.ref_freqs, estimate.times, estimate.freqs
                ),
            }
        )
    return rows


def cache_path(example, directory, model):
    source = example.audio_path.resolve()
    stat = source.stat()
    model_path = Path(model)
    model_stamp = None
    if model_path.is_dir():
        model_stamp = [
            (str(p.relative_to(model_path)), p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(model_path.rglob("*"))
            if p.is_file()
        ]
    signature = json.dumps(
        [1, str(source), stat.st_size, stat.st_mtime_ns, model, model_stamp]
    )
    key = hashlib.sha256(signature.encode()).hexdigest()
    return directory / f"{key}.npz"


def run_track(example, path, thresholds, model):
    global _detector
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            times, hz, confidence = (data[k] for k in ("times", "hz", "confidence"))
    else:
        if _detector is None:
            _detector = Spice()
            _detector.HUB_URL = model
            _detector.ensure_available()
        import tensorflow as tf

        audio, sr = example.audio(Spice.input_sr)
        output = _detector._load().signatures["serving_default"](
            tf.constant(audio, tf.float32)
        )
        hz = Spice._to_hz(np.asarray(output["pitch"]).reshape(-1))
        confidence = 1.0 - np.asarray(output["uncertainty"]).reshape(-1)
        times = np.arange(hz.size, dtype=np.float64) * Spice.HOP / sr
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npz")
        np.savez(temporary, times=times, hz=hz, confidence=confidence)
        temporary.replace(path)
    return score_thresholds(example, times, hz, confidence, thresholds)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=["urmp", "coco", "bach10"], default=["urmp"]
    )
    parser.add_argument("--per-instrument", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS
    )
    parser.add_argument(
        "--model",
        default=Spice.HUB_URL,
        help="TF Hub handle or a local SavedModel directory",
    )
    parser.add_argument(
        "--output", type=Path, default=RESULTS_ROOT / "pitch" / "spice_confidence_sweep"
    )
    args = parser.parse_args()
    if args.workers < 1 or args.per_instrument < 1:
        parser.error("workers and per-instrument must be positive")
    if not all((np.isfinite(t) and 0 <= t <= 1 for t in args.thresholds)):
        parser.error("thresholds must be finite values between 0 and 1")
    thresholds = sorted(set(args.thresholds) | {0.9})
    from benchmarks.modules.pitch.datasets.URMP import URMP
    from benchmarks.modules.pitch.datasets.CocoChorales import CocoChorales
    from benchmarks.modules.pitch.datasets.AudioAnnot import AudioAnnot

    factories = {"urmp": URMP, "coco": CocoChorales, "bach10": AudioAnnot}
    examples = []
    for name in dict.fromkeys(args.datasets):
        dataset = factories[name](per_instrument=args.per_instrument, seed=args.seed)
        tracks = dataset.tracks()
        if not tracks:
            parser.error(f"no tracks found for {name}")
        examples.extend((dataset.example(track) for track in tracks))
    model = str(Path(args.model).resolve()) if Path(args.model).is_dir() else args.model
    jobs = [(e, cache_path(e, args.output / "raw_cache", model)) for e in examples]
    resolved_model = model
    if any((not path.exists() for _, path in jobs)):
        try:
            detector = Spice()
            detector.ensure_available()
            import tensorflow_hub as hub

            resolved_model = hub.resolve(model)
            if not (Path(resolved_model) / "saved_model.pb").is_file():
                raise RuntimeError(f"no saved_model.pb in {resolved_model}")
        except Exception as exc:
            parser.exit(
                1,
                f"SPICE model unavailable: {exc}\nUse --model /path/to/a/valid/SavedModel to run inference.\n",
            )
    args.output.mkdir(parents=True, exist_ok=True)
    print(
        f"{len(jobs)} tracks × {len(thresholds)} thresholds; {sum((p.exists() for _, p in jobs))} cached tracks",
        flush=True,
    )
    rows = []
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(run_track, e, p, thresholds, resolved_model) for e, p in jobs
        ]
        for i, future in enumerate(as_completed(futures), 1):
            rows.extend(future.result())
            print(f"{i}/{len(jobs)} tracks scored", flush=True)
    frame = pd.DataFrame(rows).sort_values(["dataset", "track_id", "threshold"])
    summary = frame.groupby(["dataset", "threshold"])[METRICS].mean().reset_index()
    summary["tracks"] = frame.groupby(["dataset", "threshold"]).size().to_numpy()
    baseline = summary[summary.threshold == 0.9].set_index("dataset")
    for metric, label in [("Raw Pitch Accuracy", "rpa"), ("Overall Accuracy", "oa")]:
        summary[f"{label}_delta_vs_0.9"] = summary[metric] - summary.dataset.map(
            baseline[metric]
        )
    frame.to_csv(args.output / "rows.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    (args.output / "run.json").write_text(
        json.dumps(
            {
                **vars(args),
                "output": str(args.output),
                "thresholds": thresholds,
                "aggregation": "equal-weight mean over tracks within each dataset",
                "evaluation": "confidence-gated Hz; exploratory sample, no holdout",
                "tracks": [e.track_id for e in examples],
            },
            indent=2,
        )
        + "\n"
    )
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    for dataset, group in summary.groupby("dataset"):
        for metric in METRICS[:2]:
            best = group.loc[group[metric].idxmax()]
            print(
                f"{dataset}: best {metric} = {best[metric]:.4f} at confidence >= {best.threshold:g}"
            )
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
