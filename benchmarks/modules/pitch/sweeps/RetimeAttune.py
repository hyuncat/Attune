from __future__ import annotations

"Parallel retiming for cached Attune benchmark rows.\n\nThis does not rescore pitch accuracy. It keeps the existing cached pitch\nestimates/metrics, remeasures compute time for the selected Attune rows, updates\nthe raw-output CSV timing columns, and rebuilds ``pitch_benchmarks.csv`` with\nthroughput as ``sum(audio) / sum(compute)``.\n\nThe recomputed values use worker-local process CPU time. Wall time is written\nonly as a diagnostic, matching the main pitch benchmark.\n\nTypical full retime:\n\n    at-venv/bin/python -m benchmarks.modules.pitch.sweeps.RetimeAttune --workers 10\n\nFast smoke test with no writes:\n\n    at-venv/bin/python -m benchmarks.modules.pitch.sweeps.RetimeAttune --max-tracks 4         --no-cache-write --no-csv-write --no-summary\n"
import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")
import argparse
import csv
import lzma
import math
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _bootstrap_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "app.py").is_file() and (candidate / "benchmarks").is_dir():
            return candidate
    raise RuntimeError("could not locate Attune repo root")


_ROOT = _bootstrap_repo_root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker
from benchmarks.modules.pitch.PitchCache import PitchCache
from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase
from benchmarks.modules.pitch.competitors.Attune import AttuneRealtime
from benchmarks.modules.pitch.datasets.CocoChorales import CocoChorales

PITCH_AUDIO_SECONDS_COL = PitchBenchmarker.AUDIO_SECONDS_COL
PITCH_COMPUTE_COL = PitchBenchmarker.COMPUTE_COL
PITCH_WALL_COMPUTE_COL = PitchBenchmarker.WALL_COMPUTE_COL
PITCH_COMPUTE_CLOCK_COL = PitchBenchmarker.COMPUTE_CLOCK_COL
PITCH_REALTIME_COL = PitchBenchmarker.REALTIME_COL
PITCH_RAW_STAGE = PitchCache.RAW
PITCH_SMOOTHED_STAGE = PitchCache.SMOOTHED


@dataclass(frozen=True)
class RetimeItem:
    track_id: str
    split: str
    track: str
    instrument: str
    voice: int
    audio_seconds: float
    fmin: float
    fmax: float


@dataclass(frozen=True)
class RetimeResult:
    track_id: str
    detector_time: float
    smoother_time: float | None
    detector_wall_time: float
    smoother_wall_time: float | None
    audio_seconds: float


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        return (list(reader.fieldnames or []), list(reader))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def load_items(
    raw_root: Path,
    dataset_glob: str,
    track_ids: set[str] | None,
    max_tracks: int | None,
) -> list[RetimeItem]:
    seen: set[str] = set()
    items: list[RetimeItem] = []
    for csv_path in sorted((raw_root / "attune_realtime").glob(dataset_glob)):
        _, rows = _read_csv(csv_path)
        for row in rows:
            track_id = row["Track ID"]
            if track_id in seen:
                continue
            if track_ids is not None and track_id not in track_ids:
                continue
            seen.add(track_id)
            items.append(
                RetimeItem(
                    track_id=track_id,
                    split=row["Split"],
                    track=row["Track"],
                    instrument=row["Instrument"],
                    voice=int(row["Voice"]),
                    audio_seconds=_float(row, PITCH_AUDIO_SECONDS_COL),
                    fmin=_float(row, "Fmin"),
                    fmax=_float(row, "Fmax"),
                )
            )
            if max_tracks is not None and len(items) >= max_tracks:
                return items
    return items


def _cache_smoother_time(
    coco: CocoChorales, track_id: str
) -> tuple[float, float] | None:
    cache_path = coco.cache_path_for_track(track_id)
    stages, metadata = PitchCache(cache_path)._payload()
    timing = metadata.get(PITCH_SMOOTHED_STAGE, {})
    if (
        PITCH_SMOOTHED_STAGE not in stages
        or timing.get("compute_clock") != PitchDetectorBase.COMPUTE_CLOCK
    ):
        return None
    return (
        float(timing.get("pitch_smoother_compute_time", 0.0) or 0.0),
        float(timing.get("wall_pitch_smoother_compute_time", 0.0) or 0.0),
    )


def _write_cache_metadata(
    coco: CocoChorales,
    track_id: str,
    detector_time: float,
    smoother_time: float | None,
    detector_wall_time: float,
    smoother_wall_time: float | None,
) -> None:
    cache = PitchCache(coco.cache_path_for_track(track_id))
    cache_path = cache.path
    with cache._lock():
        stages, metadata = cache._payload()
        raw_meta = metadata.setdefault(PITCH_RAW_STAGE, {})
        raw_meta["pitch_detector_compute_time"] = detector_time
        raw_meta["pitch_smoother_compute_time"] = 0.0
        raw_meta["pitch_compute_time"] = detector_time
        raw_meta["wall_pitch_detector_compute_time"] = detector_wall_time
        raw_meta["wall_pitch_smoother_compute_time"] = 0.0
        raw_meta["wall_pitch_compute_time"] = detector_wall_time
        raw_meta["compute_clock"] = PitchDetectorBase.COMPUTE_CLOCK
        if PITCH_SMOOTHED_STAGE in stages and smoother_time is not None:
            smooth_meta = metadata.setdefault(PITCH_SMOOTHED_STAGE, {})
            smooth_meta["pitch_detector_compute_time"] = detector_time
            smooth_meta["pitch_smoother_compute_time"] = smoother_time
            smooth_meta["pitch_compute_time"] = detector_time + smoother_time
            smooth_meta["wall_pitch_detector_compute_time"] = detector_wall_time
            smooth_meta["wall_pitch_smoother_compute_time"] = smoother_wall_time or 0.0
            smooth_meta["wall_pitch_compute_time"] = detector_wall_time + (
                smoother_wall_time or 0.0
            )
            smooth_meta["compute_clock"] = PitchDetectorBase.COMPUTE_CLOCK
        out_payload = {
            "version": PitchCache.VERSION,
            "metadata": metadata,
            "stages": stages,
        }
        tmp = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
        try:
            with lzma.open(tmp, "wb") as fh:
                pickle.dump(out_payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(cache_path)
        finally:
            tmp.unlink(missing_ok=True)


def retime_one(
    item: RetimeItem, root: str | None, smoothed_source: str, write_cache: bool
) -> RetimeResult:
    coco = CocoChorales(root=root)
    attune = AttuneRealtime()
    wav = (
        coco.root
        / "materialized"
        / item.split
        / item.track
        / "stems_audio"
        / f"{item.voice}_{item.instrument}.wav"
    )
    cfg = attune.config_for(item.fmin, item.fmax)
    rec = attune.recording_for(cfg)
    rec.audio_data = coco.load_resampled_audio(wav, cfg.sr)
    recompute_smoother = smoothed_source == "recompute"
    stages = attune.detect_stages(rec, smooth=recompute_smoother)
    timing = stages.timing
    raw_timing = timing[PITCH_RAW_STAGE]
    detector_time = float(raw_timing["pitch_compute_time"])
    detector_wall_time = float(raw_timing["wall_pitch_compute_time"])
    smoother_time: float | None
    smoother_wall_time: float | None
    if recompute_smoother:
        smooth_timing = timing[PITCH_SMOOTHED_STAGE]
        smoother_time = float(smooth_timing["pitch_smoother_compute_time"])
        smoother_wall_time = float(smooth_timing["wall_pitch_smoother_compute_time"])
    else:
        cached_smoother = _cache_smoother_time(coco, item.track_id)
        if cached_smoother is not None:
            smoother_time, smoother_wall_time = cached_smoother
        else:
            _, smoother_time, smoother_wall_time = PitchDetectorBase.measure(
                lambda: rec.pitch_smoother.smooth(
                    stages.data[PITCH_RAW_STAGE].data, verbose=False
                )
            )
    if write_cache:
        _write_cache_metadata(
            coco,
            item.track_id,
            detector_time,
            smoother_time,
            detector_wall_time,
            smoother_wall_time,
        )
    return RetimeResult(
        track_id=item.track_id,
        detector_time=detector_time,
        smoother_time=smoother_time,
        detector_wall_time=detector_wall_time,
        smoother_wall_time=smoother_wall_time,
        audio_seconds=item.audio_seconds,
    )


def update_attune_csvs(
    raw_root: Path, dataset_glob: str, timings: dict[str, RetimeResult]
) -> None:
    for model in ("attune_realtime", "attune_adhoc"):
        model_dir = raw_root / model
        if not model_dir.is_dir():
            continue
        for csv_path in sorted(model_dir.glob(dataset_glob)):
            fieldnames, rows = _read_csv(csv_path)
            for column in (PITCH_WALL_COMPUTE_COL, PITCH_COMPUTE_CLOCK_COL):
                if column not in fieldnames:
                    fieldnames.append(column)
            changed = False
            for row in rows:
                result = timings.get(row["Track ID"])
                if result is None:
                    continue
                if model == "attune_realtime":
                    compute = result.detector_time
                    wall_compute = result.detector_wall_time
                elif result.smoother_time is not None:
                    compute = result.detector_time + result.smoother_time
                    wall_compute = result.detector_wall_time + (
                        result.smoother_wall_time or 0.0
                    )
                else:
                    continue
                audio = _float(row, PITCH_AUDIO_SECONDS_COL)
                row[PITCH_COMPUTE_COL] = repr(float(compute))
                row[PITCH_WALL_COMPUTE_COL] = repr(float(wall_compute))
                row[PITCH_COMPUTE_CLOCK_COL] = PitchDetectorBase.COMPUTE_CLOCK
                row[PITCH_REALTIME_COL] = repr(
                    audio / compute
                    if compute > 0 and math.isfinite(audio)
                    else float("nan")
                )
                changed = True
            if changed:
                _write_csv(csv_path, fieldnames, rows)


def _summary_model_order(summary_path: Path, raw_root: Path) -> list[str]:
    if summary_path.exists():
        _, rows = _read_csv(summary_path)
        return [row["model"] for row in rows if row.get("model")]
    return [p.name for p in sorted(raw_root.iterdir()) if p.is_dir()]


def rebuild_summary(raw_root: Path, summary_path: Path, dataset_glob: str) -> None:
    model_order = _summary_model_order(summary_path, raw_root)
    metric_cols = PitchBenchmarker.METRICS
    fieldnames = ["model", "Tracks", *metric_cols, PITCH_REALTIME_COL]
    summary_rows: list[dict[str, Any]] = []
    for model in model_order:
        rows: list[dict[str, str]] = []
        model_dir = raw_root / model
        if not model_dir.is_dir():
            continue
        for csv_path in sorted(model_dir.glob(dataset_glob)):
            _, part = _read_csv(csv_path)
            rows.extend(part)
        if not rows:
            continue
        rec: dict[str, Any] = {"model": model, "Tracks": len(rows)}
        for col in metric_cols:
            vals = [float(row[col]) for row in rows if row.get(col) not in ("", None)]
            rec[col] = sum(vals) / len(vals) if vals else float("nan")
        timed_rows = [
            row
            for row in rows
            if row.get(PITCH_COMPUTE_CLOCK_COL) == PitchDetectorBase.COMPUTE_CLOCK
        ]
        total_audio = sum((_float(row, PITCH_AUDIO_SECONDS_COL) for row in timed_rows))
        total_compute = sum((_float(row, PITCH_COMPUTE_COL) for row in timed_rows))
        rec[PITCH_REALTIME_COL] = (
            total_audio / total_compute if total_compute > 0 else float("nan")
        )
        summary_rows.append(rec)
    _write_csv(summary_path, fieldnames, summary_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None, help="CocoChorales root override")
    parser.add_argument(
        "--raw-root",
        default=str(_ROOT / "benchmarks" / "results" / "pitch" / "raw_outputs"),
        help="raw output root containing attune_realtime/ and attune_adhoc/",
    )
    parser.add_argument(
        "--summary",
        default=str(
            _ROOT / "benchmarks" / "results" / "pitch" / "pitch_benchmarks.csv"
        ),
        help="summary CSV to rebuild",
    )
    parser.add_argument(
        "--dataset-glob",
        default="coco_*.csv",
        help="which raw-output CSVs to retime/summarize (default: coco_*.csv)",
    )
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 4))
    parser.add_argument("--max-tracks", type=int, default=None, help="smoke-test cap")
    parser.add_argument(
        "--track-id",
        action="append",
        dest="track_ids",
        help="retime only this Track ID; repeatable",
    )
    parser.add_argument(
        "--smoothed-source",
        choices=["recompute", "cached"],
        default="recompute",
        help="recompute smoother timing too, or only retime raw detector and reuse cached smoother time (default: recompute)",
    )
    parser.add_argument(
        "--no-cache-write",
        action="store_true",
        help="do not update .pitch.pkl.xz timing metadata",
    )
    parser.add_argument(
        "--no-csv-write", action="store_true", help="do not update raw-output CSVs"
    )
    parser.add_argument(
        "--no-summary", action="store_true", help="do not rebuild pitch_benchmarks.csv"
    )
    parser.add_argument(
        "--allow-partial-write",
        action="store_true",
        help="allow writes with --max-tracks/--track-id",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print selected work and exit"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_root = Path(args.raw_root)
    summary_path = Path(args.summary)
    track_ids = set(args.track_ids) if args.track_ids else None
    items = load_items(raw_root, args.dataset_glob, track_ids, args.max_tracks)
    partial = bool(args.max_tracks is not None or track_ids)
    will_write = not (args.no_cache_write and args.no_csv_write and args.no_summary)
    print(f"raw root:   {raw_root}")
    print(f"summary:    {summary_path}")
    print(f"csvs:       {args.dataset_glob}")
    print(f"workers:    {args.workers}")
    print(f"smoothed:   {args.smoothed_source}")
    print(f"tracks:     {len(items)}")
    print(
        f"writes:     cache={('no' if args.no_cache_write else 'yes')}, csv={('no' if args.no_csv_write else 'yes')}, summary={('no' if args.no_summary else 'yes')}"
    )
    if not items:
        print("no Attune rows selected")
        return 1
    if args.dry_run:
        for item in items[:10]:
            print(f"  {item.track_id}")
        if len(items) > 10:
            print(f"  ... {len(items) - 10} more")
        return 0
    if partial and will_write and (not args.allow_partial_write):
        print(
            "refusing partial writes; rerun with --allow-partial-write, or add --no-cache-write --no-csv-write --no-summary for a write-free smoke test",
            file=sys.stderr,
        )
        return 2
    started = time.perf_counter()
    timings: dict[str, RetimeResult] = {}
    errors: list[tuple[str, str]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                retime_one,
                item,
                args.root,
                args.smoothed_source,
                not args.no_cache_write,
            ): item
            for item in items
        }
        for done, fut in enumerate(as_completed(futures), start=1):
            item = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:
                errors.append((item.track_id, repr(exc)))
                print(f"[err] {item.track_id}: {exc!r}", file=sys.stderr, flush=True)
                continue
            timings[result.track_id] = result
            if done % 25 == 0 or done == len(items):
                elapsed = time.perf_counter() - started
                rate = done / elapsed if elapsed else 0.0
                eta = (len(items) - done) / rate if rate else 0.0
                smooth = (
                    ""
                    if result.smoother_time is None
                    else f" smooth={result.smoother_time:.4f}s"
                )
                print(
                    f"[retime] {done:4d}/{len(items)} elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m last={result.track_id} raw={result.detector_time:.4f}s{smooth}",
                    flush=True,
                )
    if errors:
        print(
            f"\n{len(errors)} track(s) failed; not writing CSV/summary.",
            file=sys.stderr,
        )
        for track_id, msg in errors[:20]:
            print(f"  - {track_id}: {msg}", file=sys.stderr)
        return 1
    if not args.no_csv_write:
        update_attune_csvs(raw_root, args.dataset_glob, timings)
        print(f"updated raw-output CSV timing for {len(timings)} Attune row(s)")
    if not args.no_summary:
        rebuild_summary(raw_root, summary_path, args.dataset_glob)
        print(f"rebuilt summary -> {summary_path}")
    total_audio = sum((r.audio_seconds for r in timings.values()))
    total_raw = sum((r.detector_time for r in timings.values()))
    print(
        f"Attune real-time retimed throughput: {total_audio / total_raw:.6g}x ({total_audio:.3f}s audio / {total_raw:.3f}s compute)"
    )
    if any((r.smoother_time is not None for r in timings.values())):
        total_smoothed = sum(
            (r.detector_time + (r.smoother_time or 0.0) for r in timings.values())
        )
        print(
            f"Attune ad-hoc retimed throughput: {total_audio / total_smoothed:.6g}x ({total_audio:.3f}s audio / {total_smoothed:.3f}s compute)"
        )
    print(f"done in {(time.perf_counter() - started) / 60:.1f}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
