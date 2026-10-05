from __future__ import annotations

"Parallel, resumable time-aware alignment parameter sweep.\n\nThe notebook is intentionally only a front end for this module.  Each worker\nloads one CocoChorales stem's cached pitch track, detects notes once, and then\nevaluates the complete parameter list for that track.  This avoids both the\nnotebook-kernel state dependency and the old parameter-by-parameter reload\npattern.\n"
import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")
import hashlib
import io
import json
import multiprocessing
import random
import sys
import traceback
from collections import Counter
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Sequence
import mir_eval
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from benchmarks.paths import REPO_ROOT
from benchmarks.paths import SWEEP_RESULTS_ROOT
from benchmarks.paths import ensure_repo_on_path

ensure_repo_on_path()
from algorithms.Config import Config
from algorithms.MistakeDetector import MistakeDetector
from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker
from benchmarks.modules.mistake.datasets.MistakeInjector import MistakeInjector
from benchmarks.modules.note.NoteBenchmarker import CocoNoteBenchmarker
from benchmarks.modules.note.NoteDetectorBase import NoteDetectorBase

RUNNER_VERSION = 2
PARAM_COLUMNS = [
    "alpha_onset",
    "alpha_duration",
    "gamma_pitch",
    "gamma_time",
    "ins_base",
    "del_base",
]
ALIGNMENT_METRICS = [
    "pair_precision",
    "pair_recall",
    "pair_f1",
    "false_mistakes_per_ref",
    "indels_per_ref",
]
MIXED_ERROR_TYPES = ("substitution", "deletion", "insertion", "short", "long")
MIXED_PITCH_TYPES = ("substitution", "deletion", "insertion")
MIXED_DURATION_TYPES = ("short", "long")
MIXED_COUNT_COLUMNS = [
    f"mixed_{group}_{suffix}"
    for group in ("error", "pitch", "duration", "correct")
    for suffix in ("tp", "fp", "fn")
] + [
    "mixed_source_pair_tp",
    "mixed_source_pair_total",
    "mixed_detected_indels",
    "mixed_truth_events",
]
MIXED_METRICS = [
    f"mixed_{group}_{metric}"
    for group in ("error", "pitch", "duration", "correct")
    for metric in ("precision", "recall", "f1")
] + ["mixed_source_pair_recall", "mixed_indels_per_event"]


@dataclass(frozen=True)
class SweepOptions:
    split: str = "test"
    seed: int = 0
    per_stratum: int = 10
    tune_per_stratum: int = 8
    onset_tolerance_sec: float = 0.05
    pitch_tolerance_cents: float = 50.0
    offset_ratio: float = 0.2
    mixed_mistake_rate: float = 0.25
    stage2_top_raw: int = 5
    stage2_top_accepted: int = 5
    holdout_top: int = 10
    max_strata: int | None = None
    stage1_limit: int | None = None


def production_params(config: Config | None = None) -> dict[str, float]:
    current = config or Config()
    return {
        "alpha_onset": float(current.alignment_alpha_onset),
        "alpha_duration": float(current.alignment_alpha_duration),
        "gamma_pitch": float(current.alignment_gamma_pitch),
        "gamma_time": float(current.alignment_gamma_time),
        "ins_base": float(current.ins_cost),
        "del_base": float(current.del_cost),
    }


def first_stage_grid(limit: int | None = None) -> list[dict[str, float]]:
    grid = [
        {
            "alpha_onset": float(alpha_onset),
            "alpha_duration": float(1.0 - alpha_onset),
            "gamma_pitch": float(gamma_pitch),
            "gamma_time": float(gamma_time),
            "ins_base": 5.0,
            "del_base": 5.0,
        }
        for alpha_onset, gamma_pitch, gamma_time in product(
            (0.0, 0.25, 0.5, 0.75, 1.0),
            (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
            (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0),
        )
    ]
    if limit is None or limit >= len(grid):
        return grid
    selected = grid[: max(0, limit - 1)]
    prod = production_params()
    if prod not in selected:
        selected.append(prod)
    return selected


def _param_key(params: dict[str, Any] | pd.Series) -> tuple[float, ...]:
    return tuple((float(params[name]) for name in PARAM_COLUMNS))


def _records_by_stratum(
    bench: CocoNoteBenchmarker, options: SweepOptions
) -> tuple[int, list[dict[str, Any]]]:
    records = bench.coco.read_manifest(options.split)
    groups: dict[tuple[str, str], list[tuple[Any, Path, Path, Path]]] = defaultdict(
        list
    )
    for record in records:
        wav = bench.coco.local_wav_path(record)
        midi = bench.local_midi_path(record)
        cache = bench.coco.cache_path_for_track(record.track_id)
        if wav is not None and midi is not None and cache.exists():
            groups[record.ensemble, record.instrument].append(
                (record, wav, midi, cache)
            )
    strata = sorted(groups)
    if options.max_strata is not None:
        strata = strata[: options.max_strata]

    def assigned_role(record: Any) -> str:
        digest = hashlib.sha256(
            f"{options.seed}:{record.split}:{record.track}".encode()
        ).digest()
        bucket = int.from_bytes(digest[:8], "big") % options.per_stratum
        return "tune" if bucket < options.tune_per_stratum else "holdout"

    rng = random.Random(options.seed)
    selected: list[dict[str, Any]] = []
    for stratum in strata:
        candidates = sorted(groups[stratum], key=lambda row: row[0].track_id)
        rng.shuffle(candidates)
        role_quotas = (
            ("tune", options.tune_per_stratum, 0),
            (
                "holdout",
                options.per_stratum - options.tune_per_stratum,
                options.tune_per_stratum,
            ),
        )
        for role, quota, rank_offset in role_quotas:
            accepted = 0
            for record, wav, midi, cache in candidates:
                if assigned_role(record) != role:
                    continue
                if not bench.has_pitch_cache(cache, smooth=True):
                    continue
                selected.append(
                    {
                        "track": record.track,
                        "track_id": record.track_id,
                        "ensemble": record.ensemble,
                        "instrument": record.instrument,
                        "wav": str(wav),
                        "midi": str(midi),
                        "cache": str(cache),
                        "stratum_rank": rank_offset + accepted,
                        "role": role,
                    }
                )
                accepted += 1
                if accepted == quota:
                    break
            if accepted != quota:
                raise RuntimeError(
                    f"only found {accepted}/{quota} cached {role} stems for {stratum}"
                )
    return (len(records), selected)


def select_sample(options: SweepOptions) -> pd.DataFrame:
    bench = CocoNoteBenchmarker(onset_tolerance=options.onset_tolerance_sec)
    manifest_count, rows = _records_by_stratum(bench, options)
    sample = pd.DataFrame(rows)
    sample.attrs["manifest_count"] = manifest_count
    tune_tracks = set(sample.loc[sample.role == "tune", "track"])
    holdout_tracks = set(sample.loc[sample.role == "holdout", "track"])
    if not tune_tracks.isdisjoint(holdout_tracks):
        raise AssertionError("chorale-track leakage across tuning and holdout roles")
    return sample


def _safe_prf(predicted: set[Any], oracle: set[Any]) -> tuple[float, float, float]:
    tp = len(predicted & oracle)
    precision = tp / len(predicted) if predicted else float(not oracle)
    recall = tp / len(oracle) if oracle else float(not predicted)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return (precision, recall, f1)


def _prepare_track(row: dict[str, Any], options: SweepOptions) -> dict[str, Any]:
    bench = CocoNoteBenchmarker(onset_tolerance=options.onset_tolerance_sec)
    recording, ref_intervals, ref_pitches, _ = bench._prepare_note_recording(
        row["wav"], row["midi"], align="identity", needs_audio=False
    )
    recording.reset_analysis()
    recording.transition_detector.clear_transitions(recording.pitch_data.data)
    recording.update_min_note_length()
    recording.note_data = recording.note_detector.detect_notes(
        recording.pitch_data.data
    )
    detected = NoteDetectorBase.clone_note_data(recording.note_data)
    reference = NoteDetectorBase.clone_note_data(
        recording.score_data.clipped_note_data(channel=recording.active_instrument)
    )
    est_intervals, est_pitches = bench.notedata_to_intervals(detected, recording.config)
    latency = bench._latency_offset(ref_intervals[:, 0], est_intervals[:, 0])
    if latency:
        notes = detected.read(i=0, j=len(detected.times))
        for note in notes:
            note.start_time += latency
            note.end_time += latency
        detected.load_data({note.start_time: note for note in notes})
        est_intervals, est_pitches = bench.notedata_to_intervals(
            detected, recording.config
        )
    alignment_oracle_pairs = set(
        mir_eval.transcription.match_notes(
            ref_intervals,
            np.ones(len(ref_intervals)),
            est_intervals,
            np.ones(len(est_intervals)),
            onset_tolerance=options.onset_tolerance_sec,
            pitch_tolerance=options.pitch_tolerance_cents,
            offset_ratio=options.offset_ratio,
        )
    )
    pitch_pairs = set(
        mir_eval.transcription.match_notes(
            ref_intervals,
            ref_pitches,
            est_intervals,
            est_pitches,
            onset_tolerance=options.onset_tolerance_sec,
            pitch_tolerance=options.pitch_tolerance_cents,
            offset_ratio=options.offset_ratio,
        )
    )
    note_precision = (
        len(pitch_pairs) / len(est_intervals)
        if len(est_intervals)
        else float(not len(ref_intervals))
    )
    note_recall = (
        len(pitch_pairs) / len(ref_intervals)
        if len(ref_intervals)
        else float(not len(est_intervals))
    )
    note_f1 = (
        2 * note_precision * note_recall / (note_precision + note_recall)
        if note_precision + note_recall
        else 0.0
    )
    return {
        **{key: row[key] for key in ("track_id", "ensemble", "instrument", "role")},
        "config": recording.config,
        "detected": detected,
        "reference": reference,
        "alignment_oracle_pairs": alignment_oracle_pairs,
        "reference_notes": len(ref_intervals),
        "estimated_notes": len(est_intervals),
        "pitch_aware_note_f1": note_f1,
        "latency_shift_sec": latency,
    }


def _config_for(item: dict[str, Any], params: dict[str, float]) -> Config:
    return replace(
        item["config"],
        alignment_alpha_onset=params["alpha_onset"],
        alignment_alpha_duration=params["alpha_duration"],
        alignment_gamma_pitch=params["gamma_pitch"],
        alignment_gamma_time=params["gamma_time"],
        ins_cost=params["ins_base"],
        del_cost=params["del_base"],
    )


def _score_alignment(item: dict[str, Any], params: dict[str, float]) -> dict[str, Any]:
    detector = MistakeDetector(config=_config_for(item, params))
    alignment = detector.detect_mistakes(
        user_notes=item["detected"], score_notes=item["reference"]
    )
    pairs = alignment.pairs
    mistakes = alignment.pitch_mistakes
    est_notes = list(item["detected"].data.values())
    ref_notes = list(item["reference"].data.values())
    est_index = {id(note): index for index, note in enumerate(est_notes)}
    ref_index = {id(note): index for index, note in enumerate(ref_notes)}
    predicted = {
        (ref_index[id(score_note)], est_index[id(user_note)])
        for user_note, score_note in pairs
        if user_note is not None and score_note is not None
    }
    precision, recall, f1 = _safe_prf(predicted, item["alignment_oracle_pairs"])
    reference_count = max(1, item["reference_notes"])
    indels = sum((mistake.type in {"insertion", "deletion"} for mistake in mistakes))
    return {
        **params,
        **{key: item[key] for key in ("track_id", "ensemble", "instrument", "role")},
        "pair_precision": precision,
        "pair_recall": recall,
        "pair_f1": f1,
        "false_mistakes_per_ref": len(mistakes) / reference_count,
        "indels_per_ref": indels / reference_count,
    }


def _mixed_seed(seed: int, track_id: str) -> int:
    digest = hashlib.sha256(f"mixed:{seed}:{track_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _prepare_mixed(
    item: dict[str, Any], options: SweepOptions
) -> tuple[Any, list[dict[str, Any]]]:
    current = Config()
    injector = MistakeInjector(
        mistake_rate=options.mixed_mistake_rate,
        weights=(0.2, 0.2, 0.2, 0.2, 0.2),
        duration_error_min_sec=max(0.3, current.timing_tolerance + 0.05),
        timing_std_ms=0.0,
        duration_std=0.0,
    )
    return injector.inject(
        item["reference"],
        np.random.default_rng(_mixed_seed(options.seed, item["track_id"])),
    )


def _sum_counts(
    counts: dict[str, tuple[int, int, int]], mistake_types: Sequence[str]
) -> tuple[int, int, int]:
    selected = [counts.get(mistake_type, (0, 0, 0)) for mistake_type in mistake_types]
    return tuple((sum(values) for values in zip(*selected)))


def _score_mixed(
    item: dict[str, Any],
    performed: Any,
    truth: list[dict[str, Any]],
    params: dict[str, float],
) -> dict[str, Any]:
    config = _config_for(item, params)
    detector = MistakeDetector(config=config)
    alignment = detector.detect_mistakes(
        user_notes=performed, score_notes=item["reference"]
    )
    pairs = alignment.pairs
    pitch_mistakes = alignment.pitch_mistakes
    timing_mistakes = alignment.timing_mistakes
    counts = MistakeBenchmarker.score_symbolic(
        [*pitch_mistakes, *timing_mistakes], truth, pairs=pairs, canonical=False
    )
    result: dict[str, Any] = {
        **params,
        **{key: item[key] for key in ("track_id", "ensemble", "instrument", "role")},
    }
    for group, types in (
        ("error", MIXED_ERROR_TYPES),
        ("pitch", MIXED_PITCH_TYPES),
        ("duration", MIXED_DURATION_TYPES),
    ):
        tp, fp, fn = _sum_counts(counts, types)
        result.update(
            {f"mixed_{group}_tp": tp, f"mixed_{group}_fp": fp, f"mixed_{group}_fn": fn}
        )
    correct_tp, correct_fp, correct_fn = counts.get("correct", (0, 0, 0))
    result.update(
        {
            "mixed_correct_tp": correct_tp,
            "mixed_correct_fp": correct_fp,
            "mixed_correct_fn": correct_fn,
        }
    )
    source_ids = {
        int(note.source_score_id)
        for note in performed.read(i=0, j=len(performed.times))
        if getattr(note, "source_score_id", None) is not None
    }
    retained = {
        int(user_note.source_score_id)
        for user_note, score_note in pairs
        if user_note is not None
        and score_note is not None
        and (getattr(user_note, "source_score_id", None) == score_note.id)
    }
    result.update(
        {
            "mixed_source_pair_tp": len(retained),
            "mixed_source_pair_total": len(source_ids),
            "mixed_detected_indels": sum(
                (
                    mistake.type in {"insertion", "deletion"}
                    for mistake in pitch_mistakes
                )
            ),
            "mixed_truth_events": len(truth),
        }
    )
    return result


_WORKER_PARAMS: list[dict[str, float]] = []
_WORKER_OPTIONS = SweepOptions()


def _init_worker(params: list[dict[str, float]], options_dict: dict[str, Any]) -> None:
    global _WORKER_PARAMS, _WORKER_OPTIONS
    _WORKER_PARAMS = params
    _WORKER_OPTIONS = SweepOptions(**options_dict)


def _score_track_worker(row: dict[str, Any]) -> dict[str, Any]:
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            item = _prepare_track(row, _WORKER_OPTIONS)
            performed, truth = _prepare_mixed(item, _WORKER_OPTIONS)
            alignment_rows = [
                _score_alignment(item, params) for params in _WORKER_PARAMS
            ]
            mixed_rows = [
                _score_mixed(item, performed, truth, params)
                for params in _WORKER_PARAMS
            ]
        track_meta = {
            key: item[key]
            for key in (
                "track_id",
                "ensemble",
                "instrument",
                "role",
                "reference_notes",
                "estimated_notes",
                "pitch_aware_note_f1",
                "latency_shift_sec",
            )
        }
        track_meta.update(
            {f"truth_{k}": v for k, v in Counter((e["type"] for e in truth)).items()}
        )
        return {
            "ok": True,
            "track_id": row["track_id"],
            "alignment_rows": alignment_rows,
            "mixed_rows": mixed_rows,
            "track_meta": track_meta,
        }
    except Exception as exc:
        return {
            "ok": False,
            "track_id": row.get("track_id", ""),
            "error": "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
        }


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def _signature(
    stage: str,
    tracks: Sequence[dict[str, Any]],
    params: Sequence[dict[str, float]],
    options: SweepOptions,
) -> str:
    payload = {
        "version": RUNNER_VERSION,
        "stage": stage,
        "tracks": [row["track_id"] for row in tracks],
        "params": params,
        "options": asdict(options),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _run_stage(
    stage: str,
    tracks: Sequence[dict[str, Any]],
    params: list[dict[str, float]],
    output_dir: Path,
    workers: int,
    resume: bool,
    options: SweepOptions,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    alignment_path = output_dir / f"{stage}_alignment_rows.csv"
    mixed_path = output_dir / f"{stage}_mixed_rows.csv"
    track_path = output_dir / f"{stage}_track_meta.csv"
    metadata_path = output_dir / f"{stage}_metadata.json"
    signature = _signature(stage, tracks, params, options)
    if resume and all(
        (
            path.exists()
            for path in (alignment_path, mixed_path, track_path, metadata_path)
        )
    ):
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("signature") == signature:
            print(f"[{stage}] loading checkpoint")
            return (
                pd.read_csv(alignment_path),
                pd.read_csv(mixed_path),
                pd.read_csv(track_path),
            )
    alignment_rows: list[dict[str, Any]] = []
    mixed_rows: list[dict[str, Any]] = []
    track_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    options_dict = asdict(options)
    if workers <= 1:
        _init_worker(params, options_dict)
        results: Iterable[dict[str, Any]] = map(_score_track_worker, tracks)
        for result in tqdm(results, total=len(tracks), desc=stage):
            if result["ok"]:
                alignment_rows.extend(result["alignment_rows"])
                mixed_rows.extend(result["mixed_rows"])
                track_rows.append(result["track_meta"])
            else:
                errors.append(
                    {"track_id": result["track_id"], "error": result["error"]}
                )
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_init_worker,
            initargs=(params, options_dict),
        ) as pool:
            results = pool.map(_score_track_worker, tracks, chunksize=1)
            for result in tqdm(results, total=len(tracks), desc=stage):
                if result["ok"]:
                    alignment_rows.extend(result["alignment_rows"])
                    mixed_rows.extend(result["mixed_rows"])
                    track_rows.append(result["track_meta"])
                else:
                    errors.append(
                        {"track_id": result["track_id"], "error": result["error"]}
                    )
    if errors:
        _atomic_csv(pd.DataFrame(errors), output_dir / f"{stage}_errors.csv")
        first = errors[0]
        raise RuntimeError(
            f"{stage}: {len(errors)} track worker(s) failed; first={first['track_id']}\n{first['error']}"
        )
    alignment = pd.DataFrame(alignment_rows)
    mixed = pd.DataFrame(mixed_rows)
    track_meta = pd.DataFrame(track_rows)
    _atomic_csv(alignment, alignment_path)
    _atomic_csv(mixed, mixed_path)
    _atomic_csv(track_meta, track_path)
    metadata_path.write_text(
        json.dumps(
            {
                "signature": signature,
                "stage": stage,
                "workers": workers,
                "tracks": len(tracks),
                "parameters": len(params),
            },
            indent=2,
        )
    )
    return (alignment, mixed, track_meta)


def summarize_alignment(rows: pd.DataFrame) -> pd.DataFrame:
    per_stratum = rows.groupby(
        PARAM_COLUMNS + ["ensemble", "instrument"], as_index=False
    )[ALIGNMENT_METRICS].mean()
    summary = per_stratum.groupby(PARAM_COLUMNS, as_index=False)[
        ALIGNMENT_METRICS
    ].mean()
    summary["_alpha_imbalance"] = abs(summary.alpha_onset - 0.5)
    summary = summary.sort_values(
        [
            "pair_f1",
            "false_mistakes_per_ref",
            "indels_per_ref",
            "gamma_pitch",
            "gamma_time",
            "_alpha_imbalance",
        ],
        ascending=[False, True, True, True, True, True],
    )
    return summary.drop(columns="_alpha_imbalance").reset_index(drop=True)


def _add_prf_columns(frame: pd.DataFrame, group: str) -> None:
    tp = frame[f"mixed_{group}_tp"].to_numpy(dtype=float)
    fp = frame[f"mixed_{group}_fp"].to_numpy(dtype=float)
    fn = frame[f"mixed_{group}_fn"].to_numpy(dtype=float)
    precision = np.divide(tp, tp + fp, out=np.ones_like(tp), where=tp + fp > 0)
    recall = np.divide(tp, tp + fn, out=np.ones_like(tp), where=tp + fn > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=precision + recall > 0,
    )
    frame[f"mixed_{group}_precision"] = precision
    frame[f"mixed_{group}_recall"] = recall
    frame[f"mixed_{group}_f1"] = f1


def summarize_mixed(rows: pd.DataFrame) -> pd.DataFrame:
    per_stratum = rows.groupby(
        PARAM_COLUMNS + ["ensemble", "instrument"], as_index=False
    )[MIXED_COUNT_COLUMNS].sum()
    for group in ("error", "pitch", "duration", "correct"):
        _add_prf_columns(per_stratum, group)
    per_stratum["mixed_source_pair_recall"] = (
        per_stratum.mixed_source_pair_tp / per_stratum.mixed_source_pair_total
    )
    per_stratum["mixed_indels_per_event"] = (
        per_stratum.mixed_detected_indels / per_stratum.mixed_truth_events
    )
    return (
        per_stratum.groupby(PARAM_COLUMNS, as_index=False)[MIXED_METRICS]
        .mean()
        .reset_index(drop=True)
    )


def _join_summaries(alignment: pd.DataFrame, mixed: pd.DataFrame) -> pd.DataFrame:
    return alignment.merge(mixed, on=PARAM_COLUMNS, validate="one_to_one")


def _row_for(frame: pd.DataFrame, params: dict[str, float]) -> pd.Series:
    mask = np.logical_and.reduce(
        [frame[name].eq(value) for name, value in params.items()]
    )
    matches = frame.loc[mask]
    if matches.empty:
        raise RuntimeError(f"parameter row missing: {params}")
    return matches.iloc[0]


def accepted_vs_production(joint: pd.DataFrame) -> pd.DataFrame:
    production = _row_for(joint, production_params())
    return joint[
        (joint.mixed_error_f1 >= production.mixed_error_f1)
        & (joint.mixed_correct_f1 >= production.mixed_correct_f1)
        & (joint.mixed_source_pair_recall >= production.mixed_source_pair_recall)
        & (joint.false_mistakes_per_ref <= production.false_mistakes_per_ref)
        & (joint.indels_per_ref <= production.indels_per_ref)
    ].sort_values(
        ["pair_f1", "mixed_error_f1", "false_mistakes_per_ref", "indels_per_ref"],
        ascending=[False, False, True, True],
    )


def _unique_params(
    rows: Iterable[dict[str, Any] | pd.Series]
) -> list[dict[str, float]]:
    unique: dict[tuple[float, ...], dict[str, float]] = {}
    for row in rows:
        params = {name: float(row[name]) for name in PARAM_COLUMNS}
        unique[_param_key(params)] = params
    return list(unique.values())


def _stage2_grid(
    joint_stage1: pd.DataFrame, options: SweepOptions
) -> list[dict[str, float]]:
    raw = joint_stage1.sort_values("pair_f1", ascending=False).head(
        options.stage2_top_raw
    )
    accepted = accepted_vs_production(joint_stage1).head(options.stage2_top_accepted)
    seeds = _unique_params([*raw.to_dict("records"), *accepted.to_dict("records")])
    expanded = []
    for seed in seeds:
        for ins_base, del_base in product((2.5, 5.0, 10.0), repeat=2):
            expanded.append({**seed, "ins_base": ins_base, "del_base": del_base})
    return _unique_params(expanded)


def _combine_summaries(*frames: pd.DataFrame) -> pd.DataFrame:
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(
            ["pair_f1", "false_mistakes_per_ref", "indels_per_ref"],
            ascending=[False, True, True],
        )
        .drop_duplicates(PARAM_COLUMNS, keep="first")
        .reset_index(drop=True)
    )


def _write_summary(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    _atomic_csv(frame, path)
    return frame


def run_sweep(
    *,
    output_dir: Path | str | None = None,
    workers: int | None = None,
    resume: bool = True,
    options: SweepOptions | None = None,
) -> dict[str, Any]:
    options = options or SweepOptions()
    if not 0 < options.tune_per_stratum < options.per_stratum:
        raise ValueError("tune_per_stratum must be between 0 and per_stratum")
    workers = max(1, int(workers or max(1, (os.cpu_count() or 4) - 1)))
    output_dir = Path(output_dir or SWEEP_RESULTS_ROOT / "mistake")
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = select_sample(options)
    _atomic_csv(sample, output_dir / "sample.csv")
    tune_tracks = sample.loc[sample.role == "tune"].to_dict("records")
    holdout_tracks = sample.loc[sample.role == "holdout"].to_dict("records")
    print(
        f"sample: {len(sample)} stems / {sample.groupby(['ensemble', 'instrument']).ngroups} strata ({len(tune_tracks)} tune, {len(holdout_tracks)} holdout)"
    )
    print(f"workers: {workers}; output: {output_dir}")
    stage1_params = first_stage_grid(options.stage1_limit)
    print(f"stage 1: {len(stage1_params)} configurations")
    stage1_alignment_rows, stage1_mixed_rows, stage1_track_meta = _run_stage(
        "stage1_tune", tune_tracks, stage1_params, output_dir, workers, resume, options
    )
    stage1_alignment = _write_summary(
        summarize_alignment(stage1_alignment_rows),
        output_dir / "stage1_alignment_summary.csv",
    )
    stage1_mixed = _write_summary(
        summarize_mixed(stage1_mixed_rows), output_dir / "stage1_mixed_summary.csv"
    )
    joint_stage1 = _write_summary(
        _join_summaries(stage1_alignment, stage1_mixed),
        output_dir / "stage1_joint_summary.csv",
    )
    stage2_params = _stage2_grid(joint_stage1, options)
    print(f"stage 2: {len(stage2_params)} configurations")
    stage2_alignment_rows, stage2_mixed_rows, stage2_track_meta = _run_stage(
        "stage2_tune", tune_tracks, stage2_params, output_dir, workers, resume, options
    )
    stage2_alignment = _write_summary(
        summarize_alignment(stage2_alignment_rows),
        output_dir / "stage2_alignment_summary.csv",
    )
    stage2_mixed = _write_summary(
        summarize_mixed(stage2_mixed_rows), output_dir / "stage2_mixed_summary.csv"
    )
    joint_stage2 = _write_summary(
        _join_summaries(stage2_alignment, stage2_mixed),
        output_dir / "stage2_joint_summary.csv",
    )
    joint_tune = _combine_summaries(joint_stage1, joint_stage2)
    _write_summary(joint_tune, output_dir / "tune_candidates.csv")
    accepted_tune = accepted_vs_production(joint_tune)
    raw_best = joint_tune.sort_values("pair_f1", ascending=False).head(1)
    candidate_rows = [
        *accepted_tune.head(options.holdout_top).to_dict("records"),
        *raw_best.to_dict("records"),
        production_params(),
    ]
    holdout_params = _unique_params(candidate_rows)
    print(f"holdout: {len(holdout_params)} accepted/diagnostic configurations")
    holdout_alignment_rows, holdout_mixed_rows, holdout_track_meta = _run_stage(
        "holdout", holdout_tracks, holdout_params, output_dir, workers, resume, options
    )
    holdout_alignment = _write_summary(
        summarize_alignment(holdout_alignment_rows),
        output_dir / "holdout_alignment_summary.csv",
    )
    holdout_mixed = _write_summary(
        summarize_mixed(holdout_mixed_rows), output_dir / "holdout_mixed_summary.csv"
    )
    joint_holdout = _write_summary(
        _join_summaries(holdout_alignment, holdout_mixed),
        output_dir / "holdout_candidates.csv",
    )
    accepted_holdout = accepted_vs_production(joint_holdout)
    recommendation = (
        accepted_holdout.iloc[0]
        if not accepted_holdout.empty
        else _row_for(joint_holdout, production_params())
    )
    recommendation_params = {
        name: float(recommendation[name]) for name in PARAM_COLUMNS
    }
    (output_dir / "recommendation.json").write_text(
        json.dumps(
            {
                "runner_version": RUNNER_VERSION,
                "workers": workers,
                "options": asdict(options),
                "production": production_params(),
                "recommended": recommendation_params,
            },
            indent=2,
        )
    )
    metadata = pd.concat(
        [stage1_track_meta, stage2_track_meta, holdout_track_meta], ignore_index=True
    ).drop_duplicates("track_id")
    _atomic_csv(metadata, output_dir / "track_meta.csv")
    print("recommended:", recommendation_params)
    return {
        "output_dir": output_dir,
        "sample": sample,
        "track_meta": metadata,
        "stage1": joint_stage1,
        "stage2": joint_stage2,
        "tune": joint_tune,
        "holdout": joint_holdout,
        "accepted_tune": accepted_tune,
        "accepted_holdout": accepted_holdout,
        "recommendation": recommendation_params,
    }


def load_results(output_dir: Path | str | None = None) -> dict[str, Any]:
    output_dir = Path(output_dir or SWEEP_RESULTS_ROOT / "mistake")
    recommendation_path = output_dir / "recommendation.json"
    return {
        "output_dir": output_dir,
        "sample": pd.read_csv(output_dir / "sample.csv"),
        "track_meta": pd.read_csv(output_dir / "track_meta.csv"),
        "stage1": pd.read_csv(output_dir / "stage1_joint_summary.csv"),
        "stage2": pd.read_csv(output_dir / "stage2_joint_summary.csv"),
        "tune": pd.read_csv(output_dir / "tune_candidates.csv"),
        "holdout": pd.read_csv(output_dir / "holdout_candidates.csv"),
        "recommendation": json.loads(recommendation_path.read_text())["recommended"],
    }


__all__ = [
    "ALIGNMENT_METRICS",
    "MIXED_METRICS",
    "PARAM_COLUMNS",
    "SweepOptions",
    "accepted_vs_production",
    "first_stage_grid",
    "load_results",
    "production_params",
    "run_sweep",
    "select_sample",
    "summarize_alignment",
    "summarize_mixed",
]
