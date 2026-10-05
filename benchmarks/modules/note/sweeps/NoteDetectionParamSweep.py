from __future__ import annotations

"Parallel CocoChorales note-detection method and parameter sweeps.\n\nEvery worker loads one materialized stem's cached *smoothed* pYIN track once,\nthen evaluates every requested variant against the stem MIDI. Pitch detection\nis never run by this module. Every method uses the production\n``unvoiced_prob < unv_thresh`` frame gate. Stage one compares PELT/KernelCPD\ncosts with an explicit one-frame candidate-boundary hop. Stage two fixes linear\nKernelCPD and runs the full Cartesian product of three interpretable parameters:\npitch step, score-relative minimum note length, and majority-silence duration.\nTransition processing and post-hoc adjacent merging are disabled.\n"
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
from contextlib import redirect_stderr
from contextlib import redirect_stdout
import json
import multiprocessing
import random
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from dataclasses import asdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any
from typing import Sequence
import numpy as np
import pandas as pd
import ruptures as rpt
from tqdm.auto import tqdm
from benchmarks.paths import SWEEP_RESULTS_ROOT
from benchmarks.paths import ensure_repo_on_path

ensure_repo_on_path()
from algorithms.Config import Config
from benchmarks.modules.note.NoteBenchmarker import CocoNoteBenchmarker
from benchmarks.modules.note.NoteDetectorBase import NoteDetectorBase
from benchmarks.modules.note.competitors.Ruptures import Ruptures

RUNNER_VERSION = 4
DEFAULT_OUTPUT_DIR = SWEEP_RESULTS_ROOT / "note"
FIXED_PARAMETER_METHOD = "kernelcpd-linear"
METHOD_CONFIGS: dict[str, dict[str, Any]] = {
    "pelt-l1": {"ruptures_algorithm": "pelt", "model": "l1", "jump": 1},
    "pelt-l2": {"ruptures_algorithm": "pelt", "model": "l2", "jump": 1},
    "pelt-rbf": {"ruptures_algorithm": "pelt", "model": "rbf", "jump": 1},
    "kernelcpd-linear": {"ruptures_algorithm": "kernelcpd", "model": "l2", "jump": 1},
    "kernelcpd-gaussian": {
        "ruptures_algorithm": "kernelcpd",
        "model": "rbf",
        "jump": 1,
    },
    "kernelcpd-cosine": {
        "ruptures_algorithm": "kernelcpd",
        "model": "cosine",
        "jump": 1,
    },
}
PARAM_COLUMNS = [
    "method",
    "unv_thresh",
    "pitch_step_semitones",
    "min_note_length_factor",
    "min_silence_duration_ms",
]
SWEEP_PARAM_COLUMNS = [
    "pitch_step_semitones",
    "min_note_length_factor",
    "min_silence_duration_ms",
]


@dataclass(frozen=True)
class ParameterAxes:
    pitch_step_semitones: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    min_note_length_factor: tuple[float, ...] = (0.5, 0.75, 1.0)
    min_silence_duration_ms: tuple[float, ...] = (3.0, 10.0, 20.0, 40.0)


@dataclass(frozen=True)
class SweepOptions:
    split: str = "test"
    seed: int = 0
    per_stratum: int = 10
    tune_per_stratum: int = 8
    onset_tolerance_sec: float = 0.05
    max_strata: int | None = None
    max_tracks: int | None = None


def production_parameters() -> dict[str, Any]:
    return {
        "method": FIXED_PARAMETER_METHOD,
        "unv_thresh": Config.unv_thresh,
        "pitch_step_semitones": Config.pitch_thresh,
        "min_note_length_factor": Config.min_note_length_factor,
        "min_silence_duration_ms": Config.min_silence_duration_ms,
    }


def parameter_grid(axes: ParameterAxes | None = None) -> list[dict[str, Any]]:
    """Return the full Cartesian product for the three defined parameters."""
    axes = axes or ParameterAxes()
    if any((value <= 0 for value in axes.pitch_step_semitones)):
        raise ValueError("pitch_step_semitones values must be positive")
    if any((value <= 0 for value in axes.min_note_length_factor)):
        raise ValueError("min_note_length_factor values must be positive")
    if any((value <= 0 for value in axes.min_silence_duration_ms)):
        raise ValueError("min_silence_duration_ms values must be positive")
    values = [getattr(axes, name) for name in SWEEP_PARAM_COLUMNS]
    return [
        dict(zip(SWEEP_PARAM_COLUMNS, combination)) for combination in product(*values)
    ]


def method_variants(methods: Sequence[str] | None = None) -> list[dict[str, Any]]:
    selected = list(methods or METHOD_CONFIGS)
    unknown = [name for name in selected if name not in METHOD_CONFIGS]
    if unknown:
        raise ValueError(f"unknown method(s): {', '.join(unknown)}")
    return [
        {"variant": name, "method": name, **METHOD_CONFIGS[name]} for name in selected
    ]


def parameter_variants(grid: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    method_config = METHOD_CONFIGS[FIXED_PARAMETER_METHOD]
    return [
        {
            "variant": f"params-{index:04d}",
            "method": FIXED_PARAMETER_METHOD,
            **method_config,
            **params,
        }
        for index, params in enumerate(grid)
    ]


def _role_for_track(track: str, options: SweepOptions) -> str:
    digest = hashlib.sha256(f"{options.seed}:{track}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % options.per_stratum
    return "tune" if bucket < options.tune_per_stratum else "holdout"


def select_sample(options: SweepOptions) -> pd.DataFrame:
    if not 0 < options.tune_per_stratum < options.per_stratum:
        raise ValueError("tune_per_stratum must be between 1 and per_stratum - 1")
    bench = CocoNoteBenchmarker(onset_tolerance=options.onset_tolerance_sec)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in bench.coco.read_manifest(options.split):
        wav = bench.coco.local_wav_path(record)
        midi = bench.local_midi_path(record)
        cache = bench.coco.cache_path_for_track(record.track_id)
        if wav is None or midi is None or (not cache.exists()):
            continue
        groups[record.ensemble, record.instrument].append(
            {
                "track": record.track,
                "track_id": record.track_id,
                "ensemble": record.ensemble,
                "instrument": record.instrument,
                "wav": str(wav),
                "midi": str(midi),
                "cache": str(cache),
            }
        )
    strata = sorted(groups)
    if options.max_strata is not None:
        strata = strata[: options.max_strata]
    rng = random.Random(options.seed)
    selected: list[dict[str, Any]] = []
    for stratum in strata:
        candidates = sorted(groups[stratum], key=lambda row: row["track_id"])
        rng.shuffle(candidates)
        for role, quota in (
            ("tune", options.tune_per_stratum),
            ("holdout", options.per_stratum - options.tune_per_stratum),
        ):
            role_rows = []
            for row in candidates:
                if _role_for_track(row["track"], options) != role:
                    continue
                if not bench.has_pitch_cache(row["cache"], smooth=True):
                    continue
                role_rows.append(row)
                if len(role_rows) == quota:
                    break
            if len(role_rows) < quota:
                raise RuntimeError(
                    f"only found {len(role_rows)}/{quota} cached {role} stems for {stratum}"
                )
            for rank, row in enumerate(role_rows[:quota]):
                selected.append({**row, "role": role, "stratum_rank": rank})
    sample = pd.DataFrame(selected)
    sample["_role_order"] = sample["role"].map({"tune": 0, "holdout": 1})
    sample = (
        sample.sort_values(["_role_order", "ensemble", "instrument", "track_id"])
        .drop(columns="_role_order")
        .reset_index(drop=True)
    )
    if options.max_tracks is not None:
        sample = sample.iloc[: options.max_tracks].copy()
    if sample.empty:
        raise RuntimeError(
            "no materialized CocoChorales stems with smoothed-pYIN caches were found"
        )
    tune_tracks = set(sample.loc[sample.role == "tune", "track"])
    holdout_tracks = set(sample.loc[sample.role == "holdout", "track"])
    if not tune_tracks.isdisjoint(holdout_tracks):
        raise AssertionError("chorale leakage between tuning and holdout samples")
    return sample


def _penalty(recording: Any, params: dict[str, Any]) -> float:
    min_frames = recording.config.min_note_pitch_frames(
        factor=float(params["min_note_length_factor"])
    )
    return 0.5 * min_frames * float(params["pitch_step_semitones"]) ** 2


def _variant_parameters(variant: dict[str, Any]) -> dict[str, Any]:
    parameters = production_parameters()
    parameters["method"] = variant["method"]
    parameters.update(
        {key: variant[key] for key in SWEEP_PARAM_COLUMNS if key in variant}
    )
    return parameters


def _score_notes(
    bench: CocoNoteBenchmarker,
    recording: Any,
    ref_intervals: np.ndarray,
    ref_pitches: np.ndarray,
    audio_seconds: float,
    variant: dict[str, Any],
    parameters: dict[str, Any],
    notes: Any,
    note_compute_time: float,
) -> dict[str, Any]:
    est_intervals, est_pitches = bench.notedata_to_intervals(notes, recording.config)
    if len(est_intervals) == 0:
        precision = recall = f_measure = overlap = 0.0
    else:
        precision, recall, f_measure, overlap = bench._eval_intervals(
            ref_intervals,
            ref_pitches,
            est_intervals,
            est_pitches,
            align="identity",
            latency_align=True,
            trim_boundaries=True,
            onset_tolerance=bench.onset_tolerance,
        )
    realtime_factor = (
        audio_seconds / note_compute_time
        if note_compute_time > 0 and np.isfinite(audio_seconds)
        else float("nan")
    )
    return {
        **{key: variant[key] for key in variant if key != "variant"},
        "variant": variant["variant"],
        **parameters,
        "Precision": float(precision),
        "Recall": float(recall),
        "F-measure": float(f_measure),
        "Average Overlap Ratio": float(overlap),
        "Estimated Notes": int(len(est_intervals)),
        "Reference Notes": int(len(ref_intervals)),
        "note_compute_time": float(note_compute_time),
        "audio_seconds": float(audio_seconds),
        "realtime_factor": float(realtime_factor),
        "error": None,
    }


def _score_variant(
    bench: CocoNoteBenchmarker,
    recording: Any,
    ref_intervals: np.ndarray,
    ref_pitches: np.ndarray,
    audio_seconds: float,
    variant: dict[str, Any],
) -> dict[str, Any]:
    detector = NoteDetectorBase(recording)
    started = time.perf_counter()
    parameters = _variant_parameters(variant)
    notes = detector.detect(
        method="ruptures",
        ruptures_algorithm=variant["ruptures_algorithm"],
        model=variant["model"],
        jump=1,
        pen=_penalty(recording, parameters),
        pitch_step_semitones=float(parameters["pitch_step_semitones"]),
        min_note_length_factor=float(parameters["min_note_length_factor"]),
        min_silence_duration_ms=float(parameters["min_silence_duration_ms"]),
        exclude_transitions=False,
        merge_adjacent=False,
    )
    note_compute_time = time.perf_counter() - started
    return _score_notes(
        bench,
        recording,
        ref_intervals,
        ref_pitches,
        audio_seconds,
        variant,
        parameters,
        notes,
        note_compute_time,
    )


def _is_parameter_batch(variants: Sequence[dict[str, Any]]) -> bool:
    return bool(variants) and all(
        (
            variant.get("method") == FIXED_PARAMETER_METHOD
            and all((name in variant for name in SWEEP_PARAM_COLUMNS))
            for variant in variants
        )
    )


def _score_parameter_batch(
    bench: CocoNoteBenchmarker,
    recording: Any,
    ref_intervals: np.ndarray,
    ref_pitches: np.ndarray,
    audio_seconds: float,
    variants: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reuse each linear-kernel fit across all pitch-step penalties.

    Runs and KernelCPD cost matrices depend on the score-relative minimum size
    and silence window, but not on the penalty's pitch-step value. The shared
    preparation time is divided evenly among the variants that reuse it so the
    timing columns remain additive rather than counting the same work repeatedly.
    """
    detector = Ruptures(recording)
    groups: dict[tuple[float, float], list[tuple[dict[str, Any], dict[str, Any]]]] = (
        defaultdict(list)
    )
    for variant in variants:
        parameters = _variant_parameters(variant)
        key = (
            float(parameters["min_note_length_factor"]),
            float(parameters["min_silence_duration_ms"]),
        )
        groups[key].append((variant, parameters))
    scored: dict[str, dict[str, Any]] = {}
    for (factor, silence_ms), group in groups.items():
        shared_started = time.perf_counter()
        detector._configure_note_segmentation(
            min_note_length_factor=factor, min_silence_duration_ms=silence_ms
        )
        min_size = detector._pelt_min_size_from_score()
        runs = detector._pelt_runs(recording.pitch_data)
        fitted_runs: list[tuple[list[Any], Any | None]] = []
        for run in runs:
            signal = detector._feature_matrix(
                run, feature_names=("pitch",), standardize=False
            )
            fitted = None
            if len(run) >= 2 * min_size:
                try:
                    fitted = rpt.KernelCPD(
                        kernel="linear", min_size=min_size, jump=1
                    ).fit(signal)
                except (rpt.exceptions.BadSegmentationParameters, ValueError):
                    fitted = None
            fitted_runs.append((run, fitted))
        shared_time = time.perf_counter() - shared_started
        shared_time_per_variant = shared_time / len(group)
        for variant, parameters in group:
            predict_started = time.perf_counter()
            penalty = _penalty(recording, parameters)
            breakpoints_by_run: list[list[int]] = []
            for run, fitted in fitted_runs:
                if fitted is None:
                    breakpoints = [len(run)]
                else:
                    try:
                        breakpoints = detector._sanitize_bkps(
                            fitted.predict(pen=penalty), len(run)
                        )
                    except (rpt.exceptions.BadSegmentationParameters, ValueError):
                        breakpoints = [len(run)]
                breakpoints_by_run.append(breakpoints)
            notes = detector._notes_from_run_breakpoints(
                runs, breakpoints_by_run, merge_adjacent=False
            )
            note_compute_time = (
                shared_time_per_variant + time.perf_counter() - predict_started
            )
            scored[variant["variant"]] = _score_notes(
                bench,
                recording,
                ref_intervals,
                ref_pitches,
                audio_seconds,
                variant,
                parameters,
                notes,
                note_compute_time,
            )
    return [scored[variant["variant"]] for variant in variants]


def _run_track(
    row: dict[str, Any], variants: list[dict[str, Any]], onset_tolerance_sec: float
) -> list[dict[str, Any]]:
    """Evaluate one materialized track without interleaved worker logging."""
    with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
        return _run_track_impl(row, variants, onset_tolerance_sec)


def _run_track_impl(
    row: dict[str, Any], variants: list[dict[str, Any]], onset_tolerance_sec: float
) -> list[dict[str, Any]]:
    bench = CocoNoteBenchmarker(onset_tolerance=onset_tolerance_sec)
    try:
        recording, ref_intervals, ref_pitches, audio_seconds = (
            bench._prepare_note_recording(
                row["wav"], row["midi"], align="identity", needs_audio=False
            )
        )
    except Exception as exc:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        return [
            {
                **{
                    key: row[key]
                    for key in ("track", "track_id", "ensemble", "instrument", "role")
                },
                "variant": variant["variant"],
                "method": variant["method"],
                "error": tb,
            }
            for variant in variants
        ]
    if _is_parameter_batch(variants):
        metadata = {
            key: row[key]
            for key in ("track", "track_id", "ensemble", "instrument", "role")
        }
        try:
            return [
                {**metadata, **result}
                for result in _score_parameter_batch(
                    bench,
                    recording,
                    ref_intervals,
                    ref_pitches,
                    audio_seconds,
                    variants,
                )
            ]
        except Exception as exc:
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            return [
                {
                    **metadata,
                    **{
                        key: value for key, value in variant.items() if key != "variant"
                    },
                    "variant": variant["variant"],
                    "error": tb,
                }
                for variant in variants
            ]
    output: list[dict[str, Any]] = []
    for variant in variants:
        metadata = {
            key: row[key]
            for key in ("track", "track_id", "ensemble", "instrument", "role")
        }
        try:
            output.append(
                {
                    **metadata,
                    **_score_variant(
                        bench,
                        recording,
                        ref_intervals,
                        ref_pitches,
                        audio_seconds,
                        variant,
                    ),
                }
            )
        except Exception as exc:
            output.append(
                {
                    **metadata,
                    **{
                        key: value for key, value in variant.items() if key != "variant"
                    },
                    "variant": variant["variant"],
                    "error": "".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    ),
                }
            )
    return output


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _signature(
    stage: str,
    sample: pd.DataFrame,
    variants: Sequence[dict[str, Any]],
    options: SweepOptions,
) -> str:
    payload = {
        "runner_version": RUNNER_VERSION,
        "stage": stage,
        "tracks": sample.track_id.tolist(),
        "variants": variants,
        "options": asdict(options),
    }
    encoded = json.dumps(payload, sort_keys=True, default=_json_default).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _checkpoint_path(directory: Path, track_id: str) -> Path:
    safe = track_id.replace("/", "_")
    digest = hashlib.sha256(track_id.encode()).hexdigest()[:10]
    return directory / f"{safe}__{digest}.json"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=_json_default))
    temporary.replace(path)


def _parallel_stage(
    stage: str,
    sample: pd.DataFrame,
    variants: list[dict[str, Any]],
    output_dir: Path,
    workers: int,
    options: SweepOptions,
    force: bool,
) -> pd.DataFrame:
    signature = _signature(stage, sample, variants, options)
    checkpoint_dir = output_dir / "checkpoints" / f"{stage}-{signature}"
    all_rows: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for row in sample.to_dict("records"):
        checkpoint = _checkpoint_path(checkpoint_dir, row["track_id"])
        if checkpoint.exists() and (not force):
            all_rows.extend(json.loads(checkpoint.read_text()))
        else:
            pending.append(row)
    if pending:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=max(1, workers), mp_context=context
        ) as pool:
            futures = {
                pool.submit(_run_track, row, variants, options.onset_tolerance_sec): row
                for row in pending
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc=f"{stage} tracks"
            ):
                row = futures[future]
                try:
                    track_rows = future.result()
                except Exception as exc:
                    track_rows = [
                        {
                            **{
                                key: row[key]
                                for key in (
                                    "track",
                                    "track_id",
                                    "ensemble",
                                    "instrument",
                                    "role",
                                )
                            },
                            "variant": variant["variant"],
                            "method": variant["method"],
                            "error": "".join(
                                traceback.format_exception(
                                    type(exc), exc, exc.__traceback__
                                )
                            ),
                        }
                        for variant in variants
                    ]
                _write_json(
                    _checkpoint_path(checkpoint_dir, row["track_id"]), track_rows
                )
                all_rows.extend(track_rows)
    rows = pd.DataFrame(all_rows)
    rows.to_csv(output_dir / f"{stage}_rows.csv", index=False)
    _write_json(
        output_dir / f"{stage}_metadata.json",
        {
            "signature": signature,
            "runner_version": RUNNER_VERSION,
            "tracks": len(sample),
            "variants": len(variants),
            "workers": workers,
            "options": asdict(options),
        },
    )
    return rows


def summarize(rows: pd.DataFrame, parameter_columns: Sequence[str]) -> pd.DataFrame:
    valid = rows.loc[rows["error"].isna()].copy()
    if valid.empty:
        return pd.DataFrame()
    keys = ["role", *parameter_columns]
    metric_columns = [
        "Precision",
        "Recall",
        "F-measure",
        "Average Overlap Ratio",
        "Estimated Notes",
        "Reference Notes",
        "note_compute_time",
        "realtime_factor",
    ]
    strata = (
        valid.groupby([*keys, "ensemble", "instrument"], dropna=False)[metric_columns]
        .mean()
        .reset_index()
    )
    summary = strata.groupby(keys, dropna=False)[metric_columns].mean().reset_index()
    track_counts = valid.groupby(keys, dropna=False).track_id.nunique().rename("Tracks")
    error_counts = (
        rows.groupby(keys, dropna=False)
        .error.apply(lambda col: col.notna().sum())
        .rename("Errors")
    )
    summary = summary.merge(track_counts.reset_index(), on=keys, how="left")
    summary = summary.merge(error_counts.reset_index(), on=keys, how="left")
    summary["note_count_ratio"] = summary["Estimated Notes"] / summary[
        "Reference Notes"
    ].replace(0, np.nan)
    return summary.sort_values(
        ["role", "F-measure", "Average Overlap Ratio", "realtime_factor"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)


def choose_method(summary: pd.DataFrame) -> str:
    tune = summary.loc[summary.role == "tune"].sort_values(
        ["F-measure", "Average Overlap Ratio", "realtime_factor"],
        ascending=[False, False, False],
    )
    if tune.empty:
        raise RuntimeError("method sweep produced no valid tuning rows")
    return str(tune.iloc[0].method)


def choose_parameters(summary: pd.DataFrame) -> dict[str, Any]:
    tune = summary.loc[summary.role == "tune"].copy()
    if tune.empty:
        raise RuntimeError("parameter sweep produced no valid tuning rows")
    tune["_note_count_error"] = (tune["note_count_ratio"] - 1.0).abs()
    tune["_production_distance"] = 0.0
    defaults = production_parameters()
    for name in SWEEP_PARAM_COLUMNS:
        span = float(tune[name].max() - tune[name].min())
        scale = span if span > 0 else 1.0
        tune["_production_distance"] += (
            tune[name].astype(float) - float(defaults[name])
        ).abs() / scale
    tune = tune.sort_values(
        [
            "F-measure",
            "Average Overlap Ratio",
            "_note_count_error",
            "_production_distance",
            *SWEEP_PARAM_COLUMNS,
        ],
        ascending=[False, False, True, True, True, True, True],
        kind="mergesort",
    )
    row = tune.iloc[0]
    recommendation = {
        name: row[name].item() if isinstance(row[name], np.generic) else row[name]
        for name in PARAM_COLUMNS
    }
    recommendation["method"] = str(recommendation["method"])
    return recommendation


def run_method_sweep(
    output_dir: Path | str | None = None,
    workers: int | None = None,
    options: SweepOptions | None = None,
    methods: Sequence[str] | None = None,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    options = options or SweepOptions()
    sample = select_sample(options)
    sample.to_csv(output_dir / "sample.csv", index=False)
    variants = method_variants(methods)
    rows = _parallel_stage(
        "method",
        sample,
        variants,
        output_dir,
        workers or max(1, (os.cpu_count() or 4) - 1),
        options,
        force,
    )
    summary = summarize(rows, ["method", "ruptures_algorithm", "model", "jump"])
    summary.to_csv(output_dir / "method_summary.csv", index=False)
    selected = choose_method(summary)
    _write_json(
        output_dir / "selected_method.json",
        {
            "method": selected,
            "selection_role": "tune",
            "selection_metric": "F-measure",
            "all_candidate_hops": 1,
        },
    )
    return (rows, summary, selected)


def load_selected_method(output_dir: Path | str | None = None) -> str:
    path = Path(output_dir or DEFAULT_OUTPUT_DIR) / "selected_method.json"
    if not path.exists():
        raise FileNotFoundError("run the method sweep first, or pass --fixed-method")
    return str(json.loads(path.read_text())["method"])


def run_parameter_sweep(
    output_dir: Path | str | None = None,
    workers: int | None = None,
    options: SweepOptions | None = None,
    axes: ParameterAxes | None = None,
    fixed_method: str | None = None,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    options = options or SweepOptions()
    sample = select_sample(options)
    sample.to_csv(output_dir / "sample.csv", index=False)
    if fixed_method not in (None, FIXED_PARAMETER_METHOD):
        raise ValueError(
            f"parameter sweeps are fixed to {FIXED_PARAMETER_METHOD!r}, not {fixed_method!r}"
        )
    grid = parameter_grid(axes=axes)
    variants = parameter_variants(grid)
    rows = _parallel_stage(
        "parameter",
        sample,
        variants,
        output_dir,
        workers or max(1, (os.cpu_count() or 4) - 1),
        options,
        force,
    )
    summary = summarize(rows, PARAM_COLUMNS)
    summary.to_csv(output_dir / "parameter_summary.csv", index=False)
    recommendation = choose_parameters(summary)
    _write_json(
        output_dir / "recommendation.json",
        {
            **recommendation,
            "selection_role": "tune",
            "selection_metric": "F-measure",
            "grid_mode": "full_cartesian",
            "transitions": False,
            "merge_adjacent": False,
            "tie_breaker": "note_count_bias_then_proximity_to_production_defaults",
        },
    )
    return (rows, summary, recommendation)


def load_results(output_dir: Path | str | None = None) -> dict[str, Any]:
    directory = Path(output_dir or DEFAULT_OUTPUT_DIR)
    result: dict[str, Any] = {}
    for name in (
        "sample",
        "method_rows",
        "method_summary",
        "parameter_rows",
        "parameter_summary",
    ):
        path = directory / f"{name}.csv"
        if path.exists():
            result[name] = pd.read_csv(path)
    for name in ("selected_method", "recommendation"):
        path = directory / f"{name}.json"
        if path.exists():
            result[name] = json.loads(path.read_text())
    return result
