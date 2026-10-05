"""Cached-pitch note-parameter sweep on one completed injected mistake corpus.

No pitch inference, audio rendering, model loading or production edits. Candidates
are exploratory on the same sample. The saved unrefined baseline must reproduce.
"""

from __future__ import annotations
import hashlib
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from contextlib import redirect_stdout
from contextlib import redirect_stderr
from dataclasses import dataclass
from dataclasses import asdict
from importlib.metadata import version
from functools import lru_cache
from itertools import product
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from benchmarks.paths import REPO_ROOT
from benchmarks.modules.mistake.MistakeCache import MistakeCache
from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker

PARAMS = ["cap_ms", "score_factor", "pitch_step", "silence_ms"]
TOLERANCES = (0.05, 0.1, 0.2)


@dataclass(frozen=True)
class Axes:
    cap_ms: tuple = (30.0, 50.0, 60.0, 80.0, 100.0, 150.0, None)
    score_factor: tuple = (0.25, 0.5, 0.6, 0.75, 1.0)
    pitch_step: tuple = (0.25, 0.5, 0.75, 1.0, 1.5)
    silence_ms: tuple = (3.0, 10.0, 20.0, 40.0)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def grid(axes=Axes(), baseline=None):
    baseline = baseline or dict(
        cap_ms=None, score_factor=0.6, pitch_step=0.75, silence_ms=10.0
    )
    choices = asdict(axes)
    for name, values in choices.items():
        if not values or any(
            (
                v is None
                and name != "cap_ms"
                or (v is not None and (not np.isfinite(v) or v <= 0))
                for v in values
            )
        ):
            raise ValueError(f"Invalid axis: {name}")
    candidates = [
        baseline,
        *[dict(zip(PARAMS, values)) for values in product(*choices.values())],
    ]
    unique = {}
    for p in candidates:
        key = digest(p)[:16]
        unique.setdefault(key, dict(variant=key, baseline=p == baseline, **p))
    return list(unique.values())


def effective_seconds(shortest, factor, cap_ms):
    relative = shortest * factor
    return relative if cap_ms is None else min(relative, cap_ms / 1000.0)


def load_cases(run_dir):
    run_dir = Path(run_dir).resolve()
    meta = json.loads((run_dir / "run.json").read_text())
    if meta["status"] != "complete":
        raise ValueError("The source comparison must be complete")
    rows = pd.read_csv(run_dir / "rows.csv")
    rows = rows[
        (rows.method == "Attune (no refinement)")
        & (rows.input == "detected")
        & (rows.metric == "audio_pitch")
        & np.isclose(rows.tolerance, 0.1)
    ]
    if rows.empty or rows.duplicated("case_id").any():
        raise ValueError("Missing or duplicate saved baseline cases")
    clean = rows[rows.rate == 0]
    if (clean.groupby("source")[["tp", "fp", "fn"]].nunique() > 1).any().any():
        raise ValueError("Clean-seed controls differ")
    rows = pd.concat([rows[rows.rate > 0], clean.drop_duplicates("source")])
    configs = [json.loads(c) for c in rows.config]
    baselines = [
        dict(
            cap_ms=c.get("min_note_length_cap", 0.0) * 1000 or None,
            score_factor=c["min_note_length_factor"],
            pitch_step=c["pitch_thresh"],
            silence_ms=c["min_silence_duration_ms"],
        )
        for c in configs
    ]
    if any((b != baselines[0] for b in baselines)):
        raise ValueError("Source cases use different note-detection baselines")
    cases = []
    for r in rows.to_dict("records"):
        folder = run_dir / "cases" / r["case_id"]
        manifest = json.loads((folder / "manifest.json").read_text())
        paths = dict(
            source=manifest["source_midi"],
            performed=manifest["midi"],
            pitches=manifest["pitch_data"],
            truth=str(folder / "net_truth.json"),
        )
        cases.append(
            dict(
                case_id=r["case_id"],
                instrument=r["instrument"],
                seed=int(r["seed"]),
                rate=float(r["rate"]),
                config=json.loads(r["config"]),
                paths=paths,
                hashes={k: file_hash(v) for k, v in paths.items()},
                expected=[int(r[k]) for k in ("tp", "fp", "fn")],
            )
        )
    return (cases, baselines[0])


def _init_worker():
    global _LIMITER
    from threadpoolctl import threadpool_limits

    _LIMITER = threadpool_limits(limits=1)


def _evaluate_case(task):
    output = Path(task["output"])
    case, variants = (task["case"], task["variants"])
    checkpoint = output / "checkpoints" / f"{case['case_id']}--{task['group']}.json"
    if checkpoint.exists() and task["resume"]:
        saved = json.loads(checkpoint.read_text())
        if saved["signature"] == task["signature"]:
            expected = {v["variant"] for v in variants}
            if (
                len(saved["rows"]) != len(expected)
                or {r["variant"] for r in saved["rows"]} != expected
            ):
                raise ValueError(f"Incomplete checkpoint: {checkpoint}")
            return saved["rows"]
    log = output / "logs" / f"{case['case_id']}--{task['group']}.log"
    log.parent.mkdir(exist_ok=True)
    with log.open("w") as stream, redirect_stdout(stream), redirect_stderr(stream):
        rows = _case_rows(case, variants, task["case_signature"])
    MistakeCache.atomic_json(checkpoint, dict(signature=task["signature"], rows=rows))
    return rows


@lru_cache(maxsize=1)
def _prepare_case(case_json, case_signature):
    """One loaded case per process, reused across adjacent parameter jobs.

    The signature includes input/code hashes; workers are scoped to one sweep.
    Keeping a single case bounds memory independently of the corpus size.
    """
    from algorithms.Config import Config
    from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker
    from benchmarks.modules.mistake.provenance.PipelineAudit import sequence
    from benchmarks.modules.mistake.MistakeDetectorBase import MistakeDetectorBase
    from benchmarks.modules.note.NoteBenchmarker import OneInstrumentScoreData
    from benchmarks.modules.pitch.PitchCache import PitchCache

    case = json.loads(case_json)
    cfg = Config(**case["config"])
    cfg.min_note_length_cap = 0.0
    bench = MistakeBenchmarker()
    reference = OneInstrumentScoreData(case["paths"]["source"]).note_data
    performed = OneInstrumentScoreData(case["paths"]["performed"]).note_data
    truth = json.loads(Path(case["paths"]["truth"]).read_text())
    events = MistakeDetectorBase.truth_events(truth["net_truth"], reference, performed)
    rec = bench.attune.recording_for(
        cfg, score_data=OneInstrumentScoreData(case["paths"]["source"])
    )
    cached = PitchCache(case["paths"]["pitches"]).read(PitchCache.SMOOTHED, cfg)
    if cached is None:
        raise ValueError(f"Missing smoothed pitches: {case['paths']['pitches']}")
    rec.pitch_data = cached[0]
    rec.pitch_data.end_index = len(rec.pitch_data.data)
    rec.transition_detector.clear_transitions(rec.pitch_data.data)
    rec.resize_score(to_span="pitch", include_transitions=False)
    rec.update_min_note_length()
    shortest = cfg.min_note_length
    perf = sequence(performed)
    return (cfg, bench, reference, performed, events, rec, shortest, perf)


def _case_rows(case, variants, case_signature=""):
    from benchmarks.modules.mistake.provenance.PipelineAudit import sequence
    from benchmarks.modules.mistake.provenance.PipelineAudit import note_matches
    from benchmarks.modules.mistake.MistakeDetectorBase import MistakeDetectorBase
    from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker

    cfg, bench, reference, performed, events, rec, shortest, perf = _prepare_case(
        json.dumps(case, sort_keys=True), case_signature
    )
    cache, rows = ({}, [])
    for v in variants:
        seconds = effective_seconds(shortest, v["score_factor"], v["cap_ms"])
        cfg.min_note_length = shortest
        cfg.min_note_length_factor = (
            v["score_factor"]
            if v["cap_ms"] is None
            else min(v["score_factor"], v["cap_ms"] / 1000.0 / shortest)
        )
        cfg.pitch_thresh, cfg.min_silence_duration_ms = (
            v["pitch_step"],
            v["silence_ms"],
        )
        frames = cfg.min_note_pitch_frames(cfg.min_note_length_factor)
        silence_frames = rec.note_detector.silence_window_frames()
        key = (frames, v["pitch_step"], silence_frames)
        reused = key in cache
        if not reused:
            started = perf_counter()
            rec.reset_analysis()
            rec.resize_score(to_span="pitch", include_transitions=False)
            rec.update_min_note_length()
            rec.note_data = rec.note_detector.detect_notes(rec.pitch_data.data)
            bench._trim_boundary_notes(rec.note_data, performed)
            estimates = sequence(rec.note_data)
            result = dict(
                detected_notes=len(estimates),
                performed_notes=len(perf),
                score_notes=len(reference.times),
            )
            for tolerance in TOLERANCES:
                tag = int(round(tolerance * 1000))
                matches = note_matches(perf, estimates, tolerance)
                for k, value in zip(
                    ("tp", "fp", "fn"),
                    (
                        len(matches),
                        len(estimates) - len(matches),
                        len(perf) - len(matches),
                    ),
                ):
                    result[f"note{tag}_{k}"] = value
            matches = note_matches(perf, estimates, 0.1, offsets=True)
            for k, value in zip(
                ("tp", "fp", "fn"),
                (len(matches), len(estimates) - len(matches), len(perf) - len(matches)),
            ):
                result[f"note_offsets_{k}"] = value
            if rec.note_data.times:
                rec.resize_score(to_span="onset")
            rec.detect_mistakes()
            predictions = MistakeDetectorBase.predicted_events(
                MistakeDetectorBase.with_reference_score_ids(
                    MistakeBenchmarker.label_pairs(rec.alignment.pairs, cfg),
                    reference,
                    rec.score_data.note_data,
                ),
                reference,
            )
            for tolerance in TOLERANCES:
                tag = int(round(tolerance * 1000))
                counts = MistakeDetectorBase.score_events(
                    predictions, events, tolerance
                )
                for metric, triple in counts.items():
                    for k, value in zip(("tp", "fp", "fn"), triple):
                        result[f"{metric}{tag}_{k}"] = value
            result["evaluation_seconds"] = perf_counter() - started
            cache[key] = result
        result = cache[key]
        if (
            v["baseline"]
            and [result[f"audio_pitch100_{k}"] for k in ("tp", "fp", "fn")]
            != case["expected"]
        ):
            raise ValueError(
                f"Baseline does not reproduce saved counts: {case['case_id']}"
            )
        rows.append(
            dict(
                case_id=case["case_id"],
                instrument=case["instrument"],
                seed=case["seed"],
                rate=case["rate"],
                **v,
                shortest_score_seconds=shortest,
                effective_min_ms=1000 * seconds,
                min_frames=frames,
                silence_frames=silence_frames,
                reused_effective_variant=reused,
                **result,
            )
        )
    return rows


def summarize(rows):
    result = []
    for variant, group in rows.groupby("variant", sort=False):
        first = group.iloc[0]
        entry = {k: first[k] for k in ["variant", "baseline", *PARAMS]}
        for condition, selected in [
            ("injected", group[group.rate > 0]),
            ("clean", group[group.rate == 0]),
        ]:
            entry[f"{condition}_cases"] = len(selected)
            for prefix in (
                "note50",
                "note100",
                "note200",
                "note_offsets",
                "audio_pitch50",
                "audio_pitch100",
                "audio_pitch200",
                "audio_extra100",
                "audio_missed100",
            ):
                tp, fp, fn = (
                    int(selected[f"{prefix}_{k}"].sum()) for k in ("tp", "fp", "fn")
                )
                entry.update(
                    {
                        f"{condition}_{prefix}_{k}": value
                        for k, value in zip(("tp", "fp", "fn"), (tp, fp, fn))
                    }
                )
                entry[f"{condition}_{prefix}_f1"] = (
                    200 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan
                )
                if prefix == "audio_pitch100":
                    entry[f"{condition}_precision"] = (
                        100 * tp / (tp + fp) if tp + fp else np.nan
                    )
                    entry[f"{condition}_recall"] = (
                        100 * tp / (tp + fn) if tp + fn else np.nan
                    )
            score_notes = selected.score_notes.sum()
            entry[f"{condition}_false_alarms_per_100_notes"] = (
                100 * entry[f"{condition}_audio_pitch100_fp"] / score_notes
                if score_notes
                else np.nan
            )
        result.append(entry)
    summary = pd.DataFrame(result)
    base = summary[summary.baseline].iloc[0]
    summary["clean_safe"] = (
        summary.clean_audio_pitch100_fp <= base.clean_audio_pitch100_fp
    )
    summary["mistake_f1_gain_pp"] = (
        summary.injected_audio_pitch100_f1 - base.injected_audio_pitch100_f1
    )
    return summary.sort_values(
        ["clean_safe", "injected_audio_pitch100_f1", "injected_note100_f1"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def parameter_jobs(variants):
    """Keep equivalent duration settings together while distributing costly cases.

    Each pitch-step/silence group contains every cap/factor combination, so the
    frame-equivalence cache still removes duplicate segmentation work. Default:
    20 independent groups per case rather than a single long case task.
    """
    groups = {}
    for v in variants:
        key = digest(dict(pitch_step=v["pitch_step"], silence_ms=v["silence_ms"]))[:12]
        groups.setdefault(key, []).append(v)
    return groups


def run_sweep(
    run_dir, output, *, axes=Axes(), workers=None, resume=True, max_cases=None
):
    workers = max(1, os.cpu_count() or 8) if workers is None else workers
    if workers < 1:
        raise ValueError("workers must be positive")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases, baseline = load_cases(run_dir)
    if max_cases is not None:
        cases = cases[:max_cases]
    if not cases:
        raise ValueError("No cases selected")
    variants = grid(axes, baseline)
    from benchmarks.modules.mistake.sweeps.RefinementComparison import fingerprint

    code = fingerprint()
    for name in [
        "benchmarks/modules/note/sweeps/InjectedNoteSweep.py",
        "benchmarks/modules/mistake/provenance/PipelineAudit.py",
        "benchmarks/modules/mistake/MistakeBenchmarker.py",
        "benchmarks/modules/mistake/MistakeDetectorBase.py",
        "benchmarks/modules/note/NoteBenchmarker.py",
        "benchmarks/modules/pitch/PitchCache.py",
        "app_logic/midi/ScoreData.py",
        "app_logic/user/ds/PitchData.py",
        "app_logic/NoteData.py",
    ]:
        code[name] = file_hash(REPO_ROOT / name)
    packages = {
        name: version(name)
        for name in ["numpy", "scipy", "ruptures", "music21", "pretty_midi"]
    }
    groups = parameter_jobs(variants)
    job_count = len(cases) * len(groups)
    metadata = dict(
        source_run=str(Path(run_dir).resolve()),
        source_run_sha256=file_hash(Path(run_dir) / "run.json"),
        source_rows_sha256=file_hash(Path(run_dir) / "rows.csv"),
        axes=asdict(axes),
        baseline=baseline,
        cases=cases,
        variants=variants,
        code=code,
        packages=packages,
        workers=min(workers, job_count),
        jobs=job_count,
        scheduling="case × pitch-step × silence; largest pitch caches first",
        status="running",
        objective="pooled missed/extra F1 at 100ms/50c; fixed unrefined alignment",
        scope="same-sample exploratory tuning; no held-out generalization claim",
        omitted="vibrato annotation, repeat correction, pitch/model inference",
    )
    MistakeCache.atomic_json(output / "run.json", metadata)
    tasks = []
    for c in sorted(
        cases, key=lambda c: Path(c["paths"]["pitches"]).stat().st_size, reverse=True
    ):
        case_signature = digest(dict(case=c, code=code, packages=packages))
        for group, chunk in groups.items():
            tasks.append(
                dict(
                    case=c,
                    variants=chunk,
                    group=group,
                    output=str(output),
                    resume=resume,
                    case_signature=case_signature,
                    signature=digest(
                        dict(case_signature=case_signature, variants=chunk)
                    ),
                )
            )
    print(
        f"{len(cases)} cases × {len(variants)} variants = {len(cases) * len(variants):,} evaluations; {job_count} jobs across {metadata['workers']} workers",
        flush=True,
    )
    started, all_rows = (perf_counter(), [])
    try:
        with MistakeBenchmarker.single_thread_environment(), ProcessPoolExecutor(
            max_workers=metadata["workers"],
            mp_context=mp.get_context("spawn"),
            initializer=_init_worker,
        ) as pool:
            futures = [pool.submit(_evaluate_case, task) for task in tasks]
            print(f"0/{job_count} jobs complete", end="", flush=True)
            try:
                for completed, future in enumerate(as_completed(futures), 1):
                    batch = future.result()
                    all_rows.extend(batch)
                    print(
                        f"\r{completed}/{job_count} jobs complete", end="", flush=True
                    )
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
            finally:
                print()
        rows = (
            pd.DataFrame(all_rows)
            .sort_values(["case_id", "variant"])
            .reset_index(drop=True)
        )
        if (
            len(rows) != len(cases) * len(variants)
            or rows.duplicated(["case_id", "variant"]).any()
        ):
            raise ValueError("Sweep results are incomplete or duplicated")
        summary = summarize(rows)
        rows.to_csv(output / "rows.csv", index=False)
        summary.to_csv(output / "summary.csv", index=False)
        metadata.update(
            status="complete",
            wall_seconds=perf_counter() - started,
            evaluations=len(rows),
            unique_effective_evaluations=int((~rows.reused_effective_variant).sum()),
        )
        MistakeCache.atomic_json(output / "run.json", metadata)
        return (rows, summary)
    except BaseException:
        metadata["status"] = "failed"
        MistakeCache.atomic_json(output / "run.json", metadata)
        raise
