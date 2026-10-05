"""Alignment-only continuation of a completed InjectedNoteSweep (note2).

Segmentation and score timing are computed once per case using the selected
saved variant. Baseline counts must reproduce before evaluating new costs.
"""

from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict, dataclass
from importlib.metadata import version
from itertools import product
import json
import multiprocessing as mp
import os
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from benchmarks.modules.note import InjectedNoteSweep as notes
from benchmarks.modules.mistake.MistakeCache import MistakeCache
from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker
from benchmarks.paths import REPO_ROOT

PARAMS = [
    "alignment_gamma_pitch",
    "alignment_gamma_time",
    "alignment_alpha_onset",
    "alignment_alpha_duration",
    "ins_cost",
    "del_cost",
]


@dataclass(frozen=True)
class Axes:
    alignment_gamma_pitch: tuple = (1.0, 2.0, 3.0, 4.0, 6.0)
    alignment_gamma_time: tuple = (0.5, 1.0, 2.0, 4.0)
    alignment_alpha_onset: tuple = (0.0, 0.25, 0.5, 1.0)
    ins_cost: tuple = (3.0, 5.0, 7.0)
    del_cost: tuple = (3.0, 5.0, 7.0)


def grid(baseline, axes=Axes()):
    choices = asdict(axes)
    for name, values in choices.items():
        if not values or any(
            (
                not np.isfinite(v)
                or v < 0
                or (name == "alignment_alpha_onset" and v > 1)
                for v in values
            )
        ):
            raise ValueError(f"Invalid axis: {name}")
    candidates = [baseline]
    candidates += [dict(baseline, alignment_gamma_pitch=p) for p in (2.0, 4.0)]
    for values in product(*choices.values()):
        p = dict(zip(choices, values))
        p["alignment_alpha_duration"] = 1.0 - p["alignment_alpha_onset"]
        candidates.append(p)
    unique = {}
    for p in candidates:
        key = notes.digest(p)[:16]
        unique.setdefault(key, dict(variant=key, baseline=p == baseline, **p))
    return list(unique.values())


def load_source(note_run, note_variant=None):
    """Select the saved clean-safe leader, or an explicit note2 variant ID."""
    note_run = Path(note_run).resolve()
    meta = json.loads((note_run / "run.json").read_text())
    if meta["status"] != "complete":
        raise ValueError("Complete note2 before running the alignment sweep")
    summary = pd.read_csv(note_run / "summary.csv")
    eligible = summary[summary.clean_safe].sort_values(
        ["injected_audio_pitch100_f1", "injected_note100_f1"],
        ascending=False,
        kind="stable",
    )
    selected = (
        eligible.iloc[0]
        if note_variant is None
        else summary.set_index("variant", drop=False).loc[note_variant]
    )
    fixed = {
        p: None if pd.isna(selected[p]) else float(selected[p]) for p in notes.PARAMS
    }
    saved = pd.read_csv(note_run / "rows.csv")
    saved = saved[saved.variant == selected.variant]
    cases, _ = notes.load_cases(meta["source_run"])
    if saved.case_id.duplicated().any() or set(saved.case_id) != {
        c["case_id"] for c in cases
    }:
        raise ValueError(
            "Selected note variant must cover the complete source corpus exactly once"
        )
    original = {c["case_id"]: c for c in meta["cases"]}
    baseline = {p: float(cases[0]["config"][p]) for p in PARAMS}
    for case in cases:
        old = original[case["case_id"]]
        if case["hashes"] != old["hashes"] or case["config"] != old["config"]:
            raise ValueError(f"Source changed since note2: {case['case_id']}")
        if any((float(case["config"][p]) != baseline[p] for p in PARAMS)):
            raise ValueError("Source cases have different alignment baselines")
        r = saved.set_index("case_id").loc[case["case_id"]]
        case["expected"] = [int(r[f"audio_pitch100_{k}"]) for k in ("tp", "fp", "fn")]
        case["expected_metrics"] = {
            k: int(r[k]) for k in saved.columns if k.endswith(("_tp", "_fp", "_fn"))
        }
    return (cases, fixed, baseline, str(selected.variant))


def _evaluate(task):
    case, variants = (task["case"], task["variants"])
    output = Path(task["output"])
    checkpoint = output / "checkpoints" / f"{case['case_id']}.json"
    if task["resume"] and checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        if saved["signature"] == task["signature"]:
            if len(saved["rows"]) != len(variants) or {
                r["variant"] for r in saved["rows"]
            } != {v["variant"] for v in variants}:
                raise ValueError(f"Incomplete checkpoint: {checkpoint}")
            return saved["rows"]
    log = output / "logs" / f"{case['case_id']}.log"
    log.parent.mkdir(exist_ok=True)
    with log.open("w") as stream, redirect_stdout(stream), redirect_stderr(stream):
        rows = _case_rows(task)
    MistakeCache.atomic_json(checkpoint, dict(signature=task["signature"], rows=rows))
    return rows


def _case_rows(task):
    from benchmarks.modules.mistake.MistakeDetectorBase import MistakeDetectorBase
    from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker

    case = task["case"]
    notes._prepare_case.cache_clear()
    fixed = dict(variant=task["note_variant"], baseline=True, **task["fixed_notes"])
    base = notes._case_rows(case, [fixed], task["signature"])[0]
    for name, expected in case["expected_metrics"].items():
        if base[name] != expected:
            raise ValueError(
                f"note2 baseline mismatch: {case['case_id']} {name}: {base[name]} != {expected}"
            )
    cfg, _, reference, _, events, rec, _, _ = notes._prepare_case(
        json.dumps(case, sort_keys=True), task["signature"]
    )
    rows = []
    for v in task["variants"]:
        for name in PARAMS:
            setattr(cfg, name, v[name])
        started = perf_counter()
        rec.detect_mistakes()
        predictions = MistakeDetectorBase.predicted_events(
            MistakeDetectorBase.with_reference_score_ids(
                MistakeBenchmarker.label_pairs(rec.alignment.pairs, cfg),
                reference,
                rec.score_data.note_data,
            ),
            reference,
        )
        row = dict(base, **v)
        for tolerance in notes.TOLERANCES:
            for metric, triple in MistakeDetectorBase.score_events(
                predictions, events, tolerance
            ).items():
                for k, value in zip(("tp", "fp", "fn"), triple):
                    row[f"{metric}{round(tolerance * 1000)}_{k}"] = value
        if v["baseline"] and any(
            (row[k] != value for k, value in case["expected_metrics"].items())
        ):
            raise ValueError(f"Alignment baseline mismatch: {case['case_id']}")
        row["alignment_seconds"] = perf_counter() - started
        rows.append(row)
    notes._prepare_case.cache_clear()
    return rows


def summarize(rows):
    summary = notes.summarize(rows)
    params = rows[["variant", *PARAMS]].drop_duplicates()
    return (
        summary.merge(params, on="variant", validate="one_to_one")
        .sort_values(
            ["clean_safe", "injected_audio_pitch100_f1", "variant"],
            ascending=[False, False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def run_sweep(
    note_run, output, *, note_variant=None, axes=Axes(), workers=None, resume=True
):
    cases, fixed, baseline, selected = load_source(note_run, note_variant)
    variants = grid(baseline, axes)
    workers = max(1, os.cpu_count() or 1) if workers is None else workers
    if workers < 1:
        raise ValueError("workers must be positive")
    output, note_run = (Path(output).resolve(), Path(note_run).resolve())
    if output == note_run or output in note_run.parents:
        raise ValueError("Output must be separate from the source note sweep")
    output.mkdir(parents=True, exist_ok=True)
    from benchmarks.modules.mistake.sweeps.RefinementComparison import fingerprint

    code = fingerprint()
    for name in [
        "benchmarks/modules/mistake/sweeps/AlignmentParamSweep.py",
        "benchmarks/modules/note/sweeps/InjectedNoteSweep.py",
        "benchmarks/modules/mistake/MistakeDetectorBase.py",
        "benchmarks/modules/mistake/MistakeBenchmarker.py",
        "benchmarks/modules/mistake/provenance/PipelineAudit.py",
        "benchmarks/modules/note/NoteBenchmarker.py",
        "benchmarks/modules/pitch/PitchCache.py",
        "app_logic/NoteData.py",
        "app_logic/midi/ScoreData.py",
        "app_logic/user/ds/PitchData.py",
    ]:
        code[name] = notes.file_hash(REPO_ROOT / name)
    packages = {
        p: version(p)
        for p in [
            "numpy",
            "scipy",
            "pandas",
            "ruptures",
            "music21",
            "pretty_midi",
            "mir_eval",
        ]
    }
    provenance = dict(
        code=code,
        packages=packages,
        fixed_notes=fixed,
        note_variant=selected,
        source_hashes={
            p: notes.file_hash(note_run / p)
            for p in ("run.json", "rows.csv", "summary.csv")
        },
    )
    metadata = dict(
        **provenance,
        note_run=str(note_run),
        baseline=baseline,
        axes=asdict(axes),
        cases=cases,
        variants=variants,
        workers=min(workers, len(cases)),
        status="running",
        objective="pooled missed/extra event F1 at 100ms/50c; clean false alarms <= baseline",
        scope="same-sample exploratory tuning; fixed segmentation, no repeat refinement",
    )
    MistakeCache.atomic_json(output / "run.json", metadata)
    tasks = [
        dict(
            case=c,
            variants=variants,
            fixed_notes=fixed,
            note_variant=selected,
            output=str(output),
            resume=resume,
            signature=notes.digest(dict(case=c, variants=variants, **provenance)),
        )
        for c in sorted(
            cases,
            key=lambda c: Path(c["paths"]["pitches"]).stat().st_size,
            reverse=True,
        )
    ]
    started, all_rows = (perf_counter(), [])
    print(
        f"{len(cases)} cases × {len(variants)} alignment settings; {metadata['workers']} workers",
        flush=True,
    )
    try:
        with MistakeBenchmarker.single_thread_environment(), ProcessPoolExecutor(
            max_workers=metadata["workers"],
            mp_context=mp.get_context("spawn"),
            initializer=notes._init_worker,
        ) as pool:
            futures = [pool.submit(_evaluate, task) for task in tasks]
            try:
                for completed, future in enumerate(as_completed(futures), 1):
                    all_rows.extend(future.result())
                    print(
                        f"\r{completed}/{len(tasks)} cases complete", end="", flush=True
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
            raise ValueError("Incomplete or duplicate sweep results")
        summary = summarize(rows)
        rows.to_csv(output / "rows.csv", index=False)
        summary.to_csv(output / "summary.csv", index=False)
        metadata.update(
            status="complete",
            wall_seconds=perf_counter() - started,
            evaluations=len(rows),
        )
        MistakeCache.atomic_json(output / "run.json", metadata)
        return (rows, summary)
    except BaseException:
        metadata.update(status="failed", wall_seconds=perf_counter() - started)
        MistakeCache.atomic_json(output / "run.json", metadata)
        raise
