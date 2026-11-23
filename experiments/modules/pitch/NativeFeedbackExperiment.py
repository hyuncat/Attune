"""Selective, resumable 449-track experiment; historical suites are read-only."""
import json
from zipfile import BadZipFile
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.modules.pitch import FeedbackLatency
from experiments.modules.pitch.PitchBenchmarker import PitchBenchmarker
from experiments.modules.pitch.PitchCache import PitchCache
from experiments.modules.pitch.datasets.Bach10 import Bach10
from experiments.modules.pitch.datasets.URMP import URMP
from experiments.modules.pitch.datasets.CocoChorales import CocoChorales

METHODS = (*PitchBenchmarker.PitchStreaming.METHODS, "crepe_tiny")
CHANGED = {"pyin_framewise", "praat", "crepe_tiny"}
EXPERIMENT = "native_feedback_2026_11_v1"


def cached_row(runner, directory, method, example):
    path = runner._checkpoint_path(directory, method, example)
    reasons = []
    evidence = dict(reusable_predictions=False, reusable_timing_sidecars=False, scoring_valid=False)
    for candidate in PitchCache._checkpoint_candidates(path):
        if not candidate.exists():
            continue
        try:
            rows = json.loads(candidate.read_text())
            if len(rows) != 1:
                raise ValueError("expected one checkpoint row")
            row = rows[0]
            if (row.get("model"), row.get("dataset"), row.get("track_id")) != (method, example.dataset, example.track_id):
                raise ValueError("checkpoint identity mismatch")
            if method == "attune" and not (
                row.get("volume_reference") == "running_p95"
                and row.get("pitch_cache_version") == PitchCache.VERSION
                and np.isclose(row.get("dynamic_volume_floor_ratio", np.nan), 0.04)
                and np.isclose(row.get("unvoiced_probability_threshold", np.nan), 0.985)
            ):
                raise ValueError("Attune is not current running-p95 live configuration")
            evidence["reusable_predictions"] = True
            evidence["scoring_valid"] = PitchBenchmarker.current_scores(pd.DataFrame([row]))
            FeedbackLatency.read(row)
            evidence["reusable_timing_sidecars"] = True
            if not PitchBenchmarker.current_scores(pd.DataFrame([row])):
                with np.load(row["latency_samples_path"], allow_pickle=False) as data:
                    mask = example.ref_times <= float(row["audio_seconds"]) + 1e-9
                    row.update(PitchBenchmarker.score_frames(
                        example.ref_times[mask], example.ref_freqs[mask],
                        data["estimate_times_seconds"], data["estimate_frequencies_hz"]))
            evidence["scoring_valid"] = True
            return row, "compatible checkpoint and timing sidecar", evidence
        except (ValueError, KeyError, OSError, TypeError, EOFError, BadZipFile) as exc:
            reasons.append(f"{candidate}: {exc}")
    return None, "; ".join(reasons) or "no checkpoint for this adapter identity", evidence


def prepare(notebook):
    suites, plan = [], []
    for dataset, per_instrument, expected in ((Bach10.name, None, 40), (URMP.name, None, 149), (CocoChorales.name, 20, 260)):
        options = notebook.options(datasets=(dataset,), materialize=dataset == CocoChorales.name,
                                   instruments=(), per_instrument=per_instrument, max_tracks=None)
        benchmarker = PitchBenchmarker(options)
        examples = [benchmarker.dataset(t.dataset).example(t) for t in benchmarker.tracks()]
        if len(examples) != expected:
            raise ValueError(f"{dataset}: expected {expected} frozen-selection tracks, got {len(examples)}")
        directory = notebook.runs_root / f"streaming_{dataset}" / EXPERIMENT
        runner = PitchBenchmarker.PitchStreaming(options, PitchBenchmarker.StreamingConfig(
            workers=notebook.config.streaming_workers, native_feedback_policy=True))
        cached = {}
        for method in METHODS:
            for example in examples:
                row, reason, evidence = cached_row(runner, directory / "checkpoints", method, example)
                key = (method, example.track_id)
                if row is not None:
                    cached[key] = row
                plan.append(dict(dataset=dataset, model=method, track_id=example.track_id,
                    **evidence,
                    reporting_only=row is not None, inference_miss=row is None,
                    blocked=row is None and method not in CHANGED, reason=reason))
        suites.append((directory, runner, examples, cached))
    return suites, pd.DataFrame(plan)


def run(notebook, *, dry_run=True):
    from IPython.display import display
    suites, plan = prepare(notebook)
    root = notebook.runs_root / EXPERIMENT
    root.mkdir(parents=True, exist_ok=True)
    plan.to_csv(root / "dry_run_plan.csv", index=False)
    display(plan.groupby("model", sort=False)[[
        "reusable_predictions", "reusable_timing_sidecars", "reporting_only", "inference_miss", "blocked"]].sum())
    misses = plan[plan.inference_miss]
    print(f"Exact method/track plan: {root / 'dry_run_plan.csv'} ({len(misses)} inference misses)")
    if not misses.empty:
        # Full manifest preserves every reason without truncation by pandas display.
        for (method, reason), group in misses.groupby(["model", "reason"], sort=False):
            print(f"  {method}: {len(group)} track(s): {reason}")
    if dry_run:
        return plan
    if plan.blocked.any():
        raise RuntimeError("Unchanged-method evidence is missing/incompatible. See dry_run_plan.csv; no inference launched.")
    if notebook.config.force_rerun:
        raise ValueError("Selective experiment requires force_rerun=False")
    results = []
    for directory, runner, examples, cached in suites:
        rows = list(cached.values())
        for method in METHODS:
            missing = [e for e in examples if (method, e.track_id) not in cached]
            if not missing:
                continue
            # Only changed methods reach inference, one checkpoint per method/track.
            fresh = runner.run(missing, methods=[method], checkpoint_dir=directory / "checkpoints", force=True)
            if len(fresh) != len(missing):
                raise RuntimeError(f"{method}: incomplete replay; rerun to resume completed checkpoints")
            rows.extend(fresh.to_dict("records"))
        frame = pd.DataFrame(rows).sort_values(["dataset", "track_id", "model"])
        directory.mkdir(parents=True, exist_ok=True)
        frame.to_csv(directory / "rows.csv", index=False)
        results.append(notebook.SuiteResult(directory.name, directory / "rows.csv", frame,
                                             PitchBenchmarker.pool_scores(frame)))
    table = report(results, root)
    display(table)
    print(FeedbackLatency.CAPTION)
    return tuple(results), table


def report(results, root):
    root = Path(root)
    rows = pd.concat([r.rows for r in results], ignore_index=True)
    if rows.duplicated(["dataset", "track_id", "model"]).any():
        raise ValueError("Duplicate method/track rows")
    timing, diagnostics = [], []
    for method, group in rows.groupby("model", sort=False):
        ages, lengths = [], []
        failures = 0
        for row in group.to_dict("records"):
            a, w, startup = FeedbackLatency.read(row, root / "derived_feedback")
            ages.append(a)
            lengths.append(w)
            failures += not len(w)
            diagnostics.append(dict(model=method, dataset=row["dataset"], track_id=row["track_id"],
                startup_seconds=startup, evaluated_seconds=float(w.sum()),
                audio_seconds=row["audio_seconds"], coverage_failure=not len(w)))
        ages, lengths = np.concatenate(ages), np.concatenate(lengths)
        timing.append(dict(model=method, recordings=len(group), timing_coverage_failures=failures,
            evaluated_seconds=float(lengths.sum()), evaluated_duration_coverage=float(lengths.sum()) / group.audio_seconds.sum(),
            **{"P95 feedback latency (ms)": FeedbackLatency.percentile(ages, lengths)}))
    table = PitchBenchmarker.pool_scores(rows).join(pd.DataFrame(timing).set_index("model"))
    table.to_csv(root / "comparison.csv")
    pd.DataFrame(diagnostics).to_csv(root / "timing_coverage.csv", index=False)
    geometry = ["model", "dataset", "track_id", "stream_adapter", "frame_size_samples", "hop_size_samples",
                "sample_rate_hz", "update_interval_ms", "algorithmic_lookahead_ms"]
    rows[geometry].to_csv(root / "geometry.csv", index=False)
    (root / "manifest.json").write_text(json.dumps(dict(experiment=EXPERIMENT, metric=FeedbackLatency.VERSION,
        clock=FeedbackLatency.CLOCK, caption=FeedbackLatency.CAPTION, changed_methods=sorted(CHANGED),
        selection=rows[["dataset", "track_id"]].drop_duplicates().to_dict("records")), indent=2))
    return table
