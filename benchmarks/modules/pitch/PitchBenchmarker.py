"""PitchBenchmarker implementation and owned benchmark helpers."""

from __future__ import annotations
from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase
import argparse
import contextlib
import math
import multiprocessing
import os
import queue as queue_module
import shutil
import sys
import time
import traceback
import warnings
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import ClassVar

PITCHBENCHMARKER_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(PITCHBENCHMARKER_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(PITCHBENCHMARKER_REPO_ROOT))
for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
):
    os.environ.setdefault(_variable, "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import pandas as pd
from benchmarks.modules.pitch.datasets.Bach10 import Bach10
from benchmarks.modules.pitch.datasets.AudioAnnot import AudioAnnot
from benchmarks.modules.pitch.datasets.CocoChorales import CocoChorales
from benchmarks.modules.pitch.datasets.PitchDataset import PitchDataset
from benchmarks.modules.pitch.datasets.PitchDataset import PitchTrack
from benchmarks.modules.pitch.datasets.URMP import URMP
from benchmarks.paths import RESULTS_ROOT
import hashlib
import itertools
import mir_eval
import numpy as np

SCORING_VERSION = "frame_counts_10ms_v1"
HOP_SECONDS = 0.01
METRIC_COUNTS = {
    "Raw Pitch Accuracy": ("pitch_correct_frames", "voiced_frame_count"),
    "Raw Chroma Accuracy": ("chroma_correct_frames", "voiced_frame_count"),
    "Overall Accuracy": ("overall_correct_frames", "frame_count"),
    "Voicing Recall": ("voiced_detected_frames", "voiced_frame_count"),
    "Voicing False Alarm": ("false_alarm_frames", "unvoiced_frame_count"),
}
COUNT_COLUMNS = tuple(
    dict.fromkeys((c for pair in METRIC_COUNTS.values() for c in pair))
)
import json
from concurrent.futures import as_completed
from dataclasses import asdict
from algorithms.PitchSmoother import VoicingSmoother
from algorithms.PitchSmoother import PitchSmoother
from benchmarks.modules.pitch.competitors.PYIN import PYINPitchSmoother
from benchmarks.modules.pitch.competitors.Attune import Attune
from benchmarks.paths import REPO_ROOT

SELECTION = (
    REPO_ROOT
    / "benchmarks/results/pitch/pyin_ablation/pyin_voicing_controllers_v2_128hop/balanced/floor_0p02__ceiling_p95__unvoiced_0p85__switch_0p02/pitch_only_confidence_v1_gap_0.05s/runner_v2_tracks_18_seed_0_repeats_1_workers_2/selection.csv"
)
OUTPUT = REPO_ROOT / "benchmarks/results/pitch/joint_vs_current_support_timing"
REPEATS = 3
from tqdm.auto import tqdm
import lzma
import soundfile as sf
from algorithms.Config import Config
from algorithms.PitchDetector import PitchDetector
from zipfile import BadZipFile
from collections.abc import Collection
from dataclasses import replace
from scipy import ndimage
from benchmarks.modules.pitch.competitors.Praat import Praat
from benchmarks.modules.pitch.competitors.PYIN import PYIN

_pitchstreaming_streaming_worker_state = None


class PitchBenchmarker:
    """Owns benchmark execution, parallel workers, scoring, and timing."""

    @staticmethod
    def score_frames(ref_times, ref_freqs, est_times, est_freqs):
        """Return mir_eval scores and sufficient counts on its uniform 10 ms grid.

        Use mir_eval's negative-frequency convention too: raw pitch/chroma can
        credit a latent frequency even when its estimated voicing is negative.
        """
        if not len(ref_times):
            raise ValueError("Cannot score an empty reference")
        if not len(est_times):
            est_times, est_freqs = (np.array([0.0]), np.array([0.0]))
        arrays = mir_eval.melody.to_cent_voicing(
            np.asarray(ref_times),
            np.asarray(ref_freqs),
            np.asarray(est_times),
            np.asarray(est_freqs),
            hop=HOP_SECONDS,
        )
        ref_v, ref_c, est_v, est_c = arrays
        voiced = int(np.count_nonzero(ref_v))
        total = len(ref_v)
        counts = dict(
            frame_count=total,
            voiced_frame_count=voiced,
            unvoiced_frame_count=total - voiced,
        )
        functions = {
            "Raw Pitch Accuracy": lambda: mir_eval.melody.raw_pitch_accuracy(*arrays),
            "Raw Chroma Accuracy": lambda: mir_eval.melody.raw_chroma_accuracy(*arrays),
            "Overall Accuracy": lambda: mir_eval.melody.overall_accuracy(*arrays),
            "Voicing Recall": lambda: mir_eval.melody.voicing_recall(ref_v, est_v),
            "Voicing False Alarm": lambda: mir_eval.melody.voicing_false_alarm(
                ref_v, est_v
            ),
        }
        scores = {}
        for metric, (numerator, denominator) in METRIC_COUNTS.items():
            value = float(functions[metric]())
            counts[numerator] = int(round(value * counts[denominator]))
            scores[metric] = value if counts[denominator] else float("nan")
        fingerprint = hashlib.sha256(
            np.asarray(ref_v, dtype="<f8").tobytes()
            + np.asarray(ref_c, dtype="<f8").tobytes()
        ).hexdigest()
        return {
            **scores,
            **counts,
            "scoring_version": SCORING_VERSION,
            "evaluation_hop_seconds": HOP_SECONDS,
            "reference_hash": fingerprint,
        }

    @staticmethod
    def current_scores(rows):
        """Whether every row has the current, complete sufficient statistics."""
        return rows.empty or (
            set(COUNT_COLUMNS).issubset(rows.columns)
            and "scoring_version" in rows
            and rows["scoring_version"].eq(SCORING_VERSION).all()
            and rows[list(COUNT_COLUMNS)].notna().all().all()
        )

    @staticmethod
    def pool_scores(rows, by="model"):
        """Sum each metric's own numerator and denominator before dividing."""
        if rows.empty:
            return pd.DataFrame(columns=list(METRIC_COUNTS))
        if not PitchBenchmarker.current_scores(rows):
            raise ValueError(
                "Pitch rows lack current frame counts; rerun the suite to rescore cached estimates."
            )
        totals = rows.groupby(by, sort=False, dropna=False)[list(COUNT_COLUMNS)].sum()
        table = pd.DataFrame(index=totals.index)
        for metric, (numerator, denominator) in METRIC_COUNTS.items():
            table[metric] = totals[numerator] / totals[denominator].where(
                totals[denominator] > 0
            )
        return table

    @staticmethod
    def paired_comparisons(
        rows,
        *,
        methods,
        baseline="attune",
        metrics=("Overall Accuracy",),
        n_resamples=9999,
        seed=0,
        alpha=0.05,
    ):
        """Paired label swaps and stratified cluster bootstrap of pooled differences.

        All methods use the common completed track intersection. Source-piece IDs
        identify clusters; callers may override them to link known shared pieces.
        Offline and online must be invoked separately (separate Holm families).
        Differences and confidence limits are percentage points, Attune minus peer.
        """
        if n_resamples < 99:
            raise ValueError("Use at least 99 resamples")
        methods = tuple(dict.fromkeys(methods))
        if baseline not in methods or len(methods) < 2:
            raise ValueError("Select the baseline and at least one competitor")
        if any((metric not in METRIC_COUNTS for metric in metrics)):
            raise ValueError("Unknown pitch metric")
        selected = rows.loc[rows["model"].isin(methods)].copy()
        required = {
            "dataset",
            "track_id",
            "source_piece",
            "reference_hash",
            "execution_mode",
        }
        if not required.issubset(
            selected.columns
        ) or not PitchBenchmarker.current_scores(selected):
            raise ValueError(
                "Paired analysis needs current frame counts, reference hashes and source-piece IDs"
            )
        if selected["execution_mode"].nunique() != 1:
            raise ValueError("Analyze offline and online results separately")
        if selected[list(required)].isna().any().any():
            raise ValueError("Paired analysis metadata contains missing values")
        keys = ["dataset", "track_id"]
        if selected.duplicated(["model", *keys]).any():
            raise ValueError(
                "Duplicate method/track rows: choose one suite/condition per recording"
            )
        missing = set(methods) - set(selected["model"])
        if missing:
            raise ValueError(
                f"No completed rows for requested methods: {sorted(missing)}"
            )
        coverage = selected.groupby(keys)["model"].nunique()
        common = coverage.index[coverage.eq(len(methods))]
        if not len(common):
            raise ValueError("No recordings completed by every selected method")
        selected = selected.set_index(keys).loc[common].reset_index()
        for column in (
            "reference_hash",
            "source_piece",
            "frame_count",
            "voiced_frame_count",
            "unvoiced_frame_count",
        ):
            if selected.groupby(keys)[column].nunique().gt(1).any():
                raise ValueError(f"Paired recordings disagree on {column}")
        reference = (
            selected.loc[selected.model.eq(baseline)].sort_values(keys).set_index(keys)
        )
        group_ids = reference["source_piece"].astype(str).to_numpy()
        groups = np.unique(group_ids)
        if len(groups) < 2:
            raise ValueError(
                "Paired inference requires at least two independent source pieces"
            )
        group_datasets = {
            g: tuple(
                sorted(reference.loc[group_ids == g].reset_index().dataset.unique())
            )
            for g in groups
        }
        strata = [
            np.array([i for i, g in enumerate(groups) if group_datasets[g] == label])
            for label in sorted(set(group_datasets.values()))
        ]
        rng = np.random.default_rng(seed)
        records = []
        for competitor in methods:
            if competitor == baseline:
                continue
            peer = (
                selected.loc[selected.model.eq(competitor)]
                .set_index(keys)
                .loc[reference.index]
            )
            for metric in metrics:
                numerator, denominator = METRIC_COUNTS[metric]
                a = reference[numerator].to_numpy(float)
                b = peer[numerator].to_numpy(float)
                d = reference[denominator].to_numpy(float)
                cluster_delta = np.array(
                    [(a - b)[group_ids == g].sum() for g in groups]
                )
                cluster_denom = np.array([d[group_ids == g].sum() for g in groups])
                total = cluster_denom.sum()
                record = dict(
                    method=competitor,
                    metric=metric,
                    tracks=len(reference),
                    source_pieces=len(groups),
                    excluded_tracks=len(coverage) - len(common),
                    eligible_frames=int(total),
                    seed=seed,
                    bootstrap_resamples=n_resamples,
                )
                if not total:
                    records.append(
                        {
                            **record,
                            "difference_pp": np.nan,
                            "ci_low_pp": np.nan,
                            "ci_high_pp": np.nan,
                            "p_value": np.nan,
                            "permutations": 0,
                        }
                    )
                    continue
                observed_sum = cluster_delta.sum()
                exact = 2 ** len(groups) <= n_resamples
                permutations = 2 ** len(groups) if exact else n_resamples
                extreme = 0
                signs_iter = (
                    itertools.product((-1.0, 1.0), repeat=len(groups))
                    if exact
                    else None
                )
                for start in range(0, permutations, 256):
                    count = min(256, permutations - start)
                    signs = (
                        np.array(list(itertools.islice(signs_iter, count)))
                        if exact
                        else rng.choice((-1.0, 1.0), size=(count, len(groups)))
                    )
                    extreme += int(
                        np.count_nonzero(
                            np.abs(signs @ cluster_delta) >= abs(observed_sum) - 1e-10
                        )
                    )
                p_value = (
                    extreme / permutations
                    if exact
                    else (extreme + 1) / (permutations + 1)
                )
                bootstrap = []
                for start in range(0, n_resamples, 256):
                    count = min(256, n_resamples - start)
                    sampled = np.concatenate(
                        [rng.choice(s, size=(count, len(s))) for s in strata], axis=1
                    )
                    denominator_draws = cluster_denom[sampled].sum(axis=1)
                    valid = denominator_draws > 0
                    bootstrap.extend(
                        (
                            cluster_delta[sampled].sum(axis=1)[valid]
                            / denominator_draws[valid]
                        ).tolist()
                    )
                low, high = (
                    np.quantile(bootstrap, [0.025, 0.975])
                    if bootstrap
                    else (np.nan, np.nan)
                )
                records.append(
                    {
                        **record,
                        "difference_pp": 100 * observed_sum / total,
                        "ci_low_pp": 100 * low,
                        "ci_high_pp": 100 * high,
                        "p_value": p_value,
                        "permutations": permutations,
                    }
                )
        result = pd.DataFrame(records)
        valid = result["p_value"].dropna().sort_values()
        adjusted = np.minimum(
            1.0, np.maximum.accumulate(valid.to_numpy() * np.arange(len(valid), 0, -1))
        )
        result["p_holm"] = pd.Series(adjusted, index=valid.index)
        result["significant"] = result["p_holm"].lt(alpha)
        return (result, selected)

    @staticmethod
    def default_pitch_workers() -> int:
        """Leave two logical CPUs for the desktop; RAM is checked when work starts."""
        return max(1, (os.cpu_count() or 2) - 2)

    @staticmethod
    def memory_limited_workers(requested: int, *, gib_per_worker: float = 2.0) -> int:
        """Reserve desktop headroom and budget each process's model/audio workspace.

        These are scheduling estimates, not hard process memory limits. Recheck for
        each offline batch / streaming method so other applications are accounted for.
        """
        try:
            import psutil

            memory = psutil.virtual_memory()
            reserve = max(2 * 2**30, memory.total * 0.15)
            budget = max(0, memory.available - reserve)
            affordable = max(1, int(budget / (gib_per_worker * 2**30)))
        except (ImportError, OSError):
            affordable = 2
        return max(1, min(requested, affordable))

    @staticmethod
    def joint_timing_worker(payload):
        from benchmarks.modules.pitch.PitchCache import PitchCache
        from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase

        example = PitchDetectorBase.PitchExample(**payload)
        adapter = Attune()
        config = adapter.config_for(example.fmin, example.fmax)
        hit = adapter.cache(example).read(PitchCache.RAW, config)
        if hit is None:
            recording = adapter.recording_for(config)
            recording.audio_data = adapter._audio_data(example, config)
            raw = list(
                adapter.detect_stages(recording, smooth=False).data[PitchCache.RAW].data
            )
        else:
            raw = list(hit[0].data)
        joint = PYINPitchSmoother(config=config)
        current = PitchSmoother(config=config)
        gate = VoicingSmoother(config=config)
        methods = {
            "joint_pyin": lambda: joint.smooth(raw),
            "current_attune": lambda: gate.smooth(current.smooth(raw)),
        }
        joint.smooth(raw[:32])
        gate.smooth(current.smooth(raw[:32]))
        timings = {name: [] for name in methods}
        outputs = {}
        offset = sum(example.track_id.encode()) % 2
        for repeat in range(REPEATS):
            names = list(methods)
            if (repeat + offset) % 2:
                names.reverse()
            for name in names:
                outputs[name], cpu, wall = adapter.measure(methods[name])
                timings[name].append([cpu, wall])
        rows = []
        for name, pitches in outputs.items():
            data = adapter._pitch_data(pitches, config, 0.0)
            times, frequencies = adapter.melody(data, config)
            cpu, wall = np.median(timings[name], axis=0)
            estimate = PitchDetectorBase.PitchEstimate.build(
                times,
                adapter.constrain_freqs_to_range(
                    frequencies, example.fmin, example.fmax
                ),
                cpu,
                metadata={
                    "compute_clock": adapter.COMPUTE_CLOCK,
                    "wall_pitch_compute_time": wall,
                },
            )
            row = PitchBenchmarker().score(name, example, estimate)
            row.update(
                smoother_cpu_seconds=cpu,
                smoother_wall_seconds=wall,
                timing_samples=json.dumps(timings[name]),
                frames=len(raw),
                raw_cache_hit=hit is not None,
                sr=config.sr,
                hop_length=config.h1,
                frame_length=config.w1,
                repeats=REPEATS,
            )
            rows.append(row)
        return rows

    @staticmethod
    def joint_timing_main():
        OUTPUT.mkdir(parents=True, exist_ok=True)
        selection = pd.read_csv(SELECTION)
        selection.to_csv(OUTPUT / "selection.csv", index=False)
        bench = PitchBenchmarker()
        examples = []
        for dataset, group in selection.groupby("dataset"):
            source = bench.dataset(dataset)
            lookup = {t.track_id: t for t in source.tracks()}
            examples.extend((source.example(lookup[name]) for name in group.track_id))
        (OUTPUT / "metadata.json").write_text(
            json.dumps(
                {
                    "tracks": len(examples),
                    "repeats": REPEATS,
                    "workers": 2,
                    "numerical_threads_per_worker": 1,
                    "clock": "process_cpu",
                    "speedup": "median across tracks of (median joint CPU / median Attune CPU)",
                    "accuracy": "mean and median per-track overall accuracy reported separately",
                    "baseline": "Local librosa 0.11-compatible joint HMM, including its own voicing decisions",
                    "experiment": PitchSmoother.METHOD_VERSION
                    + " plus final VoicingSmoother",
                    "shared": "Identical ordinary pYIN candidates, pitch grid, rate, hop and frame size",
                    "timed": "post-hoc smoothing and output voicing only",
                    "excluded": [
                        "frontend",
                        "cache I/O",
                        "initialization",
                        "warmup",
                        "scoring",
                        "process startup",
                    ],
                    "historical_comparison": "Earlier 3.69x pilot applied identical external voicing gates to both outputs; this run uses the actual joint-model voiced/unvoiced decisions.",
                },
                indent=2,
            )
        )
        rows = []
        with ProcessPoolExecutor(
            max_workers=2, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = [
                pool.submit(PitchBenchmarker.joint_timing_worker, asdict(e))
                for e in examples
            ]
            for future in as_completed(futures):
                pair = future.result()
                rows.extend(pair)
                pd.DataFrame(rows).to_csv(OUTPUT / "rows.csv", index=False)
                print(
                    f"{len(rows) // 2}/{len(examples)} completed: {pair[0]['dataset']} {pair[0]['track_id']}",
                    flush=True,
                )
        frame = pd.DataFrame(rows)
        summary = frame.groupby("model")[
            [
                "Overall Accuracy",
                "Raw Pitch Accuracy",
                "Voicing Recall",
                "Voicing False Alarm",
                "smoother_cpu_seconds",
            ]
        ].agg(["mean", "median"])
        summary.to_csv(OUTPUT / "summary.csv")
        paired = frame.pivot(
            index=["dataset", "track_id"],
            columns="model",
            values="smoother_cpu_seconds",
        )
        paired["speedup"] = paired.joint_pyin / paired.current_attune
        paired.to_csv(OUTPUT / "paired_timing.csv")
        print(summary.to_string(), flush=True)
        print("Median paired speedup:", paired.speedup.median(), flush=True)

    @staticmethod
    def _select_tracks(tracks, max_tracks, seed):
        """Deterministic dataset-balanced pilot; no accuracy/duration-based selection."""
        tracks = sorted(tracks, key=lambda t: (t.dataset, t.track_id))
        if max_tracks is None or max_tracks >= len(tracks):
            return tracks
        rng = np.random.default_rng(seed)
        groups = []
        for dataset in sorted({t.dataset for t in tracks}):
            group = [t for t in tracks if t.dataset == dataset]
            groups.append([group[i] for i in rng.permutation(len(group))])
        selected = []
        while len(selected) < max_tracks:
            for group in groups:
                if group and len(selected) < max_tracks:
                    selected.append(group.pop())
        return sorted(selected, key=lambda t: (t.dataset, t.track_id))

    @staticmethod
    def _smoother_track_worker(
        example_payload,
        repeats,
        max_gap_seconds,
        use_cache,
        force,
        confidence_emissions=False,
    ):
        from benchmarks.modules.pitch.PitchCache import PitchCache
        from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase

        example_payload = dict(example_payload)
        if isinstance(example_payload.get("degradation"), dict):
            example_payload["degradation"] = PitchDetectorBase.AudioDegradation(
                **example_payload["degradation"]
            )
        example = PitchDetectorBase.PitchExample(**example_payload)
        adapter = Attune()
        benchmarker = PitchBenchmarker()
        rows = []
        config = adapter.config_for(example.fmin, example.fmax)
        hit = (
            adapter.cache(example).read(PitchCache.RAW, config)
            if use_cache and (not force)
            else None
        )
        if hit is None:
            recording = adapter.recording_for(config)
            recording.audio_data = adapter._audio_data(example, config)
            stages = adapter.detect_stages(recording, smooth=False)
            raw = list(stages.data[PitchCache.RAW].data)
        else:
            raw = list(hit[0].data)
        controller = VoicingSmoother(config=config)
        expected_mask = controller.decode(raw)
        baseline = PitchSmoother(mode="joint", config=config)
        experiment = PitchSmoother(
            mode="pitch_only",
            config=config,
            max_gap_seconds=max_gap_seconds,
            confidence_emissions=confidence_emissions,
        )
        methods = {
            "legacy_joint_defaults": lambda: controller.smooth(baseline.smooth(raw)),
            "pitch_only": lambda: experiment.smooth(raw),
        }
        timings = {name: [] for name in methods}
        outputs = {}
        controller.smooth(baseline.smooth(raw[:32]))
        experiment.smooth(raw[:32])
        for repeat in range(repeats):
            order = list(methods) if repeat % 2 == 0 else list(reversed(methods))
            for name in order:
                outputs[name], cpu, wall = adapter.measure(methods[name])
                timings[name].append((cpu, wall))
        for name, output in outputs.items():
            mask = np.asarray([p is not None and p.value != -1 for p in output])
            np.testing.assert_array_equal(mask, expected_mask)
            data = adapter._pitch_data(output, config, 0.0)
            times, frequencies = adapter.melody(data, config)
            cpu, wall = np.median(timings[name], axis=0)
            estimate = PitchDetectorBase.PitchEstimate.build(
                times,
                adapter.constrain_freqs_to_range(
                    frequencies, example.fmin, example.fmax
                ),
                float(cpu),
                metadata={
                    "compute_clock": adapter.COMPUTE_CLOCK,
                    "wall_pitch_compute_time": float(wall),
                    "timing_scope": "posthoc_smoothing_only",
                },
            )
            row = benchmarker.score(name, example, estimate)
            row.update(
                variant=name,
                smoother_cpu_seconds=cpu,
                smoother_wall_seconds=wall,
                frames=len(raw),
                timing_scope="posthoc_smoothing_only",
                timing_repeats=repeats,
                numerical_threads=1,
                confidence_emissions=(
                    confidence_emissions if name == "pitch_only" else False
                ),
                smoother_cpu_samples=json.dumps(
                    [sample[0] for sample in timings[name]]
                ),
                smoother_wall_samples=json.dumps(
                    [sample[1] for sample in timings[name]]
                ),
                voiced_frames=int(expected_mask.sum()),
                sr=config.sr,
                frame_length=config.w1,
                hop_length=config.h1,
                max_gap_seconds=max_gap_seconds if name == "pitch_only" else 0.0,
            )
            rows.append(row)
        return rows

    @staticmethod
    def run_smoother_ablation(
        ablation,
        repeats=3,
        *,
        max_gap_seconds=0.0,
        workers=1,
        max_tracks=None,
        confidence_emissions=False,
    ):
        """Frozen pre-promotion joint defaults vs PitchSmoother; no threshold search.

        Cached raw evidence is allowed; smoothing is always rerun. Timings exclude
        construction, frontend and warmup, and include the final voicing controller.
        """
        if not isinstance(workers, int) or workers < 1:
            raise ValueError("workers must be a positive integer")
        if max_tracks is not None and (
            not isinstance(max_tracks, int) or max_tracks < 1
        ):
            raise ValueError("max_tracks must be a positive integer or None")
        if repeats < 1:
            raise ValueError("repeats must be positive")
        max_gap_seconds = float(max_gap_seconds)
        if not np.isfinite(max_gap_seconds) or max_gap_seconds < 0:
            raise ValueError("max_gap_seconds must be finite and nonnegative")
        tracks = PitchBenchmarker._select_tracks(
            ablation.tracks, max_tracks, ablation.config.seed
        )
        if not tracks:
            raise ValueError("No tracks selected")
        examples = [ablation.benchmarker.dataset(t.dataset).example(t) for t in tracks]
        directory = (
            "pitch_only_smoother_v1"
            if max_gap_seconds == 0
            else f"pitch_only_gap_bridge_v1_{max_gap_seconds!r}s"
        )
        if confidence_emissions:
            directory = f"pitch_only_confidence_v1_gap_{max_gap_seconds!r}s"
        root = (
            ablation.run_root
            / directory
            / f"runner_v3_legacy_reference_tracks_{max_tracks}_seed_{ablation.config.seed}_repeats_{repeats}_workers_{workers}"
        )
        root.mkdir(parents=True, exist_ok=True)
        (root / "timing_metadata.json").write_text(
            json.dumps(
                {
                    "compute_clock": Attune.COMPUTE_CLOCK,
                    "parallelism": "spawned_processes" if workers > 1 else "serial",
                    "workers": workers,
                    "numerical_threads": 1,
                    "repeats": repeats,
                    "confidence_emissions": bool(confidence_emissions),
                    "confidence_rule": (
                        "clip(1 - raw_unvoiced_prob, 0, 1)"
                        if confidence_emissions
                        else None
                    ),
                    "max_gap_seconds": max_gap_seconds,
                    "warmup_max_frames": 32,
                    "speedup_basis": "paired_median_process_cpu_seconds",
                    "scope": "posthoc_smoothing_and_final_voicing_only",
                    "excluded": [
                        "frontend",
                        "cache_io",
                        "worker_startup",
                        "warmup",
                        "scoring",
                        "queue_wait",
                        "result_serialization",
                    ],
                    "wall_time_role": "diagnostic_only; includes scheduling contention",
                },
                indent=2,
            )
            + "\n"
        )
        pd.DataFrame(
            [{"dataset": e.dataset, "track_id": e.track_id} for e in examples]
        ).to_csv(root / "selection.csv", index=False)
        print(
            f"{len(examples)} tracks; {workers} workers; {repeats} timed pass(es) per method. Outputs: {root}"
        )
        rows = []

        def collect(pair):
            rows.extend(pair)
            partial = root / "partial_rows.tmp.csv"
            pd.DataFrame(rows).to_csv(partial, index=False)
            partial.replace(root / "partial_rows.csv")

        args = [
            (
                asdict(e),
                repeats,
                max_gap_seconds,
                ablation.config.use_cache,
                ablation.config.force,
                confidence_emissions,
            )
            for e in examples
        ]
        if workers == 1:
            for payload in tqdm(args, desc="Pitch-only smoother vs legacy joint"):
                collect(PitchBenchmarker._smoother_track_worker(*payload))
        else:
            pool = ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn")
            )
            futures = []
            try:
                futures = [
                    pool.submit(PitchBenchmarker._smoother_track_worker, *payload)
                    for payload in args
                ]
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Pitch-only smoother vs legacy joint",
                ):
                    collect(future.result())
            except BaseException:
                for future in futures:
                    future.cancel()
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            else:
                pool.shutdown()
        results = (
            pd.DataFrame(rows)
            .sort_values(["dataset", "track_id", "variant"])
            .reset_index(drop=True)
        )
        results.to_csv(root / "rows.csv", index=False)
        metrics = [
            "Raw Pitch Accuracy",
            "Overall Accuracy",
            "Voicing Recall",
            "Voicing False Alarm",
            "smoother_cpu_seconds",
            "smoother_wall_seconds",
        ]
        summary = results.groupby("variant")[metrics].mean()
        summary.to_csv(root / "summary.csv")
        effects = ablation.effects(
            results,
            (
                (
                    "Pitch-only smoothing",
                    "Pre-promotion joint defaults; fixed identical voicing",
                    "pitch_only",
                    "legacy_joint_defaults",
                ),
            ),
        )
        effects.to_csv(root / "paired_effects.csv", index=False)
        paired = results.pivot(
            index=["dataset", "track_id"],
            columns="variant",
            values="smoother_cpu_seconds",
        )
        paired["speedup"] = paired.legacy_joint_defaults / paired.pitch_only
        paired.to_csv(root / "paired_timing.csv")
        return (results, summary, effects, paired)

    class observation_floor_CandidateMassSmoother(PitchSmoother):
        """Experimental raw candidate emissions, keeping production transitions/gaps."""

        def __init__(self, config):
            super().__init__(
                mode="pitch_only",
                config=config,
                max_gap_seconds=0.05,
                confidence_emissions=False,
            )

        def observation_probabilities(self, pitches):
            observations = np.zeros((self.n_pitch_bins, len(pitches)))
            for frame, pitch in enumerate(pitches):
                if pitch is not None:
                    for midi, probability in pitch.candidate_pitches:
                        observations[self._midi_to_bin(midi), frame] = probability
            mass = observations.sum(axis=0)
            observations[:, mass == 0] = 1.0 / self.n_pitch_bins
            return (observations, mass)

        def smooth(self, pitches, show_progress=False, verbose=False):
            return self._smooth_pitch_path(pitches, show_progress, verbose)

    @staticmethod
    def observation_floor_compare(raw, config):
        gate = VoicingSmoother(config=config)
        mask = gate.decode(raw)
        models = {
            "production": PitchSmoother(
                mode="pitch_only",
                config=config,
                max_gap_seconds=0.05,
                confidence_emissions=True,
            ),
            "no_uniform_floor": PitchBenchmarker.observation_floor_CandidateMassSmoother(
                config=config
            ),
        }
        outputs = {}
        for name, model in models.items():
            outputs[name] = gate.smooth(model.smooth(raw))
            np.testing.assert_array_equal(
                [p is not None and p.value != -1 for p in outputs[name]], mask
            )
        return (outputs, mask)

    @staticmethod
    def observation_floor_demo(output):
        audio = REPO_ROOT / "resources/demo/fugue/fugue_rec.mp3"
        with lzma.open(audio.with_name(".fugue_rec.json.xz"), "rt") as handle:
            settings = json.load(handle)["config"]
        config = Config()
        for name in (
            "sr",
            "w1",
            "h1",
            "fmin",
            "fmax",
            "tuning",
            "unv_thresh",
            "min_volume",
            "posthoc_unv_thresh",
            "posthoc_min_volume",
        ):
            value = settings[name]
            setattr(config, name, int(value) if name in ("sr", "w1", "h1") else value)
        samples, config.sr = sf.read(audio)
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        raw = PitchDetector(config=config).detect_pitches(samples)
        outputs, mask = PitchBenchmarker.observation_floor_compare(raw, config)
        normalized = PitchSmoother(
            mode="pitch_only",
            config=config,
            max_gap_seconds=0.05,
            confidence_emissions=False,
        )
        normalized_output = normalized.smooth(raw)
        equivalent = bool(
            np.array_equal(
                [p.value for p in normalized_output],
                [p.value for p in outputs["no_uniform_floor"]],
            )
        )
        normalized_values = np.array([p.value for p in normalized_output])
        raw_values = np.array([p.value for p in outputs["no_uniform_floor"]])
        times = np.array([p.time for p in raw])
        frame_data = {
            "time_seconds": times,
            "voiced": mask,
            "raw_unvoiced_probability": [p.unvoiced_prob for p in raw],
            "normalized_no_floor": normalized_values,
        }
        summary = []
        for name, pitches in outputs.items():
            values = np.array([p.value for p in pitches])
            distance = np.array(
                [
                    min(
                        (abs(value - c[0]) for c in p.candidate_pitches), default=np.inf
                    )
                    for value, p in zip(values, raw)
                ]
            )
            frame_data[name] = values
            summary.append(
                {
                    "variant": name,
                    "voiced_frames": int(mask.sum()),
                    "unsupported_frames_5c": int((mask & (distance > 0.051)).sum()),
                    "unsupported_frames_50c": int((mask & (distance > 0.5)).sum()),
                }
            )
        pd.DataFrame(frame_data).to_csv(output / "fugue_frames.csv", index=False)
        pd.DataFrame(summary).to_csv(output / "fugue_summary.csv", index=False)
        metadata = {
            "normalized_and_raw_paths_equal_on_fugue": equivalent,
            "normalized_vs_raw_different_voiced_frames": int(
                (mask & (abs(normalized_values - raw_values) > 0.051)).sum()
            ),
            "normalization_caveat": "Zero-probability floors in the decoder can break exact path equivalence when supported transitions are impossible.",
            "config": {k: getattr(config, k) for k in settings if hasattr(config, k)},
            "fugue_accuracy": "No performed-F0 ground truth; support counts are not accuracy.",
        }
        (output / "demo_metadata.json").write_text(json.dumps(metadata, indent=2))
        print(pd.DataFrame(summary).to_string(index=False), flush=True)
        print("Normalized and raw-candidate paths equal:", equivalent, flush=True)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, sharey=True)
        cx, cy = ([], [])
        for pitch in raw:
            for midi, probability in pitch.candidate_pitches:
                if probability > 0:
                    cx.append(pitch.time)
                    cy.append(midi)
        for ax, (name, pitches) in zip(axes, outputs.items()):
            ax.scatter(cx, cy, s=2, color="lightgray", label="Raw candidates")
            values = np.array([p.value for p in pitches])
            ax.plot(times, np.where(mask, values, np.nan), ".", ms=2, label=name)
            ax.set_ylabel("MIDI pitch")
            ax.legend()
            ax.grid(alpha=0.2)
        axes[-1].set_xlabel("Seconds")
        fig.tight_layout()
        fig.savefig(output / "fugue_comparison.png", dpi=150)
        plt.close(fig)

    @staticmethod
    def observation_floor_benchmark(
        output,
        per_instrument,
        seed,
        compare_fn=observation_floor_compare,
        baseline_name="production",
        experiment_name="no_uniform_floor",
    ):
        from benchmarks.modules.pitch.competitors.Attune import Attune
        from benchmarks.modules.pitch.datasets.URMP import URMP
        from benchmarks.modules.pitch.datasets.CocoChorales import CocoChorales
        from benchmarks.modules.pitch.PitchCache import PitchCache

        PitchCache = PitchCache
        from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker

        PitchBenchmarker = PitchBenchmarker
        from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase

        PitchEstimate = PitchDetectorBase.PitchEstimate
        adapter, scorer = (Attune(), PitchBenchmarker())
        rows, selection = ([], [])
        for dataset in (
            URMP(per_instrument=per_instrument, seed=seed),
            CocoChorales(per_instrument=per_instrument, seed=seed),
        ):
            tracks = dataset.tracks()
            if not tracks:
                raise RuntimeError(f"No tracks available for {dataset.name}")
            for track in tracks:
                example = dataset.example(track)
                config = adapter.config_for(example.fmin, example.fmax)
                hit = PitchCache(example.stage_cache_path).read(PitchCache.RAW, config)
                if hit is None:
                    recording = adapter.recording_for(config)
                    recording.audio_data = adapter._audio_data(example, config)
                    stages = adapter.detect_stages(recording, smooth=False)
                    raw = list(stages.data[PitchCache.RAW].data)
                else:
                    raw = list(hit[0].data)
                outputs, _ = compare_fn(raw, config)
                selection.append(
                    {
                        "dataset": dataset.name,
                        "track_id": track.track_id,
                        "raw_cache_hit": hit is not None,
                    }
                )
                for name, pitches in outputs.items():
                    data = adapter._pitch_data(pitches, config, 0.0)
                    times, frequencies = adapter.melody(data, config)
                    estimate = PitchEstimate.build(
                        times,
                        adapter.constrain_freqs_to_range(
                            frequencies, example.fmin, example.fmax
                        ),
                        0.0,
                    )
                    rows.append(scorer.score(name, example, estimate))
                pd.DataFrame(rows).to_csv(output / "benchmark_rows.csv", index=False)
                pd.DataFrame(selection).to_csv(output / "selection.csv", index=False)
                print("Completed", dataset.name, track.track_id, flush=True)
        frame = pd.DataFrame(rows)
        metrics = [
            "Overall Accuracy",
            "Raw Pitch Accuracy",
            "Raw Chroma Accuracy",
            "Voicing Recall",
            "Voicing False Alarm",
        ]
        summary = frame.groupby(["dataset", "model"])[metrics].mean()
        summary.to_csv(output / "benchmark_summary.csv")
        paired = frame.pivot(
            index=["dataset", "track_id"], columns="model", values="Overall Accuracy"
        )
        paired["delta_percentage_points"] = 100 * (
            paired[experiment_name] - paired[baseline_name]
        )
        paired.to_csv(output / "paired_accuracy.csv")
        print(summary.to_string(), flush=True)

    @staticmethod
    def observation_floor_main():
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--benchmarks", action="store_true")
        parser.add_argument("--per-instrument", type=int, default=1)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument(
            "--output",
            type=Path,
            default=REPO_ROOT / "benchmarks/results/pitch/observation_floor_ablation",
        )
        args = parser.parse_args()
        if args.per_instrument < 1:
            parser.error("--per-instrument must be positive")
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "run_config.json").write_text(
            json.dumps(
                {
                    "per_instrument": args.per_instrument,
                    "seed": args.seed,
                    "benchmarks": args.benchmarks,
                    "gap_seconds": 0.05,
                    "experimental_emission": "raw candidate mass; no voiced-frame uniform floor",
                    "scope": "Exploratory pilot; identical evidence, transitions, gates, and gap bridging",
                    "production_defaults_changed": False,
                },
                indent=2,
            )
        )
        PitchBenchmarker.observation_floor_demo(args.output)
        if args.benchmarks:
            PitchBenchmarker.observation_floor_benchmark(
                args.output, args.per_instrument, args.seed
            )

    @staticmethod
    def candidate_support_compare(raw, config):
        model = PitchSmoother(config=config)
        gate = VoicingSmoother(config=config)
        baseline = gate.smooth(model._smooth_pitch_path(raw))
        filtered = gate.smooth(model.smooth(raw))
        for before, after in zip(baseline, filtered):
            if after is not None and after.value != -1:
                assert before.value == after.value
        return (
            {"before_support_gate": baseline, "support_gate": filtered},
            gate.decode(raw),
        )

    @staticmethod
    def candidate_support_main():
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--benchmarks", action="store_true")
        parser.add_argument("--per-instrument", type=int, default=1)
        parser.add_argument("--seed", type=int, default=0)
        args = parser.parse_args()
        if args.per_instrument < 1:
            parser.error("--per-instrument must be positive")
        output = REPO_ROOT / "benchmarks/results/pitch/candidate_support_ablation"
        output.mkdir(parents=True, exist_ok=True)
        (output / "run_config.json").write_text(json.dumps(vars(args), indent=2))
        audio = REPO_ROOT / "resources/demo/fugue/fugue_rec.mp3"
        with lzma.open(audio.with_name(".fugue_rec.json.xz"), "rt") as handle:
            settings = json.load(handle)["config"]
        config = Config()
        for name in (
            "sr",
            "w1",
            "h1",
            "fmin",
            "fmax",
            "tuning",
            "unv_thresh",
            "min_volume",
            "posthoc_unv_thresh",
            "posthoc_min_volume",
        ):
            value = settings[name]
            setattr(config, name, int(value) if name in ("sr", "w1", "h1") else value)
        audio_data, config.sr = sf.read(audio)
        if audio_data.ndim == 2:
            audio_data = audio_data.mean(axis=1)
        raw = PitchDetector(config=config).detect_pitches(audio_data)
        outputs, _ = PitchBenchmarker.candidate_support_compare(raw, config)
        frame = pd.DataFrame(
            {
                "time_seconds": [p.time for p in raw],
                **{
                    name: [p.value for p in pitches]
                    for name, pitches in outputs.items()
                },
            }
        )
        frame.to_csv(output / "fugue_frames.csv", index=False)
        summary = {name: int((frame[name] != -1).sum()) for name in outputs}
        model = PitchSmoother(config=config)
        for pitch in outputs["support_gate"]:
            if pitch.value != -1:
                assert any(
                    (
                        prob > 0
                        and model._midi_to_bin(midi) == model._midi_to_bin(pitch.value)
                        for midi, prob in pitch.candidate_pitches
                    )
                )
        (output / "fugue_summary.json").write_text(json.dumps(summary, indent=2))
        print("Fugue voiced frames:", summary, flush=True)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, sharey=True)
        for ax, name in zip(axes, outputs):
            ax.plot(frame.time_seconds, frame[name].where(frame[name] != -1), ".", ms=2)
            ax.set_title(name)
            ax.set_ylabel("MIDI pitch")
            ax.grid(alpha=0.2)
        axes[-1].set_xlabel("Seconds")
        fig.tight_layout()
        fig.savefig(output / "fugue_comparison.png", dpi=150)
        plt.close(fig)
        if args.benchmarks:
            PitchBenchmarker.observation_floor_benchmark(
                output,
                args.per_instrument,
                args.seed,
                compare_fn=PitchBenchmarker.candidate_support_compare,
                baseline_name="before_support_gate",
                experiment_name="support_gate",
            )

    @staticmethod
    def default_streaming_workers() -> int:
        return PitchBenchmarker.default_pitch_workers()

    @dataclass(frozen=True)
    class StreamingConfig:
        """Controls for low-latency replay; full audio unless explicitly capped."""

        workers: int = field(
            default_factory=lambda: PitchBenchmarker.default_streaming_workers()
        )
        max_seconds: float = math.inf
        transition_seconds: float = 0.05
        deep_rest_seconds: float = 0.25
        event_search_seconds: float = 0.5

        def __post_init__(self) -> None:
            if self.workers < 1:
                raise ValueError("workers must be at least 1")
            if self.max_seconds <= 0:
                raise ValueError("max_seconds must be positive")

    class PitchStreaming:
        """Replay audio through causal or practical rolling adapters."""

        VERSION = (
            "native_center_v9_pyin_joint_state_latency_samples__" + SCORING_VERSION
        )
        METHODS = ("attune", "pyin_framewise", "praat", "swiftf0", "spice")
        MATCHED_METHODS = ("attune", "praat")
        ROLLING_METHODS = ("swiftf0", "spice")
        APP_SR = Attune.DEFAULT_CONFIG.sr
        MODEL_SR = 16000
        ROLLING_GEOMETRY = {
            "swiftf0": (5120, 256, 256, 10),
            "spice": (1024, 512, 512, 1),
        }

        def __init__(
            self,
            options: PitchBenchmarker.Options,
            config: StreamingConfig | None = None,
        ) -> None:
            self.options = options
            self.config = config or PitchBenchmarker.StreamingConfig()
            self.reporter = PitchBenchmarker(options)

        def run(
            self,
            examples: list[PitchExample],
            methods: Collection[str] | None = None,
            *,
            checkpoint_dir: Path | None = None,
            force: bool = False,
        ) -> pd.DataFrame:
            """Return one canonical benchmark row per method and excerpt."""
            from benchmarks.modules.pitch.PitchCache import PitchCache

            requested = (
                set(self.METHODS)
                if methods is None
                else {methods} if isinstance(methods, str) else set(methods)
            )
            selected = [method for method in self.METHODS if method in requested]
            unknown = requested.difference(self.METHODS)
            if unknown:
                raise ValueError(
                    f"unknown streaming method(s): {', '.join(sorted(unknown))}"
                )
            if not selected:
                raise ValueError(
                    "methods cannot be empty; omit it to select all methods"
                )
            if not examples:
                return pd.DataFrame()
            workers = min(self.config.workers, len(examples))
            print(
                f"Streaming replay: {workers} worker(s); serial frames per track; dedicated-worker CPU latency simulation",
                flush=True,
            )
            rows: list[dict[str, Any]] = []
            total = len(examples) * len(selected)
            completed = 0
            for method in selected:
                workers = min(
                    len(examples),
                    PitchBenchmarker.memory_limited_workers(
                        self.config.workers,
                        gib_per_worker=2.0 if method in self.ROLLING_METHODS else 1.0,
                    ),
                )
                print(
                    f"  {method}: {workers} worker(s) within the RAM budget", flush=True
                )
                progress = PitchBenchmarker.Progress(compact=True)
                jobs = []
                reused = 0
                for example in examples:
                    path = self._checkpoint_path(checkpoint_dir, method, example)
                    cached = None
                    if path is not None and (not force):
                        for candidate_path in PitchCache._checkpoint_candidates(path):
                            try:
                                candidate = json.loads(candidate_path.read_text())
                                if candidate and all(
                                    (
                                        row.get("model") == method
                                        and row.get("track_id") == example.track_id
                                        and (row.get("dataset") == example.dataset)
                                        and PitchCache.has_latency_samples(row)
                                        for row in candidate
                                    )
                                ):
                                    cached = candidate
                                    break
                            except (ValueError, OSError, TypeError, AttributeError):
                                pass
                    if cached is not None:
                        rows.extend(cached)
                        completed += 1
                        reused += 1
                    else:
                        jobs.append((example, path))
                started = time.monotonic()
                last_finished = (
                    f"{reused} cached; loading model" if jobs else f"{reused} cached"
                )

                def render():
                    progress.completed = min(total, completed + (1 if jobs else 0))
                    progress.update_compact(
                        total,
                        method,
                        f"{completed}/{total} finished | {int(time.monotonic() - started)}s | {last_finished}",
                        count=0,
                    )

                def accept(example, result):
                    nonlocal completed, last_finished
                    for row in result:
                        row.setdefault("streaming_workers", workers)
                    rows.extend(result)
                    completed += 1
                    status = "finished" if result else "skipped"
                    last_finished = f"{status}: {example.track_id}"
                    render()

                render()
                try:
                    if not jobs:
                        continue
                    context = multiprocessing.get_context("spawn")
                    pool = ProcessPoolExecutor(
                        max_workers=min(workers, len(jobs)),
                        mp_context=context,
                        initializer=PitchBenchmarker._initialize_streaming_worker,
                        initargs=(
                            dict(vars(self.options)),
                            asdict(self.config),
                            [method],
                        ),
                    )
                    try:
                        remaining = iter(jobs)
                        futures = {}

                        def submit_next():
                            job = next(remaining, None)
                            if job is not None:
                                example, path = job
                                future = pool.submit(
                                    PitchBenchmarker._streaming_example_worker,
                                    asdict(example),
                                    path,
                                )
                                futures[future] = example

                        for _ in range(min(workers, len(jobs))):
                            submit_next()
                        while futures:
                            done, _ = wait(
                                futures, timeout=1.0, return_when=FIRST_COMPLETED
                            )
                            for future in done:
                                accept(futures.pop(future), future.result())
                                submit_next()
                            if not done:
                                render()
                    except BaseException:
                        for process in list(pool._processes.values()):
                            process.terminate()
                        pool.shutdown(wait=True, cancel_futures=True)
                        raise
                    else:
                        pool.shutdown(wait=True)
                finally:
                    progress.finish()
            for row in rows:
                row.setdefault("streaming_workers", workers)
            result = pd.DataFrame(rows)
            if not result.empty:
                result = result.sort_values(
                    ["dataset", "track_id", "model"]
                ).reset_index(drop=True)
            return result

        def _checkpoint_path(self, directory, method, example):
            from benchmarks.modules.pitch.PitchCache import PitchCache

            return PitchCache.streaming_checkpoint_path(
                self, directory, method, example
            )

        def _prepare_detectors(
            self, selected: Collection[str]
        ) -> dict[str, PitchDetectorBase]:
            if "praat" in selected:
                PitchBenchmarker.detector_for("praat", self.options).ensure_available()
            return self._load_rolling_detectors(selected)

        def _run_example(
            self,
            example: PitchExample,
            selected: Collection[str],
            rolling: dict[str, PitchDetectorBase],
        ) -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            matched = [method for method in self.MATCHED_METHODS if method in selected]
            if matched:
                rows.extend(self._run_matched_example(example, matched))
            if "pyin_framewise" in selected:
                rows.append(self._run_pyin_framewise(example))
            for method in self.ROLLING_METHODS:
                if method not in selected:
                    continue
                if method in rolling:
                    rows.append(
                        self._run_rolling_example(method, rolling[method], example)
                    )
                detail = (
                    example.track_id
                    if method in rolling
                    else f"{example.track_id} (skipped)"
                )
            return rows

        def _load_rolling_detectors(
            self, methods: Collection[str]
        ) -> dict[str, PitchDetectorBase]:
            """Load and warm optional models once, outside timed stream updates."""
            from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase

            loaded: dict[str, PitchDetectorBase] = {}
            for method in self.ROLLING_METHODS:
                if method not in methods:
                    continue
                try:
                    detector = PitchBenchmarker.detector_for(method, self.options)
                    detector.ensure_available()
                    frame_size, _, _, _ = self.ROLLING_GEOMETRY[method]
                    detector.predict(
                        np.zeros(frame_size, dtype=np.float32),
                        self.MODEL_SR,
                        20.0,
                        5000.0,
                    )
                    loaded[method] = detector
                except PitchDetectorBase.Unavailable as exc:
                    print(f"  skipped {method}: {str(exc).splitlines()[0]}")
                except Exception as exc:
                    print(f"  skipped {method}: model warm-up failed: {exc}")
            return loaded

        def _run_matched_example(
            self, example: PitchExample, methods: Collection[str]
        ) -> list[dict[str, Any]]:
            from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase

            audio, sr = example.audio(self.APP_SR)
            audio, duration = self._crop_audio(audio, sr)
            attune = Attune()
            config = attune.config_for(example.fmin, example.fmax, sr=sr)
            yin = attune.recording_for(config).pitch_detector
            praat = Praat.Streaming(config=config)
            frame_size = int(yin.FRAME_SIZE)
            hop_size = int(yin.HOP_SIZE)
            if len(audio) < frame_size:
                raise ValueError(
                    f"{example.track_id}: {duration:.3f}s excerpt is shorter than the shared {frame_size}-sample frame"
                )
            frames = np.lib.stride_tricks.sliding_window_view(audio, frame_size)[
                ::hop_size
            ]
            tracks = {
                method: {
                    "times": [],
                    "freqs": [],
                    "latencies": [],
                    "walls": [],
                    "cpus": [],
                    "cpu": 0.0,
                    "wall": 0.0,
                    "worker_ready": 0.0,
                }
                for method in self.MATCHED_METHODS
                if method in methods
            }
            if "attune" in methods:
                yin.detect_pitch(frames[0], 0.0)
                yin = attune.recording_for(config).pitch_detector
            if "praat" in methods:
                praat.detect_pitch(frames[0], 0.0)
                praat = Praat.Streaming(config=config)
            for frame_index, frame in enumerate(frames):
                frame_start = frame_index * hop_size / sr
                timestamp = frame_start + 0.5 * yin.INTEGRATION_SIZE / sr
                available = frame_start + frame_size / sr
                if "attune" in methods:
                    yin_pitch, yin_cpu, yin_wall = PitchDetectorBase.measure(
                        lambda frame=frame, frame_start=frame_start: yin.detect_pitch(
                            frame, frame_start
                        )
                    )
                    yin_voiced = (
                        yin_pitch.value != -1
                        and yin_pitch.unvoiced_prob < config.unv_thresh
                    )
                    yin_freq = (
                        config.midi_to_freq(yin_pitch.value) if yin_voiced else 0.0
                    )
                    self._append_frame(
                        tracks["attune"],
                        timestamp,
                        yin_freq,
                        available,
                        yin_cpu,
                        yin_wall,
                    )
                if "praat" in methods:
                    praat_pitch, praat_cpu, praat_wall = PitchDetectorBase.measure(
                        lambda frame=frame, frame_start=frame_start: praat.detect_pitch(
                            frame, frame_start
                        )
                    )
                    praat_voiced = (
                        praat_pitch.value != -1
                        and praat_pitch.unvoiced_prob < config.unv_thresh
                    )
                    praat_freq = (
                        config.midi_to_freq(praat_pitch.value) if praat_voiced else 0.0
                    )
                    self._append_frame(
                        tracks["praat"],
                        timestamp,
                        praat_freq,
                        available,
                        praat_cpu,
                        praat_wall,
                    )
            deadline = hop_size / sr
            rows = []
            for method in self.MATCHED_METHODS:
                if method not in methods:
                    continue
                track = tracks[method]
                melody = (
                    np.asarray(track["times"], dtype=np.float64),
                    np.asarray(track["freqs"], dtype=np.float64),
                )
                rows.append(
                    self._row(
                        method,
                        example,
                        duration,
                        melody,
                        float(track["cpu"]),
                        float(track["wall"]),
                        np.asarray(track["latencies"], dtype=np.float64),
                        np.asarray(track["cpus"], dtype=np.float64),
                        deadline,
                        frame_size,
                        int(yin.INTEGRATION_SIZE),
                        hop_size,
                        sr,
                        stream_adapter="Matched app frame",
                        model_hop_size=hop_size,
                        algorithmic_lookahead=(frame_size - 0.5 * yin.INTEGRATION_SIZE)
                        / sr,
                    )
                )
            return rows

        def _run_pyin_framewise(self, example: PitchExample) -> dict[str, Any]:
            """Select raw pYIN candidates on the production capture grid; no HMM."""
            from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase

            detector = PYIN(step_seconds=self.options.step_seconds)
            detector.ensure_available()
            audio, sr = example.audio(detector.input_sr)
            audio, duration = self._crop_audio(audio, sr)
            frame_size = detector.FRAME_LENGTH
            hop_size = detector.DEFAULT_CONFIG.h1
            fmin, fmax = detector.clamp_range(example.fmin, example.fmax)
            if len(audio) < frame_size:
                raise ValueError(
                    f"{example.track_id}: {duration:.3f}s excerpt is shorter than the pYIN {frame_size}-sample frame"
                )
            detector.predict_frame(audio[:frame_size], sr, fmin, fmax, hop_size)
            track = self._empty_track()
            for end in range(frame_size, len(audio) + 1, hop_size):
                start = end - frame_size
                frame = audio[start:end]
                frequency, cpu, wall = PitchDetectorBase.measure(
                    lambda frame=frame: detector.predict_frame(
                        frame, sr, fmin, fmax, hop_size
                    )
                )
                frequency = detector.constrain_freqs_to_range(
                    [frequency], example.fmin, example.fmax
                )[0]
                timestamp = start / sr + 0.5 * frame_size / sr
                self._append_frame(
                    track, timestamp, float(frequency), end / sr, cpu, wall
                )
            melody = (
                np.asarray(track["times"], dtype=np.float64),
                np.asarray(track["freqs"], dtype=np.float64),
            )
            return self._row(
                "pyin_framewise",
                example,
                duration,
                melody,
                float(track["cpu"]),
                float(track["wall"]),
                np.asarray(track["latencies"], dtype=np.float64),
                np.asarray(track["cpus"], dtype=np.float64),
                hop_size / sr,
                frame_size,
                frame_size,
                hop_size,
                sr,
                stream_adapter="Frame-restarted public API",
                model_hop_size=hop_size,
                algorithmic_lookahead=0.5 * frame_size / sr,
            )

        def _run_rolling_example(
            self, method: str, detector: PitchDetectorBase, example: PitchExample
        ) -> dict[str, Any]:
            """Commit one model-native center estimate per rolling update."""
            from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase

            audio, sr = example.audio(self.MODEL_SR)
            audio, duration = self._crop_audio(audio, sr)
            frame_size, update_size, model_hop, output_index = self.ROLLING_GEOMETRY[
                method
            ]
            if len(audio) < frame_size:
                raise ValueError(
                    f"{example.track_id}: {duration:.3f}s excerpt is shorter than the {method} {frame_size}-sample receptive frame"
                )
            track = self._empty_track()
            local_timestamp: float | None = None
            for end in range(frame_size, len(audio) + 1, update_size):
                start = end - frame_size
                frame = audio[start:end]
                (times, freqs), cpu, wall = PitchDetectorBase.measure(
                    lambda frame=frame: detector.predict(
                        frame, sr, example.fmin, example.fmax
                    )
                )
                if output_index >= len(times):
                    raise RuntimeError(
                        f"{method} returned {len(times)} outputs for its {frame_size}-sample streaming frame; expected index {output_index}"
                    )
                local_timestamp = float(times[output_index])
                selected = detector.constrain_freqs_to_range(
                    [freqs[output_index]], example.fmin, example.fmax
                )[0]
                timestamp = start / sr + local_timestamp
                available = end / sr
                self._append_frame(
                    track, timestamp, float(selected), available, cpu, wall
                )
            assert local_timestamp is not None
            melody = (
                np.asarray(track["times"], dtype=np.float64),
                np.asarray(track["freqs"], dtype=np.float64),
            )
            return self._row(
                method,
                example,
                duration,
                melody,
                float(track["cpu"]),
                float(track["wall"]),
                np.asarray(track["latencies"], dtype=np.float64),
                np.asarray(track["cpus"], dtype=np.float64),
                update_size / sr,
                frame_size,
                None,
                update_size,
                sr,
                stream_adapter="Rolling center frame",
                model_hop_size=model_hop,
                algorithmic_lookahead=frame_size / sr - local_timestamp,
            )

        def _crop_audio(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
            duration = min(self.config.max_seconds, len(audio) / float(sr))
            cropped = np.ascontiguousarray(
                audio[: int(math.floor(duration * sr))], dtype=np.float32
            )
            return (cropped, duration)

        @staticmethod
        def _empty_track() -> dict[str, Any]:
            return {
                "times": [],
                "freqs": [],
                "latencies": [],
                "walls": [],
                "cpus": [],
                "cpu": 0.0,
                "wall": 0.0,
                "worker_ready": 0.0,
            }

        @staticmethod
        def _append_frame(
            track: dict[str, Any],
            timestamp: float,
            frequency: float,
            available: float,
            cpu: float,
            wall: float,
        ) -> None:
            ready = max(float(available), float(track["worker_ready"])) + cpu
            track["worker_ready"] = ready
            track["times"].append(float(timestamp))
            track["freqs"].append(float(frequency))
            track["latencies"].append(max(0.0, ready - timestamp))
            track["walls"].append(float(wall))
            track["cpus"].append(float(cpu))
            track["cpu"] += float(cpu)
            track["wall"] += float(wall)

        def _row(
            self,
            method: str,
            example: PitchExample,
            duration: float,
            melody: Melody,
            cpu: float,
            wall: float,
            latencies: np.ndarray,
            call_cpus: np.ndarray,
            deadline_seconds: float,
            frame_size: int,
            integration_size: int | None,
            hop_size: int,
            sr: int,
            *,
            stream_adapter: str,
            model_hop_size: int,
            algorithmic_lookahead: float,
        ) -> dict[str, Any]:
            from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase

            ref_mask = example.ref_times <= duration + 1e-09
            cropped = replace(
                example,
                ref_times=example.ref_times[ref_mask],
                ref_freqs=example.ref_freqs[ref_mask],
            )
            estimate = PitchDetectorBase.PitchEstimate.build(
                *melody,
                cpu,
                metadata={
                    "wall_pitch_compute_time": wall,
                    "compute_clock": PitchDetectorBase.COMPUTE_CLOCK,
                },
            )
            row = self.reporter.score(method, cropped, estimate)
            row.update(
                {
                    "_latency_samples": {
                        "output_latency_ms": latencies * 1000.0,
                        "call_cpu_ms": call_cpus * 1000.0,
                        "estimate_times_seconds": np.asarray(melody[0]),
                    },
                    "result_label": f"{example.dataset}_streaming",
                    "audio_seconds": float(duration),
                    "realtime_factor": duration / cpu if cpu > 0 else float("nan"),
                    "execution_mode": "Streaming",
                    "latency_clock": "dedicated-worker process CPU simulation",
                    "stream_adapter": stream_adapter,
                    "frame_size_samples": int(frame_size),
                    "integration_size_samples": (
                        int(integration_size)
                        if integration_size is not None
                        else float("nan")
                    ),
                    "hop_size_samples": int(hop_size),
                    "model_hop_size_samples": int(model_hop_size),
                    "sample_rate_hz": int(sr),
                    "update_interval_ms": float(deadline_seconds) * 1000.0,
                    "algorithmic_lookahead_ms": float(algorithmic_lookahead) * 1000.0,
                    "dynamic_volume_floor_ratio": (
                        float(Attune.DEFAULT_CONFIG.min_volume)
                        if method == "attune"
                        else float("nan")
                    ),
                    "unvoiced_probability_threshold": (
                        float(Attune.DEFAULT_CONFIG.unv_thresh)
                        if method == "attune"
                        else float("nan")
                    ),
                    **self._diagnostics(
                        cropped.ref_times,
                        cropped.ref_freqs,
                        estimate.times,
                        estimate.freqs,
                    ),
                    "median_output_latency_ms": self._percentile(
                        latencies * 1000.0, 50
                    ),
                    "p95_output_latency_ms": self._percentile(latencies * 1000.0, 95),
                    "deadline_miss_rate": (
                        float(np.mean(call_cpus > deadline_seconds))
                        if call_cpus.size
                        else float("nan")
                    ),
                }
            )
            return row

        def _diagnostics(
            self,
            ref_times: np.ndarray,
            ref_freqs: np.ndarray,
            est_times: np.ndarray,
            est_freqs: np.ndarray,
        ) -> dict[str, float]:
            ref_v, ref_c, est_v, est_c = mir_eval.melody.to_cent_voicing(
                ref_times, ref_freqs, est_times, est_freqs
            )
            ref_v = np.asarray(ref_v, dtype=bool)
            est_v = np.asarray(est_v, dtype=bool)
            both = ref_v & est_v
            errors = np.abs(np.asarray(est_c)[both] - np.asarray(ref_c)[both])
            hop = self._hop_seconds(ref_times)
            distance = ndimage.distance_transform_edt(~ref_v) * hop
            false_alarm = ~ref_v & est_v
            near = ~ref_v & (distance <= self.config.transition_seconds)
            deep = ~ref_v & (distance > self.config.deep_rest_seconds)
            onset, hangover = self._event_delays(
                ref_v, est_v, hop, self.config.event_search_seconds
            )
            return {
                "median_abs_cents": self._percentile(errors, 50),
                "p95_abs_cents": self._percentile(errors, 95),
                "gross_pitch_error_rate": (
                    float(np.mean(errors > 50.0)) if errors.size else float("nan")
                ),
                "transition_vfa": self._rate(false_alarm, near),
                "deep_rest_vfa": self._rate(false_alarm, deep),
                "median_onset_acquisition_ms": self._percentile(onset * 1000.0, 50),
                "p95_onset_acquisition_ms": self._percentile(onset * 1000.0, 95),
                "median_release_hangover_ms": self._percentile(hangover * 1000.0, 50),
                "p95_release_hangover_ms": self._percentile(hangover * 1000.0, 95),
                "p95_false_alarm_burst_ms": self._percentile(
                    self._run_lengths(false_alarm) * hop * 1000.0, 95
                ),
            }

        @staticmethod
        def _event_delays(
            ref_voiced: np.ndarray,
            est_voiced: np.ndarray,
            hop: float,
            search_seconds: float,
        ) -> tuple[np.ndarray, np.ndarray]:
            search = max(1, int(round(search_seconds / hop)))
            onsets = np.flatnonzero(ref_voiced & ~np.r_[False, ref_voiced[:-1]])
            offsets = np.flatnonzero(~ref_voiced & np.r_[False, ref_voiced[:-1]])

            def delays(events: np.ndarray, target: np.ndarray) -> np.ndarray:
                found: list[float] = []
                for event in events:
                    tail = np.flatnonzero(target[event : event + search + 1])
                    if tail.size:
                        found.append(float(tail[0]) * hop)
                return np.asarray(found, dtype=np.float64)

            return (delays(onsets, est_voiced), delays(offsets, ~est_voiced))

        @staticmethod
        def _run_lengths(mask: np.ndarray) -> np.ndarray:
            labels, count = ndimage.label(mask)
            if count == 0:
                return np.array([], dtype=np.float64)
            return np.bincount(labels.ravel())[1:].astype(np.float64)

        @staticmethod
        def _hop_seconds(times: np.ndarray) -> float:
            differences = np.diff(np.asarray(times, dtype=np.float64))
            differences = differences[np.isfinite(differences) & (differences > 0)]
            return float(np.median(differences)) if differences.size else 0.01

        @staticmethod
        def _rate(numerator_mask: np.ndarray, denominator_mask: np.ndarray) -> float:
            denominator = int(np.count_nonzero(denominator_mask))
            if denominator == 0:
                return float("nan")
            return float(
                np.count_nonzero(numerator_mask & denominator_mask) / denominator
            )

        @staticmethod
        def _percentile(values: np.ndarray, percentile: float) -> float:
            values = np.asarray(values, dtype=np.float64)
            values = values[np.isfinite(values)]
            return (
                float(np.percentile(values, percentile))
                if values.size
                else float("nan")
            )

    @staticmethod
    def _initialize_streaming_worker(options, config, selected):
        global _pitchstreaming_streaming_worker_state
        runner = PitchBenchmarker.PitchStreaming(
            PitchBenchmarker.Options(**options),
            PitchBenchmarker.StreamingConfig(**config),
        )
        _pitchstreaming_streaming_worker_state = (
            runner,
            selected,
            runner._prepare_detectors(selected),
        )

    @staticmethod
    def _streaming_example_worker(example, checkpoint_path):
        from benchmarks.modules.pitch.PitchCache import PitchCache
        from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase

        runner, selected, rolling = _pitchstreaming_streaming_worker_state
        if example["degradation"] is not None:
            example["degradation"] = PitchDetectorBase.AudioDegradation(
                **example["degradation"]
            )
        rows = runner._run_example(
            PitchDetectorBase.PitchExample(**example), selected, rolling
        )
        PitchCache._write_streaming_checkpoint(checkpoint_path, rows)
        return rows

    "The one runnable entry point: plan, run in parallel, score, report."
    METRICS: ClassVar[list[str]] = [
        "Raw Pitch Accuracy",
        "Raw Chroma Accuracy",
        "Overall Accuracy",
        "Voicing Recall",
        "Voicing False Alarm",
    ]
    COMPUTE_COL: ClassVar[str] = "Compute Time (s)"
    WALL_COMPUTE_COL: ClassVar[str] = "Wall Time (s)"
    COMPUTE_CLOCK_COL: ClassVar[str] = "Compute Clock"
    REALTIME_COL: ClassVar[str] = "Audio(s)/Compute(s)"
    AUDIO_SECONDS_COL: ClassVar[str] = "Audio Length (s)"
    DISPLAY_NAMES: ClassVar[dict[str, str]] = {
        "track_id": "Track ID",
        "realtime_factor": REALTIME_COL,
        "audio_seconds": AUDIO_SECONDS_COL,
        "pitch_compute_time": COMPUTE_COL,
        "wall_pitch_compute_time": WALL_COMPUTE_COL,
        "compute_clock": COMPUTE_CLOCK_COL,
        "split": "Split",
        "track": "Track",
        "piece_number": "Piece Number",
        "ensemble": "Ensemble",
        "ensemble_size": "Ensemble Size",
        "instrument": "Instrument",
        "instrument_code": "Instrument Code",
        "voice": "Voice",
        "f0_voice": "F0_voice",
        "from_cache": "From Cache",
        "fmin": "Fmin",
        "fmax": "Fmax",
        "degradation": "Degradation",
        "degradation_seed": "Degradation Seed",
        "snr_db": "SNR (dB)",
        "snr_label": "SNR",
        "source_track_id": "Source Track ID",
        "execution_mode": "Execution Mode",
        "stream_adapter": "Stream Adapter",
        "frame_size_samples": "Frame Size (samples)",
        "integration_size_samples": "Integration Size (samples)",
        "hop_size_samples": "Hop Size (samples)",
        "model_hop_size_samples": "Model Hop Size (samples)",
        "sample_rate_hz": "Sample Rate (Hz)",
        "update_interval_ms": "Update Interval (ms)",
        "algorithmic_lookahead_ms": "Algorithmic Look-ahead (ms)",
        "dynamic_volume_floor_ratio": "Dynamic Volume Floor Ratio",
        "median_abs_cents": "Median Absolute Error (cents)",
        "p95_abs_cents": "P95 Absolute Error (cents)",
        "gross_pitch_error_rate": "Gross Pitch Error Rate",
        "transition_vfa": "Transition VFA",
        "deep_rest_vfa": "Deep-rest VFA",
        "median_onset_acquisition_ms": "Median Onset Acquisition (ms)",
        "p95_onset_acquisition_ms": "P95 Onset Acquisition (ms)",
        "median_release_hangover_ms": "Median Release Hangover (ms)",
        "p95_release_hangover_ms": "P95 Release Hangover (ms)",
        "p95_false_alarm_burst_ms": "P95 False-alarm Burst (ms)",
        "median_output_latency_ms": "Median Output Latency (ms)",
        "p95_output_latency_ms": "P95 Output Latency (ms)",
        "deadline_miss_rate": "Deadline Miss Rate",
    }
    RESULT_COLUMNS: ClassVar[list[str]] = [
        "Track ID",
        *METRICS[3:],
        *METRICS[:3],
        REALTIME_COL,
        AUDIO_SECONDS_COL,
        COMPUTE_COL,
        WALL_COMPUTE_COL,
        COMPUTE_CLOCK_COL,
        "Split",
        "Track",
        "Piece Number",
        "Ensemble",
        "Ensemble Size",
        "Instrument",
        "Instrument Code",
        "Voice",
        "F0_voice",
        "From Cache",
        "Fmin",
        "Fmax",
        "Degradation",
        "Degradation Seed",
        "SNR (dB)",
        "SNR",
        "Source Track ID",
        "Execution Mode",
        "Stream Adapter",
        "Frame Size (samples)",
        "Integration Size (samples)",
        "Hop Size (samples)",
        "Model Hop Size (samples)",
        "Sample Rate (Hz)",
        "Update Interval (ms)",
        "Algorithmic Look-ahead (ms)",
        "Volume Floor (dBFS)",
        "Median Absolute Error (cents)",
        "P95 Absolute Error (cents)",
        "Gross Pitch Error Rate",
        "Transition VFA",
        "Deep-rest VFA",
        "Median Onset Acquisition (ms)",
        "P95 Onset Acquisition (ms)",
        "Median Release Hangover (ms)",
        "P95 Release Hangover (ms)",
        "P95 False-alarm Burst (ms)",
        "Median Output Latency (ms)",
        "P95 Output Latency (ms)",
        "Deadline Miss Rate",
    ]

    @dataclass(frozen=True)
    class Options:
        """Everything a spawned worker needs to rebuild datasets and detectors."""

        datasets: tuple[str, ...] = ("bach10-mf0-synth",)
        root: str | None = None
        split: str = "test"
        instruments: tuple[str, ...] = ()
        ensembles: tuple[str, ...] = ()
        max_tracks: int | None = None
        per_stratum: int | None = None
        per_instrument: int | None = None
        seed: int = 0
        materialize: bool = False
        f0_fps: float | None = None
        use_cache: bool = True
        confidence: float | None = None
        step_seconds: float = 0.01
        crepe_capacity: str = "full"
        rmvpe_checkpoint: str | None = None
        rmvpe_module: str | None = None
        attune_unv_thresh: float | None = None
        attune_min_volume: float | None = None
        verbose_algorithms: bool = False
        noise_snrs: tuple[float, ...] = ()
        quiet_runtime: bool = False
        force_reanalysis: bool = False

    @dataclass(frozen=True)
    class Job:
        """One method over a slice of tracks -- the unit of parallelism."""

        method: str
        first_index: int
        tracks: tuple[PitchTrack, ...]

        @property
        def key(self) -> str:
            if not self.tracks:
                return f"{self.method}:empty"
            first, last = (self.tracks[0], self.tracks[-1])
            return f"{self.method}:{first.dataset}:{first.track_id}:{last.dataset}:{last.track_id}:{len(self.tracks)}"

    @dataclass(frozen=True)
    class Outcome:
        """What a worker reports back for one job."""

        status: str
        method: str
        rows: list[dict[str, Any]] = field(default_factory=list)
        errors: list[tuple[str, str, str, str]] = field(default_factory=list)
        seconds: float = 0.0
        skip_reason: str | None = None

    class Progress:
        """Multi-worker terminal display or one notebook-friendly progress line."""

        SPINNER: ClassVar[str] = "|/-\\"

        def __init__(self, enabled: bool = True, compact: bool = False) -> None:
            self.enabled = enabled
            self.compact = enabled and compact
            self.is_tty = enabled and sys.stdout.isatty()
            self.active: dict[int, dict[str, Any]] = {}
            self.live_lines = 0
            self.last_render = 0.0
            self.transport_failed = False
            self.completed = 0
            self.compact_width = 0

        @staticmethod
        def event(
            channel: Any | None,
            kind: str,
            index: int,
            total: int,
            method: str,
            track_id: str,
            ok: bool | None = None,
        ) -> None:
            """Worker side: post one event, ignoring a torn-down queue."""
            if channel is None:
                return
            try:
                channel.put(
                    {
                        "event": kind,
                        "pid": os.getpid(),
                        "index": index,
                        "total": total,
                        "method": method,
                        "track_id": track_id,
                        "ok": ok,
                    }
                )
            except (BrokenPipeError, EOFError, OSError):
                return

        def drain(self, channel: Any | None) -> bool:
            if not self.enabled or channel is None:
                return False
            changed = False
            while True:
                try:
                    self._handle(channel.get_nowait())
                    changed = True
                except queue_module.Empty:
                    break
                except (BrokenPipeError, EOFError, OSError):
                    self.transport_failed = True
                    self.enabled = False
                    self.clear()
                    return changed
            if changed or (self.is_tty and self.active):
                self.render()
            return changed

        def _handle(self, event: dict[str, Any]) -> None:
            pid = int(event.get("pid", 0) or 0)
            item = {
                "index": int(event.get("index", 0) or 0),
                "total": int(event.get("total", 0) or 0),
                "method": str(event.get("method", "")),
                "track_id": str(event.get("track_id", "")),
            }
            if event.get("event") == "start":
                self.active[pid] = item
            elif event.get("event") == "done":
                item = self.active.pop(pid, None) or item
                if self.compact:
                    self.update_compact(item["total"], item["method"], item["track_id"])
                    return
                self.clear()
                print(self._line(item, "✓" if event.get("ok") else "X"), flush=True)

        def update_compact(
            self, total: int, method: str, track_id: str, count: int = 1
        ) -> None:
            """Advance the same plain-text counter used by preliminary suites."""
            if not self.enabled:
                return
            self.completed += count
            now = time.monotonic()
            if (
                not self.is_tty
                and count == 0
                and self.compact_width
                and (self.completed == getattr(self, "last_compact_completed", None))
                and (now - self.last_render < 30)
            ):
                return
            self.last_render = now
            self.last_compact_completed = self.completed
            message = f"[{self.completed:>4}/{total}] {method}: {track_id}"
            self.compact_width = max(self.compact_width, len(message))
            print(f"\r{message:<{self.compact_width}}", end="", flush=True)

        def render(self) -> None:
            if self.compact or not self.is_tty:
                return
            now = time.perf_counter()
            if now - self.last_render < 0.08 and self.live_lines:
                return
            self.clear()
            spinner = self.SPINNER[int(now * 10) % len(self.SPINNER)]
            lines = [self._line(item, spinner) for item in self.active.values()]
            if lines:
                sys.stdout.write("\n".join(lines) + "\n")
                sys.stdout.flush()
            self.live_lines = len(lines)
            self.last_render = now

        def clear(self) -> None:
            if self.compact:
                return
            if not self.is_tty or self.live_lines <= 0:
                self.live_lines = 0
                return
            for _ in range(self.live_lines):
                sys.stdout.write("\x1b[1A\x1b[2K")
            sys.stdout.flush()
            self.live_lines = 0

        def finish(self) -> None:
            """End a compact line, or clear the multi-line terminal display."""
            if self.compact:
                if self.compact_width:
                    print()
                    self.compact_width = 0
                return
            self.clear()

        @staticmethod
        def _line(item: dict[str, Any], status: str) -> str:
            line = f"[{item.get('index', 0)}/{item.get('total', 0)}] {item.get('method', '')}: {item.get('track_id', '')} [{status}]"
            width = shutil.get_terminal_size((120, 20)).columns
            return line[: width - 1] if width > 20 and len(line) > width else line

    @staticmethod
    def available_detectors() -> list[PitchDetectorBase]:
        from benchmarks.modules.pitch.competitors.Attune import Attune
        from benchmarks.modules.pitch.competitors.BasicPitch import BasicPitch
        from benchmarks.modules.pitch.competitors.Crepe import Crepe, TorchCrepe
        from benchmarks.modules.pitch.competitors.Penn import Penn
        from benchmarks.modules.pitch.competitors.Praat import Praat
        from benchmarks.modules.pitch.competitors.Rmvpe import Rmvpe
        from benchmarks.modules.pitch.competitors.Spice import Spice
        from benchmarks.modules.pitch.competitors.SwiftF0 import SwiftF0
        from benchmarks.modules.pitch.competitors.PYIN import PYIN

        return [
            PYIN(),
            Attune(),
            BasicPitch(),
            Praat(),
            SwiftF0(),
            Crepe(),
            TorchCrepe(),
            Spice(),
            Penn(),
            Rmvpe(),
        ]

    @classmethod
    def default_detectors(cls) -> list[PitchDetectorBase]:
        """Omit the heavy optional PyTorch models; --methods still reaches them."""
        heavy = {"torchcrepe", "penn"}
        return [d for d in cls.available_detectors() if d.name not in heavy]

    @classmethod
    def detector_for(cls, method: str, options: "PitchBenchmarker.Options"):
        by_name = {d.name: type(d) for d in cls.available_detectors()}
        if method not in by_name:
            raise KeyError(f"unknown method {method!r}")
        kwargs: dict[str, Any] = {"step_seconds": options.step_seconds}
        if options.confidence is not None:
            kwargs["confidence"] = options.confidence
        if method in ("crepe", "torchcrepe"):
            kwargs["model_capacity"] = options.crepe_capacity
        if method == "rmvpe":
            kwargs["checkpoint"] = options.rmvpe_checkpoint
            kwargs["module"] = options.rmvpe_module
        detector = by_name[method](**kwargs)
        if method.startswith("attune") and hasattr(detector, "config_overrides"):
            if options.attune_unv_thresh is not None:
                detector.config_overrides["unv_thresh"] = options.attune_unv_thresh
            if options.attune_min_volume is not None:
                detector.config_overrides["min_volume"] = options.attune_min_volume
            detector.algorithm_verbose = options.verbose_algorithms
        return detector

    @classmethod
    def dataset_for(cls, key: str, options: "PitchBenchmarker.Options") -> PitchDataset:
        if key == CocoChorales.name:
            return CocoChorales(
                root=options.root,
                f0_fps=options.f0_fps,
                split=options.split,
                per_stratum=options.per_stratum,
                per_instrument=options.per_instrument,
                seed=options.seed,
                max_tracks=options.max_tracks,
                ensembles=options.ensembles,
                instruments=options.instruments,
                materialize=options.materialize,
                noise_snrs=options.noise_snrs,
            )
        if key == Bach10.name:
            return Bach10(
                instruments=options.instruments,
                max_tracks=options.max_tracks,
                per_instrument=options.per_instrument,
                seed=options.seed,
            )
        if key == URMP.name:
            return URMP(
                instruments=options.instruments,
                ensembles=options.ensembles,
                max_tracks=options.max_tracks,
                per_instrument=options.per_instrument,
                seed=options.seed,
            )
        return AudioAnnot(
            key,
            instruments=list(options.instruments),
            max_tracks=options.max_tracks,
            per_instrument=options.per_instrument,
            seed=options.seed,
        )

    def __init__(self, options: "PitchBenchmarker.Options | None" = None) -> None:
        self.options = options or self.Options()
        self.results_dir = RESULTS_ROOT / "pitch"
        self._datasets: dict[str, PitchDataset] = {}

    def dataset(self, key: str) -> PitchDataset:
        """Datasets are cached per key so a manifest is only read once per process."""
        if key not in self._datasets:
            self._datasets[key] = self.dataset_for(key, self.options)
        return self._datasets[key]

    def tracks(self) -> list[PitchTrack]:
        found = [
            track
            for key in self.options.datasets
            for track in self.dataset(key).tracks()
        ]
        return sorted(found, key=lambda track: (track.dataset, track.track_id))

    def is_cached(self, method: str, track: PitchTrack) -> bool:
        detector = self.detector_for(method, self.options)
        return detector.has_cache(self.dataset(track.dataset), track)

    def plan(
        self,
        methods: Sequence[str],
        tracks: Sequence[PitchTrack],
        workers: int,
        tracks_per_job: int = 0,
        skip_cached: bool = False,
        cache_only: bool = False,
    ) -> tuple[list["PitchBenchmarker.Job"], dict[str, dict[str, int]]]:
        """Split the work into per-method jobs, honouring what is already cached."""
        jobs: list[PitchBenchmarker.Job] = []
        counts: dict[str, dict[str, int]] = {}
        index = 1
        for method in methods:
            cached = [track for track in tracks if self.is_cached(method, track)]
            if cache_only:
                selected = list(cached)
            elif skip_cached:
                selected = [t for t in tracks if t not in cached]
            else:
                selected = list(tracks)
            counts[method] = {
                "total": len(tracks),
                "cached": len(cached),
                "selected": len(selected),
            }
            size = (
                tracks_per_job
                if tracks_per_job > 0
                else max(1, math.ceil(len(selected) / max(1, workers)))
            )
            for start in range(0, len(selected), size):
                jobs.append(
                    self.Job(
                        method, index + start, tuple(selected[start : start + size])
                    )
                )
            index += len(selected)
        return (jobs, counts)

    def score(
        self, method: str, example: PitchExample, estimate: PitchEstimate
    ) -> dict[str, Any]:
        """One CSV row: mir_eval melody metrics plus timing and track metadata."""
        from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase

        metrics = PitchBenchmarker.score_frames(
            example.ref_times, example.ref_freqs, estimate.times, estimate.freqs
        )
        seconds = example.audio_seconds
        compute = float(estimate.compute_seconds)
        compute_clock = str(
            estimate.metadata.get(
                "compute_clock", PitchDetectorBase.LEGACY_COMPUTE_CLOCK
            )
        )
        wall_compute = float(estimate.metadata.get("wall_pitch_compute_time", compute))
        return {
            **metrics,
            **{
                key: value
                for key, value in estimate.metadata.items()
                if isinstance(value, (int, float))
            },
            "pitch_compute_time": compute,
            "wall_pitch_compute_time": wall_compute,
            "compute_clock": compute_clock,
            "from_cache": bool(estimate.from_cache),
            "fmin": float(example.fmin),
            "fmax": float(example.fmax),
            "model": method,
            "execution_mode": "Offline",
            "source_piece": str(
                example.metadata.get("source_piece")
                or f"{example.dataset}:{example.metadata.get('track', example.track_id)}"
            ),
            "dataset": example.dataset,
            "track_id": example.track_id,
            "result_label": example.metadata.get("result_label", example.dataset),
            "audio_seconds": seconds,
            "realtime_factor": (
                seconds / compute
                if compute_clock == PitchDetectorBase.COMPUTE_CLOCK
                and compute > 0
                and seconds
                else float("nan")
            ),
            **{
                key: value
                for key, value in example.metadata.items()
                if key != "result_label"
            },
        }

    @staticmethod
    def _worker_options_payload(options: "PitchBenchmarker.Options") -> dict[str, Any]:
        """Plain-data boundary resilient to IPython ``autoreload``.

        A nested dataclass instance created before IPython reloads this module
        is no longer identical to the class reachable by its pickle name.
        Workers only need its field values, so never send that class instance
        through the multiprocessing queue.
        """
        return dict(vars(options))

    @staticmethod
    def _worker_job_payload(job: "PitchBenchmarker.Job") -> dict[str, Any]:
        tracks = []
        for track in job.tracks:
            degradation = track.degradation
            tracks.append(
                {
                    "track_id": track.track_id,
                    "dataset": track.dataset,
                    "audio_path": str(track.audio_path),
                    "annot_path": str(track.annot_path),
                    "metadata": dict(track.metadata),
                    "degradation": (
                        None
                        if degradation is None
                        else {
                            "snr_db": float(degradation.snr_db),
                            "seed": int(degradation.seed),
                        }
                    ),
                }
            )
        return {
            "method": job.method,
            "first_index": int(job.first_index),
            "tracks": tracks,
        }

    @classmethod
    def _worker_job_from_payload(
        cls, payload: dict[str, Any]
    ) -> "PitchBenchmarker.Job":
        from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase

        tracks = []
        for item in payload["tracks"]:
            degradation = item.get("degradation")
            tracks.append(
                PitchTrack(
                    track_id=str(item["track_id"]),
                    dataset=str(item["dataset"]),
                    audio_path=Path(item["audio_path"]),
                    annot_path=Path(item["annot_path"]),
                    metadata=dict(item.get("metadata") or {}),
                    degradation=(
                        None
                        if degradation is None
                        else PitchDetectorBase.AudioDegradation(**degradation)
                    ),
                )
            )
        return cls.Job(
            method=str(payload["method"]),
            first_index=int(payload["first_index"]),
            tracks=tuple(tracks),
        )

    @staticmethod
    @contextlib.contextmanager
    def silence_runtime(enabled: bool = True):
        """Suppress third-party warnings/logging while retaining scored errors.

        Python redirection catches ordinary library chatter; temporarily
        redirecting file descriptors 1/2 also catches native TensorFlow/TFLite
        messages. Each benchmark worker is single-threaded, so the scoped fd
        redirection cannot swallow output from another job.
        """
        if not enabled:
            yield
            return
        import logging

        previous_logging_disable = logging.root.manager.disable
        saved_fds: list[tuple[int, int]] = []
        devnull_fd: int | None = None
        try:
            for stream in (sys.stdout, sys.stderr):
                try:
                    stream.flush()
                except Exception:
                    pass
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            for fd in (1, 2):
                try:
                    saved_fds.append((fd, os.dup(fd)))
                    os.dup2(devnull_fd, fd)
                except OSError:
                    pass
            logging.disable(logging.CRITICAL)
            with warnings.catch_warnings(), open(
                os.devnull, "w", encoding="utf-8"
            ) as devnull, contextlib.redirect_stdout(
                devnull
            ), contextlib.redirect_stderr(
                devnull
            ):
                warnings.simplefilter("ignore")
                yield
        finally:
            logging.disable(previous_logging_disable)
            for fd, saved_fd in saved_fds:
                try:
                    os.dup2(saved_fd, fd)
                finally:
                    os.close(saved_fd)
            if devnull_fd is not None:
                os.close(devnull_fd)

    @staticmethod
    def run_job(
        options: "PitchBenchmarker.Options | dict[str, Any]",
        job: "PitchBenchmarker.Job | dict[str, Any]",
        verbose: bool = False,
        channel: Any = None,
        total: int = 0,
        cache_only: bool = False,
    ) -> "PitchBenchmarker.Outcome":
        """Worker entry point: score one method over one slice of tracks."""
        from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase

        if isinstance(options, dict):
            options = PitchBenchmarker.Options(**options)
        if isinstance(job, dict):
            job = PitchBenchmarker._worker_job_from_payload(job)
        started = time.perf_counter()
        benchmarker = PitchBenchmarker(options)
        rows: list[dict[str, Any]] = []
        errors: list[tuple[str, str, str, str]] = []
        try:
            with benchmarker.silence_runtime(options.quiet_runtime):
                detector = benchmarker.detector_for(job.method, options)
                detector.ensure_available()
        except PitchDetectorBase.Unavailable as exc:
            return PitchBenchmarker.Outcome(
                "skip",
                job.method,
                seconds=time.perf_counter() - started,
                skip_reason=str(exc),
            )
        except Exception as exc:
            trace = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
            return PitchBenchmarker.Outcome(
                "err",
                job.method,
                errors=[(job.method, "", "", trace)],
                seconds=time.perf_counter() - started,
            )
        for offset, track in enumerate(job.tracks):
            index = job.first_index + offset
            PitchBenchmarker.Progress.event(
                channel, "start", index, total, job.method, track.track_id
            )
            try:
                with benchmarker.silence_runtime(options.quiet_runtime):
                    dataset = benchmarker.dataset(track.dataset)
                    if cache_only and (not detector.has_cache(dataset, track)):
                        PitchBenchmarker.Progress.event(
                            channel,
                            "done",
                            index,
                            total,
                            job.method,
                            track.track_id,
                            ok=False,
                        )
                        continue
                    example = dataset.example(track)
                    estimate = detector.estimate(
                        example,
                        use_cache=options.use_cache or cache_only,
                        force_reanalysis=options.force_reanalysis and (not cache_only),
                    )
                    rows.append(benchmarker.score(job.method, example, estimate))
                PitchBenchmarker.Progress.event(
                    channel, "done", index, total, job.method, track.track_id, ok=True
                )
                if verbose and channel is None:
                    row = rows[-1]
                    print(
                        f"[{job.method}] {offset + 1:>4}/{len(job.tracks)} {track.dataset:18s} {track.track_id[:36]:36s} RPA={row['Raw Pitch Accuracy']:.3f} OA={row['Overall Accuracy']:.3f} {row['realtime_factor']:.0f}xRT",
                        flush=True,
                    )
            except PitchDetectorBase.Unavailable as exc:
                PitchBenchmarker.Progress.event(
                    channel, "done", index, total, job.method, track.track_id, ok=False
                )
                return PitchBenchmarker.Outcome(
                    "skip",
                    job.method,
                    rows,
                    errors,
                    time.perf_counter() - started,
                    str(exc),
                )
            except Exception as exc:
                PitchBenchmarker.Progress.event(
                    channel, "done", index, total, job.method, track.track_id, ok=False
                )
                trace = "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                )
                errors.append((job.method, track.dataset, track.track_id, trace))
                if not options.quiet_runtime:
                    print(
                        f"[{job.method}] {track.dataset} / {track.track_id} ERROR: {exc!r}",
                        file=sys.stderr,
                        flush=True,
                    )
        return PitchBenchmarker.Outcome(
            "ok", job.method, rows, errors, time.perf_counter() - started
        )

    def run(
        self,
        jobs: Sequence["PitchBenchmarker.Job"],
        workers: int = 1,
        batch_size: int = 0,
        watchdog: float = 1200.0,
        max_attempts: int = 2,
        verbose: bool = True,
        progress: bool = True,
        compact_progress: bool = False,
        cache_only: bool = False,
    ) -> tuple[pd.DataFrame, list[tuple[str, str, str, str]], dict[str, str]]:
        """Drive the jobs, one fresh pool per batch, re-queueing anything stuck."""
        rows: list[dict[str, Any]] = []
        errors: list[tuple[str, str, str, str]] = []
        skipped: dict[str, str] = {}
        if workers == 1:
            total = sum((len(job.tracks) for job in jobs))
            for job in jobs:
                outcome = self.run_job(
                    self.options, job, verbose, None, total, cache_only
                )
                rows.extend(outcome.rows)
                errors.extend(outcome.errors)
                if outcome.status == "skip":
                    skipped[outcome.method] = (
                        outcome.skip_reason or "dependency unavailable"
                    )
                    print(f"[{outcome.method}] SKIPPED -- {skipped[outcome.method]}")
                elif outcome.status == "err":
                    for *_, trace in outcome.errors:
                        print(trace.rstrip(), file=sys.stderr, flush=True)
            return (pd.DataFrame(rows), errors, skipped)
        batch_size = batch_size if batch_size > 0 else max(1, workers * 2)
        attempts: dict[str, int] = {}
        pending_jobs = list(jobs)
        total = sum((len(job.tracks) for job in jobs))
        completed = 0
        started = time.perf_counter()
        manager = multiprocessing.Manager() if progress else None
        channel = manager.Queue() if manager is not None else None
        renderer = self.Progress(enabled=progress, compact=compact_progress)

        def log(tag: str, method: str, count: int, seconds: float) -> None:
            nonlocal completed
            completed += count
            elapsed = time.perf_counter() - started
            rate = completed / elapsed if elapsed else 0.0
            eta = (total - completed) / rate if rate else 0.0
            print(
                f"[{completed:>4}/{total}] {tag:7s} {method:14s} {count:>4} track(s) in {self.duration(seconds)} | elapsed {self.duration(elapsed)} eta {self.duration(eta)}",
                flush=True,
            )

        try:
            while pending_jobs:
                batch, pending_jobs = (
                    pending_jobs[:batch_size],
                    pending_jobs[batch_size:],
                )
                runnable = []
                for job in batch:
                    attempts[job.key] = attempts.get(job.key, 0) + 1
                    if attempts[job.key] > max_attempts:
                        note = f"gave up after {max_attempts} attempts (kept hanging)"
                        errors.extend(
                            (
                                (job.method, t.dataset, t.track_id, note)
                                for t in job.tracks
                            )
                        )
                        renderer.finish()
                        log("GIVEUP", job.method, len(job.tracks), 0.0)
                    else:
                        runnable.append(job)
                if not runnable:
                    continue
                batch_workers = min(
                    len(runnable), PitchBenchmarker.memory_limited_workers(workers)
                )
                print(
                    f"Pitch batch: {batch_workers}/{workers} worker(s) within the RAM budget",
                    flush=True,
                )
                pool = ProcessPoolExecutor(
                    max_workers=batch_workers,
                    mp_context=multiprocessing.get_context("spawn"),
                )
                options_payload = self._worker_options_payload(self.options)
                futures = {
                    pool.submit(
                        PitchBenchmarker.run_job,
                        options_payload,
                        self._worker_job_payload(job),
                        verbose,
                        channel,
                        total,
                        cache_only,
                    ): job
                    for job in runnable
                }
                try:
                    waiting = set(futures)
                    last_activity = time.perf_counter()
                    while waiting:
                        done, waiting = wait(
                            waiting,
                            timeout=0.1 if channel is not None else watchdog,
                            return_when=FIRST_COMPLETED,
                        )
                        if renderer.drain(channel):
                            last_activity = time.perf_counter()
                        if renderer.transport_failed:
                            channel = None
                        if not done:
                            if time.perf_counter() - last_activity < watchdog:
                                continue
                            stuck = [futures[f] for f in waiting]
                            renderer.finish()
                            print(
                                f"\n!! watchdog: no progress in {self.duration(watchdog)} -- re-queueing {len(stuck)} in-flight job(s).",
                                file=sys.stderr,
                                flush=True,
                            )
                            pending_jobs.extend(stuck)
                            break
                        last_activity = time.perf_counter()
                        broke = False
                        for future in done:
                            job = futures[future]
                            try:
                                outcome = future.result()
                            except BrokenProcessPool:
                                broke = True
                                break
                            rows.extend(outcome.rows)
                            errors.extend(outcome.errors)
                            if outcome.status == "skip":
                                skipped[outcome.method] = (
                                    outcome.skip_reason or "dependency unavailable"
                                )
                                renderer.finish()
                                log(
                                    "SKIP",
                                    outcome.method,
                                    len(job.tracks),
                                    outcome.seconds,
                                )
                                print(
                                    f"[{outcome.method}] SKIPPED -- {skipped[outcome.method]}",
                                    flush=True,
                                )
                            elif outcome.status == "ok":
                                if not progress:
                                    log(
                                        "OK",
                                        job.method,
                                        len(outcome.rows) + len(outcome.errors),
                                        outcome.seconds,
                                    )
                            else:
                                renderer.finish()
                                log("ERR", job.method, len(job.tracks), outcome.seconds)
                                for *_, trace in outcome.errors:
                                    print(trace.rstrip(), file=sys.stderr, flush=True)
                        if broke:
                            stuck = [job, *[futures[f] for f in waiting]]
                            renderer.finish()
                            print(
                                f"\n!! pool broke (a worker died) -- re-queueing {len(stuck)} unfinished job(s).",
                                file=sys.stderr,
                                flush=True,
                            )
                            pending_jobs.extend(stuck)
                            break
                finally:
                    renderer.drain(channel)
                    renderer.clear()
                    self._teardown(pool)
        finally:
            renderer.finish()
            if manager is not None:
                try:
                    manager.shutdown()
                except (BrokenPipeError, EOFError, OSError):
                    pass
        return (pd.DataFrame(rows), errors, skipped)

    @staticmethod
    def _teardown(pool: ProcessPoolExecutor) -> None:
        children = list((getattr(pool, "_processes", None) or {}).values())
        pool.shutdown(wait=False, cancel_futures=True)
        for child in children:
            if child.is_alive():
                child.terminate()
        for child in children:
            child.join(timeout=10)
            if child.is_alive():
                child.kill()

    @staticmethod
    def duration(seconds: float) -> str:
        seconds = int(seconds)
        hours, rest = divmod(seconds, 3600)
        minutes, secs = divmod(rest, 60)
        return (
            f"{hours:d}h{minutes:02d}m{secs:02d}s"
            if hours
            else f"{minutes:d}m{secs:02d}s"
        )

    @classmethod
    def display(cls, frame: pd.DataFrame) -> pd.DataFrame:
        """Rename and order the columns the way the published CSVs read."""
        out = frame.rename(
            columns={
                source: target
                for source, target in cls.DISPLAY_NAMES.items()
                if source in frame.columns and target not in frame.columns
            }
        ).drop(
            columns=[
                column
                for column in (
                    "pitch_detector_compute_time",
                    "pitch_smoother_compute_time",
                    "wall_pitch_detector_compute_time",
                    "wall_pitch_smoother_compute_time",
                    "result_label",
                )
                if column in frame.columns
            ],
            errors="ignore",
        )
        ordered = [column for column in cls.RESULT_COLUMNS if column in out.columns]
        return out[ordered + [c for c in out.columns if c not in ordered]]

    def write_raw_outputs(self, raw: pd.DataFrame) -> list[Path]:
        """One CSV per (method, result label), e.g. coco_violin or bach10-mf0-synth."""
        written: list[Path] = []
        for (method, label), rows in raw.groupby(["model", "result_label"], sort=True):
            out_path = self.results_dir / "raw_outputs" / str(method) / f"{label}.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            table = self.display(
                rows.drop(columns=["model", "dataset"], errors="ignore")
            )
            table = (
                table.set_index("Track ID") if "Track ID" in table.columns else table
            )
            table.to_csv(out_path)
            written.append(out_path)
            print(f"wrote {len(table)} rows -> {out_path}")
        return written

    def summarize(self, raw: pd.DataFrame, method_order: Sequence[str]) -> pd.DataFrame:
        """Per-method frame-pooled metrics, with throughput as sum(audio)/sum(compute)."""
        from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase

        if raw.empty or "model" not in raw.columns:
            return pd.DataFrame(columns=["Tracks", *self.METRICS, self.REALTIME_COL])
        table_source = self.display(raw)
        grouped = table_source.groupby("model", sort=False)
        table = PitchBenchmarker.pool_scores(table_source)
        table.insert(0, "Tracks", grouped.size())
        timed_source = table_source
        if self.COMPUTE_CLOCK_COL in timed_source.columns:
            timed_source = timed_source.loc[
                timed_source[self.COMPUTE_CLOCK_COL] == PitchDetectorBase.COMPUTE_CLOCK
            ]
        if {self.AUDIO_SECONDS_COL, self.COMPUTE_COL}.issubset(timed_source.columns):
            totals = timed_source.groupby("model", sort=False)[
                [self.AUDIO_SECONDS_COL, self.COMPUTE_COL]
            ].sum(numeric_only=True)
            table[self.REALTIME_COL] = (
                totals[self.AUDIO_SECONDS_COL] / totals[self.COMPUTE_COL]
            ).where(totals[self.COMPUTE_COL] > 0)
        ordered = [m for m in method_order if m in table.index]
        table = table.reindex(ordered + [m for m in table.index if m not in ordered])
        table.index.name = "model"
        return table

    @staticmethod
    def snr_label(snr: float) -> str:
        return "clean" if snr == math.inf else f"{snr:g}dB"

    def summarize_degradation(
        self, raw: pd.DataFrame, method_order: Sequence[str], snr_order: Sequence[float]
    ) -> pd.DataFrame:
        """Per-method, per-SNR frame-pooled metrics for a degradation run."""
        from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase

        if raw.empty or not {"model", "snr_label"}.issubset(raw.columns):
            return pd.DataFrame(columns=["Tracks", *self.METRICS, self.REALTIME_COL])
        grouped = raw.groupby(["model", "snr_label"], sort=False)
        table = PitchBenchmarker.pool_scores(raw, ["model", "snr_label"])
        table.insert(0, "Tracks", grouped.size())
        timed_source = raw
        if "compute_clock" in timed_source.columns:
            timed_source = timed_source.loc[
                timed_source["compute_clock"] == PitchDetectorBase.COMPUTE_CLOCK
            ]
        if {"audio_seconds", "pitch_compute_time"}.issubset(timed_source.columns):
            totals = timed_source.groupby(["model", "snr_label"], sort=False)[
                ["audio_seconds", "pitch_compute_time"]
            ].sum(numeric_only=True)
            table[self.REALTIME_COL] = (
                totals["audio_seconds"] / totals["pitch_compute_time"]
            ).where(totals["pitch_compute_time"] > 0)
        requested = [
            (method, self.snr_label(snr))
            for method in method_order
            for snr in snr_order
        ]
        order = [key for key in requested if key in table.index]
        order.extend((key for key in table.index if key not in order))
        table = table.reindex(order)
        table.index.names = ("model", "SNR")
        return table

    def write_summary(self, table: pd.DataFrame) -> Path:
        """Merge this run's methods into the shared comparison CSV, keeping the rest."""
        out_path = self.results_dir / "pitch_frame_weighted_benchmarks.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged = table
        if out_path.exists():
            existing = pd.read_csv(out_path)
            if "model" in existing.columns:
                existing = existing.set_index("model")
            elif len(existing.columns):
                existing = existing.rename(
                    columns={existing.columns[0]: "model"}
                ).set_index("model")
            existing.index = existing.index.astype(str)
            existing = existing[~existing.index.duplicated(keep="last")]
            merged = existing.reindex(
                columns=list(table.columns)
                + [c for c in existing.columns if c not in table.columns]
            )
            for model, row in table.iterrows():
                merged.loc[model, table.columns] = row.to_numpy()
        merged.index.name = "model"
        merged.to_csv(out_path)
        print(f"\nwrote summary -> {out_path}")
        return out_path

    def write_degradation_summary(self, table: pd.DataFrame) -> Path:
        """Merge condition rows without overwriting the clean leaderboard."""
        out_path = self.results_dir / "pitch_frame_weighted_degradation_benchmarks.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged = table
        if out_path.exists():
            existing = pd.read_csv(out_path)
            if {"model", "SNR"}.issubset(existing.columns):
                existing = existing.set_index(["model", "SNR"])
                merged = pd.concat([existing, table])
                merged = merged[~merged.index.duplicated(keep="last")]
        merged.to_csv(out_path)
        print(f"\nwrote degradation summary -> {out_path}")
        return out_path

    @staticmethod
    def parse_noise_snrs(value: str) -> tuple[float, ...]:
        """Parse ``clean,20,10`` into ordered, unique SNR conditions."""
        parsed: list[float] = []
        for token in value.split(","):
            token = token.strip().lower()
            if not token:
                continue
            snr = (
                math.inf if token in {"clean", "inf", "+inf", "none"} else float(token)
            )
            if math.isnan(snr) or snr == -math.inf:
                raise argparse.ArgumentTypeError(
                    "noise SNRs must be finite numbers or 'clean'"
                )
            if snr not in parsed:
                parsed.append(snr)
        if not parsed:
            raise argparse.ArgumentTypeError("provide at least one noise SNR")
        return tuple(parsed)

    @classmethod
    def parse_args(cls, argv: Sequence[str] | None = None) -> argparse.Namespace:
        from algorithms.Config import (
            PYIN_DEFAULT_MIN_VOLUME,
            PYIN_DEFAULT_UNV_THRESH,
            PYIN_PRAAT_MIRROR_UNV_THRESH,
        )

        names = [detector.name for detector in cls.available_detectors()]
        parser = argparse.ArgumentParser(
            description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
        )
        parser.add_argument(
            "--datasets",
            nargs="+",
            default=list(AudioAnnot.NAMES),
            help=f"corpora to run: {', '.join(AudioAnnot.NAMES)}, {Bach10.name}, {URMP.name}, {CocoChorales.name}",
        )
        parser.add_argument(
            "--methods",
            nargs="+",
            default=None,
            help=f"default omits torchcrepe and penn. choices: {', '.join(names)}",
        )
        parser.add_argument("--include-slow-methods", action="store_true")
        parser.add_argument("--instrument", action="append", dest="instruments")
        parser.add_argument("--ensemble", action="append", dest="ensembles")
        parser.add_argument("--max-tracks", type=int, default=None)
        parser.add_argument("--split", default="test", help="CocoChorales split")
        sampling = parser.add_mutually_exclusive_group()
        sampling.add_argument("--per-stratum", type=int, default=None)
        sampling.add_argument(
            "--per-instrument",
            type=int,
            default=None,
            help="seed-select this many tracks for every instrument label",
        )
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--root", default=None, help="CocoChorales root override")
        parser.add_argument("--f0-fps", type=float, default=None)
        parser.add_argument(
            "--materialize",
            action="store_true",
            help="CocoChorales: extract selected stems before benchmarking",
        )
        parser.add_argument(
            "--noise-snrs",
            type=cls.parse_noise_snrs,
            default=(),
            metavar="CLEAN,20,15,10,5",
            help="CocoChorales: benchmark deterministic white-noise degradation at these voiced-frame SNRs; include 'clean' for the baseline",
        )
        parser.add_argument(
            "--workers",
            type=int,
            default=PitchBenchmarker.default_pitch_workers(),
            help="maximum detector processes (default: CPUs minus 2; also limited by available RAM)",
        )
        parser.add_argument("--tracks-per-job", type=int, default=0)
        parser.add_argument("--batch-size", type=int, default=0)
        parser.add_argument("--watchdog", type=float, default=1200.0)
        parser.add_argument("--max-attempts", type=int, default=2)
        parser.add_argument("--no-cache", action="store_true")
        parser.add_argument("--skip-cached", action="store_true")
        parser.add_argument(
            "--cache-only",
            action="store_true",
            help="re-score existing caches without running any detector",
        )
        parser.add_argument("--confidence", type=float, default=None)
        parser.add_argument("--step", type=float, default=0.01, dest="step_seconds")
        parser.add_argument(
            "--crepe-capacity",
            default="full",
            choices=["tiny", "small", "medium", "large", "full"],
        )
        parser.add_argument("--rmvpe-checkpoint", default=None)
        parser.add_argument("--rmvpe-module", default=None)
        parser.add_argument(
            "--attune-unv-thresh",
            type=float,
            default=None,
            help=f"Attune causal unvoiced-probability threshold (default {PYIN_DEFAULT_UNV_THRESH:.3f}); Praat's voicing_threshold=0.45 corresponds to {PYIN_PRAAT_MIRROR_UNV_THRESH:.2f}",
        )
        parser.add_argument(
            "--mirror-praat-voicing",
            action="store_true",
            help="shorthand for the pYIN analogue of Praat's voicing threshold",
        )
        parser.add_argument(
            "--attune-min-volume",
            type=float,
            default=None,
            help=f"causal running-peak RMS ratio (default {PYIN_DEFAULT_MIN_VOLUME:.2f}; 0 disables)",
        )
        parser.add_argument("--write", action="store_true", help="write CSV reports")
        parser.add_argument("--quiet-tracks", action="store_true")
        parser.add_argument(
            "--quiet-runtime",
            action="store_true",
            help="suppress third-party warnings/logging while retaining scored errors",
        )
        parser.add_argument("--no-progress", action="store_true")
        parser.add_argument(
            "--compact-progress",
            action="store_true",
            help="overwrite one progress line, matching notebook output",
        )
        parser.add_argument("--algorithm-verbose", action="store_true")
        parser.add_argument(
            "--list",
            action="store_true",
            help="print the plan plus which method deps are installed, then exit",
        )
        parser.add_argument("--dry-run", action="store_true")
        return parser.parse_args(argv)

    @classmethod
    def options_from(cls, args: argparse.Namespace) -> "PitchBenchmarker.Options":
        from algorithms.Config import PYIN_PRAAT_MIRROR_UNV_THRESH

        unv_thresh = args.attune_unv_thresh
        if args.mirror_praat_voicing and unv_thresh is None:
            unv_thresh = PYIN_PRAAT_MIRROR_UNV_THRESH
        return cls.Options(
            datasets=tuple(args.datasets),
            root=args.root,
            split=args.split,
            instruments=tuple(args.instruments or ()),
            ensembles=tuple(args.ensembles or ()),
            max_tracks=args.max_tracks,
            per_stratum=args.per_stratum,
            per_instrument=args.per_instrument,
            seed=args.seed,
            materialize=args.materialize and (not args.dry_run),
            f0_fps=args.f0_fps,
            use_cache=not args.no_cache,
            confidence=args.confidence,
            step_seconds=args.step_seconds,
            crepe_capacity=args.crepe_capacity,
            rmvpe_checkpoint=args.rmvpe_checkpoint,
            rmvpe_module=args.rmvpe_module,
            attune_unv_thresh=unv_thresh,
            attune_min_volume=args.attune_min_volume,
            verbose_algorithms=args.algorithm_verbose,
            noise_snrs=tuple(args.noise_snrs),
            quiet_runtime=args.quiet_runtime,
        )

    @classmethod
    def main(cls, argv: Sequence[str] | None = None) -> int:
        from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase

        args = cls.parse_args(argv)
        available = {d.name for d in cls.available_detectors()}
        methods = list(
            args.methods
            or (
                [d.name for d in cls.available_detectors()]
                if args.include_slow_methods
                else [d.name for d in cls.default_detectors()]
            )
        )
        unknown = [method for method in methods if method not in available]
        if unknown:
            raise SystemExit(
                f"unknown methods: {unknown}; choices: {sorted(available)}"
            )
        benchmarker = cls(cls.options_from(args))
        tracks = benchmarker.tracks()
        jobs, counts = benchmarker.plan(
            methods,
            tracks,
            workers=args.workers,
            tracks_per_job=args.tracks_per_job,
            skip_cached=args.skip_cached,
            cache_only=args.cache_only,
        )
        selected = sum((count["selected"] for count in counts.values()))
        print(f"datasets:   {', '.join(args.datasets)}")
        print(f"methods:    {', '.join(methods)}")
        print(
            f"tracks:     {len(tracks)} corpus tracks | {len(methods) * len(tracks)} pairs"
        )
        if args.noise_snrs:
            conditions = ", ".join(
                (
                    "clean" if snr == math.inf else f"{snr:g}dB"
                    for snr in args.noise_snrs
                )
            )
            print(f"noise:      {conditions} (same seeded draw for every method)")
        print(
            f"workers:    {args.workers}  (BLAS threads capped to 1/worker; timing=process CPU)"
        )
        print(f"jobs:       {len(jobs)} | watchdog={cls.duration(args.watchdog)}")
        for method in methods:
            count = counts[method]
            print(
                f"  - {method:16s} {count['cached']:>4}/{count['total']:<4} cached | {count['selected']:>4} selected"
            )
        if args.list:
            print("\nmethod availability:")
            for method in methods:
                detector = cls.detector_for(method, benchmarker.options)
                try:
                    detector.ensure_available()
                    print(f"  {method:16s} OK")
                except PitchDetectorBase.Unavailable as exc:
                    print(f"  {method:16s} MISSING -- {str(exc).splitlines()[0]}")
            for track in tracks[:5]:
                print(f"  {track.track_id}  <-  {track.audio_path}")
            return 0
        if args.dry_run:
            print(f"\n[dry-run] would process {selected} pair(s); nothing detected.")
            return 0
        if not tracks:
            print("\nno tracks selected.")
            return 1
        if not jobs:
            print("\nnothing to do.")
            return 0
        print(
            f"\nstarting {selected} pair(s) at {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        started = time.perf_counter()
        raw, errors, skipped = benchmarker.run(
            jobs,
            workers=args.workers,
            batch_size=args.batch_size,
            watchdog=args.watchdog,
            max_attempts=args.max_attempts,
            verbose=not args.quiet_tracks,
            progress=not args.no_progress,
            compact_progress=args.compact_progress,
            cache_only=args.cache_only,
        )
        elapsed = time.perf_counter() - started
        if not raw.empty:
            summary = (
                benchmarker.summarize_degradation(raw, methods, args.noise_snrs)
                if args.noise_snrs
                else benchmarker.summarize(raw, methods)
            )
            with pd.option_context(
                "display.float_format", lambda v: f"{v:.4f}", "display.width", 200
            ):
                print(f"\n{'=' * 72}\n{'; '.join(args.datasets)}\n{'=' * 72}")
                print(summary.to_string())
            if args.write:
                benchmarker.write_raw_outputs(raw)
                if args.noise_snrs:
                    benchmarker.write_degradation_summary(summary)
                else:
                    benchmarker.write_summary(summary)
        print(
            f"\ndone: {len(raw)} rows, {len(errors)} track errors, {len(skipped)} skipped method(s) in {cls.duration(elapsed)}"
        )
        for method, reason in sorted(skipped.items()):
            print(f"  - {method}: {reason.splitlines()[0]}")
        for method, dataset, track_id, _ in errors:
            print(f"  ! {method} / {dataset} / {track_id}")
        return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(PitchBenchmarker.main())
