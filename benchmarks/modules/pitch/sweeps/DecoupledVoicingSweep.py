"""Cache-only pYIN pitch/voicing decoupling experiment.

The expensive pYIN frontend is never rerun.  Each selected preliminary track
loads its raw candidate cache, computes one banded pitch-only path, and reuses
that path for a small O(T) two-state voicing grid.  Cached production pYIN,
production pYIN-HMM, and Praat contours are reported beside two hybrids:

``decoupled_*``
    pYIN conditional pitch path + pYIN periodicity/volume two-state voicing.

``praat_inspired_*``
    The same cached pYIN evidence with a Praat-shaped local-strength threshold,
    hard silence floor, and additive voiced/unvoiced transition cost.  This
    path does not run Praat or recompute autocorrelation.

``praat_mask_pyin_pitch``
    A historical diagnostic: the same pYIN pitch path + Praat's literal cached
    voiced/unvoiced mask.  It is not the active hybrid benchmark method.

The script keeps tuning and holdout stems separate within every
corpus/instrument group.  It writes exact mir_eval scores for the named
finalists; the larger tuning grid uses an explicitly labelled nearest-grid
approximation only to choose those finalists.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any
import mir_eval
import numpy as np
import pandas as pd
from benchmarks.paths import SWEEP_RESULTS_ROOT
from benchmarks.modules.pitch.sweeps.DecoupledVoicing import (
    BandedPitchDecoder,
    PitchPath,
    PraatInspiredVoicingDecoder,
    PraatInspiredVoicingParameters,
    TwoStateVoicingDecoder,
    VoicingFeatures,
    VoicingParameters,
    nearest_voicing_mask,
    voiced_frequencies,
)
from benchmarks.modules.pitch.PitchCache import PitchCache
from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase
from benchmarks.modules.pitch.competitors.Attune import AttuneAdHoc, AttuneRealtime
from benchmarks.modules.pitch.competitors.Praat import Praat
from benchmarks.modules.pitch.datasets.CocoChorales import CocoChorales
from benchmarks.modules.pitch.datasets.URMP import URMP

RUNNER_VERSION = 1
DEFAULT_OUTPUT_DIR = SWEEP_RESULTS_ROOT / "pitch" / "decoupled_voicing_preliminary"
METRICS = (
    "Voicing Recall",
    "Voicing False Alarm",
    "Raw Pitch Accuracy",
    "Raw Chroma Accuracy",
    "Overall Accuracy",
)


@dataclass(frozen=True)
class SweepOptions:
    per_instrument: int = 2
    seed: int = 0
    oa_knee_tolerance: float = 0.0025


@dataclass
class TrackBundle:
    corpus: str
    track_id: str
    instrument: str
    role: str
    example: Any
    config: Any
    raw_pitches: list
    path: PitchPath
    features: VoicingFeatures
    raw_detector_cpu_seconds: float
    joint_smoother_cpu_seconds: float
    pitch_path_cpu_seconds: float
    pitch_path_wall_seconds: float


def parameter_grid() -> list[VoicingParameters]:
    """A deliberately small first-pass grid around the observed VFA knee."""
    transition_pairs = ((0.5, 0.5), (0.1, 0.1), (0.03, 0.03), (0.01, 0.03), (0.01, 0.1))
    return [
        VoicingParameters(
            max_unvoiced_prob=max_unvoiced,
            relative_volume_floor=relative_floor,
            absolute_volume_floor_dbfs=absolute_floor,
            onset_probability=onset,
            offset_probability=offset,
        )
        for max_unvoiced, relative_floor, absolute_floor, (onset, offset) in product(
            (0.55, 0.65, 0.75, 0.85, 0.9),
            (0.01, 0.02, 0.04),
            (None, -57.0, -54.0),
            transition_pairs,
        )
    ]


def praat_inspired_parameter_grid() -> list[PraatInspiredVoicingParameters]:
    """Calibrate Praat-shaped costs to pYIN candidate mass, not Praat ACF."""
    return [
        PraatInspiredVoicingParameters(
            candidate_strength_threshold=strength_threshold,
            silence_threshold=silence_threshold,
            absolute_volume_floor_dbfs=absolute_floor,
            voiced_unvoiced_cost=switch_cost,
        )
        for strength_threshold, silence_threshold, absolute_floor, switch_cost in product(
            (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45),
            (0.005, 0.01, 0.02, 0.03),
            (None, -57.0, -54.0, -51.0),
            (0.0, 0.02, 0.04, 0.08, 0.16),
        )
    ]


def _stable_role_rows(tracks: list[Any], seed: int) -> dict[str, str]:
    """Assign one deterministic tune and one holdout stem per instrument."""
    by_instrument: dict[str, list[Any]] = {}
    for track in tracks:
        instrument = str(track.metadata.get("instrument", "unknown"))
        by_instrument.setdefault(instrument, []).append(track)
    roles: dict[str, str] = {}
    for instrument, group in by_instrument.items():
        ordered = sorted(
            group,
            key=lambda track: hashlib.blake2b(
                f"{seed}\x00{instrument}\x00{track.track_id}".encode(), digest_size=8
            ).digest(),
        )
        if len(ordered) < 2:
            raise RuntimeError(
                f"{instrument}: need two tracks for tune/holdout, found {len(ordered)}"
            )
        roles[ordered[0].track_id] = "tune"
        roles[ordered[1].track_id] = "holdout"
    return roles


def _datasets(options: SweepOptions) -> list[Any]:
    return [
        CocoChorales(
            per_instrument=options.per_instrument, seed=options.seed, materialize=False
        ),
        URMP(per_instrument=options.per_instrument, seed=options.seed),
    ]


def load_bundles(options: SweepOptions) -> list[TrackBundle]:
    """Load raw caches and run exactly one pitch-only path per track."""
    attune = AttuneRealtime()
    bundles: list[TrackBundle] = []
    for dataset in _datasets(options):
        tracks = dataset.tracks()
        roles = _stable_role_rows(tracks, options.seed)
        for track in tracks:
            example = dataset.example(track)
            config = attune.config_for(example.fmin, example.fmax)
            hit = PitchCache(dataset.pitch_cache_path(track)).read(
                PitchCache.RAW, config
            )
            if hit is None:
                raise RuntimeError(
                    f"missing raw pYIN cache for {dataset.name}/{track.track_id}"
                )
            raw_pitches, raw_timing = hit
            smoothed_hit = PitchCache(dataset.pitch_cache_path(track)).read(
                PitchCache.SMOOTHED, config
            )
            if smoothed_hit is None:
                raise RuntimeError(
                    f"missing smoothed pYIN cache for {dataset.name}/{track.track_id}"
                )
            _, smoothed_timing = smoothed_hit
            cpu_started = time.process_time()
            wall_started = time.perf_counter()
            path = BandedPitchDecoder(config).decode(raw_pitches.data)
            pitch_cpu = time.process_time() - cpu_started
            pitch_wall = time.perf_counter() - wall_started
            bundles.append(
                TrackBundle(
                    corpus=dataset.name,
                    track_id=track.track_id,
                    instrument=str(track.metadata.get("instrument", "unknown")),
                    role=roles[track.track_id],
                    example=example,
                    config=config,
                    raw_pitches=raw_pitches.data,
                    path=path,
                    features=VoicingFeatures.from_pitches(raw_pitches.data),
                    raw_detector_cpu_seconds=float(
                        raw_timing["pitch_detector_compute_time"]
                    ),
                    joint_smoother_cpu_seconds=float(
                        smoothed_timing["pitch_smoother_compute_time"]
                    ),
                    pitch_path_cpu_seconds=pitch_cpu,
                    pitch_path_wall_seconds=pitch_wall,
                )
            )
    return bundles


def _fast_metrics(bundle: TrackBundle, voiced: np.ndarray) -> dict[str, float]:
    """Nearest-grid tuning proxy; finalists are rescored exactly with mir_eval."""
    path_times, path_freqs = voiced_frequencies(bundle.path, voiced, bundle.config)
    ref_times = np.asarray(bundle.example.ref_times, dtype=np.float64)
    ref_freqs = np.asarray(bundle.example.ref_freqs, dtype=np.float64)
    if path_times.size == 0:
        estimated_freqs = np.zeros_like(ref_freqs)
    else:
        nearest_voiced = nearest_voicing_mask(ref_times, path_times, path_freqs)
        estimated_freqs = np.interp(
            ref_times, path_times, path_freqs, left=0.0, right=0.0
        )
        estimated_freqs[~nearest_voiced] = 0.0
    ref_voiced = ref_freqs > 0.0
    est_voiced = estimated_freqs > 0.0
    correct_pitch = np.zeros(ref_freqs.size, dtype=bool)
    both_voiced = ref_voiced & est_voiced
    correct_pitch[both_voiced] = (
        np.abs(1200.0 * np.log2(estimated_freqs[both_voiced] / ref_freqs[both_voiced]))
        <= 50.0
    )
    chroma_error = np.zeros(ref_freqs.size, dtype=np.float64)
    chroma_error[both_voiced] = np.abs(
        1200.0 * np.log2(estimated_freqs[both_voiced] / ref_freqs[both_voiced])
    )
    chroma_error = np.minimum(
        np.mod(chroma_error, 1200.0), 1200.0 - np.mod(chroma_error, 1200.0)
    )
    correct_chroma = both_voiced & (chroma_error <= 50.0)
    voiced_count = max(1, int(np.count_nonzero(ref_voiced)))
    unvoiced_count = max(1, int(np.count_nonzero(~ref_voiced)))
    total_count = max(1, ref_freqs.size)
    return {
        "Voicing Recall": float(
            np.count_nonzero(ref_voiced & est_voiced) / voiced_count
        ),
        "Voicing False Alarm": float(
            np.count_nonzero(~ref_voiced & est_voiced) / unvoiced_count
        ),
        "Raw Pitch Accuracy": float(np.count_nonzero(correct_pitch) / voiced_count),
        "Raw Chroma Accuracy": float(np.count_nonzero(correct_chroma) / voiced_count),
        "Overall Accuracy": float(
            (
                np.count_nonzero(correct_pitch)
                + np.count_nonzero(~ref_voiced & ~est_voiced)
            )
            / total_count
        ),
    }


def _exact_metrics(
    bundle: TrackBundle, times: np.ndarray, freqs: np.ndarray
) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in mir_eval.melody.evaluate(
            bundle.example.ref_times, bundle.example.ref_freqs, times, freqs
        ).items()
    }


def _parameter_columns(parameters: VoicingParameters) -> dict[str, Any]:
    return asdict(parameters)


def tune_grid(
    bundles: list[TrackBundle], variants: list[VoicingParameters]
) -> tuple[pd.DataFrame, float]:
    rows: list[dict[str, Any]] = []
    cpu_started = time.process_time()
    for index, parameters in enumerate(variants):
        decoder = TwoStateVoicingDecoder(parameters)
        for bundle in bundles:
            if bundle.role != "tune":
                continue
            voiced = decoder.decode(bundle.features)
            rows.append(
                {
                    "variant": f"voicing-{index:03d}",
                    "corpus": bundle.corpus,
                    "track_id": bundle.track_id,
                    "instrument": bundle.instrument,
                    **_parameter_columns(parameters),
                    **_fast_metrics(bundle, voiced),
                }
            )
    return (pd.DataFrame(rows), time.process_time() - cpu_started)


def tune_praat_inspired_grid(
    bundles: list[TrackBundle], variants: list[PraatInspiredVoicingParameters]
) -> tuple[pd.DataFrame, float]:
    rows: list[dict[str, Any]] = []
    cpu_started = time.process_time()
    for index, parameters in enumerate(variants):
        for bundle in bundles:
            if bundle.role != "tune":
                continue
            decoder = PraatInspiredVoicingDecoder(
                parameters,
                time_step_seconds=float(bundle.config.h1) / float(bundle.config.sr),
            )
            rows.append(
                {
                    "variant": f"praat-inspired-{index:03d}",
                    "corpus": bundle.corpus,
                    "track_id": bundle.track_id,
                    "instrument": bundle.instrument,
                    **asdict(parameters),
                    **_fast_metrics(bundle, decoder.decode(bundle.features)),
                }
            )
    return (pd.DataFrame(rows), time.process_time() - cpu_started)


def select_finalists(
    grid_rows: pd.DataFrame, options: SweepOptions
) -> dict[str, VoicingParameters]:
    parameter_names = list(asdict(VoicingParameters()))
    summary = (
        grid_rows.groupby(["variant", *parameter_names], dropna=False)[list(METRICS)]
        .mean()
        .reset_index()
    )
    oa_row = summary.sort_values(
        ["Overall Accuracy", "Voicing False Alarm"], ascending=[False, True]
    ).iloc[0]
    eligible = summary.loc[
        summary["Overall Accuracy"]
        >= float(oa_row["Overall Accuracy"]) - options.oa_knee_tolerance
    ]
    knee_row = eligible.sort_values(
        ["Voicing False Alarm", "Overall Accuracy"], ascending=[True, False]
    ).iloc[0]

    def parameters(row: pd.Series) -> VoicingParameters:
        absolute = row["absolute_volume_floor_dbfs"]
        return VoicingParameters(
            max_unvoiced_prob=float(row["max_unvoiced_prob"]),
            relative_volume_floor=float(row["relative_volume_floor"]),
            absolute_volume_floor_dbfs=None if pd.isna(absolute) else float(absolute),
            onset_probability=float(row["onset_probability"]),
            offset_probability=float(row["offset_probability"]),
        )

    return {
        "decoupled_default": VoicingParameters(),
        "decoupled_strict": VoicingParameters(
            max_unvoiced_prob=0.65,
            relative_volume_floor=0.02,
            absolute_volume_floor_dbfs=-54.0,
        ),
        "decoupled_tuned_oa": parameters(oa_row),
        "decoupled_tuned_vfa_knee": parameters(knee_row),
    }


def select_praat_inspired_finalists(
    grid_rows: pd.DataFrame, options: SweepOptions
) -> dict[str, PraatInspiredVoicingParameters]:
    parameter_names = list(asdict(PraatInspiredVoicingParameters()))
    summary = (
        grid_rows.groupby(["variant", *parameter_names], dropna=False)[list(METRICS)]
        .mean()
        .reset_index()
    )
    oa_row = summary.sort_values(
        ["Overall Accuracy", "Voicing False Alarm"], ascending=[False, True]
    ).iloc[0]
    eligible = summary.loc[
        summary["Overall Accuracy"]
        >= float(oa_row["Overall Accuracy"]) - options.oa_knee_tolerance
    ]
    knee_row = eligible.sort_values(
        ["Voicing False Alarm", "Overall Accuracy"], ascending=[True, False]
    ).iloc[0]

    def parameters(row: pd.Series) -> PraatInspiredVoicingParameters:
        absolute = row["absolute_volume_floor_dbfs"]
        return PraatInspiredVoicingParameters(
            candidate_strength_threshold=float(row["candidate_strength_threshold"]),
            silence_threshold=float(row["silence_threshold"]),
            absolute_volume_floor_dbfs=None if pd.isna(absolute) else float(absolute),
            voiced_unvoiced_cost=float(row["voiced_unvoiced_cost"]),
        )

    return {
        "praat_inspired_tuned_oa": parameters(oa_row),
        "praat_inspired_tuned_vfa_knee": parameters(knee_row),
    }


def exact_rows(
    bundles: list[TrackBundle],
    finalists: dict[str, VoicingParameters],
    praat_inspired_finalists: dict[str, PraatInspiredVoicingParameters],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    attune = AttuneRealtime()
    smoothed = AttuneAdHoc()
    praat = Praat()
    for bundle in bundles:
        common = {
            "corpus": bundle.corpus,
            "track_id": bundle.track_id,
            "instrument": bundle.instrument,
            "role": bundle.role,
        }
        for method, detector in (
            ("attune_realtime", attune),
            ("attune_adhoc", smoothed),
            ("praat", praat),
        ):
            cache = detector.cache(bundle.example)
            cached = (
                cache.read(detector.stage, bundle.config)
                if isinstance(cache, PitchCache)
                else cache.read()
            )
            if cached is None:
                raise RuntimeError(
                    f"missing cached {method} estimate for {bundle.track_id}"
                )
            if isinstance(cache, PitchCache):
                pitch_data, timing = cached
                times, freqs = detector.melody(pitch_data, bundle.config)
                estimate = PitchDetectorBase.PitchEstimate.build(
                    times, freqs, float(timing["pitch_compute_time"]), metadata=timing
                )
                estimate = detector.constrain_estimate_to_range(
                    estimate, bundle.example.fmin, bundle.example.fmax
                )
            else:
                estimate = detector.constrain_estimate_to_range(
                    cached, bundle.example.fmin, bundle.example.fmax
                )
            rows.append(
                {
                    **common,
                    "method": method,
                    **_exact_metrics(bundle, estimate.times, estimate.freqs),
                    "compute_seconds": float(estimate.compute_seconds),
                    "postprocessor_compute_seconds": (
                        0.0
                        if method == "attune_realtime"
                        else (
                            bundle.joint_smoother_cpu_seconds
                            if method == "attune_adhoc"
                            else np.nan
                        )
                    ),
                }
            )
        praat_estimate = praat.cache(bundle.example).read()
        assert praat_estimate is not None
        praat_voiced = nearest_voicing_mask(
            bundle.path.times, praat_estimate.times, praat_estimate.freqs
        )
        times, freqs = voiced_frequencies(bundle.path, praat_voiced, bundle.config)
        rows.append(
            {
                **common,
                "method": "praat_mask_pyin_pitch",
                **_exact_metrics(bundle, times, freqs),
                "compute_seconds": bundle.raw_detector_cpu_seconds
                + bundle.pitch_path_cpu_seconds
                + float(praat_estimate.compute_seconds),
                "postprocessor_compute_seconds": bundle.pitch_path_cpu_seconds
                + float(praat_estimate.compute_seconds),
            }
        )
        for method, parameters in finalists.items():
            cpu_started = time.process_time()
            voiced = TwoStateVoicingDecoder(parameters).decode(bundle.features)
            voicing_seconds = time.process_time() - cpu_started
            times, freqs = voiced_frequencies(bundle.path, voiced, bundle.config)
            rows.append(
                {
                    **common,
                    "method": method,
                    **_parameter_columns(parameters),
                    **_exact_metrics(bundle, times, freqs),
                    "compute_seconds": bundle.raw_detector_cpu_seconds
                    + bundle.pitch_path_cpu_seconds
                    + voicing_seconds,
                    "postprocessor_compute_seconds": bundle.pitch_path_cpu_seconds
                    + voicing_seconds,
                    "pitch_path_compute_seconds": bundle.pitch_path_cpu_seconds,
                    "voicing_compute_seconds": voicing_seconds,
                }
            )
        for method, parameters in praat_inspired_finalists.items():
            cpu_started = time.process_time()
            voiced = PraatInspiredVoicingDecoder(
                parameters,
                time_step_seconds=float(bundle.config.h1) / float(bundle.config.sr),
            ).decode(bundle.features)
            voicing_seconds = time.process_time() - cpu_started
            times, freqs = voiced_frequencies(bundle.path, voiced, bundle.config)
            rows.append(
                {
                    **common,
                    "method": method,
                    **asdict(parameters),
                    **_exact_metrics(bundle, times, freqs),
                    "compute_seconds": bundle.raw_detector_cpu_seconds
                    + bundle.pitch_path_cpu_seconds
                    + voicing_seconds,
                    "postprocessor_compute_seconds": bundle.pitch_path_cpu_seconds
                    + voicing_seconds,
                    "pitch_path_compute_seconds": bundle.pitch_path_cpu_seconds,
                    "voicing_compute_seconds": voicing_seconds,
                }
            )
    return pd.DataFrame(rows)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(["corpus", "role", "method"], sort=False)
        .agg(
            Tracks=("track_id", "nunique"),
            **{metric: (metric, "mean") for metric in METRICS},
            **{"Mean Compute Seconds": ("compute_seconds", "mean")},
            **{"Mean Postprocessor Seconds": ("postprocessor_compute_seconds", "mean")},
        )
        .reset_index()
    )


def run(
    output_dir: Path | str = DEFAULT_OUTPUT_DIR, options: SweepOptions | None = None
) -> pd.DataFrame:
    options = options or SweepOptions()
    if options.per_instrument != 2:
        raise ValueError("this tune/holdout experiment requires per_instrument=2")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    bundles = load_bundles(options)
    variants = parameter_grid()
    grid_rows, grid_cpu = tune_grid(bundles, variants)
    finalists = select_finalists(grid_rows, options)
    praat_variants = praat_inspired_parameter_grid()
    praat_grid_rows, praat_grid_cpu = tune_praat_inspired_grid(bundles, praat_variants)
    praat_finalists = select_praat_inspired_finalists(praat_grid_rows, options)
    scores = exact_rows(bundles, finalists, praat_finalists)
    summary = summarize(scores)
    grid_rows.to_csv(destination / "tuning_grid_approximate.csv", index=False)
    praat_grid_rows.to_csv(
        destination / "praat_inspired_tuning_grid_approximate.csv", index=False
    )
    scores.to_csv(destination / "per_track_exact.csv", index=False)
    summary.to_csv(destination / "summary_exact.csv", index=False)
    (destination / "metadata.json").write_text(
        json.dumps(
            {
                "runner_version": RUNNER_VERSION,
                "options": asdict(options),
                "tracks": len(bundles),
                "tune_tracks": sum((bundle.role == "tune" for bundle in bundles)),
                "holdout_tracks": sum((bundle.role == "holdout" for bundle in bundles)),
                "grid_variants": len(variants),
                "grid_voicing_cpu_seconds": grid_cpu,
                "praat_inspired_grid_variants": len(praat_variants),
                "praat_inspired_grid_voicing_cpu_seconds": praat_grid_cpu,
                "pitch_path_cpu_seconds": sum(
                    (bundle.pitch_path_cpu_seconds for bundle in bundles)
                ),
                "pitch_path_wall_seconds": sum(
                    (bundle.pitch_path_wall_seconds for bundle in bundles)
                ),
                "production_joint_smoother_cpu_seconds": sum(
                    (bundle.joint_smoother_cpu_seconds for bundle in bundles)
                ),
                "pitch_path_speedup_over_joint_smoother": sum(
                    (bundle.joint_smoother_cpu_seconds for bundle in bundles)
                )
                / sum((bundle.pitch_path_cpu_seconds for bundle in bundles)),
                "finalists": {
                    name: asdict(parameters) for name, parameters in finalists.items()
                },
                "praat_inspired_finalists": {
                    name: asdict(parameters)
                    for name, parameters in praat_finalists.items()
                },
                "note": "The tuning grid uses nearest-grid proxy metrics only for selection; per_track_exact.csv and summary_exact.csv use mir_eval.melody.evaluate.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--oa-knee-tolerance", type=float, default=0.0025)
    args = parser.parse_args()
    result = run(
        args.output_dir,
        SweepOptions(seed=args.seed, oa_knee_tolerance=args.oa_knee_tolerance),
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
