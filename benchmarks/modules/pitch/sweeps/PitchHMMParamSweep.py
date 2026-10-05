from __future__ import annotations

"Parallel, resumable CocoChorales sweep for Attune's pYIN HMM.\n\nStage-1 pitch candidates are loaded from the existing ``raw`` pitch caches.\nEach worker owns one stem, loads that cache once, and evaluates the complete\nHMM grid locally. The final statistics combine all selected stems, with equal\nweight for every represented (ensemble, instrument) stratum.\n"
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
import json
import multiprocessing
import random
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Sequence
import mir_eval
import numpy as np
import pandas as pd
from hmmlearn.base import BaseHMM
from scipy.sparse import diags
from tqdm.auto import tqdm
from benchmarks.paths import REPO_ROOT, SWEEP_RESULTS_ROOT, ensure_repo_on_path

ensure_repo_on_path()
from algorithms.Config import Config
from app_logic.user.ds.PitchData import Pitch
from benchmarks.modules.pitch.PitchCache import PitchCache
from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase
from benchmarks.modules.pitch.competitors.Attune import AttuneRealtime
from benchmarks.modules.pitch.datasets.CocoChorales import CocoChorales

RUNNER_VERSION = 2
CACHE_REFRESH_VERSION = 1
DEFAULT_OUTPUT_DIR = SWEEP_RESULTS_ROOT / "pitch"
ATTUNE_OBSERVATION = "attune-uniform"
PAPER_OBSERVATION = "paper-repeated"
OBSERVATION_MODES = (ATTUNE_OBSERVATION, PAPER_OBSERVATION)
ATTUNE_HMM_MAX_JUMP_BINS = 9
ATTUNE_HMM_SWITCH_PROB = 0.0025
ATTUNE_HMM_YIN_TRUST = 0.65
PAPER_HALF_HOP_MAX_JUMP = 13
PAPER_HALF_HOP_SWITCH = 1.0 - (1.0 - 0.01) ** 0.5
PARAM_COLUMNS = ["observation_mode", "max_jump_bins", "switch_prob", "yin_trust"]
METRIC_COLUMNS = [
    "Voicing Recall",
    "Voicing False Alarm",
    "Raw Pitch Accuracy",
    "Raw Chroma Accuracy",
    "Overall Accuracy",
]


@dataclass(frozen=True)
class ParameterAxes:
    """Small grid centered on the two HMM formulations under discussion."""

    observation_mode: tuple[str, ...] = OBSERVATION_MODES
    max_jump_bins: tuple[int, ...] = (PAPER_HALF_HOP_MAX_JUMP, 25)
    switch_prob: tuple[float, ...] = (PAPER_HALF_HOP_SWITCH, 0.01)
    yin_trust: tuple[float, ...] = (0.35, 0.5, 0.65)


@dataclass(frozen=True)
class SweepOptions:
    split: str = "test"
    seed: int = 0
    per_stratum: int = 2
    tune_per_stratum: int = 1
    max_strata: int | None = None
    max_tracks: int | None = None


def production_parameters() -> dict[str, Any]:
    return {
        "observation_mode": ATTUNE_OBSERVATION,
        "max_jump_bins": ATTUNE_HMM_MAX_JUMP_BINS,
        "switch_prob": ATTUNE_HMM_SWITCH_PROB,
        "yin_trust": ATTUNE_HMM_YIN_TRUST,
    }


def paper_parameters() -> dict[str, Any]:
    """Paper HMM constants expressed on Attune's shorter frame grid."""
    return {
        "observation_mode": PAPER_OBSERVATION,
        "max_jump_bins": PAPER_HALF_HOP_MAX_JUMP,
        "switch_prob": PAPER_HALF_HOP_SWITCH,
        "yin_trust": 0.5,
    }


def parameter_grid(axes: ParameterAxes | None = None) -> list[dict[str, Any]]:
    axes = axes or ParameterAxes()
    unknown_modes = set(axes.observation_mode) - set(OBSERVATION_MODES)
    if unknown_modes:
        raise ValueError(
            "unknown observation mode(s): " + ", ".join(sorted(unknown_modes))
        )
    if any((value < 1 for value in axes.max_jump_bins)):
        raise ValueError("max_jump_bins values must be positive integers")
    if any((not 0.0 < value < 1.0 for value in axes.switch_prob)):
        raise ValueError("switch_prob values must be between zero and one")
    if any((not 0.0 < value < 1.0 for value in axes.yin_trust)):
        raise ValueError("yin_trust values must be between zero and one")
    values = [getattr(axes, name) for name in PARAM_COLUMNS]
    return [dict(zip(PARAM_COLUMNS, combination)) for combination in product(*values)]


def parameter_variants(grid: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"variant": f"hmm-{index:03d}", **params} for index, params in enumerate(grid)
    ]


def _role_for_track(track: str, options: SweepOptions) -> str:
    """Assign all stems from one chorale to the same role."""
    digest = hashlib.sha256(f"{options.seed}:{track}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % options.per_stratum
    return "tune" if bucket < options.tune_per_stratum else "holdout"


def select_sample(
    options: SweepOptions | None = None, *, require_raw_cache: bool = True
) -> pd.DataFrame:
    """Select a deterministic, stratum-complete cached subset.

    Legacy tune/holdout labels preserve the already-materialized sample but
    are not used by summary aggregation or parameter selection.
    """
    options = options or SweepOptions()
    if not 0 < options.tune_per_stratum < options.per_stratum:
        raise ValueError("tune_per_stratum must be between 1 and per_stratum - 1")
    bench = CocoChorales()
    attune = AttuneRealtime()
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in bench.read_manifest(options.split):
        wav = bench.local_wav_path(record)
        cache = bench.cache_path_for_track(record.track_id)
        f0_path = bench.f0_path_for_record(record)
        if wav is None or not f0_path.exists():
            continue
        if require_raw_cache and (not cache.exists()):
            continue
        groups[record.ensemble, record.instrument].append(
            {
                "track": record.track,
                "track_id": record.track_id,
                "ensemble": record.ensemble,
                "instrument": record.instrument,
                "stem_voice": record.stem_voice,
                "wav": str(wav),
                "f0": str(f0_path),
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
        quotas = {
            "tune": options.tune_per_stratum,
            "holdout": options.per_stratum - options.tune_per_stratum,
        }
        role_rows: dict[str, list[dict[str, Any]]] = {"tune": [], "holdout": []}
        for row in candidates:
            role = _role_for_track(row["track"], options)
            if len(role_rows[role]) >= quotas[role]:
                continue
            if require_raw_cache and (
                not attune.has_pitch_cache(row["cache"], smooth=False)
            ):
                continue
            role_rows[role].append(row)
            if all((len(role_rows[key]) >= quotas[key] for key in quotas)):
                break
        for role, quota in quotas.items():
            if len(role_rows[role]) < quota:
                raise RuntimeError(
                    f"only found {len(role_rows[role])}/{quota} cached raw-pitch stems for {stratum} role={role}"
                )
            for rank, row in enumerate(role_rows[role]):
                selected.append({**row, "role": role, "stratum_rank": rank})
    sample = pd.DataFrame(selected)
    if sample.empty:
        raise RuntimeError(
            "no materialized CocoChorales stems with raw pitch caches were found"
        )
    sample["_role_order"] = sample["role"].map({"tune": 0, "holdout": 1})
    sample = (
        sample.sort_values(["_role_order", "ensemble", "instrument", "track_id"])
        .drop(columns="_role_order")
        .reset_index(drop=True)
    )
    if options.max_tracks is not None:
        sample = sample.iloc[: options.max_tracks].copy()
    return sample


def _source_signature(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(REPO_ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:20]


def current_detector_signature() -> str:
    """Fingerprint code that determines raw Stage-1 pitch candidates."""
    return _source_signature(
        [
            REPO_ROOT / "algorithms" / "Config.py",
            REPO_ROOT / "algorithms" / "PitchDetector.py",
        ]
    )


def current_smoother_signature() -> str:
    return _source_signature(
        [
            REPO_ROOT / "algorithms" / "Config.py",
            REPO_ROOT / "algorithms" / "PitchSmoother.py",
        ]
    )


def _cache_metadata(cache_path: Path | str) -> dict[str, dict[str, Any]]:
    path = Path(cache_path)
    if not path.exists():
        return {}
    try:
        return PitchCache(path)._payload()[1]
    except Exception:
        return {}


def cache_detector_signature(cache_path: Path | str) -> str | None:
    metadata = _cache_metadata(cache_path)
    return metadata.get("raw", {}).get("detector_signature")


def cache_smoother_signature(cache_path: Path | str) -> str | None:
    metadata = _cache_metadata(cache_path)
    return metadata.get("smoothed", {}).get("smoother_signature")


def cache_matches_current_detector(cache_path: Path | str) -> bool:
    return cache_detector_signature(cache_path) == current_detector_signature()


def cache_matches_current_pipeline(cache_path: Path | str) -> bool:
    metadata = _cache_metadata(cache_path)
    return (
        metadata.get("raw", {}).get("detector_signature")
        == current_detector_signature()
        and metadata.get("smoothed", {}).get("smoother_signature")
        == current_smoother_signature()
    )


def _regenerate_pitch_cache(track_row: dict[str, Any]) -> dict[str, Any]:
    """Regenerate both raw candidates and production-smoothed pitches."""
    bench = CocoChorales()
    attune = AttuneRealtime()
    ref_times, ref_freqs = bench.load_f0(track_row["f0"], int(track_row["stem_voice"]))
    fmin, fmax = bench.search_range(ref_freqs)
    config = attune.config_for(fmin, fmax)
    recording = attune.recording_for(config)
    recording.audio_data = bench.load_resampled_audio(track_row["wav"], config.sr)
    started = time.perf_counter()
    stages = attune.detect_stages(recording, smooth=True)
    detector_signature = current_detector_signature()
    smoother_signature = current_smoother_signature()
    cache_metadata: dict[str, dict[str, Any]] = {}
    for stage, stage_timing in stages.timing.items():
        cache_metadata[stage] = {
            **stage_timing,
            "cache_refresh_version": CACHE_REFRESH_VERSION,
            "detector_signature": detector_signature,
            "smoother_signature": smoother_signature if stage == "smoothed" else None,
            "fmin": float(fmin),
            "fmax": float(fmax),
            "sr": int(config.sr),
            "w1": int(config.w1),
            "h1": int(config.h1),
            "min_volume": float(config.min_volume),
        }
    stages.timing = cache_metadata
    PitchCache(track_row["cache"]).write(stages)
    return {
        **{
            key: track_row[key]
            for key in ("track", "track_id", "ensemble", "instrument", "role", "cache")
        },
        "status": "regenerated",
        "detector_signature": detector_signature,
        "fmin": float(fmin),
        "fmax": float(fmax),
        "frames": len(stages.data[PitchCache.RAW].data),
        "compute_time": time.perf_counter() - started,
        "error": None,
    }


def regenerate_sample_caches(
    output_dir: Path | str | None = None,
    workers: int | None = None,
    options: SweepOptions | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Refresh the selected raw+smoothed caches with the current pitch code."""
    output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    options = options or SweepOptions()
    sample = select_sample(options, require_raw_cache=False)
    sample.to_csv(output_dir / "sample.csv", index=False)
    detector_signature = current_detector_signature()
    complete: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for row in sample.to_dict("records"):
        if not force and cache_matches_current_pipeline(row["cache"]):
            complete.append(
                {
                    **{
                        key: row[key]
                        for key in (
                            "track",
                            "track_id",
                            "ensemble",
                            "instrument",
                            "role",
                            "cache",
                        )
                    },
                    "status": "current",
                    "detector_signature": detector_signature,
                    "error": None,
                }
            )
        else:
            pending.append(row)
    worker_count = workers or max(1, (os.cpu_count() or 4) - 1)
    print(
        f"Pitch-cache refresh: {len(pending)} stale/missing, {len(complete)} current; {worker_count} worker processes",
        flush=True,
    )
    if pending:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=max(1, worker_count), mp_context=context
        ) as pool:
            futures = {
                pool.submit(_regenerate_pitch_cache, row): row for row in pending
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Refreshing pitch caches",
            ):
                row = futures[future]
                try:
                    complete.append(future.result())
                except Exception as exc:
                    complete.append(
                        {
                            **{
                                key: row[key]
                                for key in (
                                    "track",
                                    "track_id",
                                    "ensemble",
                                    "instrument",
                                    "role",
                                    "cache",
                                )
                            },
                            "status": "error",
                            "detector_signature": detector_signature,
                            "error": "".join(
                                traceback.format_exception(
                                    type(exc), exc, exc.__traceback__
                                )
                            ),
                        }
                    )
    rows = (
        pd.DataFrame(complete)
        .sort_values(["role", "ensemble", "instrument", "track_id"])
        .reset_index(drop=True)
    )
    rows.to_csv(output_dir / "cache_refresh.csv", index=False)
    _write_json(
        output_dir / "cache_refresh_metadata.json",
        {
            "cache_refresh_version": CACHE_REFRESH_VERSION,
            "detector_signature": detector_signature,
            "smoother_signature": current_smoother_signature(),
            "tracks": len(sample),
            "regenerated": int(rows.status.eq("regenerated").sum()),
            "already_current": int(rows.status.eq("current").sum()),
            "errors": int(rows.error.notna().sum()),
            "workers": worker_count,
            "options": asdict(options),
        },
    )
    return rows


class _PrecomputedHMM(BaseHMM):
    """Decode-only HMM whose observations are supplied by the sweep."""

    def __init__(
        self, n_components: int, startprob: np.ndarray, transmat: np.ndarray
    ) -> None:
        super().__init__(
            n_components=n_components, params="", init_params="", implementation="log"
        )
        self.startprob_ = startprob
        self.transmat_ = transmat
        self.n_features = n_components

    def _compute_log_likelihood(self, observations: np.ndarray) -> np.ndarray:
        return observations

    def _init(self, observations, lengths=None) -> None:
        pass

    def _generate_sample_from_state(self, state, random_state=None):
        raise NotImplementedError("The parameter-sweep HMM is decode-only")


class _SweepPitchSmoother:
    """Historical joint HMM, scoped to its reproducibility experiment."""

    def __init__(
        self,
        *,
        config: Config,
        observation_mode: str,
        max_jump_bins: int,
        switch_prob: float,
        yin_trust: float,
    ) -> None:
        self.observation_mode = observation_mode
        self.config = config
        self.resolution = 0.1
        midi_lo = config.freq_to_midi(config.fmin)
        midi_hi = config.freq_to_midi(config.fmax)
        self.midi_min = np.floor(midi_lo / self.resolution) * self.resolution
        midi_max = np.ceil(midi_hi / self.resolution) * self.resolution
        self.n_bins = int(round((midi_max - self.midi_min) / self.resolution)) + 1
        self.bin_midis = self.midi_min + self.resolution * np.arange(self.n_bins)
        self.n_states = 2 * self.n_bins
        self.max_jump = int(max_jump_bins)
        self.switch_prob = float(switch_prob)
        self.yin_trust = float(yin_trust)
        self._startprob = np.zeros(self.n_states, dtype=np.float64)
        self._startprob[self.n_bins :] = 1.0 / self.n_bins
        self._transmat = self._build_transition()

    def _midi_to_bin(self, midi: float) -> int | None:
        pitch_bin = int(round((float(midi) - self.midi_min) / self.resolution))
        return pitch_bin if 0 <= pitch_bin < self.n_bins else None

    def _build_transition(self) -> np.ndarray:
        offsets = np.arange(-self.max_jump, self.max_jump + 1)
        weights = self.max_jump + 1 - np.abs(offsets)
        diagonals = [
            np.full(self.n_bins - abs(offset), weight, dtype=np.float64)
            for offset, weight in zip(offsets, weights)
        ]
        pitch_transition = diags(
            diagonals, offsets, shape=(self.n_bins, self.n_bins)
        ).toarray()
        pitch_transition /= pitch_transition.sum(axis=1, keepdims=True)
        stay = 1.0 - self.switch_prob
        switch = self.switch_prob
        return np.block(
            [
                [stay * pitch_transition, switch * pitch_transition],
                [switch * pitch_transition, stay * pitch_transition],
            ]
        )

    def _observation_logprobs(self, pitches: list[Pitch]) -> np.ndarray:
        if self.observation_mode == ATTUNE_OBSERVATION:
            obs = np.zeros((len(pitches), self.n_states), dtype=np.float64)
            for frame_index, pitch in enumerate(pitches):
                voiced = np.zeros(self.n_bins, dtype=np.float64)
                if pitch is not None:
                    for midi, probability in pitch.candidate_pitches:
                        pitch_bin = self._midi_to_bin(midi)
                        if pitch_bin is not None:
                            voiced[pitch_bin] += probability
                voiced_mass = float(voiced.sum())
                prob_pitched = self.yin_trust * voiced_mass
                if voiced_mass > 0.0:
                    voiced *= prob_pitched / voiced_mass
                obs[frame_index, : self.n_bins] = voiced
                obs[frame_index, self.n_bins :] = (1.0 - prob_pitched) / self.n_bins
            np.maximum(obs, 1e-12, out=obs)
            return np.log(obs)
        if self.observation_mode != PAPER_OBSERVATION:
            raise ValueError(f"unknown observation mode: {self.observation_mode}")
        obs = np.zeros((len(pitches), self.n_states), dtype=np.float64)
        for frame_index, pitch in enumerate(pitches):
            voiced = np.zeros(self.n_bins, dtype=np.float64)
            if pitch is not None:
                for midi, probability in pitch.candidate_pitches:
                    pitch_bin = self._midi_to_bin(midi)
                    if pitch_bin is not None:
                        voiced[pitch_bin] += probability
            raw_mass = float(voiced.sum())
            voiced_mass = float(np.clip(raw_mass, 0.0, 1.0))
            if raw_mass > 0.0 and raw_mass != voiced_mass:
                voiced *= voiced_mass / raw_mass
            obs[frame_index, : self.n_bins] = self.yin_trust * voiced
            obs[frame_index, self.n_bins :] = (1.0 - self.yin_trust) * (
                1.0 - voiced_mass
            )
        np.maximum(obs, 1e-12, out=obs)
        return np.log(obs)

    def decode(self, pitches: list[Pitch]) -> np.ndarray:
        if not pitches:
            return np.empty(0, dtype=int)
        hmm = _PrecomputedHMM(self.n_states, self._startprob, self._transmat)
        _log_probability, states = hmm.decode(
            self._observation_logprobs(pitches), algorithm="viterbi"
        )
        return states

    def smooth_to_arrays(
        self, pitches: list[Pitch]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        states = self.decode(pitches)
        times = np.asarray(
            [pitch.time if pitch is not None else np.nan for pitch in pitches],
            dtype=np.float64,
        )
        voiced = states < self.n_bins
        midi = np.full(len(states), np.nan, dtype=np.float64)
        midi[voiced] = self.bin_midis[states[voiced]]
        return (times, midi, voiced)


def _evaluate_variant(
    raw_pitches: list[Pitch],
    config: Config,
    ref_times: np.ndarray,
    ref_freqs: np.ndarray,
    variant: dict[str, Any],
) -> dict[str, Any]:

    def smooth_variant():
        smoother = _SweepPitchSmoother(
            config=config,
            observation_mode=str(variant["observation_mode"]),
            max_jump_bins=int(variant["max_jump_bins"]),
            switch_prob=float(variant["switch_prob"]),
            yin_trust=float(variant["yin_trust"]),
        )
        return smoother.smooth_to_arrays(raw_pitches)

    (est_times, est_midi, voiced), smoother_seconds, smoother_wall_seconds = (
        PitchDetectorBase.measure(smooth_variant)
    )
    est_freqs = np.zeros_like(est_midi)
    est_freqs[voiced] = config.midi_to_freq(est_midi[voiced])
    metrics = {
        name: float(value)
        for name, value in mir_eval.melody.evaluate(
            ref_times, ref_freqs, est_times, est_freqs
        ).items()
    }
    return {
        **variant,
        **metrics,
        "voiced_frames": int(voiced.sum()),
        "total_frames": int(voiced.size),
        "voiced_fraction": float(voiced.mean()) if voiced.size else 0.0,
        "smoother_compute_time": smoother_seconds,
        "wall_smoother_compute_time": smoother_wall_seconds,
        "compute_clock": PitchDetectorBase.COMPUTE_CLOCK,
    }


def _run_track(
    track_row: dict[str, Any], variants: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    bench = CocoChorales()
    attune = AttuneRealtime()
    ref_times, ref_freqs = bench.load_f0(track_row["f0"], int(track_row["stem_voice"]))
    fmin, fmax = bench.search_range(ref_freqs)
    config = attune.config_for(fmin, fmax)
    raw_data, _ = attune.load_pitch_data(track_row["cache"], config, smooth=False)
    metadata = {
        key: track_row[key]
        for key in ("track", "track_id", "ensemble", "instrument", "role")
    }
    metadata.update(
        {
            "fmin": float(fmin),
            "fmax": float(fmax),
            "audio_seconds": float(ref_times[-1]) if ref_times.size else 0.0,
        }
    )
    rows: list[dict[str, Any]] = []
    for variant in variants:
        try:
            rows.append(
                {
                    **metadata,
                    **_evaluate_variant(
                        raw_data.data, config, ref_times, ref_freqs, variant
                    ),
                    "error": None,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    **metadata,
                    **variant,
                    "error": "".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    ),
                }
            )
    return rows


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _signature(
    sample: pd.DataFrame, variants: Sequence[dict[str, Any]], options: SweepOptions
) -> str:
    cache_signatures = (
        sample["detector_signature"].tolist()
        if "detector_signature" in sample
        else [cache_detector_signature(path) for path in sample["cache"]]
    )
    payload = {
        "runner_version": RUNNER_VERSION,
        "detector_signature": current_detector_signature(),
        "tracks": sample.track_id.tolist(),
        "cache_detector_signatures": cache_signatures,
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


def _parallel_sweep(
    sample: pd.DataFrame,
    variants: list[dict[str, Any]],
    output_dir: Path,
    workers: int,
    options: SweepOptions,
    force: bool,
) -> pd.DataFrame:
    signature = _signature(sample, variants, options)
    checkpoint_dir = output_dir / "checkpoints" / signature
    completed_rows: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for row in sample.to_dict("records"):
        checkpoint = _checkpoint_path(checkpoint_dir, row["track_id"])
        if checkpoint.exists() and (not force):
            completed_rows.extend(json.loads(checkpoint.read_text()))
        else:
            pending.append(row)
    if pending:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=max(1, workers), mp_context=context
        ) as pool:
            futures = {pool.submit(_run_track, row, variants): row for row in pending}
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="HMM sweep tracks"
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
                            **variant,
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
                completed_rows.extend(track_rows)
    rows = pd.DataFrame(completed_rows)
    rows.to_csv(output_dir / "rows.csv", index=False)
    _write_json(
        output_dir / "metadata.json",
        {
            "signature": signature,
            "runner_version": RUNNER_VERSION,
            "detector_signature": current_detector_signature(),
            "cache_detector_signatures": sorted(
                {
                    value
                    for value in sample.get(
                        "detector_signature", pd.Series(dtype=object)
                    ).dropna()
                }
            ),
            "tracks": len(sample),
            "variants": len(variants),
            "workers": workers,
            "options": asdict(options),
            "parameter_axes": {
                name: sorted(rows[name].dropna().unique().tolist())
                for name in PARAM_COLUMNS
                if name in rows
            },
        },
    )
    return rows


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    valid = rows.loc[rows["error"].isna()].copy()
    if valid.empty:
        return pd.DataFrame()
    keys = PARAM_COLUMNS
    numeric = [*METRIC_COLUMNS, "voiced_fraction", "smoother_compute_time"]
    strata = (
        valid.groupby([*keys, "ensemble", "instrument"], dropna=False)[numeric]
        .mean()
        .reset_index()
    )
    summary = strata.groupby(keys, dropna=False)[numeric].mean().reset_index()
    track_counts = valid.groupby(keys, dropna=False).track_id.nunique().rename("Tracks")
    error_counts = (
        rows.groupby(keys, dropna=False)
        .error.apply(lambda column: column.notna().sum())
        .rename("Errors")
    )
    summary = summary.merge(track_counts.reset_index(), on=keys, how="left")
    summary = summary.merge(error_counts.reset_index(), on=keys, how="left")
    return summary.sort_values(
        ["Overall Accuracy", "Raw Pitch Accuracy", "Voicing False Alarm"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _parameter_distance(frame: pd.DataFrame, target: dict[str, Any]) -> pd.Series:
    distance = frame["observation_mode"].ne(target["observation_mode"]).astype(float)
    distance += (
        frame["max_jump_bins"].astype(float) - target["max_jump_bins"]
    ).abs() / 25.0
    distance += (
        frame["switch_prob"].astype(float) - target["switch_prob"]
    ).abs() / 0.01
    distance += (frame["yin_trust"].astype(float) - target["yin_trust"]).abs() / 0.5
    return distance


def choose_parameters(summary: pd.DataFrame) -> dict[str, Any]:
    candidates = summary.copy()
    if candidates.empty:
        raise RuntimeError("HMM sweep produced no valid aggregate rows")
    candidates["_production_distance"] = _parameter_distance(
        candidates, production_parameters()
    )
    candidates = candidates.sort_values(
        [
            "Overall Accuracy",
            "Raw Pitch Accuracy",
            "Raw Chroma Accuracy",
            "Voicing False Alarm",
            "Voicing Recall",
            "_production_distance",
        ],
        ascending=[False, False, False, True, False, True],
        kind="mergesort",
    )
    row = candidates.iloc[0]
    return {
        name: row[name].item() if isinstance(row[name], np.generic) else row[name]
        for name in PARAM_COLUMNS
    }


def run_sweep(
    output_dir: Path | str | None = None,
    workers: int | None = None,
    options: SweepOptions | None = None,
    axes: ParameterAxes | None = None,
    force: bool = False,
    variant_limit: int | None = None,
    require_current_cache: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    options = options or SweepOptions()
    axes = axes or ParameterAxes()
    sample = select_sample(options)
    sample = sample.copy()
    sample["detector_signature"] = [
        cache_detector_signature(cache) for cache in sample["cache"]
    ]
    expected_detector_signature = current_detector_signature()
    stale_caches = sample.loc[
        sample["detector_signature"].ne(expected_detector_signature), "cache"
    ].tolist()
    if stale_caches and require_current_cache:
        raise RuntimeError(
            f"{len(stale_caches)}/{len(sample)} selected raw pitch caches were not generated by the current Stage-1 detector. Run benchmarks/notebooks/sweeps/scripts/regenerate_pitch_caches.py first."
        )
    if stale_caches:
        print(
            f"WARNING: {len(stale_caches)}/{len(sample)} selected raw pitch caches lack the current detector signature.",
            flush=True,
        )
    sample.to_csv(output_dir / "sample.csv", index=False)
    variants = parameter_variants(parameter_grid(axes))
    if variant_limit is not None:
        variants = variants[:variant_limit]
    if not variants:
        raise ValueError("the HMM parameter grid is empty")
    worker_count = workers or max(1, (os.cpu_count() or 4) - 1)
    print(
        f"HMM sweep: {len(sample)} stems x {len(variants)} variants; {worker_count} worker processes",
        flush=True,
    )
    print(f"results: {output_dir}", flush=True)
    rows = _parallel_sweep(sample, variants, output_dir, worker_count, options, force)
    summary = summarize(rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    recommendation = choose_parameters(summary)
    _write_json(
        output_dir / "recommendation.json",
        {
            **recommendation,
            "selection_scope": "all_tracks",
            "primary_metric": "Overall Accuracy",
            "tie_breakers": [
                "Raw Pitch Accuracy",
                "Raw Chroma Accuracy",
                "Voicing False Alarm",
                "Voicing Recall",
                "distance_to_production",
            ],
            "production": production_parameters(),
            "paper": paper_parameters(),
        },
    )
    return {
        "output_dir": output_dir,
        "sample": sample,
        "rows": rows,
        "summary": summary,
        "recommendation": recommendation,
    }


def load_results(output_dir: Path | str | None = None) -> dict[str, Any]:
    directory = Path(output_dir or DEFAULT_OUTPUT_DIR)
    result: dict[str, Any] = {"output_dir": directory}
    for name in ("sample", "rows", "summary"):
        path = directory / f"{name}.csv"
        if path.exists():
            result[name] = pd.read_csv(path)
    for name in ("metadata", "recommendation"):
        path = directory / f"{name}.json"
        if path.exists():
            result[name] = json.loads(path.read_text())
    return result
