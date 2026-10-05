"""PitchCache implementation and owned benchmark helpers."""

from __future__ import annotations
from dataclasses import asdict
import hashlib
import math
import pandas as pd
from pathlib import Path
from zipfile import BadZipFile
import os
import numpy as np
import json
import contextlib
import lzma
import pickle
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import ClassVar
from algorithms.Config import Config
from app_logic.user.ds.PitchData import Pitch
from app_logic.user.ds.PitchData import PitchData

try:
    import fcntl
except ImportError:
    fcntl = None


class PitchCache:
    """Storage adapters and provenance checks for all benchmark methods."""

    @staticmethod
    def _checkpoint_candidates(path):
        """Find identical per-track runs even when the suite's method list changed.

        The filename hashes the method, version, config, audio and reference identity.
        Only that exact filename is reusable; the suite folder is a report grouping.
        """
        path = Path(path)
        yield path
        checkpoint_dir = path.parent.parent
        if checkpoint_dir.name == "checkpoints":
            corpus_dir = checkpoint_dir.parent.parent
            for sibling in sorted(
                corpus_dir.glob(f"*/checkpoints/{path.parent.name}/{path.name}")
            ):
                if sibling != path:
                    yield sibling

    @staticmethod
    def read_latency_samples(row):
        """Read lossless per-update samples; summary-only checkpoints are stale."""
        with np.load(str(row["latency_samples_path"]), allow_pickle=False) as saved:
            samples = saved["output_latency_ms"]
            if (
                samples.ndim != 1
                or samples.size != int(row["latency_sample_count"])
                or (not np.all(np.isfinite(samples)))
                or np.any(samples < 0)
            ):
                raise ValueError("Invalid streaming latency samples")
            return samples

    @staticmethod
    def has_latency_samples(row):
        try:
            PitchCache.read_latency_samples(row)
            return True
        except (KeyError, OSError, ValueError, TypeError, EOFError, BadZipFile):
            return False

    @staticmethod
    def _write_streaming_checkpoint(path, rows):
        if path is None or not rows:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        for index, row in enumerate(rows):
            samples = row.pop("_latency_samples", None)
            if samples is None:
                continue
            sample_path = path.with_suffix(f".{index}.latency.npz")
            temporary = sample_path.with_suffix(f".{os.getpid()}.tmp")
            try:
                with temporary.open("wb") as output:
                    np.savez_compressed(output, **samples)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, sample_path)
            finally:
                temporary.unlink(missing_ok=True)
            row["latency_samples_path"] = str(sample_path.resolve())
            row["latency_sample_count"] = len(samples["output_latency_ms"])
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        try:
            with temporary.open("w") as output:
                json.dump(rows, output, default=lambda value: value.item())
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    "One track's raw and smoothed ``PitchData``, lzma-pickled under a file lock.\n\n    Attune writes it; the note, mistake, and vibrato benchmarks read it so they\n    never re-run pitch detection just to get a pitch track.\n    "
    VERSION: ClassVar[int] = 14
    RAW: ClassVar[str] = "raw"
    SMOOTHED: ClassVar[str] = "smoothed"

    @dataclass
    class Stages:
        """Both Attune stages from one detection pass, with their own timings."""

        data: dict[str, PitchData] = field(default_factory=dict)
        timing: dict[str, dict[str, Any]] = field(default_factory=dict)

        def __contains__(self, stage: str) -> bool:
            return stage in self.data

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    @classmethod
    def stage_name(cls, smooth: bool) -> str:
        return cls.SMOOTHED if smooth else cls.RAW

    @classmethod
    def path_for(cls, corpus_dir: Path | str, track_id: str) -> Path:
        safe_id = track_id.replace("/", "_")
        return Path(corpus_dir) / "pitch_data" / f"{safe_id}.pitch.pkl.xz"

    def has(self, stage: str) -> bool:
        if not self.path.exists():
            return False
        try:
            return stage in self._payload()[0]
        except Exception:
            return False

    def has_current_timing(self, stage: str, compute_clock: str) -> bool:
        """Whether a stage has timing safe to reuse in parallel benchmarks."""
        if not self.path.exists():
            return False
        try:
            stages, metadata = self._payload()
            current = (
                stage in stages
                and metadata.get(stage, {}).get("compute_clock") == compute_clock
            )
            if stage == self.SMOOTHED:
                current = current and (
                    self.RAW in stages
                    and metadata.get(self.RAW, {}).get("compute_clock") == compute_clock
                )
            return current
        except Exception:
            return False

    def read(
        self, stage: str, config: Config
    ) -> tuple[PitchData, dict[str, Any]] | None:
        if not self.path.exists():
            return None
        try:
            stages, metadata = self._payload()
        except Exception:
            return None
        if stage not in stages:
            return None
        body = stages[stage]
        pitch_data = PitchData(config=config)
        pitch_data.t_origin = float(body.get("t_origin", 0.0))
        pitch_data.data = [
            self._pitch_from_payload(entry, config) for entry in body.get("pitches", [])
        ]
        return (
            pitch_data,
            self.stage_timing(
                metadata.get(stage, {}),
                smooth=stage == self.SMOOTHED,
                raw_timing=metadata.get(self.RAW),
            ),
        )

    def write(self, stages: "PitchCache.Stages") -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock():
            merged: dict[str, Any] = {}
            merged_metadata: dict[str, Any] = {}
            if self.path.exists():
                try:
                    merged, merged_metadata = self._payload()
                except Exception:
                    merged, merged_metadata = ({}, {})
            for stage, pitch_data in stages.data.items():
                merged[stage] = {
                    "t_origin": float(pitch_data.t_origin),
                    "pitches": [
                        self._pitch_to_payload(pitch) for pitch in pitch_data.data
                    ],
                }
                merged_metadata[stage] = dict(stages.timing.get(stage, {}))
            payload = {
                "version": self.VERSION,
                "metadata": merged_metadata,
                "stages": merged,
            }
            tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
            try:
                with lzma.open(tmp, "wb") as handle:
                    pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
                tmp.replace(self.path)
            finally:
                tmp.unlink(missing_ok=True)
        return self.path

    @staticmethod
    def stage_timing(
        timing: dict[str, Any], smooth: bool, raw_timing: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Per-stage compute times, preferring the raw stage's own detector time."""
        raw_timing = raw_timing or {}
        detector = float(timing.get("pitch_detector_compute_time", 0.0) or 0.0)
        if smooth and raw_timing:
            detector = (
                float(raw_timing.get("pitch_detector_compute_time", 0.0) or 0.0)
                or detector
            )
        smoother = float(timing.get("pitch_smoother_compute_time", 0.0) or 0.0)
        components = detector + smoother if smooth else detector
        stored = float(timing.get("pitch_compute_time", 0.0) or 0.0)
        detector_wall = float(
            (
                raw_timing.get("wall_pitch_detector_compute_time")
                if smooth
                else timing.get("wall_pitch_detector_compute_time")
            )
            or timing.get("wall_pitch_detector_compute_time", 0.0)
            or 0.0
        )
        smoother_wall = float(
            timing.get("wall_pitch_smoother_compute_time", 0.0) or 0.0
        )
        wall_components = detector_wall + smoother_wall if smooth else detector_wall
        stored_wall = float(timing.get("wall_pitch_compute_time", 0.0) or 0.0)
        timing_clock = str(timing.get("compute_clock") or "legacy_wall")
        raw_clock = str(raw_timing.get("compute_clock") or "legacy_wall")
        compute_clock = (
            timing_clock
            if not smooth or timing_clock == raw_clock
            else "mixed_or_legacy"
        )
        return {
            "pitch_cache_version": int(
                timing.get("pitch_cache_version")
                or raw_timing.get("pitch_cache_version")
                or PitchCache.VERSION
            ),
            "pitch_detector_compute_time": detector,
            "pitch_smoother_compute_time": smoother if smooth else 0.0,
            "pitch_compute_time": components or stored,
            "wall_pitch_detector_compute_time": detector_wall,
            "wall_pitch_smoother_compute_time": smoother_wall if smooth else 0.0,
            "wall_pitch_compute_time": wall_components or stored_wall,
            "compute_clock": compute_clock,
        }

    @contextlib.contextmanager
    def _lock(self):
        lock_path = self.path.with_name(f"{self.path.name}.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "a", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _payload(self) -> tuple[dict[str, Any], dict[str, Any]]:
        with lzma.open(self.path, "rb") as handle:
            payload = pickle.load(handle)
        version = payload.get("version")
        if version not in (13, self.VERSION):
            raise ValueError(f"unsupported pitch cache version: {version}")
        stages = {
            str(stage): dict(body or {})
            for stage, body in (payload.get("stages") or {}).items()
        }
        if version == 13:
            stages.pop(self.SMOOTHED, None)
        raw_metadata = payload.get("metadata") or {}
        return (
            stages,
            {stage: dict(raw_metadata.get(stage, {}) or {}) for stage in stages},
        )

    @staticmethod
    def _pitch_to_payload(pitch: Pitch | None) -> dict[str, Any] | None:
        if pitch is None:
            return None
        pitch.ensure_compatible()
        return {
            "time": float(pitch.time),
            "value": float(pitch.value),
            "candidates": [
                (float(midi), float(prob)) for midi, prob in pitch.candidate_pitches
            ],
            "volume": float(pitch.volume),
            "unvoiced_prob": float(pitch.unvoiced_prob),
            "distance": float(getattr(pitch, "live_distance", 0.0) or 0.0),
            "align_distance": getattr(pitch, "aligned_distance", None),
            "is_transition": getattr(pitch, "is_transition", None),
        }

    @staticmethod
    def _pitch_from_payload(payload: Any, config: Config) -> Pitch | None:
        if payload is None:
            return None
        if isinstance(payload, Pitch):
            return payload.ensure_compatible(config)
        if not isinstance(payload, dict):
            candidates = (
                payload[8]
                if len(payload) > 8
                else payload[1] if len(payload) > 1 else []
            )
            pitch = Pitch(
                time=float(payload[0]) if payload else 0.0,
                candidates=[(float(midi), float(prob)) for midi, prob in candidates],
                value=(
                    float(payload[7])
                    if len(payload) > 7 and payload[7] is not None
                    else None
                ),
                volume=float(payload[2]) if len(payload) > 2 else 0.0,
                unvoiced_prob=(
                    float(payload[9])
                    if len(payload) > 9
                    else float(payload[3]) if len(payload) > 3 else 1.0
                ),
                live_distance=(
                    None
                    if len(payload) <= 4 or payload[4] is None
                    else float(payload[4])
                ),
                config=config,
            )
            pitch.aligned_distance = None if len(payload) <= 5 else payload[5]
            pitch.is_transition = None if len(payload) <= 6 else payload[6]
            return pitch.ensure_compatible(config)
        candidates = payload.get("candidate_pitches") or payload.get("candidates") or []
        posthoc_candidates = payload.get("posthoc_candidate_pitches")
        if posthoc_candidates is None:
            posthoc_candidates = payload.get("posthoc_candidates")
        if posthoc_candidates is not None:
            candidates = posthoc_candidates
        value = payload.get("value")
        pitch = Pitch(
            time=float(payload["time"]),
            candidates=[(float(midi), float(prob)) for midi, prob in candidates],
            value=None if value is None else float(value),
            volume=float(payload["volume"]),
            unvoiced_prob=float(
                payload.get("posthoc_unvoiced_prob", payload.get("unvoiced_prob", 1.0))
            ),
            live_distance=payload.get("live_distance", payload.get("distance")),
            config=config,
        )
        pitch.aligned_distance = payload.get(
            "aligned_distance", payload.get("align_distance")
        )
        pitch.is_transition = payload.get("is_transition")
        return pitch.ensure_compatible(config)

    @staticmethod
    def ablation_streaming_cache_path(
        self, example: PitchExample, variant: Variant
    ) -> Path:
        from benchmarks.modules.pitch.PitchNotebook import PitchNotebook

        return (
            self.streaming_run_root
            / "estimates"
            / variant.name
            / f"{example.dataset}__{example.safe_id}.npz"
        )

    @staticmethod
    def ablation_read_streaming_cache(self, path: Path) -> StreamingTrackResult | None:
        from benchmarks.modules.pitch.PitchNotebook import PitchNotebook
        from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase

        if not self.config.use_cache or self.config.force or (not path.exists()):
            return None
        try:
            with np.load(path, allow_pickle=False) as stored:
                if str(stored["version"].item()) != self.STREAMING_VERSION:
                    return None
                estimate = PitchDetectorBase.PitchEstimate.build(
                    stored["times"],
                    stored["freqs"],
                    float(stored["compute_time"].item()),
                    from_cache=True,
                    metadata={
                        "compute_clock": PitchDetectorBase.COMPUTE_CLOCK,
                        "wall_pitch_compute_time": float(
                            stored["wall_compute_time"].item()
                        ),
                    },
                )
                return PitchNotebook.StreamingTrackResult(
                    estimate=estimate,
                    duration=float(stored["duration"].item()),
                    sample_rate=int(stored["sample_rate"].item()),
                    frame_size=int(stored["frame_size"].item()),
                    integration_size=int(stored["integration_size"].item()),
                    hop_size=int(stored["hop_size"].item()),
                    latencies=np.asarray(stored["latencies"], dtype=np.float64),
                    call_cpus=np.asarray(stored["call_cpus"], dtype=np.float64),
                    call_walls=np.asarray(stored["call_walls"], dtype=np.float64),
                )
        except Exception:
            return None

    @staticmethod
    def ablation_write_streaming_cache(
        self, path: Path, result: StreamingTrackResult
    ) -> None:
        from benchmarks.modules.pitch.PitchNotebook import PitchNotebook

        if not self.config.use_cache:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            version=np.asarray(self.STREAMING_VERSION),
            times=result.estimate.times,
            freqs=result.estimate.freqs,
            compute_time=np.asarray(result.estimate.compute_seconds),
            wall_compute_time=np.asarray(
                result.estimate.metadata["wall_pitch_compute_time"]
            ),
            duration=np.asarray(result.duration),
            sample_rate=np.asarray(result.sample_rate),
            frame_size=np.asarray(result.frame_size),
            integration_size=np.asarray(result.integration_size),
            hop_size=np.asarray(result.hop_size),
            latencies=result.latencies,
            call_cpus=result.call_cpus,
            call_walls=result.call_walls,
        )
        temporary.replace(path)

    @staticmethod
    def ablation_streaming_threshold_evidence_path(
        self, example: PitchExample, *, prominence: bool
    ) -> Path:
        from benchmarks.modules.pitch.PitchNotebook import PitchNotebook

        candidate_mode = "prominent" if prominence else "ordinary"
        return (
            self.streaming_threshold_sweep_root
            / "evidence"
            / candidate_mode
            / f"{example.dataset}__{example.safe_id}.npz"
        )

    @staticmethod
    def ablation_read_streaming_threshold_evidence(
        self, path: Path, grid: StreamingGrid
    ) -> StreamingThresholdEvidence | None:
        from benchmarks.modules.pitch.PitchNotebook import PitchNotebook

        if not self.config.use_cache or self.config.force or (not path.exists()):
            return None
        try:
            with np.load(path, allow_pickle=False) as stored:
                if str(stored["version"].item()) != self.THRESHOLD_SWEEP_VERSION:
                    return None
                evidence = PitchNotebook.StreamingThresholdEvidence(
                    times=np.asarray(stored["times"], dtype=np.float64),
                    frequencies=np.asarray(stored["frequencies"], dtype=np.float64),
                    volumes=np.asarray(stored["volumes"], dtype=np.float64),
                    unvoiced_probabilities=np.asarray(
                        stored["unvoiced_probabilities"], dtype=np.float64
                    ),
                    duration=float(stored["duration"].item()),
                    sample_rate=int(stored["sample_rate"].item()),
                    frame_size=int(stored["frame_size"].item()),
                    integration_size=int(stored["integration_size"].item()),
                    hop_size=int(stored["hop_size"].item()),
                )
        except Exception:
            return None
        arrays = (
            evidence.times,
            evidence.frequencies,
            evidence.volumes,
            evidence.unvoiced_probabilities,
        )
        if any((len(values) != len(grid.times) for values in arrays)):
            return None
        if not np.array_equal(evidence.times, grid.times):
            return None
        if (
            evidence.sample_rate,
            evidence.frame_size,
            evidence.integration_size,
            evidence.hop_size,
        ) != (grid.sample_rate, grid.frame_size, grid.integration_size, grid.hop_size):
            return None
        if not math.isclose(
            evidence.duration, grid.duration, rel_tol=0.0, abs_tol=1e-12
        ):
            return None
        return evidence

    @staticmethod
    def ablation_write_streaming_threshold_evidence(
        self, path: Path, evidence: StreamingThresholdEvidence
    ) -> None:
        from benchmarks.modules.pitch.PitchNotebook import PitchNotebook

        if not self.config.use_cache:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            version=np.asarray(self.THRESHOLD_SWEEP_VERSION),
            times=evidence.times,
            frequencies=evidence.frequencies,
            volumes=evidence.volumes,
            unvoiced_probabilities=evidence.unvoiced_probabilities,
            duration=np.asarray(evidence.duration),
            sample_rate=np.asarray(evidence.sample_rate),
            frame_size=np.asarray(evidence.frame_size),
            integration_size=np.asarray(evidence.integration_size),
            hop_size=np.asarray(evidence.hop_size),
        )
        temporary.replace(path)

    @staticmethod
    def notebook_read_run(path: Path) -> pd.DataFrame | None:
        from benchmarks.modules.pitch.PitchNotebook import PitchNotebook

        "A finished run's rows, or None if it never completed."
        if not path.is_file():
            return None
        try:
            return pd.read_csv(path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            return None

    @staticmethod
    def streaming_checkpoint_path(self, directory, method, example):
        from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker

        if directory is None:
            return None
        stat = example.audio_path.stat()
        config = asdict(self.config)
        config.pop("workers")
        identity = {
            "version": self.VERSION,
            "method": method,
            "config": config,
            "options": vars(self.options),
            "example": asdict(example),
            "audio_size": stat.st_size,
            "audio_mtime_ns": stat.st_mtime_ns,
            "reference": hashlib.sha256(
                example.ref_times.tobytes() + example.ref_freqs.tobytes()
            ).hexdigest(),
        }
        identity["options"] = {
            k: v
            for k, v in identity["options"].items()
            if k not in ("workers", "force", "use_cache", "force_reanalysis")
        }
        identity["example"].pop("ref_times")
        identity["example"].pop("ref_freqs")
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, default=str).encode()
        ).hexdigest()
        return Path(directory) / method / f"{digest}.json"
