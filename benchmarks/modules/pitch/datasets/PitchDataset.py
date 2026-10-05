"""Common API for the corpora the pitch benchmark runs over."""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import random
from typing import Any, ClassVar
import numpy as np
import numpy.typing as npt
from benchmarks.modules.pitch.PitchCache import PitchCache
from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase
from benchmarks.paths import DATASETS_ROOT


@dataclass(frozen=True)
class PitchTrack:
    """One benchmarkable track: an audio file plus its reference annotation.

    Cheap and picklable, so a work plan can be built once and shipped to workers
    that rebuild the dataset on the other side.
    """

    track_id: str
    dataset: str
    audio_path: Path
    annot_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)
    degradation: PitchDetectorBase.AudioDegradation | None = None

    @property
    def safe_id(self) -> str:
        return self.track_id.replace("/", "_")


class PitchDataset(ABC):
    """Turns a corpus on disk into ``PitchExample``s and owns its cache layout."""

    name: str = "dataset"
    ROOT = DATASETS_ROOT
    PER_INSTRUMENT_SELECTION_POLICY: ClassVar[str] = "seeded_per_instrument_v1"

    @abstractmethod
    def tracks(self) -> list[PitchTrack]:
        """Every track this dataset offers, after the instance's own filters."""

    @abstractmethod
    def reference(self, track: PitchTrack) -> tuple[FloatArray, FloatArray]:
        """Annotated (times, freqs) with 0 Hz for unvoiced."""

    @abstractmethod
    def cache_dir(self, track: PitchTrack) -> Path:
        """Corpus directory holding this track's estimate and pitch caches."""

    def result_label(self, track: PitchTrack) -> str:
        """CSV name a track's row is grouped under."""
        return track.dataset

    @staticmethod
    def seed_for(seed: int, *parts: object) -> int:
        """Stable per-group seed, independent of ordering and Python hashing."""
        payload = "\x00".join((str(int(seed)), *(str(part) for part in parts)))
        digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=False) & (1 << 63) - 1

    @classmethod
    def sample_tracks_per_instrument(
        cls, tracks: list[PitchTrack], count: int, seed: int
    ) -> list[PitchTrack]:
        """Choose exactly ``count`` tracks per annotated instrument."""
        if count <= 0:
            raise ValueError("per-instrument track count must be positive")
        groups: dict[str, list[PitchTrack]] = defaultdict(list)
        for track in tracks:
            instrument = str(track.metadata.get("instrument", "")).strip().lower()
            if not instrument:
                raise ValueError(
                    f"{track.dataset}/{track.track_id} has no instrument metadata"
                )
            groups[instrument].append(track)
        selected: list[PitchTrack] = []
        for instrument in sorted(groups):
            group = sorted(groups[instrument], key=lambda track: track.track_id)
            random.Random(
                cls.seed_for(seed, cls.PER_INSTRUMENT_SELECTION_POLICY, instrument)
            ).shuffle(group)
            if len(group) < count:
                raise RuntimeError(
                    f"only found {len(group)}/{count} tracks for {instrument}"
                )
            selected.extend(group[:count])
        return sorted(selected, key=lambda track: track.track_id)

    def search_range(self, freqs: npt.ArrayLike) -> tuple[float, float]:
        """Pitch range detectors search, taken from the reference melody."""
        voiced = np.asarray(freqs, dtype=np.float64)
        voiced = voiced[np.isfinite(voiced) & (voiced > 0)]
        if voiced.size == 0:
            return (196.0, 3000.0)
        return (float(voiced.min()), float(voiced.max()))

    def estimate_cache_dir(self, track: PitchTrack) -> Path:
        return self.cache_dir(track) / "competitor_pitch"

    def pitch_cache_path(self, track: PitchTrack) -> Path:
        return PitchCache.path_for(self.cache_dir(track), track.track_id)

    def example(self, track: PitchTrack, tighten_range: bool = True) -> PitchExample:
        ref_times, ref_freqs = self.reference(track)
        fmin, fmax = self.search_range(ref_freqs) if tighten_range else (196.0, 3000.0)
        return PitchDetectorBase.PitchExample(
            track_id=track.track_id,
            dataset=track.dataset,
            audio_path=Path(track.audio_path),
            ref_times=ref_times,
            ref_freqs=ref_freqs,
            fmin=fmin,
            fmax=fmax,
            estimate_cache_dir=self.estimate_cache_dir(track),
            stage_cache_path=self.pitch_cache_path(track),
            metadata={**track.metadata, "result_label": self.result_label(track)},
            degradation=track.degradation,
        )
