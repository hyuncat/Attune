"""CocoChorales: per-stem pitch benchmarking over a pruned tiny download.

The useful unit is one stem, so a lightweight manifest is built from the
compressed ``main_dataset`` shards and only selected stems are materialized --
keeping the large ``mix.wav`` files and unrelated stems out of the working tree.

    main_dataset/<split>/<shard>.tar.bz2:
      <track>/stems_audio/<stem>.wav
      <track>/stems_midi/<stem>.mid
      <track>/metadata.yaml

    f0/<split>/<track>.pickle:   {voice_index: f0_hz_per_frame}
"""

from __future__ import annotations
import csv
import hashlib
import math
import os
import pickle
import random
import re
import shutil
import subprocess
import tarfile
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar
import numpy as np
import numpy.typing as npt
from benchmarks.modules.pitch.PitchCache import PitchCache
from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase
from benchmarks.modules.pitch.datasets.PitchDataset import PitchDataset, PitchTrack


class CocoChorales(PitchDataset):
    """Manifest, materialization, f0 references, and cache layout for one root."""

    name: ClassVar[str] = "coco"
    DEFAULT_ROOT_NAME: ClassVar[str] = "cocochorales_tiny"
    SPLITS: ClassVar[tuple[str, ...]] = ("train", "valid", "test")
    F0_FPS_DEFAULT: ClassVar[float] = float(os.environ.get("COCO_F0_FPS", "250"))
    BALANCED_SELECTION_POLICY: ClassVar[str] = (
        "seeded_proportional_ensemble_instrument_v1"
    )
    TRACK_RE: ClassVar[re.Pattern[str]] = re.compile(
        "^(?P<ensemble>string|brass|woodwind|random)_track\\d+$"
    )

    @dataclass(frozen=True)
    class Stem:
        """One manifest row: where a stem's audio, MIDI, and f0 live."""

        split: str
        shard: str
        track: str
        ensemble: str
        stem: str
        stem_voice: int
        f0_voice: int
        instrument: str
        wav_member: str
        midi_member: str
        metadata_member: str
        mix_midi_member: str
        f0_path: str
        FIELDS: ClassVar[tuple[str, ...]] = (
            "split",
            "shard",
            "track",
            "ensemble",
            "stem",
            "stem_voice",
            "f0_voice",
            "instrument",
            "wav_member",
            "midi_member",
            "metadata_member",
            "mix_midi_member",
            "f0_path",
        )

        @property
        def track_id(self) -> str:
            return f"{self.split}__{self.track}__{self.stem}"

        @classmethod
        def from_row(cls, row: dict[str, str]) -> "CocoChorales.Stem":
            data = dict(row)
            data["stem_voice"] = int(data["stem_voice"])
            data["f0_voice"] = int(data["f0_voice"])
            return cls(**data)

    def __init__(
        self,
        root: Path | str | None = None,
        f0_fps: float | None = None,
        split: str = "test",
        per_stratum: int | None = None,
        per_instrument: int | None = None,
        seed: int = 0,
        max_tracks: int | None = None,
        ensembles: Iterable[str] | None = None,
        instruments: Iterable[str] | None = None,
        shards: Iterable[str] | None = None,
        materialize: bool = False,
        noise_snrs: Iterable[float] | None = None,
    ) -> None:
        self.root = (
            Path(root) if root is not None else self.ROOT / self.DEFAULT_ROOT_NAME
        )
        self.f0_fps = float(self.F0_FPS_DEFAULT if f0_fps is None else f0_fps)
        self.split = split
        self.per_stratum = per_stratum
        self.per_instrument = per_instrument
        self.seed = seed
        self.max_tracks = max_tracks
        self.ensembles = list(ensembles or [])
        self.instruments = list(instruments or [])
        self.shards = list(shards or [])
        self.materialize = materialize
        self.noise_snrs = tuple((float(snr) for snr in noise_snrs or ()))

    @staticmethod
    def seed_for(seed: int, *parts: object) -> int:
        """A stable non-negative 63-bit seed, independent of hash ordering."""
        payload = "\x00".join((str(int(seed)), *(str(part) for part in parts)))
        digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=False) & (1 << 63) - 1

    @property
    def manifest_dir(self) -> Path:
        return self.root / "manifest"

    @property
    def materialized_dir(self) -> Path:
        return self.root / "materialized"

    def manifest_path(self, split: str) -> Path:
        return self.manifest_dir / f"{split}_stems.csv"

    def f0_path_for_track(self, split: str, track_name: str) -> Path | None:
        for ext in (".pickle", ".pkl", ".npz", ".npy", ".csv"):
            candidate = self.root / "f0" / split / f"{track_name}{ext}"
            if candidate.exists():
                return candidate
        return None

    def f0_path_for_record(self, record: "CocoChorales.Stem") -> Path:
        return self.root / record.f0_path

    def materialized_track_dir(self, record: "CocoChorales.Stem") -> Path:
        return self.materialized_dir / record.split / record.track

    def materialized_wav_path(self, record: "CocoChorales.Stem") -> Path:
        return (
            self.materialized_track_dir(record) / "stems_audio" / f"{record.stem}.wav"
        )

    def materialized_midi_path(self, record: "CocoChorales.Stem") -> Path:
        return self.materialized_track_dir(record) / "stems_midi" / f"{record.stem}.mid"

    def local_wav_path(self, record: "CocoChorales.Stem") -> Path | None:
        extracted = (
            self.root
            / "main_dataset"
            / record.split
            / record.track
            / "stems_audio"
            / f"{record.stem}.wav"
        )
        return next(
            (p for p in (self.materialized_wav_path(record), extracted) if p.exists()),
            None,
        )

    def cache_dir(self, track: PitchTrack) -> Path:
        return self.root

    def cache_path_for_track(self, track_id: str) -> Path:
        return PitchCache.path_for(self.root, track_id)

    def cache_path_for_wav(self, wav_path: Path | str) -> Path:
        return self.cache_path_for_track(self.track_id_for_wav(wav_path))

    def pitch_cache_path(self, track: PitchTrack) -> Path:
        return self.cache_path_for_track(track.track_id)

    def result_label(self, track: PitchTrack) -> str:
        """Coco rows are compared per instrument, not per split."""
        label = f"coco_{track.metadata.get('instrument', 'unknown')}"
        snr = track.metadata.get("snr_label")
        if snr is None:
            return label
        label += f"__snr_{str(snr).lower()}"
        degradation_seed = track.metadata.get("degradation_seed")
        return (
            label
            if track.metadata.get("degradation") == "clean"
            else f"{label}__seed_{degradation_seed}"
        )

    @classmethod
    def _split_of(cls, path: Path | str) -> str:
        parts = set(Path(path).parts)
        return next((split for split in cls.SPLITS if split in parts), "unknown")

    @staticmethod
    def _norm_member(name: str) -> str:
        clean = str(PurePosixPath(name))
        while clean.startswith("./"):
            clean = clean[2:]
        if clean in ("", ".") or clean.startswith("../") or "/../" in clean:
            raise ValueError(f"unsafe tar member path: {name!r}")
        return clean

    @staticmethod
    def voice_of(stem_path: Path | str) -> int:
        head = Path(stem_path).stem.split("_", 1)[0]
        return int(head) if head.isdigit() else 0

    @staticmethod
    def instrument_of(stem_path: Path | str) -> str:
        parts = Path(stem_path).stem.split("_", 1)
        return parts[1] if len(parts) == 2 else "unknown"

    @classmethod
    def ensemble_of_track(cls, track_name: str) -> str:
        match = cls.TRACK_RE.match(track_name)
        return match.group("ensemble") if match else track_name.split("_track")[0]

    @classmethod
    def ensemble_of(cls, wav_path: Path | str) -> str:
        return cls.ensemble_of_track(Path(wav_path).parent.parent.name)

    @staticmethod
    def f0_voice_of(stem_voice: int) -> int:
        return stem_voice - 1 if 1 <= stem_voice <= 4 else stem_voice

    def track_id_for_wav(self, wav_path: Path | str) -> str:
        wav_path = Path(wav_path)
        track_dir = wav_path.parent.parent
        return f"{self._split_of(track_dir)}__{track_dir.name}__{wav_path.stem}"

    def meta_for_wav(self, wav_path: Path | str) -> dict[str, Any]:
        stem_voice = self.voice_of(wav_path)
        track_dir = Path(wav_path).parent.parent
        return {
            "split": self._split_of(track_dir),
            "track": track_dir.name,
            "ensemble": self.ensemble_of(wav_path),
            "instrument": self.instrument_of(wav_path),
            "voice": stem_voice,
            "f0_voice": self.f0_voice_of(stem_voice),
        }

    def read_manifest(self, split: str = "test") -> list["CocoChorales.Stem"]:
        path = self.manifest_path(split)
        if not path.exists():
            return []
        with path.open(newline="") as fh:
            return [self.Stem.from_row(row) for row in csv.DictReader(fh)]

    def write_manifest(
        self, records: Sequence["CocoChorales.Stem"], split: str
    ) -> Path:
        path = self.manifest_path(split)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.Stem.FIELDS)
            writer.writeheader()
            for record in records:
                writer.writerow(asdict(record))
        return path

    def load_or_build_manifest(
        self, split: str = "test", rebuild: bool = False
    ) -> list["CocoChorales.Stem"]:
        if split == "all":
            return [
                record
                for part in self.SPLITS
                for record in self.load_or_build_manifest(part, rebuild=rebuild)
            ]
        existing = self.read_manifest(split)
        if existing and (not rebuild):
            return existing
        return self.build_manifest(split=split)

    def build_manifest(
        self, split: str = "test", write: bool = True
    ) -> list["CocoChorales.Stem"]:
        """Scan retained main_dataset shards and write one row per usable stem."""
        records: dict[str, CocoChorales.Stem] = {}
        for record in self._scan_extracted(split):
            records[record.track_id] = record
        for record in self._scan_shards(split):
            records[record.track_id] = record
        out = sorted(records.values(), key=lambda r: (r.split, r.track, r.stem))
        if write:
            self.write_manifest(out, split)
        return out

    def _scan_extracted(self, split: str):
        base = self.root / "main_dataset"
        if not base.is_dir():
            return
        for stems_audio in sorted(base.glob("**/stems_audio")):
            track_dir = stems_audio.parent
            track_split = self._split_of(track_dir)
            if split not in ("all", track_split) or not self.TRACK_RE.match(
                track_dir.name
            ):
                continue
            f0_src = self.f0_path_for_track(track_split, track_dir.name)
            if f0_src is None:
                continue
            for wav in sorted(stems_audio.glob("*.wav")):
                midi = track_dir / "stems_midi" / f"{wav.stem}.mid"
                if not midi.exists():
                    midi = track_dir / "stems_MIDI" / f"{wav.stem}.mid"
                stem_voice = self.voice_of(wav)
                yield self.Stem(
                    split=track_split,
                    shard="",
                    track=track_dir.name,
                    ensemble=self.ensemble_of_track(track_dir.name),
                    stem=wav.stem,
                    stem_voice=stem_voice,
                    f0_voice=self.f0_voice_of(stem_voice),
                    instrument=self.instrument_of(wav),
                    wav_member=str(wav.relative_to(self.root)),
                    midi_member=(
                        str(midi.relative_to(self.root)) if midi.exists() else ""
                    ),
                    metadata_member=str(
                        (track_dir / "metadata.yaml").relative_to(self.root)
                    ),
                    mix_midi_member=str((track_dir / "mix.mid").relative_to(self.root)),
                    f0_path=str(f0_src.relative_to(self.root)),
                )

    def _scan_shards(self, split: str):
        base = self.root / "main_dataset"
        if not base.is_dir():
            return
        for shard in sorted(base.glob("**/*.tar.bz2")):
            shard_split = self._split_of(shard)
            if split not in ("all", shard_split):
                continue
            shard_rel = str(shard.relative_to(self.root))
            print(f"scanning {shard_rel}", flush=True)
            listing = subprocess.Popen(
                [shutil.which("bsdtar") or "tar", "-tjf", str(shard)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                env={**os.environ, "LC_ALL": "C"},
            )
            assert listing.stdout is not None
            for line in listing.stdout:
                raw = line.strip()
                if raw in ("", ".", "./"):
                    continue
                member = self._norm_member(raw)
                parts = PurePosixPath(member).parts
                if len(parts) != 3 or parts[1] != "stems_audio":
                    continue
                if not parts[2].lower().endswith(".wav"):
                    continue
                track, _, wav_name = parts
                if not self.TRACK_RE.match(track):
                    continue
                f0_src = self.f0_path_for_track(shard_split, track)
                if f0_src is None:
                    continue
                stem = Path(wav_name).stem
                stem_voice = self.voice_of(stem)
                yield self.Stem(
                    split=shard_split,
                    shard=shard_rel,
                    track=track,
                    ensemble=self.ensemble_of_track(track),
                    stem=stem,
                    stem_voice=stem_voice,
                    f0_voice=self.f0_voice_of(stem_voice),
                    instrument=self.instrument_of(stem),
                    wav_member=member,
                    midi_member=f"{track}/stems_midi/{stem}.mid",
                    metadata_member=f"{track}/metadata.yaml",
                    mix_midi_member=f"{track}/mix.mid",
                    f0_path=str(f0_src.relative_to(self.root)),
                )
            if listing.wait() != 0:
                raise RuntimeError(f"tar listing failed for {shard}")

    def select_records(
        self,
        split: str | None = None,
        per_stratum: int | None = None,
        per_instrument: int | None = None,
        seed: int | None = None,
        max_tracks: int | None = None,
        ensembles: Iterable[str] | None = None,
        instruments: Iterable[str] | None = None,
        shards: Iterable[str] | None = None,
        rebuild_manifest: bool = False,
        balanced: bool = False,
    ) -> list["CocoChorales.Stem"]:
        split = self.split if split is None else split
        per_stratum = self.per_stratum if per_stratum is None else per_stratum
        per_instrument = (
            self.per_instrument if per_instrument is None else per_instrument
        )
        seed = self.seed if seed is None else seed
        max_tracks = self.max_tracks if max_tracks is None else max_tracks
        ensembles = self.ensembles if ensembles is None else ensembles
        instruments = self.instruments if instruments is None else instruments
        shards = self.shards if shards is None else shards
        records = self.load_or_build_manifest(split, rebuild=rebuild_manifest)
        if shards:
            records = [r for r in records if self._matches_shard(r, shards)]
        if ensembles:
            want = {value.lower() for value in ensembles}
            records = [r for r in records if r.ensemble.lower() in want]
        if instruments:
            want = {value.lower() for value in instruments}
            records = [r for r in records if r.instrument.lower() in want]
        if per_stratum is not None and per_instrument is not None:
            raise ValueError("per_stratum and per_instrument are mutually exclusive")
        if per_stratum is not None:
            records = self.sample_records(records, per_stratum=per_stratum, seed=seed)
        if per_instrument is not None:
            records = self.sample_records_per_instrument(
                records, count=per_instrument, seed=seed
            )
        if max_tracks is not None:
            records = (
                self.sample_balanced_records(records, count=max_tracks, seed=seed)
                if balanced
                else records[:max_tracks]
            )
        return records

    @staticmethod
    def _matches_shard(record: "CocoChorales.Stem", shards: Iterable[str]) -> bool:
        name = Path(record.shard).name
        return any(
            (
                want and (want in (record.shard, name) or record.shard.endswith(want))
                for want in (value.strip() for value in shards)
            )
        )

    @staticmethod
    def sample_records(
        records: Sequence["CocoChorales.Stem"], per_stratum: int, seed: int = 0
    ) -> list["CocoChorales.Stem"]:
        rng = random.Random(seed)
        groups: dict[tuple[str, str], list[CocoChorales.Stem]] = defaultdict(list)
        for record in records:
            groups[record.ensemble, record.instrument].append(record)
        out: list[CocoChorales.Stem] = []
        for key in sorted(groups):
            group = sorted(groups[key], key=lambda r: r.track_id)
            rng.shuffle(group)
            out.extend(group[:per_stratum])
        return sorted(out, key=lambda r: r.track_id)

    @staticmethod
    def _proportional_quotas(group_sizes: dict[str, int], count: int) -> dict[str, int]:
        """Largest-remainder allocation, with no randomness of its own."""
        if count < 0:
            raise ValueError("count must be non-negative")
        available = sum(group_sizes.values())
        target = min(int(count), available)
        if target == 0 or available == 0:
            return {key: 0 for key in group_sizes}
        exact = {key: target * size / available for key, size in group_sizes.items()}
        quotas = {
            key: min(group_sizes[key], int(math.floor(value)))
            for key, value in exact.items()
        }
        remaining = target - sum(quotas.values())
        order = sorted(
            group_sizes, key=lambda key: (-(exact[key] - math.floor(exact[key])), key)
        )
        while remaining:
            progressed = False
            for key in order:
                if quotas[key] >= group_sizes[key]:
                    continue
                quotas[key] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
            if not progressed:
                raise RuntimeError("could not allocate balanced record quota")
        return quotas

    @classmethod
    def sample_records_per_instrument(
        cls, records: Sequence["CocoChorales.Stem"], count: int, seed: int = 0
    ) -> list["CocoChorales.Stem"]:
        """Seed-select equal stem counts for every instrument label."""
        if count <= 0:
            raise ValueError("per-instrument stem count must be positive")
        groups: dict[str, list[CocoChorales.Stem]] = defaultdict(list)
        for record in records:
            groups[record.instrument.lower()].append(record)
        selected: list[CocoChorales.Stem] = []
        for instrument in sorted(groups):
            group = sorted(groups[instrument], key=lambda record: record.track_id)
            random.Random(
                cls.seed_for(seed, cls.PER_INSTRUMENT_SELECTION_POLICY, instrument)
            ).shuffle(group)
            chosen: list[CocoChorales.Stem] = []
            used_sources: set[tuple[str, str]] = set()
            for record in group:
                source = (record.split, record.track)
                if source in used_sources:
                    continue
                chosen.append(record)
                used_sources.add(source)
                if len(chosen) == count:
                    break
            if len(chosen) < count:
                picked = {record.track_id for record in chosen}
                chosen.extend(
                    (record for record in group if record.track_id not in picked)
                )
                chosen = chosen[:count]
            if len(chosen) < count:
                raise RuntimeError(
                    f"only found {len(chosen)}/{count} stems for {instrument}"
                )
            selected.extend(chosen)
        return sorted(selected, key=lambda record: record.track_id)

    @classmethod
    def sample_balanced_records(
        cls, records: Sequence["CocoChorales.Stem"], count: int, seed: int = 0
    ) -> list["CocoChorales.Stem"]:
        """Seed-sample stems while fixing ensemble and instrument composition.

        Both quota levels follow their manifest proportions; only the concrete
        track chosen inside each (ensemble, instrument) stratum depends on seed.
        """
        if count <= 0:
            return []
        by_ensemble: dict[str, list[CocoChorales.Stem]] = defaultdict(list)
        for record in records:
            by_ensemble[record.ensemble].append(record)
        ensemble_quotas = cls._proportional_quotas(
            {key: len(group) for key, group in by_ensemble.items()}, count
        )
        selected: list[CocoChorales.Stem] = []
        used_tracks: set[tuple[str, str]] = set()
        for ensemble in sorted(by_ensemble):
            by_instrument: dict[str, list[CocoChorales.Stem]] = defaultdict(list)
            for record in by_ensemble[ensemble]:
                by_instrument[record.instrument].append(record)
            instrument_quotas = cls._proportional_quotas(
                {key: len(group) for key, group in by_instrument.items()},
                ensemble_quotas[ensemble],
            )
            for instrument in sorted(by_instrument):
                group = sorted(by_instrument[instrument], key=lambda r: r.track_id)
                random.Random(
                    cls.seed_for(
                        seed, cls.BALANCED_SELECTION_POLICY, ensemble, instrument
                    )
                ).shuffle(group)
                quota = instrument_quotas[instrument]
                chosen = [
                    record
                    for record in group
                    if (record.split, record.track) not in used_tracks
                ][:quota]
                if len(chosen) < quota:
                    picked = {record.track_id for record in chosen}
                    chosen.extend((r for r in group if r.track_id not in picked))
                    chosen = chosen[:quota]
                selected.extend(chosen)
                used_tracks.update(((r.split, r.track) for r in chosen))
        return sorted(selected, key=lambda record: record.track_id)

    def records_to_tracks(
        self, records: Sequence["CocoChorales.Stem"]
    ) -> list[PitchTrack]:
        tracks: list[PitchTrack] = []
        for record in records:
            wav = self.local_wav_path(record)
            if wav is None:
                continue
            base = PitchTrack(
                track_id=record.track_id,
                dataset=self.name,
                audio_path=wav,
                annot_path=self.f0_path_for_record(record),
                metadata=self.meta_for_wav(wav),
            )
            if not self.noise_snrs:
                tracks.append(base)
                continue
            for snr_db in self.noise_snrs:
                condition = PitchDetectorBase.AudioDegradation(
                    snr_db=snr_db,
                    seed=self.seed_for(self.seed, record.track_id, snr_db),
                )
                track_id = (
                    base.track_id
                    if condition.is_clean
                    else f"{base.track_id}__{condition.cache_tag}__seed_{self.seed}"
                )
                tracks.append(
                    PitchTrack(
                        track_id=track_id,
                        dataset=base.dataset,
                        audio_path=base.audio_path,
                        annot_path=base.annot_path,
                        metadata={
                            **base.metadata,
                            "source_track_id": base.track_id,
                            "degradation": (
                                "clean"
                                if condition.is_clean
                                else PitchDetectorBase.AudioDegradation.VERSION
                            ),
                            "degradation_seed": condition.seed,
                            "snr_db": condition.snr_db,
                            "snr_label": condition.label,
                        },
                        degradation=condition,
                    )
                )
        return tracks

    def tracks(self) -> list[PitchTrack]:
        records = self.select_records()
        if self.materialize:
            written = self.materialize_records(records)
            print(f"materialized {len(written)} file(s) for {self.split}", flush=True)
        return self.records_to_tracks(records)

    def materialize_records(
        self,
        records: Sequence["CocoChorales.Stem"],
        include_mix_midi: bool = False,
        force: bool = False,
    ) -> list[Path]:
        """Extract only the selected stem files from retained shards."""
        by_shard: dict[str, list[CocoChorales.Stem]] = defaultdict(list)
        for record in records:
            if record.shard:
                by_shard[record.shard].append(record)
        written: list[Path] = []
        for shard_rel, shard_records in sorted(by_shard.items()):
            needed: dict[str, Path] = {}
            for record in shard_records:
                members = [
                    record.wav_member,
                    record.midi_member,
                    record.metadata_member,
                ]
                if include_mix_midi:
                    members.append(record.mix_midi_member)
                for member in members:
                    if not member:
                        continue
                    dest = (
                        self.materialized_dir / record.split / self._norm_member(member)
                    )
                    if force or not dest.exists():
                        needed[self._norm_member(member)] = dest
            if not needed:
                continue
            print(f"materializing {len(needed)} file(s) from {shard_rel}", flush=True)
            with tarfile.open(self.root / shard_rel, "r:bz2") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    dest = needed.pop(self._norm_member(member.name), None)
                    if dest is None:
                        continue
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    source = archive.extractfile(member)
                    if source is None:
                        continue
                    with source, dest.open("wb") as out:
                        shutil.copyfileobj(source, out)
                    written.append(dest)
                    if not needed:
                        break
        return written

    @staticmethod
    def _flatten(values: Any) -> FloatArray:
        array = np.squeeze(np.asarray(values, dtype=np.float64))
        if array.ndim > 1:
            array = array.reshape(array.shape[0], -1)[:, 0]
        return array.reshape(-1)

    def _f0_from_obj(self, obj: Any, voice_idx: int) -> FloatArray:
        if isinstance(obj, np.ndarray) and obj.dtype == object and (obj.ndim == 0):
            obj = obj.item()
        if isinstance(obj, dict):
            for key in (self.f0_voice_of(voice_idx), voice_idx):
                for actual in (key, str(key)):
                    if actual in obj:
                        value = obj[actual]
                        if isinstance(value, dict):
                            value = value.get(
                                "f0_hz", value.get("f0", next(iter(value.values())))
                            )
                        return self._flatten(value)
            for key in ("f0_hz", "f0"):
                if key in obj:
                    return self._flatten(obj[key])
            return self._flatten(next(iter(obj.values())))
        if isinstance(obj, np.ndarray) and obj.dtype != object:
            if obj.ndim == 2 and obj.shape[1] == 4 and (obj.shape[0] > 4):
                return obj[:, self.f0_voice_of(voice_idx)].astype(float).reshape(-1)
        return self._flatten(obj)

    def load_f0(
        self, f0_src: Path | str, voice_idx: int
    ) -> tuple[FloatArray, FloatArray]:
        path = Path(f0_src)
        suffix = path.suffix.lower()
        if suffix == ".npz":
            stored = np.load(path, allow_pickle=True)
            keys = (str(self.f0_voice_of(voice_idx)), str(voice_idx), "f0_hz", "f0")
            key = next((k for k in keys if k in stored.files), stored.files[0])
            freqs = self._f0_from_obj(stored[key], voice_idx)
        elif suffix == ".csv":
            data = np.loadtxt(path, delimiter=",", ndmin=2)
            if data.shape[1] >= 2:
                times, freqs = (data[:, 0].astype(float), data[:, 1].astype(float))
                return (times, np.where(np.isfinite(freqs) & (freqs > 0), freqs, 0.0))
            freqs = self._flatten(data)
        else:
            if suffix == ".npy":
                obj = np.load(path, allow_pickle=True)
            else:
                with path.open("rb") as fh:
                    obj = pickle.load(fh)
            freqs = self._f0_from_obj(obj, voice_idx)
        freqs = np.where(np.isfinite(freqs) & (freqs > 0), freqs, 0.0)
        return (np.arange(freqs.size, dtype=np.float64) / self.f0_fps, freqs)

    def reference(self, track: PitchTrack) -> tuple[FloatArray, FloatArray]:
        return self.load_f0(track.annot_path, self.voice_of(track.audio_path))

    def search_range(
        self,
        freqs: npt.ArrayLike,
        low_percentile: float = 1.0,
        high_percentile: float = 99.5,
        pad_semitones: float = 2.0,
        floor_hz: float = 30.0,
        ceiling_hz: float = 3000.0,
    ) -> tuple[float, float]:
        """Percentile range: synthesized f0 tails would otherwise widen the search."""
        voiced = np.asarray(freqs, dtype=np.float64)
        voiced = voiced[np.isfinite(voiced) & (voiced > floor_hz)]
        if voiced.size == 0:
            return (196.0, 3000.0)
        low, high = np.percentile(voiced, [low_percentile, high_percentile])
        pad = 2 ** (pad_semitones / 12.0)
        return (float(max(floor_hz, low / pad)), float(min(ceiling_hz, high * pad)))

    def load_resampled_audio(self, wav_path: Path | str, target_sr: int):
        """An ``AudioData`` at ``target_sr``, for the Attune pipeline."""
        import librosa
        from algorithms.Config import Config
        from app_logic.user.ds.AudioData import AudioData

        samples, _ = librosa.load(str(wav_path), sr=int(target_sr), mono=True)
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        audio = AudioData(config=Config(sr=int(target_sr)))
        audio.data = samples
        audio.sr = int(target_sr)
        audio.capacity = samples.size
        audio.end_index = samples.size
        audio.t_origin = 0.0
        return audio

    def plan(self) -> None:
        """Print what the current filters select and how much is already local."""
        everything = self.load_or_build_manifest(self.split)
        selected = self.select_records()
        materialized = sum((1 for r in selected if self.local_wav_path(r) is not None))
        cached = sum(
            (1 for r in selected if self.cache_path_for_track(r.track_id).exists())
        )
        counts: dict[tuple[str, str], int] = defaultdict(int)
        for record in selected:
            counts[record.ensemble, record.instrument] += 1
        print(f"root:         {self.root}")
        print(f"manifest:     {self.manifest_path(self.split)}")
        print(f"split:        {self.split}  |  f0 fps: {self.f0_fps:g}")
        print(f"stems:        {len(everything)} in manifest | {len(selected)} selected")
        print(f"materialized: {materialized}/{len(selected)} selected stems")
        print(f"cached:       {cached}/{len(selected)} selected stems")
        for shard in sorted({r.shard for r in selected if r.shard}):
            print(f"  shard: {shard}")
        if self.per_stratum:
            print(
                f"sample:       <= {self.per_stratum} per (ensemble,instrument), seed={self.seed}"
            )
        print("\nstrata (ensemble, instrument -> stems):")
        for key in sorted(counts):
            print(f"  {key[0]:9s} {key[1]:16s} {counts[key]:>5}")

    def probe(self) -> None:
        """Report which parts of the download are present and parseable."""
        print(f"root: {self.root}  (exists={self.root.is_dir()})")
        for component in (
            "main_dataset",
            "f0",
            "manifest",
            "materialized",
            "pitch_data",
        ):
            path = self.root / component
            tars = len(list(path.glob("**/*.tar.bz2"))) if path.is_dir() else 0
            files = len(list(path.glob("**/*"))) if path.is_dir() else 0
            print(
                f"  {component:16s} present={path.is_dir()!s:5s} tar.bz2={tars:<4} entries={files}"
            )
        f0_files = sorted((self.root / "f0").glob("**/*.pickle"))
        if f0_files:
            print(f"\nf0 sample: {f0_files[0].relative_to(self.root)}")
            _, freqs = self.load_f0(f0_files[0], voice_idx=1)
            voiced = freqs[freqs > 0]
            print(
                f"  frames={freqs.size}  dur~{freqs.size / self.f0_fps:.2f}s @ {self.f0_fps:g}fps  voiced={np.mean(freqs > 0):.1%}  Hz[min..max]={(voiced.min() if voiced.size else 0):.1f}..{(voiced.max() if voiced.size else 0):.1f}"
            )

    def prune_f0_to_manifest(self, dry_run: bool = False) -> int:
        """Delete f0 pickles no manifest row references."""
        keep = {
            self.f0_path_for_record(record).resolve()
            for record in self.load_or_build_manifest(self.split)
        }
        f0_dir = self.root / "f0" / self.split
        if not f0_dir.is_dir():
            return 0
        stale = [p for p in f0_dir.glob("*.pickle") if p.resolve() not in keep]
        for path in stale:
            if not dry_run:
                path.unlink()
        return len(stale)
