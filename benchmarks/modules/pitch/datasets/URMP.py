"""University of Rochester Multi-Modal Music Performance (URMP) stems."""

from __future__ import annotations
from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar
import numpy as np
import benchmarks.modules.pitch.PitchDetectorBase as _api_PitchDetectorBase
from benchmarks.modules.pitch.datasets.PitchDataset import PitchDataset, PitchTrack


class URMP(PitchDataset):
    """Real monophonic instrument stems with frame-level F0 annotations.

    The official download is laid out as one directory per piece.  Every
    ``AuSep_<part>_<instrument>_<piece>.wav`` stem is paired with the matching
    ``F0s_<part>_<instrument>_<piece>.txt`` annotation in that directory.
    """

    name: ClassVar[str] = "urmp"
    INSTRUMENT_NAMES: ClassVar[dict[str, str]] = {
        "bn": "bassoon",
        "cl": "clarinet",
        "db": "double bass",
        "fl": "flute",
        "hn": "horn",
        "ob": "oboe",
        "sax": "saxophone",
        "tba": "tuba",
        "tbn": "trombone",
        "tpt": "trumpet",
        "va": "viola",
        "vc": "cello",
        "vn": "violin",
    }
    ENSEMBLE_NAMES: ClassVar[dict[int, str]] = {
        2: "duet",
        3: "trio",
        4: "quartet",
        5: "quintet",
    }

    def __init__(
        self,
        root: Path | str | None = None,
        instruments: Iterable[str] | None = None,
        ensembles: Iterable[str] | None = None,
        max_tracks: int | None = None,
        per_instrument: int | None = None,
        seed: int = 0,
    ) -> None:
        self.root = Path(root) if root is not None else self.ROOT / self.name
        self.instruments = self._instrument_codes(instruments or ())
        self.ensembles = self._ensemble_sizes(ensembles or ())
        self.max_tracks = max_tracks
        self.per_instrument = per_instrument
        self.seed = seed

    @classmethod
    def _instrument_codes(cls, values: Iterable[str]) -> set[str]:
        aliases = {
            cls._normal(value): code
            for code, name in cls.INSTRUMENT_NAMES.items()
            for value in (code, name)
        }
        return {aliases.get(cls._normal(value), cls._normal(value)) for value in values}

    @classmethod
    def _ensemble_sizes(cls, values: Iterable[str]) -> set[int | str]:
        aliases = {
            cls._normal(value): size
            for size, name in cls.ENSEMBLE_NAMES.items()
            for value in (str(size), f"{size}-part", name)
        }
        return {aliases.get(cls._normal(value), cls._normal(value)) for value in values}

    @staticmethod
    def _normal(value: str) -> str:
        normalized = value.strip().lower().replace("_", " ").replace("-", " ")
        return " ".join(normalized.split())

    @staticmethod
    def _stem_fields(wav: Path) -> tuple[int, str, int, str]:
        payload = wav.stem.removeprefix("AuSep_")
        part, instrument, piece_number, piece = payload.split("_", 3)
        return (int(part), instrument.lower(), int(piece_number), piece)

    def tracks(self) -> list[PitchTrack]:
        if not self.root.is_dir():
            return []
        found: list[PitchTrack] = []
        for piece_dir in sorted(
            (path for path in self.root.iterdir() if path.is_dir())
        ):
            stems = sorted(piece_dir.glob("AuSep_*.wav"))
            if not stems:
                continue
            ensemble_size = len(stems)
            if self.ensembles and ensemble_size not in self.ensembles:
                continue
            ensemble = self.ENSEMBLE_NAMES.get(ensemble_size, f"{ensemble_size}-part")
            for wav in stems:
                try:
                    part, instrument_code, piece_number, piece = self._stem_fields(wav)
                except (TypeError, ValueError):
                    continue
                annotation = wav.with_name(f"F0s_{wav.stem.removeprefix('AuSep_')}.txt")
                if not annotation.is_file():
                    continue
                if self.instruments and instrument_code not in self.instruments:
                    continue
                found.append(
                    PitchTrack(
                        track_id=f"{piece_dir.name}/{wav.stem}",
                        dataset=self.name,
                        audio_path=wav,
                        annot_path=annotation,
                        metadata={
                            "track": piece,
                            "piece_number": piece_number,
                            "ensemble": ensemble,
                            "ensemble_size": ensemble_size,
                            "instrument": self.INSTRUMENT_NAMES.get(
                                instrument_code, instrument_code
                            ),
                            "instrument_code": instrument_code,
                            "voice": part,
                        },
                    )
                )
        if self.per_instrument is not None:
            found = self.sample_tracks_per_instrument(
                found, count=self.per_instrument, seed=self.seed
            )
        return found[: self.max_tracks] if self.max_tracks is not None else found

    def reference(self, track: PitchTrack) -> tuple[FloatArray, FloatArray]:
        table = np.loadtxt(track.annot_path, dtype=np.float64, ndmin=2)
        if table.shape[1] < 2:
            raise ValueError(f"{track.annot_path} must contain time and F0 columns")
        times = np.asarray(table[:, 0], dtype=np.float64)
        freqs = np.asarray(table[:, 1], dtype=np.float64)
        freqs = np.where(np.isfinite(freqs) & (freqs > 0), freqs, 0.0)
        return (times, freqs)

    def cache_dir(self, track: PitchTrack) -> Path:
        return self.root
