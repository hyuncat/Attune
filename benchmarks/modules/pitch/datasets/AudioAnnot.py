"""Corpora laid out as ``<root>/<name>/{audio/*.wav, annot/*.csv}``."""

from __future__ import annotations
from pathlib import Path
from typing import ClassVar
import numpy as np
import benchmarks.modules.pitch.PitchDetectorBase as _api_PitchDetectorBase
from benchmarks.modules.pitch.datasets.PitchDataset import PitchDataset, PitchTrack


class AudioAnnot(PitchDataset):
    """The synthesized monophonic corpora (MedleyDB stems/melodies, Bach10)."""

    NAMES: ClassVar[tuple[str, ...]] = (
        "mdb-stem-synth",
        "mdb-melody-synth",
        "bach10-mf0-synth",
    )
    BACH10_INSTRUMENT_NAMES: ClassVar[dict[str, str]] = {"saxphone": "saxophone"}

    def __init__(
        self,
        name: str = "bach10-mf0-synth",
        root: Path | str | None = None,
        instruments: list[str] | None = None,
        max_tracks: int | None = None,
        per_instrument: int | None = None,
        seed: int = 0,
    ) -> None:
        self.name = name
        self.root = Path(root) if root is not None else self.ROOT / name
        self.instruments = [value.lower() for value in instruments or []]
        self.max_tracks = max_tracks
        self.per_instrument = per_instrument
        self.seed = seed

    def metadata_for_wav(self, wav: Path) -> dict[str, str]:
        if self.name != "bach10-mf0-synth":
            return {}
        instrument = wav.stem.removesuffix(".RESYN").rsplit("_", 1)[-1].lower()
        return {"instrument": self.BACH10_INSTRUMENT_NAMES.get(instrument, instrument)}

    def tracks(self) -> list[PitchTrack]:
        found = [
            PitchTrack(
                track_id=wav.stem,
                dataset=self.name,
                audio_path=wav,
                annot_path=self.root / "annot" / f"{wav.stem}.csv",
                metadata=self.metadata_for_wav(wav),
            )
            for wav in sorted((self.root / "audio").glob("*.wav"))
            if (self.root / "annot" / f"{wav.stem}.csv").exists()
        ]
        if self.instruments:
            found = [
                track
                for track in found
                if any(
                    (
                        name in track.track_id.lower()
                        or name == track.metadata.get("instrument", "").lower()
                        for name in self.instruments
                    )
                )
            ]
        if self.per_instrument is not None:
            found = self.sample_tracks_per_instrument(
                found, count=self.per_instrument, seed=self.seed
            )
        return found[: self.max_tracks] if self.max_tracks is not None else found

    def reference(self, track: PitchTrack) -> tuple[FloatArray, FloatArray]:
        import pandas as pd

        table = pd.read_csv(
            track.annot_path, sep="\\s+|,", header=None, engine="python"
        )
        freqs = table[1].to_numpy(np.float64)
        return (table[0].to_numpy(np.float64), np.where(freqs > 0, freqs, 0.0))

    def cache_dir(self, track: PitchTrack) -> Path:
        return self.root
