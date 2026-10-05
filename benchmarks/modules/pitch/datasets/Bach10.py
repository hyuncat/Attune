"""Original Bach10 v1.1 isolated recordings and fractional-MIDI F0 references.

Source: https://labsites.rochester.edu/air/resource.html
The release uses 46 ms windows, a 10 ms hop and a first center at 23 ms.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from benchmarks.modules.pitch.datasets.PitchDataset import PitchDataset, PitchTrack


class Bach10(PitchDataset):
    name = "bach10-original"
    INSTRUMENTS = ("violin", "clarinet", "saxphone", "bassoon")

    def __init__(
        self, root=None, instruments=(), max_tracks=None, per_instrument=None, seed=0
    ):
        self.root = Path(root) if root is not None else self.ROOT / self.name
        self.instruments = {
            s.lower().replace("saxphone", "saxophone") for s in instruments
        }
        self.max_tracks = max_tracks
        self.per_instrument = per_instrument
        self.seed = seed

    def tracks(self):
        found = []
        for annotation in sorted(self.root.glob("*/*-GTF0s.mat")):
            piece = annotation.name.removesuffix("-GTF0s.mat")
            for row, suffix in enumerate(self.INSTRUMENTS):
                instrument = suffix.replace("saxphone", "saxophone")
                if self.instruments and instrument not in self.instruments:
                    continue
                wav = annotation.with_name(f"{piece}-{suffix}.wav")
                if not wav.is_file():
                    raise FileNotFoundError(f"Missing original Bach10 stem: {wav}")
                found.append(
                    PitchTrack(
                        track_id=f"{piece}-{suffix}",
                        dataset=self.name,
                        audio_path=wav,
                        annot_path=annotation,
                        metadata={
                            "instrument": instrument,
                            "track": piece,
                            "f0_row": row,
                            "audio_source": "original_recording",
                        },
                    )
                )
        if not found:
            raise FileNotFoundError(
                f"Original Bach10 is missing at {self.root}. Extract Bach10 v1.1 "
                "here with its ten piece directories and *-GTF0s.mat files. "
                "See https://labsites.rochester.edu/air/resource.html. "
                "Resynthesized audio is not used as a fallback."
            )
        if self.per_instrument is not None:
            found = self.sample_tracks_per_instrument(
                found, self.per_instrument, self.seed
            )
        return found[: self.max_tracks] if self.max_tracks is not None else found

    def reference(self, track):
        pitches = np.asarray(loadmat(track.annot_path)["GTF0s"], dtype=float)
        if pitches.ndim != 2 or pitches.shape[0] != 4:
            raise ValueError(f"Expected four instrument rows in {track.annot_path}")
        midi = pitches[int(track.metadata["f0_row"])]
        freqs = np.zeros_like(midi)
        voiced = np.isfinite(midi) & (midi > 0)
        freqs[voiced] = 440.0 * 2.0 ** ((midi[voiced] - 69.0) / 12.0)
        return 0.023 + np.arange(len(midi)) * 0.010, freqs

    def cache_dir(self, track):
        return self.root
