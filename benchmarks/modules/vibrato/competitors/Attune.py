from __future__ import annotations

import numpy as np

from algorithms.Config import Config
from algorithms.VibratoDetector import VibratoDetector
from app_logic.NoteData import Note, NoteData
from app_logic.user.ds.PitchData import Pitch, PitchData
from benchmarks.modules.vibrato.VibratoDetectorBase import (
    VibratoDetectorBase,
    VibratoEstimate,
    VibratoExample,
)


class Attune(VibratoDetectorBase):
    """Adapter for Attune's production whole-note vibrato detector."""

    name = "attune"
    description = "Attune whole-note time-varying variable-projection model"
    scores_center = True

    def __init__(self, **config_overrides: float | bool) -> None:
        self.config_overrides = dict(config_overrides)

    def set_config_overrides(self, **overrides: float | bool) -> None:
        self.config_overrides = dict(overrides)

    @staticmethod
    def _config(frame_rate: float, **overrides: float | bool) -> Config:
        hop = 128
        sample_rate = int(round(frame_rate * hop))
        if not np.isclose(sample_rate / hop, frame_rate, atol=1e-6, rtol=0.0):
            raise ValueError(
                f"frame rate {frame_rate:g} cannot be represented on the benchmark grid"
            )
        parameters = dict(
            sr=sample_rate,
            h1=hop,
            w1=0,
            fmin=20.0,
            fmax=5000.0,
            min_volume=0.0,
        )
        parameters.update(overrides)
        return Config(**parameters)

    @staticmethod
    def _inputs(
        example: VibratoExample,
        config: Config,
    ) -> tuple[PitchData, NoteData]:
        pitch_data = PitchData(config)
        pitch_data.t_origin = float(example.times[0])
        transitions = (
            np.asarray(example.transition_mask, dtype=bool)
            if example.transition_mask is not None
            else np.zeros(len(example.times), dtype=bool)
        )
        pitches: list[Pitch] = []
        for time, value, is_transition in zip(
            example.times, example.pitch_midi, transitions
        ):
            voiced = bool(np.isfinite(value))
            pitch = Pitch(
                time=float(time),
                volume=1.0 if voiced else 0.0,
                unvoiced_prob=0.0 if voiced else 1.0,
                live_distance=0.0,
                config=config,
                candidates=[(float(value), 1.0)] if voiced else [],
                value=float(value) if voiced else -1.0,
            )
            pitch.is_transition = bool(is_transition and voiced)
            pitches.append(pitch)
        pitch_data.load(pitches)

        note_data = NoteData()
        raw_bounds = example.metadata.get("analysis_note_bounds")
        if not isinstance(raw_bounds, (list, tuple)) or not raw_bounds:
            note_start, note_end = example.scored_time_bounds
            center = example.center_midi[
                example.score_mask & np.isfinite(example.center_midi)
            ]
            representative = float(np.median(center)) if center.size else 60.0
            raw_bounds = [(note_start, note_end, representative)]
        for index, bounds in enumerate(raw_bounds):
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 3:
                raise ValueError(
                    "analysis_note_bounds entries must be (start, end, midi) triples"
                )
            start, end, midi = map(float, bounds)
            if not np.all(np.isfinite((start, end, midi))) or end <= start:
                raise ValueError(f"invalid analysis note bounds: {bounds!r}")
            note_data.write_note(
                Note(
                    i=index,
                    start_time=start,
                    end_time=end,
                    midi_num=[midi],
                )
            )
        return pitch_data, note_data

    def _implementation(self, config: Config) -> VibratoDetector:
        return VibratoDetector(config=config)

    def estimate(self, example: VibratoExample) -> VibratoEstimate:
        config = self._config(example.frame_rate, **self.config_overrides)
        pitch_data, note_data = self._inputs(example, config)
        implementation = self._implementation(config)
        data = implementation.detect(pitch_data, note_data=note_data)
        n = len(example.times)

        def take(source: np.ndarray, fill: float) -> np.ndarray:
            output = np.full(n, fill, dtype=np.float64)
            count = min(n, data.computed_until, len(source))
            output[:count] = source[:count]
            return output

        rates = take(data.rates, 0.0)
        widths = take(data.widths, 0.0)
        qualities = take(data.qualities, 0.0)
        centers = take(data.centers, np.nan)
        rates[~np.isfinite(rates)] = 0.0
        widths[~np.isfinite(widths)] = 0.0
        fit_confidence = np.ones(n, dtype=np.float64)
        for start, end in implementation._note_frame_spans(pitch_data, note_data, n):
            lo = max(0, start)
            hi = min(n, end)
            fit_confidence[lo:hi] = implementation._onset_confidence(
                hi - lo, example.frame_rate
            )
        return VibratoEstimate(
            rates,
            widths,
            (rates > 0.0) & (widths > 0.0),
            centers if np.isfinite(centers).any() else None,
            qualities,
            metadata={"fit_confidence": fit_confidence},
        )
