from __future__ import annotations
from typing import Any
from typing import Sequence
import numpy as np
import ruptures as rpt
from app_logic.NoteData import Note
from app_logic.NoteData import NoteData
from app_logic.user.ds.PitchData import Pitch
from app_logic.user.ds.PitchData import PitchData
from benchmarks.modules.note.NoteDetectorBase import NoteDetectorBase


class SlopeWindow(NoteDetectorBase):
    """Historical slope-aware window segmentation."""

    def detect(
        self,
        w2: int = 29,
        h2: int | None = None,
        slope_thresh: float | None = None,
        pitch_thresh: float = 0.75,
        unv_ratio: float = 0.8,
        refine_onsets: bool = True,
        exclude_transitions: bool = True,
        **_unused: Any,
    ) -> NoteData:
        """The original (pre-PELT) sliding-window, slope-aware note detector.

        Faithful port of Attune's production ``NoteDetector.detect_notes`` at commit
        41958c5: slide a width-``w2`` window (stride ``h2``) over the pitch track;
        a window starts/continues a note when it is flat-and-voiced, and a *new*
        note begins only when the window's median pitch differs from the running
        note by more than ``pitch_thresh`` AND the window itself is flat or
        unvoiced (so mid-slide windows never spawn notes). The hop-quantized
        boundaries are then pulled onto their true onsets by ``refine_onsets``.

        Config params ``w2``/``h2``/``slope_thresh``/``pitch_thresh``/``unv_ratio``
        no longer exist on the live Config, so they are passed explicitly here with
        the 41958c5 defaults (w2=29, h2=w2-10, slope_thresh=0.75/w2,
        pitch_thresh=0.75, unv_ratio=0.8). ``find_best_w2`` is intentionally not
        run: it re-fits the score/mistake pipeline to pick w2, which would tune the
        detector against the reference. A fixed w2 keeps the comparison honest.
        """
        pitch_data = self.recording.pitch_data
        pitches = pitch_data.data
        w = max(2, int(w2))
        hop = max(1, int(h2) if h2 is not None else w - 10)
        slope_cut = 0.75 / w if slope_thresh is None else float(slope_thresh)
        unv_thresh = self.config.unv_thresh
        if exclude_transitions:
            self._prepare_transition_flags(True)
        else:
            for p in pitches:
                if p is not None:
                    p.is_transition = False

        def is_unvoiced(window: list[Pitch]) -> bool:
            probs = [p.unvoiced_prob if p else 1.0 for p in window]
            return sum((pr > unv_thresh for pr in probs)) > unv_ratio * len(probs)

        def handle_window(window: list[Pitch]):
            slope, _ = self._get_slope(window)
            is_flat = abs(slope) < slope_cut
            is_unv = is_unvoiced(window)
            med = self._get_median_pitches(window)
            return (is_flat, is_unv, med)

        nd = NoteData()
        prev_note = None
        prev_time = None
        note_index = 0
        last_i = last_t = None
        for i in range(0, len(pitches) - w - 1, hop):
            window = pitches[i : i + w]
            if not window or window[0] is None:
                continue
            t = window[0].time
            last_i, last_t = (i, t)
            is_flat, is_unv, med = handle_window(window)
            if prev_note is None:
                if is_unv:
                    prev_note = [-1, -1, -1]
                elif is_flat:
                    prev_note = med
                prev_time = t
                continue
            if abs(prev_note[0] - med[0]) > pitch_thresh:
                if not is_flat and (not is_unv):
                    continue
                nd.write_note(
                    Note(
                        i=note_index,
                        start_time=prev_time,
                        end_time=t,
                        midi_num=prev_note,
                    )
                )
                note_index += 1
                prev_note = [-1, -1, -1] if is_unv else med
                prev_time = t
        if (
            prev_note is not None
            and prev_time is not None
            and (last_t is not None)
            and (last_t > prev_time)
        ):
            nd.write_note(
                Note(
                    i=note_index,
                    start_time=prev_time,
                    end_time=last_t,
                    midi_num=prev_note,
                )
            )
        if refine_onsets and len(nd.times) >= 2:
            nd = self._refine_onsets_from_pitch_data(nd, pitch_data)
        return self._reindex(nd)

    def _get_slope(self, pitches: Sequence[Pitch]) -> tuple[float, float]:
        mask = np.asarray(
            [
                p is not None
                and p.value != -1
                and (p.unvoiced_prob < self.config.unv_thresh)
                for p in pitches
            ],
            dtype=bool,
        )
        if not mask.any():
            return (0.0, 0.0)
        x_all = np.linspace(start=0, stop=len(pitches), num=len(pitches))
        x = x_all[mask]
        y = np.asarray([p.value for p, keep in zip(pitches, mask) if keep], dtype=float)
        slope, intercept = np.linalg.lstsq(
            np.vstack([x, np.ones_like(x)]).T, y, rcond=None
        )[0]
        return (float(slope), float(intercept))

    @staticmethod
    def _changepoint(signal: np.ndarray) -> int | None:
        if len(signal) < 2 or np.ptp(signal) == 0:
            return None
        algo = rpt.Dynp(model="l2", min_size=1, jump=1).fit(signal.reshape(-1, 1))
        return int(algo.predict(n_bkps=1)[0])

    def _find_pitch_crossing(
        self, pitches: Sequence[Pitch], lo: int, hi: int
    ) -> int | None:
        idx = [
            k for k in range(lo, hi) if self._frame_pitch(pitches[k]) not in (None, -1)
        ]
        if len(idx) < 2:
            return None
        signal = np.asarray([self._frame_pitch(pitches[k]) for k in idx], dtype=float)
        split = self._changepoint(signal)
        return idx[split] if split is not None else None

    def _find_voicing_change(
        self, pitches: Sequence[Pitch], lo: int, hi: int
    ) -> int | None:
        signal = np.asarray(
            [
                0.0 if self._frame_pitch(pitches[k]) in (None, -1) else 1.0
                for k in range(lo, hi)
            ],
            dtype=float,
        )
        split = self._changepoint(signal)
        return lo + split if split is not None else None

    def _refine_onsets_from_pitch_data(
        self, note_data: NoteData, pitch_data: PitchData
    ) -> NoteData:
        notes = note_data.read(i=0, j=len(note_data.times))
        if len(notes) < 2:
            return note_data
        pitches = pitch_data.data
        radius = self.ONSET_REFINE_RADIUS
        frame_dt = self.config.h1 / self.config.sr
        n_frames = len(pitches)
        for a, b in zip(notes, notes[1:]):
            bound_idx = pitch_data.time_to_index(a.end_time)
            a_start = pitch_data.time_to_index(a.start_time)
            b_end = pitch_data.time_to_index(b.end_time)
            lo = max(a_start + 1, bound_idx - radius)
            hi = min(b_end, bound_idx + radius, n_frames)
            if lo >= hi:
                continue
            a_voiced = bool(a.midi_num and a.midi_num[0] != -1)
            b_voiced = bool(b.midi_num and b.midi_num[0] != -1)
            if a_voiced and b_voiced:
                k = self._find_pitch_crossing(pitches, lo, hi)
            elif a_voiced != b_voiced:
                k = self._find_voicing_change(pitches, lo, hi)
            else:
                k = None
            if k is None:
                continue
            new_t = pitch_data.t_origin + k * frame_dt
            if a.start_time < new_t < b.end_time:
                a.end_time = new_t
                b.start_time = new_t
        refined = NoteData()
        for idx, note in enumerate(notes):
            note.id = idx
            refined.write_note(note)
        return refined

    ONSET_REFINE_RADIUS = 29
