import numpy as np
from PyQt6.QtCore import pyqtSignal, QObject

from app_logic.NoteData import Note, NoteData
from app_logic.user.ds.PitchData import Pitch, PitchData
from app_logic.user.ds.Recording import Recording
from app_logic.midi.ScoreData import ScoreData
from algorithms.Config import Config


class NoteDetector3(QObject):
    """
    Score-guided note detector (offline).

    Walks the pitch frames and the MIDI score notes with two pointers
    (j over frames, i_score over score notes). For each frame:
      1. The 'expected' MIDI is read from score[i_score].
      2. The pitch candidate closest in probability that lies within
         `prox_window` semitones of `expected` is chosen; otherwise we
         fall back to the most probable candidate.
      3. Segmentation is done string-edit-style:
            - chosen ~= expected           -> extend current segment
            - chosen ~= score[i_score+1]   -> user advanced; flush + i_score++
            - chosen far from both         -> insertion; flush, don't advance
            - sustained unvoiced           -> flush as rest
            - frame time > score[i_score].end_time + grace
              with no progress             -> deletion; advance i_score silently
            - chosen drifts from segment median by > pitch_thresh
                                           -> safety-net flush (score-blind)
    """

    note_detected = pyqtSignal(float)

    def __init__(self, recording: Recording = None, config: Config = None,
                 parent: QObject | None = None):
        super().__init__(parent)
        if recording is None and config is None:
            raise ValueError("Must provide either a recording or a config.")

        self.recording = recording
        self.config = recording.config if recording else config

        self.UNV_THRESH = self.config.unv_thresh
        self.PITCH_THRESH = self.config.pitch_thresh
        self.TOLERANCE = self.config.tolerance
        self.PROX_WINDOW = self.config.prox_window

        # frames of sustained-unvoiced before flushing the current segment
        self.MIN_UNV_RUN = max(2, self.config.h2 // 2)
        # extra time past a score note's end before we consider it 'deleted'
        self.DELETION_GRACE = 0.15  # seconds

    # ------------------------------------------------------------- selection

    def _pick_candidate(self, p: Pitch, expected: float | None) -> float:
        """
        Pick the MIDI value for frame `p` given an `expected` MIDI from the
        score (or None when no score note covers this frame).

        - Unvoiced -> -1
        - If `expected` is given, prefer the most-probable candidate within
          PROX_WINDOW semitones of expected.
        - Otherwise fall back to the top candidate.
        """
        if p is None or not p.candidates:
            return -1.0
        if p.unvoiced_prob > self.UNV_THRESH:
            return -1.0

        if expected is not None:
            # candidates are already sorted by descending probability
            for midi, _prob in p.candidates:
                if midi == -1:
                    continue
                if abs(midi - expected) <= self.PROX_WINDOW:
                    return float(midi)

        return float(p.candidates[0][0])

    # ----------------------------------------------------------- segmentation

    def _flush(self, nd: NoteData, note_idx: int,
               seg_pitches: list[float], seg_start_t: float,
               end_t: float) -> int:
        """Write the current segment to nd and return the next note_idx."""
        if seg_start_t is None or end_t is None or end_t <= seg_start_t:
            return note_idx

        if not seg_pitches or all(m == -1 for m in seg_pitches):
            midi_num = [-1.0, -1.0, -1.0]
        else:
            voiced = [m for m in seg_pitches if m != -1]
            m = float(np.median(voiced)) if voiced else -1.0
            midi_num = [m, m, m]

        n = Note(i=note_idx, start_time=seg_start_t,
                 end_time=end_t, midi_num=midi_num)
        nd.write_note(n)
        return note_idx + 1

    # ------------------------------------------------------------- detection

    def detect_notes(self, pitch_data: PitchData) -> NoteData:
        """
        Offline score-guided note detection.

        If no score is available on the recording, falls back to a
        score-blind pass that simply picks `candidates[0]` per frame.
        """
        nd = NoteData()
        frames: list[Pitch] = [p for p in pitch_data.data if p is not None]
        if not frames:
            return nd

        # pull score notes for the active instrument, if any
        score_notes: NoteData | None = None
        if self.recording is not None and self.recording.score_data is not None:
            sd: ScoreData = self.recording.score_data
            channel = self.recording.active_instrument
            score_notes = sd.note_datas.get(channel) if sd.note_datas else None
        n_score = len(score_notes.times) if score_notes else 0

        i_score = 0  # pointer into score_notes
        note_idx = 0

        seg_pitches: list[float] = []
        seg_start_t: float | None = None
        seg_match: int | None = None  # which score note this segment belongs to
        unv_run = 0  # consecutive unvoiced frames

        def expected_at(i: int) -> float | None:
            if score_notes is None or i >= n_score:
                return None
            return score_notes.read_note(i=i).midi_num[0]

        for p in frames:
            t = p.time
            expected = expected_at(i_score)
            next_expected = expected_at(i_score + 1)

            # time-based catch-up (deletion): advance score pointer past notes
            # whose window has clearly passed while we made no contact with them.
            while (score_notes is not None
                   and i_score < n_score
                   and t > score_notes.read_note(i=i_score).end_time + self.DELETION_GRACE
                   and (seg_match is None or seg_match != i_score)):
                i_score += 1
                expected = expected_at(i_score)
                next_expected = expected_at(i_score + 1)

            chosen = self._pick_candidate(p, expected)

            # --- start-of-stream ---
            if seg_start_t is None:
                seg_start_t = t
                seg_pitches = [chosen]
                if chosen != -1 and expected is not None and abs(chosen - expected) <= self.TOLERANCE:
                    seg_match = i_score
                else:
                    seg_match = None
                unv_run = 1 if chosen == -1 else 0
                continue

            # --- unvoiced frame ---
            if chosen == -1:
                unv_run += 1
                if unv_run >= self.MIN_UNV_RUN and any(m != -1 for m in seg_pitches):
                    # close out the previous voiced segment, start an unvoiced one
                    note_idx = self._flush(nd, note_idx, seg_pitches, seg_start_t, t)
                    if seg_match is not None:
                        i_score = max(i_score, seg_match + 1)
                    seg_pitches = [-1.0]
                    seg_start_t = t
                    seg_match = None
                else:
                    seg_pitches.append(chosen)
                continue
            unv_run = 0

            # --- voiced frame: classify against expected / next expected ---
            close_to_curr = (expected is not None
                             and abs(chosen - expected) <= self.TOLERANCE)
            close_to_next = (next_expected is not None
                             and abs(chosen - next_expected) <= self.TOLERANCE)

            # running median of voiced pitches in the segment (safety net)
            voiced_seg = [m for m in seg_pitches if m != -1]
            seg_med = float(np.median(voiced_seg)) if voiced_seg else None
            drifted = (seg_med is not None
                       and abs(chosen - seg_med) > self.PITCH_THRESH)

            if close_to_curr and seg_match == i_score:
                # extending the current matched segment
                seg_pitches.append(chosen)
                continue

            if close_to_next:
                # user advanced -> flush, advance pointer, start fresh
                note_idx = self._flush(nd, note_idx, seg_pitches, seg_start_t, t)
                i_score += 1
                seg_pitches = [chosen]
                seg_start_t = t
                seg_match = i_score  # now locked onto the new expected
                continue

            if close_to_curr:
                # segment wasn't matched yet but this frame matches expected;
                # close any in-progress mismatch (insertion) and lock on
                if seg_pitches and seg_med is not None and not drifted:
                    seg_pitches.append(chosen)
                    seg_match = i_score
                else:
                    note_idx = self._flush(nd, note_idx, seg_pitches, seg_start_t, t)
                    seg_pitches = [chosen]
                    seg_start_t = t
                    seg_match = i_score
                continue

            # far from both expected and next_expected
            if drifted:
                # safety net: pitch boundary even without score guidance
                note_idx = self._flush(nd, note_idx, seg_pitches, seg_start_t, t)
                if seg_match is not None:
                    # we'd been tracking score[i_score]; that note is done
                    i_score = max(i_score, seg_match + 1)
                seg_pitches = [chosen]
                seg_start_t = t
                seg_match = None  # insertion until proven otherwise
            else:
                # noisy frame but not a boundary; just accumulate
                seg_pitches.append(chosen)

        # --- flush the final segment ---
        end_t = frames[-1].time
        self._flush(nd, note_idx, seg_pitches, seg_start_t, end_t)
        return nd

    # ----------------------------------------------------- real-time (unused)

    def run(self, start_time: float = None):
        raise NotImplementedError("NoteDetector3 is offline-only.")

    def stop(self):
        pass
