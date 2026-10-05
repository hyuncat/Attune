"""Single-pass tempo fitting must preserve performance notes and realign score references."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import pretty_midi

from app_logic.NoteData import Note, NoteData
from app_logic.user.ds.Recording import Recording
from benchmarks.modules.note.NoteBenchmarker import OneInstrumentScoreData


class RobustFitFlowTests(unittest.TestCase):
    def test_missing_anchors_preserve_initial_alignment(self):
        rec = Recording()
        initial = rec.alignment
        rec.detect_mistakes = Mock()
        self.assertFalse(rec.refit_score_alignment_once())
        self.assertIs(rec.alignment, initial)
        rec.detect_mistakes.assert_not_called()

    def test_one_fit_then_one_alignment_without_refinement(self):
        rec = Recording()
        calls = []
        rec.resize_score_to_aligned_onsets = Mock(
            side_effect=lambda: calls.append("fit") or True
        )
        rec.detect_mistakes = Mock(side_effect=lambda **kw: calls.append("align"))
        rec.repeat_splitter.check_mistakes = Mock()
        self.assertTrue(rec.refit_score_alignment_once(verbose=True))
        self.assertEqual(calls, ["fit", "align"])
        rec.detect_mistakes.assert_called_once_with(verbose=True)
        rec.repeat_splitter.check_mistakes.assert_not_called()

    def test_robust_fit_replaces_biased_endpoint_fit_without_moving_notes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "score.mid"
            midi = pretty_midi.PrettyMIDI()
            instrument = pretty_midi.Instrument(40)
            instrument.notes = [
                pretty_midi.Note(90, 60 + i, float(i), i + 0.5) for i in range(8)
            ]
            midi.instruments.append(instrument)
            midi.write(str(path))
            rec = Recording(score_data=OneInstrumentScoreData(path))
            rec.note_data = NoteData()
            score = list(rec.score_data.note_data.data.values())
            original_bpm = rec.score_data.bpm
            for i, n in enumerate(score):
                onset = 2 + 1.2 * n.start_time + (0.4 if i == len(score) - 1 else 0)
                rec.note_data.write_note(
                    Note(i, onset, onset + 1.2 * n.duration(), list(n.midi_num))
                )
            before = [
                (id(n), n.start_time, n.end_time, tuple(n.midi_num))
                for n in rec.note_data.data.values()
            ]
            rec.resize_score(to_span="onset")
            rec.detect_mistakes()
            endpoint_bpm = rec.score_data.bpm
            self.assertTrue(rec.refit_score_alignment_once())
            self.assertAlmostEqual(rec.score_data.bpm, original_bpm / 1.2, places=5)
            self.assertNotAlmostEqual(endpoint_bpm, rec.score_data.bpm, places=3)
            self.assertEqual(
                before,
                [
                    (id(n), n.start_time, n.end_time, tuple(n.midi_num))
                    for n in rec.note_data.data.values()
                ],
            )
            current_ids = {id(n) for n in rec.score_data.note_data.data.values()}
            self.assertTrue(
                all(s is None or id(s) in current_ids for u, s in rec.alignment.pairs)
            )
            rec.stabilize_score_alignment()
            self.assertEqual(
                before,
                [
                    (id(n), n.start_time, n.end_time, tuple(n.midi_num))
                    for n in rec.note_data.data.values()
                ],
            )


if __name__ == "__main__":
    unittest.main()
