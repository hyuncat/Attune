"""Production note orchestration checks; no audio analysis or benchmark runs."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from app_logic.user.ds.Recording import Recording
from benchmarks.modules.vibrato.competitors.Attune import Attune


class ProductionNotesTest(unittest.TestCase):
    def test_vibrato_receives_final_recovered_notes(self):
        rec = Mock()
        initial, recovered = object(), object()
        rec.note_data = initial
        def refine(**kwargs):
            rec.note_data = recovered
        rec.align_score_and_refine.side_effect = refine
        def vibrato(**kwargs):
            self.assertIs(rec.note_data, recovered)
        rec.recompute_vibrato.side_effect = vibrato
        self.assertIs(Recording.analyze_notes(rec), recovered)
        self.assertEqual([call[0] for call in rec.mock_calls], [
            'reset_analysis', 'detect_notes', 'align_score_and_refine',
            'update_alignment_distances', 'recompute_vibrato'])

    def test_no_detected_notes_does_not_create_whole_track_note(self):
        example = SimpleNamespace(times=np.arange(20),
                                  metadata={'analysis_note_bounds': []})
        estimate = Attune().estimate(example)
        self.assertFalse(np.any(estimate.detected))


if __name__ == '__main__':
    unittest.main()
