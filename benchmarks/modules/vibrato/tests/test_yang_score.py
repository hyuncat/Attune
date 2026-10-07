"""Synthetic plumbing checks only: no dataset analysis or F1 benchmark."""
import pickle
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import pretty_midi

from benchmarks.modules.vibrato.tests.YangScore import (
    melody_spans, excerpts, match, write_midi, freeze_pitch, thaw_pitch,
)


class YangScoreTest(unittest.TestCase):
    def test_skyline_preserves_same_pitch_reattacks_and_rests(self):
        part = pretty_midi.Instrument(0)
        part.notes = [pretty_midi.Note(90, p, a, b) for a, b, p in
                      [(0, 1, 60), (1, 2, 60), (3, 4, 62), (0, 4, 48)]]
        spans = melody_spans(SimpleNamespace(instruments=[part]))
        self.assertEqual([s[:3] for s in spans],
                         [(0, 1, 60), (1, 2, 60), (2, 3, 48), (3, 4, 62)])
        part.notes.pop()
        self.assertEqual([s[:3] for s in melody_spans(SimpleNamespace(instruments=[part]))],
                         [(0, 1, 60), (1, 2, 60), (3, 4, 62)])

    def test_warp_keeps_symbolic_rhythm_separate_and_last_note(self):
        spans = [(10, 11, 60, (0, 0)), (11, 12, 60, (0, 1))]
        path = np.array([[10, 2, 60, 60], [11, 4, 60, 60], [11.9, 5, 60, 60]])
        score, bounds = excerpts(spans, path, 2, 5.1)
        self.assertEqual(score, [(0, 1, 62), (1, 2, 62)])
        np.testing.assert_allclose(bounds, [(2, 4, 62), (4, 5.1, 62)])

    def test_midi_round_trip_preserves_repeats(self):
        from app_logic.midi.ScoreData import ScoreData
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'repeats.mid'
            expected = [(0, .5, 60), (.5, 1, 60), (1.25, 2, 62)]
            write_midi(expected, path)
            score = ScoreData(path)
            notes = list(score.note_datas[score.active_instrument].data.values())
            self.assertEqual(len(notes), 3)
            np.testing.assert_allclose([(n.start_time, n.end_time, n.midi_num[0]) for n in notes], expected, atol=.001)

    def test_dtw_covers_query_including_tall_cost_matrix(self):
        spans = [(i, i+1, pitch, (0, i)) for i, pitch in enumerate([60, 64, 62, 67])]
        # Longer audio than score exercises librosa's path-axis flip.
        for factor in [.7, 1.5]:
            times = np.arange(0, 4*factor, .01)
            pitch = np.array([spans[min(3, int(t/factor))][2] for t in times], float)
            path, info = match(spans, times, pitch, shifts=[0])
            self.assertAlmostEqual(path[0, 1], 0)
            self.assertGreater(path[-1, 1], times[-1]-.12)
            self.assertTrue(np.all(np.diff(path[:, :2], axis=0) > 0))
            self.assertEqual(info['transpose_semitones'], 0)

    def test_frozen_pitch_round_trip_retains_confidence_volume_and_origin(self):
        from algorithms.Config import Config
        from app_logic.user.ds.PitchData import Pitch, PitchData
        config = Config()
        pitch = Pitch(-.1, .003, .7, 0., config, [(60., .3)], 60.)
        data = PitchData(config)
        data.t_origin = -.1
        data.load([pitch, None])
        restored_config, restored = thaw_pitch(pickle.loads(pickle.dumps(freeze_pitch(config, data))))
        self.assertEqual(restored.t_origin, -.1)
        self.assertEqual(restored.frames_available(), 2)
        self.assertEqual(restored.data[0].volume, .003)
        self.assertEqual(restored.data[0].unvoiced_prob, .7)
        self.assertEqual(restored.data[0].candidate_pitches, [(60., .3)])
        self.assertIs(restored.data[0].config, restored_config)
        self.assertIsNone(restored.data[1])

    def test_unvoiced_query_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            match([(0, 1, 60, (0, 0))], np.arange(0, 1, .01), np.full(100, np.nan))


if __name__ == '__main__':
    unittest.main()
