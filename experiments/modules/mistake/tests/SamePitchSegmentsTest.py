"""Initial consolidation bounds the spread of original segment medians."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from algorithms.Config import Config
from algorithms.NoteDetector import NoteDetector
from app_logic.user.ds.PitchData import Pitch


class SamePitchSegmentTests(unittest.TestCase):
    def detect(self, values, runs=None):
        cfg = Config()
        detector = NoteDetector(SimpleNamespace(config=cfg))
        pitches = [
            Pitch(
                time=i * 0.01,
                volume=0.2,
                unvoiced_prob=0.0,
                live_distance=None,
                config=cfg,
                value=v,
                candidates=[(v, 1.0)],
            )
            for i, v in enumerate(values)
        ]
        groups = [pitches] if runs is None else [pitches[a:b] for a, b in runs]
        with patch.object(
            detector, "get_pitch_runs", return_value=groups
        ), patch.object(
            detector, "segment_breakpoints", side_effect=lambda x: [len(x) // 2, len(x)]
        ):
            notes = detector.detect_notes(pitches)
        return [notes.data[t] for t in notes.times]

    def test_bass_settling_keeps_outer_boundaries_and_raw_median(self):
        ns = self.detect([45.9] * 20 + [46.3] * 50)
        self.assertEqual(len(ns), 1)
        self.assertEqual(ns[0].midi_num, [46.3])
        self.assertAlmostEqual(ns[0].start_time, 0.0)
        self.assertGreater(ns[0].end_time, 0.68)

    def test_real_semitone_change_survives(self):
        ns = self.detect([69.0] * 40 + [70.0] * 40)
        self.assertEqual([n.midi_num[0] for n in ns], [69.0, 70.0])
        self.assertAlmostEqual(ns[0].end_time, ns[1].start_time)

    def test_close_pitches_across_rounding_boundary_merge(self):
        self.assertEqual(len(self.detect([60.4] * 40 + [60.6] * 40)), 1)

    def test_same_rounded_pitch_with_excess_spread_stays_separate(self):
        self.assertEqual(len(self.detect([59.6] * 40 + [60.4] * 40)), 2)

    def test_half_semitone_is_inclusive(self):
        self.assertEqual(len(self.detect([60.0] * 40 + [60.5] * 40)), 1)
        self.assertEqual(len(self.detect([60.0] * 40 + [60.5001] * 40)), 2)

    def test_group_extrema_prevent_chaining_in_both_directions(self):
        for medians in ([60.0, 60.4, 60.8, 61.2], [61.2, 60.8, 60.4, 60.0]):
            with self.subTest(medians=medians):
                signal = np.repeat(medians, 20).reshape(-1, 1)
                self.assertEqual(
                    NoteDetector._merge_same_pitch_segments(signal, [20, 40, 60, 80]),
                    [40, 80],
                )

    def test_group_spread_checks_both_extrema(self):
        signal = np.repeat([60.0, 60.4, 59.9], 20).reshape(-1, 1)
        self.assertEqual(
            NoteDetector._merge_same_pitch_segments(signal, [20, 40, 60]),
            [60],
        )
        signal = np.repeat([60.0, 60.4, 59.8], 20).reshape(-1, 1)
        self.assertEqual(
            NoteDetector._merge_same_pitch_segments(signal, [20, 40, 60]),
            [40, 60],
        )

    def test_separate_voiced_runs_do_not_merge(self):
        ns = self.detect([69.0] * 100, runs=[(0, 40), (60, 100)])
        self.assertEqual(len(ns), 2)
        self.assertLess(ns[0].end_time, ns[1].start_time)


if __name__ == "__main__":
    unittest.main()
