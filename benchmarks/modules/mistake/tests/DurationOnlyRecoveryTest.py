import unittest

from algorithms.RepeatSplitter import RepeatSplitter
from benchmarks.modules.mistake.tests.CollapsedRepeatsTest import notes, recording


class DurationOnlyRecoveryTests(unittest.TestCase):
    def test_pitch_error_no_longer_changes_split_choice(self):
        for pitch in (60.0, 60.1, 60.4):
            users = notes([(0, 1.6, pitch)])
            scores = notes([(0, 1, 60), (1, 2, 60)])
            rec = recording(users, scores)
            detector = rec.mistake_detector
            block = [(users[0], scores[0]), (None, scores[1])]
            original_cost = detector.get_alignment_cost(block)
            cost, cuts, _ = RepeatSplitter.recover_repeat_block(block, detector)
            self.assertAlmostEqual(cost, 6.4)
            self.assertEqual(cuts, 1)
            self.assertEqual(detector.get_alignment_cost(block), original_cost)

    def test_duration_only_ignores_onset_and_pitch_weights(self):
        users = notes([(10, 11.6, 60.4)])
        scores = notes([(0, 1, 60)])
        rec = recording(users, scores)
        rec.config.alignment_alpha_onset = 100.0
        rec.config.alignment_gamma_pitch = 100.0
        self.assertAlmostEqual(
            RepeatSplitter.duration_cost([(users[0], scores[0])], rec.mistake_detector),
            0.6,
        )

    def test_missing_repeat_stays_competitive(self):
        users = notes([(0, 1.0, 60.1)])
        scores = notes([(0, 1, 60), (1, 2, 60)])
        rec = recording(users, scores)
        detector = rec.mistake_detector
        cost, cuts, _ = RepeatSplitter.recover_repeat_block(
            [(users[0], scores[0]), (None, scores[1])], detector
        )
        self.assertEqual(cuts, 0)
        self.assertEqual(cost, 6.0)


if __name__ == "__main__":
    unittest.main()
