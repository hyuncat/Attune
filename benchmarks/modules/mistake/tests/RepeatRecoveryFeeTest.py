"""Deletion-neutral fees, partial recovery, and conservative tie-breaking."""

import unittest

from algorithms.RepeatSplitter import RepeatSplitter
from benchmarks.modules.mistake.tests.CollapsedRepeatsTest import notes, recording


class RecoveryFeeTests(unittest.TestCase):
    def recover(self, duration, score_rows, anchor=0):
        user = notes([(0, duration, 60)])
        score = notes(score_rows)
        rec = recording(user, score)
        rec.alignment = rec.mistake_detector.alignment_from_pairs(
            [(user[0] if i == anchor else None, s) for i, s in enumerate(score)]
        )
        rec.stabilize_score_alignment()
        return rec

    def test_one_second_note_does_not_invent_two_one_second_repeats(self):
        rec = self.recover(1.0, [(0, 1, 60), (1, 2, 60)])
        self.assertEqual(rec.note_data.times, [0])
        self.assertEqual(sum(u is None for u, _ in rec.alignment.pairs), 1)

    def test_two_second_note_recovers_repeat_with_exact_deletion_fee(self):
        rec = self.recover(2.0, [(0, 1, 60), (1, 2, 60)])
        self.assertEqual(rec.note_data.times, [0, 1])
        proposal = rec.repeat_splitter.proposals[0]
        self.assertEqual(proposal["recovery_fee"], 6.0)
        self.assertEqual(proposal["before_cost"], 7.0)
        self.assertEqual(proposal["candidate_cost"], 6.0)

    def test_partial_recovery_leaves_one_score_note_missing(self):
        rec = self.recover(2.0, [(0, 1, 60), (1, 2, 60), (2, 3, 60)])
        self.assertEqual(rec.note_data.times, [0, 1])
        self.assertEqual(sum(u is None for u, _ in rec.alignment.pairs), 1)
        self.assertEqual(rec.repeat_splitter.proposals[0]["recovered_notes"], 1)

    def test_equal_cost_keeps_original_segmentation(self):
        rec = self.recover(1.5, [(0, 1, 60), (1, 2, 60)])
        self.assertEqual(rec.note_data.times, [0])
        self.assertEqual(rec.repeat_splitter.proposals[0]["status"], "unchanged")

    def test_fee_uses_recovered_note_duration_not_anchor_duration(self):
        rec = self.recover(2.0, [(0, 0.5, 60), (0.5, 2, 60)], anchor=1)
        self.assertEqual(rec.note_data.times, [0, 0.5])
        self.assertEqual(rec.repeat_splitter.proposals[0]["recovery_fee"], 5.5)

    def test_existing_other_match_limits_maximum_split_count(self):
        user = notes([(0, 3, 60), (3, 4, 60)])
        score = notes([(i, i + 1, 60) for i in range(4)])
        rec = recording(user, score)
        block = [
            (user[0], score[0]),
            (None, score[1]),
            (None, score[2]),
            (user[1], score[3]),
        ]
        cost, cuts, path = RepeatSplitter.recover_repeat_block(
            block, rec.mistake_detector
        )
        self.assertEqual(cuts, 2)  # K - M = 2 additional boundaries.
        self.assertEqual([stop - start for _, _, start, stop, _ in path], [3, 1])
        self.assertEqual(cost, 12.0)

    def test_recovery_fee_cancels_arbitrary_gap_cost_changes(self):
        for deletion_base in (0.1, 5.0, 100.0):
            user = notes([(0, 1, 60)])
            score = notes([(0, 1, 60), (1, 2, 60)])
            rec = recording(user, score)
            rec.config.del_cost = deletion_base
            cost, cuts, _ = RepeatSplitter.recover_repeat_block(
                [(user[0], score[0]), (None, score[1])], rec.mistake_detector
            )
            self.assertEqual(cuts, 0)
            self.assertAlmostEqual(cost, deletion_base + 1.0)


if __name__ == "__main__":
    unittest.main()
