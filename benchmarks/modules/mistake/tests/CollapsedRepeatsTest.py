"""Regression coverage for original-note alignment and local missing-repeat recovery.

The filename is retained so existing focused validation commands keep working.
"""

import unittest
from unittest.mock import Mock, patch
from algorithms.Config import Config
from algorithms.RepeatSplitter import RepeatSplitter
from app_logic.NoteData import Note, NoteData
from app_logic.user.ds.Recording import Recording


def notes(rows):
    return [Note(i, a, b, [pitch]) for i, (a, b, pitch) in enumerate(rows)]


def recording(user, score):
    rec = Recording(config=Config())
    rec.note_data = NoteData()
    data = NoteData()
    for n in user:
        rec.note_data.write_note(n)
    for n in score:
        data.write_note(n)
    rec.score_data.note_datas = {0: data}
    rec.active_instrument = rec.score_data.active_instrument = 0
    rec.recompute_vibrato = Mock()
    return rec


class LocalRepeatTests(unittest.TestCase):

    def test_original_alignment_then_recovery_without_another_global_alignment(self):
        score = notes([(0, 1, 60), (1, 2, 60), (2, 3, 62)])
        user = notes([(0, 2, 60), (2, 3, 62)])
        rec = recording(user, score)
        original = rec.mistake_detector.get_string_edit_alignment
        rec.mistake_detector.get_string_edit_alignment = Mock(wraps=original)
        rec.detect_mistakes()
        self.assertEqual([s for _, s in rec.alignment.pairs], score)
        self.assertEqual(sum((u is None for u, _ in rec.alignment.pairs)), 1)
        matched_score = next((s for u, s in rec.alignment.pairs if u is user[0]))
        rec.stabilize_score_alignment()
        self.assertEqual(rec.note_data.times, [0, 1, 2])
        self.assertEqual(rec.mistake_detector.get_string_edit_alignment.call_count, 1)
        self.assertEqual(
            [id(s) for _, s in rec.alignment.pairs], [id(s) for s in score]
        )
        self.assertIs(rec.note_data.data[2], user[1])
        self.assertEqual(
            next((u.id for u, s in rec.alignment.pairs if s is matched_score)),
            user[0].id,
        )
        self.assertEqual(len({n.id for n in rec.note_data.data.values()}), 3)

    def test_complete_flow_uses_original_notes_on_both_alignment_calls(self):
        rec = recording(notes([(0, 2, 60)]), notes([(0, 1, 60), (1, 2, 60)]))
        rec.resize_score = Mock()
        rec.resize_score_to_aligned_onsets = Mock(return_value=True)
        detect = rec.detect_mistakes
        rec.detect_mistakes = Mock(wraps=detect)
        rec.align_score_and_refine()
        self.assertEqual(rec.detect_mistakes.call_count, 2)
        self.assertTrue(
            all(
                (
                    c.kwargs == {"verbose": False}
                    for c in rec.detect_mistakes.call_args_list
                )
            )
        )
        self.assertEqual(rec.note_data.times, [0, 1])

    def test_already_matched_repeats_across_silence_are_untouched(self):
        rec = recording(
            notes([(0, 0.6, 60), (1.0, 1.25, 60), (1.25, 2.0, 62)]),
            notes([(0, 1.0, 60), (1.0, 1.25, 60), (1.25, 2.0, 62)]),
        )
        original = rec.note_data
        rec.detect_mistakes()
        alignment = rec.alignment
        rec.stabilize_score_alignment()
        self.assertIs(rec.note_data, original)
        self.assertIs(rec.alignment, alignment)
        self.assertFalse(rec.repeat_splitter.proposals)
        rec.recompute_vibrato.assert_not_called()

    def test_same_pitch_insertion_is_a_barrier_not_an_extra_repeat_anchor(self):
        user = notes([(0, 2, 60), (2.5, 3, 60.1)])
        score = notes([(0, 1, 60), (1, 2, 60)])
        rec = recording(user, score)
        rec.alignment = rec.mistake_detector.alignment_from_pairs(
            [(user[0], score[0]), (user[1], None), (None, score[1])]
        )
        before = rec.alignment
        rec.stabilize_score_alignment()
        self.assertIs(rec.alignment, before)
        self.assertEqual(rec.note_data.times, [0, 2.5])

    def test_point_nine_semitone_match_can_supply_missing_repeat(self):
        rec = recording(notes([(0, 2, 60.9)]), notes([(0, 1, 60), (1, 2, 60)]))
        rec.detect_mistakes()
        rec.stabilize_score_alignment()
        self.assertEqual(rec.note_data.times, [0, 1])
        self.assertFalse(rec.alignment.pitch_mistakes)

    def test_recovery_uses_configured_substitution_boundary(self):
        for tolerance in (0.0, 0.25, 0.5, 1.0, 2.0):
            for delta in (max(0.0, tolerance - 0.01), tolerance, tolerance + 0.01):
                with self.subTest(tolerance=tolerance, delta=delta):
                    user = notes([(0, 2, 60 + delta)])
                    score = notes([(0, 1, 60), (1, 2, 60)])
                    rec = recording(user, score)
                    rec.config.pitch_tolerance = tolerance
                    rec.alignment = rec.mistake_detector.alignment_from_pairs(
                        [(user[0], score[0]), (None, score[1])]
                    )
                    substitution = any(
                        (m.type == "substitution" for m in rec.alignment.pitch_mistakes)
                    )
                    rec.stabilize_score_alignment()
                    self.assertEqual(len(rec.note_data.times), 1 if substitution else 2)
                    self.assertEqual(
                        sum((u is None for u, _ in rec.alignment.pairs)),
                        int(substitution),
                    )

    def test_multiple_anchors_keep_existing_boundaries(self):
        user = notes([(0, 1, 60), (1.2, 3, 60)])
        score = notes([(0, 1, 60), (1, 2, 60), (2, 3, 60)])
        rec = recording(user, score)
        rec.alignment = rec.mistake_detector.alignment_from_pairs(
            [(user[0], score[0]), (user[1], score[1]), (None, score[2])]
        )
        rec.stabilize_score_alignment()
        self.assertEqual(rec.note_data.times, [0, 1.2, 2.1])
        self.assertIs(rec.note_data.data[0], user[0])
        self.assertEqual(rec.note_data.data[0].end_time, 1)
        self.assertFalse(rec.alignment.pitch_mistakes)

    def test_increased_local_cost_rejects_recovery(self):
        rec = recording(notes([(0, 2, 60)]), notes([(0, 1, 60), (1, 2, 60)]))
        rec.detect_mistakes()
        before = rec.alignment
        with patch(
            "algorithms.RepeatSplitter.RepeatSplitter.duration_cost",
            side_effect=lambda pairs, detector: sum((u is not None for u, _ in pairs)),
        ):
            rec.stabilize_score_alignment()
        self.assertIs(rec.alignment, before)
        self.assertEqual(rec.note_data.times, [0])

    def test_unequal_proportions_preserve_pitch(self):
        rec = recording(notes([(10, 14.8, 60.2)]), notes([(0, 1, 60), (1, 4, 60)]))
        rec.detect_mistakes()
        rec.stabilize_score_alignment()
        self.assertAlmostEqual(rec.note_data.times[1], 11.2)
        self.assertTrue(
            all((n.midi_num == [60.2] for n in rec.note_data.data.values()))
        )

    def test_unmatched_group_cannot_invent_notes(self):
        rec = recording([], notes([(0, 1, 60), (1, 2, 60)]))
        rec.detect_mistakes()
        rec.stabilize_score_alignment()
        self.assertEqual(len(rec.alignment.pitch_mistakes), 2)
        self.assertFalse(rec.note_data.times)

    def test_chord_changes_rests_and_clips_bound_recovery(self):
        score = notes([(0, 1, 60), (1, 2, 60), (2.2, 3, 60)])
        score[1].midi_num = [60, 64]
        self.assertEqual(len(RepeatSplitter.groups(score)), 3)
        rec = recording(notes([(0, 2, 60)]), notes([(0, 1, 60), (1, 2, 60)]))
        rec.score_data.clip = (1, 1)
        rec.detect_mistakes()
        rec.stabilize_score_alignment()
        self.assertEqual(rec.note_data.times, [0])
        self.assertEqual([s.id for _, s in rec.alignment.pairs if s], [1])

    def test_short_pieces_reject_recovery(self):
        rec = recording(
            notes([(0, 0.05, 60)]), notes([(0, 0.02, 60), (0.02, 0.05, 60)])
        )
        rec.detect_mistakes()
        rec.stabilize_score_alignment()
        self.assertEqual(rec.note_data.times, [0])


if __name__ == "__main__":
    unittest.main()
