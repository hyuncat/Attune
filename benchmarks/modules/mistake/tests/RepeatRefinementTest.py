import unittest
from unittest.mock import Mock

from algorithms.Config import Config
from algorithms.MistakeDetector import MistakeDetector
from app_logic.NoteData import Note
from app_logic.NoteData import NoteData
from app_logic.user.ds.Recording import Recording
from algorithms.RepeatSplitter import RepeatSplitter
from benchmarks.modules.mistake.sweeps.RepeatRefinementComparison import (
    groups,
    repeat_split,
)


def notes(items):
    return [Note(i, a, b, [p]) for i, (a, b, p) in enumerate(items)]


class RepeatRefinementTests(unittest.TestCase):
    def split(self, user, score):
        return repeat_split(notes(user), notes(score), MistakeDetector(config=Config()))

    def test_supported_repeat_and_ambiguity(self):
        # Timing alone also accepts a sustained C: deliberately documents the limit.
        result, proposals = self.split([(0, 2, 60)], [(0, 1, 60), (1, 2, 60)])
        self.assertEqual(result.times, [0, 1])
        self.assertEqual(proposals[0]["status"], "split")

    def test_cost_gate_can_accept_shortened_span(self):
        result, proposals = self.split([(0, 1, 60)], [(0, 1, 60), (1, 2, 60)])
        self.assertEqual(result.times, [0, 0.5])
        self.assertLess(proposals[0]["after_cost"], proposals[0]["before_cost"])

    def test_score_proportions_scale_to_observed_span_without_onset_gate(self):
        result, _ = self.split([(10, 14.8, 60)], [(0, 1, 60), (1, 4, 60)])
        self.assertEqual(len(result.times), 2)
        self.assertAlmostEqual(result.times[1], 11.2)

    def test_increased_global_cost_rejects_split(self):
        detector = MistakeDetector(config=Config())
        # A deliberately adverse objective isolates acceptance from edit weights.
        detector.get_alignment_cost = lambda a: sum(u is not None for u, _ in a.pairs)
        result, proposals = repeat_split(
            notes([(0, 2, 60)]), notes([(0, 1, 60), (1, 2, 60)]), detector
        )
        self.assertEqual(result.times, [0])
        self.assertEqual(proposals[0]["status"], "alignment cost increased")
        self.assertEqual(proposals[0]["after_cost"], proposals[0]["before_cost"])

    def test_equal_cost_is_accepted(self):
        detector = MistakeDetector(config=Config())
        detector.get_alignment_cost = lambda a: 1.0
        result, proposals = repeat_split(
            notes([(0, 2, 60)]), notes([(0, 1, 60), (1, 2, 60)]), detector
        )
        self.assertEqual(result.times, [0, 1])
        self.assertEqual(proposals[0]["after_cost"], proposals[0]["before_cost"])

    def test_intervening_pitch_and_rests_break_groups(self):
        self.assertEqual(len(groups(notes([(0, 1, 60), (1, 2, 62), (2, 3, 60)]))), 3)
        self.assertEqual(len(groups(notes([(0, 1, 60), (1.2, 2, 60)]))), 2)

    def test_existing_boundaries_are_preserved(self):
        result, proposals = self.split(
            [(0, 1.04, 60), (1.04, 2, 60)], [(0, 1, 60), (1, 2, 60)]
        )
        self.assertEqual(result.times, [0, 1.04])
        self.assertEqual(proposals[0]["status"], "already segmented")

    def test_mismatched_pitch_cannot_create_repeat(self):
        result, _ = self.split([(0, 2, 61)], [(0, 1, 60), (1, 2, 60)])
        self.assertEqual(result.times, [0])


class ProductionRepeatTests(unittest.TestCase):
    def recording(self, user, score, clip=None):
        rec = Recording(config=Config())

        def data(items):
            result = NoteData()
            for n in notes(items):
                result.write_note(n)
            return result

        rec.note_data = data(user)
        rec.score_data.note_datas = {0: data(score)}
        rec.score_data.active_instrument = rec.active_instrument = 0
        rec.score_data.clip = clip
        rec.recompute_vibrato = Mock()
        rec.resize_score_to_aligned_onsets = Mock(
            side_effect=AssertionError("Unexpected score refit")
        )
        rec.detect_mistakes()
        return rec

    def test_default_production_entry_preserves_benchmarked_protocol(self):
        rec = self.recording([(10, 14.8, 60)], [(0, 1, 60), (1, 4, 60)])
        rec.note_data.data[10].info = {"duration": 4.8}
        self.assertIsInstance(rec.repeat_splitter, RepeatSplitter)
        before = rec.mistake_detector.get_alignment_cost(rec.alignment)
        rec.stabilize_score_alignment()
        self.assertAlmostEqual(rec.note_data.times[1], 11.2)
        self.assertLessEqual(
            rec.mistake_detector.get_alignment_cost(rec.alignment), before
        )
        rec.resize_score_to_aligned_onsets.assert_not_called()
        rec.recompute_vibrato.assert_called_once_with(note_aware=True)
        actual = {id(n) for n in rec.note_data.data.values()}
        self.assertTrue(all(n.info is None for n in rec.note_data.data.values()))
        self.assertTrue(
            all(id(u) in actual for u, s in rec.alignment.pairs if u is not None)
        )

    def test_only_clipped_score_notes_are_eligible(self):
        rec = self.recording([(0, 2, 60)], [(0, 1, 60), (1, 2, 60)], clip=(0, 0))
        initial = rec.note_data
        rec.stabilize_score_alignment()
        self.assertIs(rec.note_data, initial)
        self.assertEqual(rec.note_data.times, [0])
        rec.recompute_vibrato.assert_not_called()

    def test_no_repeat_preserves_note_and_alignment_objects(self):
        rec = self.recording([(0, 1, 60), (1, 2, 62)], [(0, 1, 60), (1, 2, 62)])
        initial, alignment = rec.note_data, rec.alignment
        rec.stabilize_score_alignment()
        self.assertIs(rec.note_data, initial)
        self.assertIs(rec.alignment, alignment)
        rec.recompute_vibrato.assert_not_called()


if __name__ == "__main__":
    unittest.main()
