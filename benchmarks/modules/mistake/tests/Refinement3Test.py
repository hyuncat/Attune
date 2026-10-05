"""Focused invariants: run with python -m unittest ...test_refinement3."""

import unittest
from types import SimpleNamespace

from algorithms.Config import Config
from notebooks.archive.MistakeChecker3 import MistakeChecker as LegacyRepeatSplitter
from algorithms.MistakeDetector import MistakeDetector
from algorithms.NoteDetector import NoteDetector
from app_logic.NoteData import Note, NoteData
from app_logic.user.ds.PitchData import Pitch, PitchData


def note(i, pitch=60, start=None, end=None):
    return Note(
        i,
        float(i if start is None else start),
        float(i + 1 if end is None else end),
        [pitch],
    )


def setup():
    config = Config()
    rec = SimpleNamespace(config=config)
    rec.mistake_detector = MistakeDetector(config=config)
    return LegacyRepeatSplitter(recording=rec), rec


class RefinementTests(unittest.TestCase):
    def test_mixed_region_includes_substitution(self):
        checker, _ = setup()
        good = lambda i: (note(i), note(i))
        pairs = [
            good(0),
            (note(1), None),
            (note(2), None),
            (None, note(3)),
            (note(4, 65), note(4)),
            (note(5), None),
            (None, note(6)),
            (None, note(7)),
            good(8),
        ]
        regions = list(checker._regions(SimpleNamespace(pairs=pairs)))
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0], (pairs, True, True))

    def test_substitutions_alone_do_not_trigger(self):
        checker, _ = setup()
        pairs = [
            (note(0), note(0)),
            *[(note(i, 65), note(i)) for i in range(1, 5)],
            (note(5), note(5)),
        ]
        self.assertEqual(list(checker._regions(SimpleNamespace(pairs=pairs))), [])

    def test_correct_pair_separates_regions_with_shared_anchor(self):
        checker, _ = setup()
        pairs = [
            (note(0), note(0)),
            (note(1), None),
            (note(2), note(2)),
            (None, note(3)),
            (note(4), note(4)),
        ]
        regions = list(checker._regions(SimpleNamespace(pairs=pairs)))
        self.assertEqual(len(regions), 2)
        self.assertIs(regions[0][0][-1][0], regions[1][0][0][0])

    def test_edge_insertions_do_not_trigger_but_deletions_do(self):
        checker, _ = setup()
        self.assertFalse(
            list(
                checker._regions(
                    SimpleNamespace(pairs=[(note(0), None), (note(1), note(1))])
                )
            )
        )
        self.assertEqual(
            len(
                list(
                    checker._regions(
                        SimpleNamespace(pairs=[(None, note(0)), (note(1), note(1))])
                    )
                )
            ),
            1,
        )

    def test_repeated_note_recovery_lowers_cost_without_moving_outer_edges(self):
        checker, rec = setup()
        checker.pd = PitchData(rec.config)
        dt = rec.config.h1 / rec.config.sr
        checker.pd.data = [
            Pitch(i * dt, 1.0, 0.0, 0.0, rec.config, value=60 if i * dt < 2 else 62)
            for i in range(int(3 / dt) + 1)
        ]
        rec.note_detector = NoteDetector(rec)
        user = [note(0, 60, 0, 2), note(1, 62, 2, 3)]
        score = [note(0, 60), note(1, 60), note(2, 62)]
        window = [(user[0], score[0]), (None, score[1]), (user[1], score[2])]
        proposal = checker._refine_region(window, True, True)
        self.assertIsNotNone(proposal)
        removed, added, saving = proposal
        self.assertEqual(len(added), 3)
        self.assertGreater(saving, 0)
        self.assertEqual(added[0].start_time, 0)
        self.assertEqual(added[-1].end_time, 3)
        self.assertAlmostEqual(added[0].end_time, 1, places=2)
        self.assertEqual(user[0].end_time, 2)  # proposal generation is nonmutating

    def test_non_improving_pass_is_rolled_back(self):
        from unittest.mock import Mock

        checker, rec = setup()
        original = NoteData()
        original.write_note(note(0))
        rec.note_data = original
        rec.alignment = rec.mistake_detector.get_string_edit_alignment(
            [note(0)], [note(0)]
        )
        rec.update_min_note_length = lambda: None
        rec.pitch_data = None
        rec.reindex_mistakes = Mock()
        rec.recompute_vibrato = Mock()
        worse = rec.mistake_detector.get_string_edit_alignment([note(0, 66)], [note(0)])
        checker._check_mistakes = Mock(return_value=(NoteData(), worse, 1))
        checker.check_mistakes()
        self.assertIs(rec.note_data, original)
        rec.reindex_mistakes.assert_not_called()

    def test_long_silence_blocks_merging(self):
        checker, rec = setup()
        checker.pd = None  # use endpoint fallback for this invariant
        sources = [note(0, 60, 0, 1), note(1, 60, 2, 3)]
        self.assertTrue(checker._crosses_blocking_silence(sources, [note(0, 60, 0, 3)]))

    def test_block_subdivision_does_not_rerun_cpd(self):
        checker, rec = setup()
        # The repeat stage computes score-proportional cut positions only.
        cuts = list(
            checker._repeat_splits(note(0, 60, 0, 3), [note(0), note(1), note(2)])
        )
        self.assertEqual(cuts, [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
