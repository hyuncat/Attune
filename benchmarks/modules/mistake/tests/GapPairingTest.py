import unittest
from algorithms.Config import Config
from algorithms.MistakeDetector import MistakeDetector
from app_logic.NoteData import Note, NoteData
from app_logic.Alignment import Alignment, Mistake
from app_logic.user.ds.Recording import Recording


class GapPairingTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.detector = MistakeDetector(config=self.cfg)

    def gaps(self, u, s, reverse=False):
        pairs = [(u, None), (None, s)]
        if reverse:
            pairs.reverse()
        return Alignment(
            self.cfg,
            notes=pairs,
            pitch_mistakes=[
                Mistake("insertion", u, None),
                Mistake("deletion", None, s),
            ],
        )

    def test_octave_error_becomes_one_substitution_without_changing_notes(self):
        u = Note(1, 7.66, 7.78, [62])
        s = Note(2, 7.60, 7.80, [74])
        for reverse in [False, True]:
            a = self.detector.reconcile_gap_pairs(self.gaps(u, s, reverse))
            self.assertEqual(a.pairs, [(u, s)])
            self.assertEqual([m.type for m in a.pitch_mistakes], ["substitution"])
            self.assertEqual(a.pitch_mistakes[0].pair_index, 0)
            self.assertIs(self.detector.reconcile_gap_pairs(a), a)

    def test_distant_disjoint_wrong_length_and_same_pitch_gaps_remain(self):
        s = Note(2, 1.0, 1.5, [72])
        for u in [
            Note(1, 2.0, 2.5, [60]),
            Note(1, 1.4, 1.6, [60]),
            Note(1, 1.0, 1.1, [60]),
            Note(1, 1.0, 1.5, [72]),
        ]:
            a = self.gaps(u, s)
            self.assertIs(self.detector.reconcile_gap_pairs(a), a)

    def test_matched_anchor_and_ambiguous_blocks_are_not_crossed(self):
        u = Note(1, 1.0, 1.3, [60])
        s = Note(2, 1.0, 1.3, [72])
        anchor = Note(3, 1.1, 1.2, [65])
        a = Alignment(self.cfg, notes=[(u, None), (anchor, anchor), (None, s)])
        self.assertIs(self.detector.reconcile_gap_pairs(a), a)
        a = Alignment(
            self.cfg, notes=[(u, None), (None, s), (Note(4, 1.1, 1.3, [61]), None)]
        )
        self.assertIs(self.detector.reconcile_gap_pairs(a), a)

    def test_explicit_gap_override_is_not_merged(self):
        a = self.gaps(Note(1, 1.0, 1.3, [60]), Note(2, 1.0, 1.3, [72]))
        a.reapply_overrides({0})
        self.assertIs(self.detector.reconcile_gap_pairs(a), a)

    def test_later_override_is_reindexed(self):
        u = Note(1, 1.0, 1.3, [60])
        s = Note(2, 1.0, 1.3, [72])
        extra = Note(3, 3.0, 3.2, [65])
        anchor = Note(4, 2.0, 2.2, [67])
        a = self.gaps(u, s)
        a = Alignment(
            self.cfg,
            notes=a.pairs + [(anchor, anchor), (extra, None)],
            pitch_mistakes=a.pitch_mistakes + [Mistake("insertion", extra, None)],
        )
        a.reapply_overrides({2})
        b = self.detector.reconcile_gap_pairs(a)
        self.assertEqual(b.overridden_pair_indices, {2})
        self.assertTrue(b.pitch_mistakes[-1].overridden)

    def test_recording_preserves_substitution_override_across_raw_realign(self):
        r = Recording()
        u = Note(1, 1.0, 1.3, [60])
        s = Note(2, 1.0, 1.3, [72])
        nd = NoteData()
        nd.write_note(u)
        sd = NoteData()
        sd.write_note(s)
        r.note_data = nd
        r.score_data.clipped_note_data = lambda **kw: sd
        r.detect_mistakes()
        r.reindex_mistakes()
        self.assertEqual(len(r.alignment.pairs), 1)
        r.overridden_mistake_indices = {0}
        r.alignment.reapply_overrides({0})
        r.detect_mistakes()
        r.reindex_mistakes()
        self.assertEqual(r.alignment.pairs, [(u, s)])
        self.assertEqual(r.overridden_mistake_indices, {0})
        self.assertTrue(r.alignment.pitch_mistakes[0].overridden)

    def test_recording_override_uses_mistake_index_after_a_clean_anchor(self):
        r = Recording()
        nd = NoteData()
        sd = NoteData()
        for n in [Note(0, 0.0, 0.3, [65]), Note(1, 1.0, 1.3, [60])]:
            nd.write_note(n)
        for n in [Note(0, 0.0, 0.3, [65]), Note(1, 1.0, 1.3, [72])]:
            sd.write_note(n)
        r.note_data = nd
        r.score_data.clipped_note_data = lambda **kw: sd
        r.detect_mistakes()
        r.reindex_mistakes()
        self.assertEqual(r.alignment.pitch_mistakes[0].pair_index, 1)
        r.overridden_mistake_indices = {0}
        r.alignment.reapply_overrides({0})
        r.detect_mistakes()
        r.reindex_mistakes()
        self.assertEqual(r.overridden_mistake_indices, {0})
        self.assertEqual(r.alignment.overridden_pair_indices, {1})


if __name__ == "__main__":
    unittest.main()
