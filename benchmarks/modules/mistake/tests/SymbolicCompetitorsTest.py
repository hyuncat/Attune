"""Contracts for symbolic-only runs and external note identity adapters."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker
from benchmarks.modules.mistake.competitors.Nakamura import Nakamura
from benchmarks.modules.mistake.tests.CompetitorComparisonTest import notes


class ExternalIdentityTest(unittest.TestCase):

    def test_nakamura_match_missing_extra_and_wrong_pitch(self):
        score = list(notes([(0, 0.5, 60), (1, 1.5, 62), (2, 2.5, 64)]).data.values())
        user = list(notes([(0, 0.5, 60), (1, 1.5, 63), (3, 3.5, 67)]).data.values())
        alignment = Nakamura.parse_corresp(
            "// header\n0 0 C4 60 64 0 0 C4 60 64\n1 1 D#4 63 64 1 1 D4 62 64\n2 3 G4 67 64 * -1 * -1 -1\n* -1 * -1 -1 2 2 E4 64 64\n"
        )
        pairs = MistakeBenchmarker.validated_pairs(alignment, score, user)
        self.assertEqual(
            pairs,
            [
                (user[0], score[0]),
                (user[1], score[1]),
                (user[2], None),
                (None, score[2]),
            ],
        )

    def test_missing_duplicate_and_invalid_ids_fail(self):
        seq = list(notes([(0, 0.5, 60)]).data.values())
        good = dict(label="match", performance_id="0", score_id="0")
        for alignment in [[], [good, good], [dict(good, score_id="-1")]]:
            with self.assertRaises(ValueError):
                MistakeBenchmarker.validated_pairs(alignment, seq, seq)
        with self.assertRaises(ValueError):
            Nakamura.parse_corresp("0 0 C4")

    def test_spr_preserves_index_identity_and_subtick_times(self):
        seq = list(notes([(0.123456789, 0.723456789, 60.2)]).data.values())
        seq[0].id = 1234
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "notes.txt"
            Nakamura.write_spr(path, seq)
            self.assertEqual(
                path.read_text().split()[:4], ["0", "0.123456789", "0.723456789", "C4"]
            )

    def test_gluenote_has_quarter_fields_and_original_indices(self):
        seq = list(notes([(1, 1.5, 60)]).data.values())
        array = MistakeBenchmarker.note_array(seq, score=True, bpm=90)
        self.assertEqual(array["onset_quarter"][0], 1.5)
        self.assertEqual(array["duration_quarter"][0], 0.75)
        self.assertEqual(array["id"][0], "0")


class SymbolicRunnerTest(unittest.TestCase):

    def test_symbolic_run_never_renders_or_extracts_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.mid"
            MistakeBenchmarker.notedata_to_pm(
                notes([(i, i + 0.5, 60 + i) for i in range(8)])
            ).write(str(source))
            output = Path(tmp) / "results"
            kwargs = dict(
                methods=("Attune (no refinement)",),
                seeds=(0,),
                rates=(0.0, 0.25),
                tolerances=(0.1,),
                parallel=False,
            )
            with patch.object(
                MistakeBenchmarker,
                "synth_midi",
                side_effect=AssertionError("Rendered audio"),
            ), patch(
                "algorithms.RepeatSplitter.RepeatSplitter.repeat_split",
                side_effect=AssertionError("Merged symbolic notes"),
            ), patch(
                "app_logic.user.ds.Recording.Recording.stabilize_score_alignment",
                side_effect=AssertionError("Refined symbolic notes"),
            ), patch(
                "app_logic.user.ds.Recording.Recording.detect_notes",
                side_effect=AssertionError("Extracted notes"),
            ):
                rows = MistakeBenchmarker.run_symbolic_comparison(
                    [source], output, **kwargs
                )
                with patch.object(
                    MistakeBenchmarker,
                    "evaluate_case",
                    side_effect=AssertionError("Recomputed checkpoint"),
                ):
                    resumed = MistakeBenchmarker.run_symbolic_comparison(
                        [source], output, **kwargs
                    )
            self.assertEqual(set(rows.input), {"oracle_notes"})
            self.assertEqual(len(rows), len(resumed))
            self.assertFalse(list(output.rglob("*.wav")))
            self.assertFalse(list(output.rglob("*.pitch.pkl.xz")))
            self.assertTrue((rows.frontend_cpu_seconds == 0).all())
            self.assertTrue((rows.segmentation_cpu_seconds == 0).all())
            clean = rows[(rows.rate == 0) & (rows.metric == "audio_pitch")]
            self.assertEqual(clean[["tp", "fp", "fn"]].sum().sum(), 0)

    def test_unsupported_input_method_combinations_fail_early(self):
        with tempfile.TemporaryDirectory() as tmp:
            for methods in [
                ("PolyTune",),
                ("Attune (Checker 3)",),
                ("Attune (repeat only)",),
            ]:
                with self.assertRaisesRegex(ValueError, "support"):
                    MistakeBenchmarker.run_symbolic_comparison([], tmp, methods=methods)


if __name__ == "__main__":
    unittest.main()
