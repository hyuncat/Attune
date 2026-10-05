"""Cost-grid invariants and pooled selection rules (no corpus required)."""

import unittest

import pandas as pd

from benchmarks.modules.mistake.sweeps.AlignmentParamSweep import Axes, grid, summarize


class AlignmentParamSweepTest(unittest.TestCase):
    baseline = dict(
        alignment_gamma_pitch=2.0,
        alignment_gamma_time=1.0,
        alignment_alpha_onset=0.0,
        alignment_alpha_duration=1.0,
        ins_cost=5.0,
        del_cost=5.0,
    )

    def test_controls_survive_custom_grid(self):
        candidates = grid(
            self.baseline,
            Axes(
                alignment_gamma_pitch=(6.0,),
                alignment_gamma_time=(3.0,),
                alignment_alpha_onset=(0.25,),
                ins_cost=(2.0,),
                del_cost=(8.0,),
            ),
        )
        self.assertEqual(len(candidates), 3)
        self.assertEqual(sum(v["baseline"] for v in candidates), 1)
        control = candidates[1]
        self.assertEqual(control["alignment_gamma_pitch"], 4.0)
        self.assertTrue(
            all(
                control[p] == value
                for p, value in self.baseline.items()
                if p != "alignment_gamma_pitch"
            )
        )
        self.assertTrue(
            all(
                v["alignment_alpha_onset"] + v["alignment_alpha_duration"] == 1
                for v in candidates
            )
        )
        self.assertEqual(candidates[-1]["ins_cost"], 2.0)
        self.assertEqual(candidates[-1]["del_cost"], 8.0)

    def test_reject_invalid_axes(self):
        for axes in (
            Axes(alignment_alpha_onset=(1.1,)),
            Axes(ins_cost=(-1.0,)),
            Axes(alignment_gamma_pitch=(float("nan"),)),
            Axes(del_cost=()),
        ):
            with self.assertRaises(ValueError):
                grid(self.baseline, axes)

    def test_clean_constraint_and_pooled_counts(self):
        rows = []
        for variant, baseline, counts in [
            ("base", True, [(1, 0, 0), (0, 9, 1), (0, 0, 0)]),
            ("candidate", False, [(1, 0, 0), (1, 0, 0), (0, 1, 0)]),
        ]:
            for case_id, (tp, fp, fn) in enumerate(counts):
                row = dict(
                    variant=variant,
                    baseline=baseline,
                    case_id=case_id,
                    rate=0.0 if case_id == 2 else 0.25,
                    cap_ms=100.0,
                    score_factor=0.25,
                    pitch_step=1.0,
                    silence_ms=40.0,
                    score_notes=10,
                    **dict(
                        self.baseline, alignment_gamma_pitch=2.0 if baseline else 4.0
                    ),
                )
                for prefix in (
                    "note50",
                    "note100",
                    "note200",
                    "note_offsets",
                    "audio_pitch50",
                    "audio_pitch100",
                    "audio_pitch200",
                    "audio_extra100",
                    "audio_missed100",
                ):
                    row.update(
                        {
                            f"{prefix}_{k}": v
                            for k, v in zip(("tp", "fp", "fn"), (tp, fp, fn))
                        }
                    )
                rows.append(row)
        summary = summarize(pd.DataFrame(rows))
        self.assertEqual(summary.iloc[0].variant, "base")
        self.assertAlmostEqual(summary.iloc[0].injected_audio_pitch100_f1, 100 / 6)
        self.assertEqual(summary.iloc[1].injected_audio_pitch100_f1, 100.0)
        self.assertFalse(summary.iloc[1].clean_safe)
        self.assertEqual(summary.iloc[1].alignment_gamma_pitch, 4.0)


if __name__ == "__main__":
    unittest.main()
