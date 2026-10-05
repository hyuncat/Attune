from __future__ import annotations
import unittest
import warnings
import mir_eval
import numpy as np
import pandas as pd
from benchmarks.modules.pitch.PitchBenchmarker import (
    HOP_SECONDS,
    METRIC_COUNTS,
    PitchBenchmarker,
    SCORING_VERSION,
)


def counted_rows(rows):
    """Test fixtures: percentages on a 100-frame, 50-voiced reference."""
    rows = rows.copy()
    rows["scoring_version"] = SCORING_VERSION
    rows["frame_count"] = 100
    rows["voiced_frame_count"] = 50
    rows["unvoiced_frame_count"] = 50
    for metric, (num, den) in METRIC_COUNTS.items():
        rows[num] = rows[metric] * rows[den] if metric in rows else 0
    return rows


class PitchMetricsTest(unittest.TestCase):

    def test_scoring_matches_mir_eval_on_shared_grid_and_negative_predictions(self):
        times = np.arange(101) * 0.01
        ref = np.where(np.arange(101) < 60, 440.0, 0.0)
        est = np.where(np.arange(101) < 20, -440.0, 880.0)
        result = PitchBenchmarker.score_frames(times, ref, times, est)
        expected = mir_eval.melody.evaluate(times, ref, times, est, hop=HOP_SECONDS)
        for metric in METRIC_COUNTS:
            self.assertAlmostEqual(result[metric], expected[metric])
        self.assertGreater(result["Raw Chroma Accuracy"], result["Raw Pitch Accuracy"])
        self.assertEqual(result["voiced_frame_count"], 60)

    def test_pooling_uses_distinct_denominators_and_long_tracks(self):

        def row(n, voiced, correct, detected, false):
            return dict(
                model="m",
                scoring_version=SCORING_VERSION,
                frame_count=n,
                voiced_frame_count=voiced,
                unvoiced_frame_count=n - voiced,
                pitch_correct_frames=correct,
                chroma_correct_frames=correct,
                overall_correct_frames=correct + n - voiced - false,
                voiced_detected_frames=detected,
                false_alarm_frames=false,
            )

        rows = pd.DataFrame([row(100, 90, 90, 90, 10), row(900, 100, 0, 50, 0)])
        actual = PitchBenchmarker.pool_scores(rows).loc["m"]
        self.assertAlmostEqual(actual["Overall Accuracy"], 0.89)
        self.assertAlmostEqual(actual["Raw Pitch Accuracy"], 90 / 190)
        self.assertAlmostEqual(actual["Voicing Recall"], 140 / 190)
        self.assertAlmostEqual(actual["Voicing False Alarm"], 10 / 810)
        with self.assertRaisesRegex(ValueError, "frame counts"):
            PitchBenchmarker.pool_scores(rows.drop(columns="voiced_frame_count"))

    def test_no_unvoiced_frames_do_not_dilute_false_alarm_rate(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = PitchBenchmarker.score_frames(
                np.arange(10) * 0.01, np.full(10, 440.0), [0.0, 0.09], [440.0, 440.0]
            )
            b = PitchBenchmarker.score_frames(
                np.arange(10) * 0.01, np.zeros(10), [0.0, 0.09], [440.0, 440.0]
            )
        rows = pd.DataFrame([dict(model="m", **a), dict(model="m", **b)])
        self.assertTrue(np.isnan(a["Voicing False Alarm"]))
        self.assertEqual(
            PitchBenchmarker.pool_scores(rows).loc["m", "Voicing False Alarm"], 1.0
        )

    @staticmethod
    def paired_rows():
        rows = []
        for piece in range(4):
            for stem in range(2):
                for method, correct in [("attune", 90), ("peer", 80)]:
                    rows.append(
                        dict(
                            model=method,
                            dataset="real",
                            track_id=f"{piece}/{stem}",
                            source_piece=f"real:{piece}",
                            reference_hash=f"{piece}/{stem}",
                            execution_mode="Offline",
                            **{"Overall Accuracy": correct / 100},
                        )
                    )
        return counted_rows(pd.DataFrame(rows))

    def test_exact_paired_test_clusters_stems_and_reproducible_intervals(self):
        rows = self.paired_rows()
        a, paired = PitchBenchmarker.paired_comparisons(
            rows, methods=["attune", "peer"], n_resamples=999, seed=7
        )
        b, _ = PitchBenchmarker.paired_comparisons(
            rows, methods=["attune", "peer"], n_resamples=999, seed=7
        )
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(len(paired), 16)
        self.assertEqual(a.iloc[0].source_pieces, 4)
        self.assertEqual(a.iloc[0].tracks, 8)
        self.assertAlmostEqual(a.iloc[0].difference_pp, 10)
        self.assertAlmostEqual(a.iloc[0].p_value, 2 / 16)
        self.assertAlmostEqual(a.iloc[0].ci_low_pp, 10)
        self.assertAlmostEqual(a.iloc[0].ci_high_pp, 10)

    def test_duplicate_frames_change_weight_but_not_independent_evidence(self):
        rows = self.paired_rows()
        before, _ = PitchBenchmarker.paired_comparisons(
            rows, methods=["attune", "peer"], n_resamples=999
        )
        for col in (
            "frame_count",
            "voiced_frame_count",
            "unvoiced_frame_count",
            *(num for num, _ in METRIC_COUNTS.values()),
        ):
            rows[col] *= 10
        after, _ = PitchBenchmarker.paired_comparisons(
            rows, methods=["attune", "peer"], n_resamples=999
        )
        self.assertEqual(before.iloc[0].p_value, after.iloc[0].p_value)
        self.assertEqual(before.iloc[0].difference_pp, after.iloc[0].difference_pp)

    def test_paired_statistic_retains_long_track_weight(self):
        rows = self.paired_rows()
        long = rows.source_piece.eq("real:0")
        rows.loc[long, "overall_correct_frames"] = np.where(
            rows.loc[long, "model"].eq("attune"), 80, 90
        )
        for column in (
            "frame_count",
            "voiced_frame_count",
            "unvoiced_frame_count",
            *(num for num, _ in METRIC_COUNTS.values()),
        ):
            rows.loc[long, column] *= 100
        result, _ = PitchBenchmarker.paired_comparisons(
            rows, methods=["attune", "peer"], n_resamples=99
        )
        pooled = PitchBenchmarker.pool_scores(rows)
        expected = 100 * (
            pooled.loc["attune", "Overall Accuracy"]
            - pooled.loc["peer", "Overall Accuracy"]
        )
        self.assertAlmostEqual(result.iloc[0].difference_pp, expected)
        self.assertLess(expected, 0)

    def test_pairing_uses_common_coverage_and_rejects_invalid_designs(self):
        rows = self.paired_rows()
        rows = rows.drop(rows.index[-1])
        result, paired = PitchBenchmarker.paired_comparisons(
            rows, methods=["attune", "peer"], n_resamples=99
        )
        self.assertEqual(result.iloc[0].tracks, 7)
        self.assertEqual(result.iloc[0].excluded_tracks, 1)
        for column, value in [
            ("reference_hash", "bad"),
            ("execution_mode", "Streaming"),
        ]:
            bad = self.paired_rows()
            bad.loc[0, column] = value
            with self.assertRaises(ValueError):
                PitchBenchmarker.paired_comparisons(
                    bad, methods=["attune", "peer"], n_resamples=99
                )
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            PitchBenchmarker.paired_comparisons(
                pd.concat([rows, rows.iloc[:1]]),
                methods=["attune", "peer"],
                n_resamples=99,
            )

    def test_holm_adjustment_and_zero_effect(self):
        rows = self.paired_rows()
        clone = rows.loc[rows.model.eq("attune")].assign(model="equal")
        result, _ = PitchBenchmarker.paired_comparisons(
            pd.concat([rows, clone]),
            methods=["attune", "peer", "equal"],
            n_resamples=999,
        )
        self.assertEqual(result.loc[result.method.eq("equal"), "p_holm"].iloc[0], 1.0)
        self.assertAlmostEqual(
            result.loc[result.method.eq("peer"), "p_holm"].iloc[0], 0.25
        )


if __name__ == "__main__":
    unittest.main()
