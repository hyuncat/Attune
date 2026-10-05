"""Check exact inference, pairing, and correction without audio/model execution."""

import itertools
import unittest
import numpy as np
import pandas as pd
from benchmarks.modules.note.NoteBenchmarker import NoteEvaluation


def example(groups=4):
    rows = []
    for i in range(groups):
        for method, tp, fp in [
            ("attune", 10, 0),
            ("basic-pitch", 6, 8),
            ("tony", 10, 0),
        ]:
            rows.append(
                dict(
                    dataset="coco",
                    track_id=str(i),
                    group=str(i),
                    method=method,
                    status="ok",
                    tp=tp,
                    fp=fp,
                    fn=10 - tp,
                    tp_offset=tp,
                    fp_offset=fp,
                    fn_offset=10 - tp,
                )
            )
    return pd.DataFrame(rows)


class NoteSignificanceTest(unittest.TestCase):

    def test_exact_swaps_and_holm_with_tied_method(self):
        result, _ = NoteEvaluation.paired_significance(
            example(), methods=("attune", "basic-pitch", "tony")
        )
        result = result.set_index("method")
        self.assertEqual(result.loc["basic-pitch", "p_value"], 2 / 16)
        self.assertEqual(result.loc["basic-pitch", "p_adj"], 4 / 16)
        self.assertEqual(result.loc["tony", "p_adj"], 1)
        self.assertTrue(result.exact.all())
        self.assertFalse(result.significant.any())

    def test_pooled_f1_recomputed_with_unequal_prediction_counts(self):
        rows = example()
        rows.loc[(rows.track_id == "0") & (rows.method == "basic-pitch"), "fp"] = 100
        a = rows.loc[rows.method == "attune", ["tp", "fp", "fn"]].to_numpy()
        b = rows.loc[rows.method == "basic-pitch", ["tp", "fp", "fn"]].to_numpy()

        def score(counts):
            tp, fp, fn = counts.sum(axis=0)
            return 2 * tp / (2 * tp + fp + fn)

        observed = score(a) - score(b)
        extreme = 0
        for bits in itertools.product((False, True), repeat=4):
            mask = np.array(bits)[:, None]
            delta = score(np.where(mask, b, a)) - score(np.where(mask, a, b))
            extreme += abs(delta) >= abs(observed) - 1e-12
        result, _ = NoteEvaluation.paired_significance(
            rows, methods=("attune", "basic-pitch")
        )
        self.assertAlmostEqual(result.iloc[0].difference_pp, 100 * observed)
        self.assertEqual(result.iloc[0].p_value, extreme / 16)

    def test_failures_and_missing_methods_use_one_common_intersection(self):
        rows = example()
        rows.loc[(rows.track_id == "0") & (rows.method == "tony"), "status"] = "error"
        rows = rows.loc[~((rows.track_id == "1") & (rows.method == "basic-pitch"))]
        result, coverage = NoteEvaluation.paired_significance(
            rows, methods=("attune", "basic-pitch", "tony")
        )
        self.assertTrue(result.tracks.eq(2).all())
        self.assertTrue(result.excluded_tracks.eq(2).all())
        self.assertTrue(coverage.paired_tracks.eq(2).all())

    def test_urmp_titles_stay_together_and_overrides_can_link_groups(self):
        rows = example()
        rows["dataset"] = "urmp"
        rows["group"] = rows.track_id.map(
            {
                "0": "01_Spring_fl_vn",
                "1": "02_Spring_vn_vc",
                "2": "03_Fugue_vn_vc",
                "3": "04_Sonata_fl_vn",
            }
        )
        result, _ = NoteEvaluation.paired_significance(
            rows, methods=("attune", "basic-pitch")
        )
        self.assertEqual(result.iloc[0].source_groups, 3)
        self.assertEqual(result.iloc[0].permutations, 8)
        result, _ = NoteEvaluation.paired_significance(
            rows,
            methods=("attune", "basic-pitch"),
            source_groups={
                ("urmp", "03_Fugue_vn_vc"): "shared",
                ("urmp", "04_Sonata_fl_vn"): "shared",
            },
        )
        self.assertEqual(result.iloc[0].source_groups, 2)

    def test_invalid_pairs_fail_loudly(self):
        rows = example()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            NoteEvaluation.paired_significance(
                pd.concat([rows, rows.iloc[:1]]), methods=("attune", "tony")
            )
        rows.loc[0, "fn"] = 1
        with self.assertRaisesRegex(ValueError, "reference-note"):
            NoteEvaluation.paired_significance(rows, methods=("attune", "tony"))

    def test_monte_carlo_reproducibility_and_metric_family(self):
        rows = example(15)
        args = dict(
            methods=("attune", "basic-pitch"),
            metrics=("f1", "f1_offset"),
            n_resamples=999,
            seed=17,
        )
        first, _ = NoteEvaluation.paired_significance(rows, **args)
        second, _ = NoteEvaluation.paired_significance(
            rows.sample(frac=1, random_state=1), **args
        )
        pd.testing.assert_frame_equal(first, second)
        self.assertFalse(first.exact.any())
        self.assertTrue(first.p_value.gt(0).all())
        self.assertTrue(first.p_adj.eq(2 * first.p_value).all())


if __name__ == "__main__":
    unittest.main()
