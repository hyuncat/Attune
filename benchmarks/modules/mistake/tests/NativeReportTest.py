"""Reporting distinguishes equal-case averages, pooled counts and empty cases."""

import json
from pathlib import Path
import tempfile
import unittest
import pandas as pd
from benchmarks.modules.mistake.MistakeNotebook import MistakeNotebook


class NativeReportTests(unittest.TestCase):

    def fixture(self, root):
        manifest = {"cases": [{"case_id": f"p/{i}"} for i in range(3)]}
        (root / "run.json").write_text(
            json.dumps({"contract": {"methods": ["Attune"], "manifest": manifest}})
        )
        rows = []
        for i, (tp, fp, fn) in enumerate([(1, 0, 0), (0, 2, 2), (0, 0, 0)]):
            rows.append(
                dict(
                    method="Attune",
                    case_id=f"p/{i}",
                    protocol="common_pooled_events",
                    metric="audio_pitch",
                    tolerance=0.1,
                    tp=tp,
                    fp=fp,
                    fn=fn,
                    f1=1.0 if i == 0 else 0.0 if i == 1 else None,
                )
            )
            rows.append(
                dict(
                    method="Attune",
                    case_id=f"p/{i}",
                    protocol="native_style_macro",
                    metric="three_class_average",
                    tolerance=0.05,
                    f1=1 / 3,
                )
            )
        pd.DataFrame(rows).to_csv(root / "rows.csv", index=False)
        return rows

    def test_all_case_mean_and_pooled_f1_are_different(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            row = MistakeNotebook.native_overview(root).iloc[0]
            self.assertEqual(row["Mean case error F1 (100 ms), %"], 66.7)
            self.assertEqual(row["Pooled error F1 (100 ms), %"], 33.3)
            self.assertEqual(row["Mean three-class F1 (50 ms), %"], 33.3)
            self.assertEqual(row["Cases"], "3/3")

    def test_partial_results_do_not_masquerade_as_full_average(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = self.fixture(root)
            pd.DataFrame(rows[:2]).to_csv(root / "rows.csv", index=False)
            row = MistakeNotebook.native_overview(root).iloc[0]
            self.assertEqual(row["Cases"], "1/3")
            self.assertTrue(pd.isna(row["Mean case error F1 (100 ms), %"]))

    def test_duplicate_case_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = self.fixture(root)
            pd.DataFrame(rows + [rows[0]]).to_csv(root / "rows.csv", index=False)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                MistakeNotebook.native_overview(root)


if __name__ == "__main__":
    unittest.main()
