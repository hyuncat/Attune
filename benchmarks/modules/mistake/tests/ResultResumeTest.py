"""Saved-run replay must not infer again or discard differing clean controls."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
from benchmarks.modules.mistake.MistakeNotebook import MistakeNotebook


class SavedResultsTests(unittest.TestCase):

    def test_real_completed_run_replays_from_checkpoints_without_running_pipeline(self):
        from benchmarks.modules.mistake.MistakeCache import MistakeCache

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp).resolve() / "source.mid"
            source.write_bytes(b"fixture")
            req = dict(
                output=tmp,
                stage="audio",
                sources=[str(source)],
                source_metadata={},
                methods=["LadderSym"],
                seeds=[0],
                rates=[0.0],
                tolerances=[0.1],
            )
            meta = dict(
                req,
                status="complete",
                code={},
                packages={},
                input_kinds=["audio"],
                checkpoint_schema=1,
            )
            MistakeCache.atomic_json(Path(tmp) / "run.json", meta)
            cp = MistakeCache(tmp, meta)
            job = cp.job(str(source), 0, 0.0, "audio", "LadderSym")
            batch = [
                dict(
                    source=str(source),
                    seed=0,
                    rate=0.0,
                    input="audio",
                    method="LadderSym",
                    case_id="x",
                    tolerance=0.1,
                    metric=m,
                    tp=0,
                    fp=1,
                    fn=0,
                    f1=0.0,
                    score_notes=8,
                    seconds=1.0,
                )
                for m in ["audio_pitch", "audio_missed", "audio_extra"]
            ]
            cp.save(job, batch)
            (Path(tmp) / "rows.csv").write_text("corrupt aggregate")
            with patch(
                "benchmarks.modules.mistake.MistakeBenchmarker.MistakeBenchmarker.run_comparison",
                side_effect=AssertionError("redetection"),
            ):
                result = MistakeNotebook.completed_results(req)
            self.assertEqual(len(result), 3)
            self.assertEqual(len(pd.read_csv(Path(tmp) / "rows.csv")), 3)
            self.assertTrue((Path(tmp) / "summary.csv").exists())
            source.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "No redetection"):
                MistakeNotebook.completed_results(req)

    def test_differing_clean_seeds_are_pooled_with_matching_denominators(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "run.json").write_text(json.dumps({"status": "complete"}))
            records = []
            for seed, fp in [(0, 0), (1, 2)]:
                for rate in [0.0, 0.25]:
                    records.append(
                        dict(
                            case_id=f"{seed}-{rate}",
                            source="stem",
                            seed=seed,
                            rate=rate,
                            method="LadderSym",
                            input="audio",
                            metric="audio_pitch",
                            tolerance=0.1,
                            tp=0,
                            fp=fp,
                            fn=0,
                            score_notes=20,
                        )
                    )
            pd.DataFrame(records).to_csv(Path(tmp, "rows.csv"), index=False)
            pd.DataFrame(records).to_csv(Path(tmp, "summary.csv"), index=False)
            clean = MistakeNotebook.overview(tmp)["clean"].loc["LadderSym"]
            self.assertEqual(clean["Clean stems"], 1)
            self.assertEqual(clean["Clean evaluations"], 2)
            self.assertEqual(clean["False alarms"], 2)
            self.assertEqual(clean["False alarms / 100 notes"], 5.0)
            self.assertEqual(clean["Stems with differing seed results"], 1)
