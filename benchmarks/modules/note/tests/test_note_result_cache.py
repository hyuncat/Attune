"""Cache compatibility and interrupted-run recovery without detector inference."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from benchmarks.modules.note.NoteNotebook import NoteNotebook
from benchmarks.modules.note.NoteNotebook import NotebookConfig
from benchmarks.modules.note.NoteCache import LEGACY_FUNCTIONS
from benchmarks.modules.note.NoteCache import LEGACY_NOTEBOOK_SHA
from benchmarks.modules.note.NoteCache import NOTEBOOK_PATH
from benchmarks.modules.note.NoteCache import REPEAT_RECOVERY_PREDICT_SHA
from benchmarks.modules.note.NoteCache import CREPE_CACHE_PREDICT_SHA
from benchmarks.modules.note.NoteCache import NoteCache


class NoteResultCacheTest(unittest.TestCase):

    def fixture(self):
        tracks = [
            dict(
                dataset="coco",
                track_id=str(i),
                group=str(i),
                instrument="violin",
                audio=f"audio{i}",
                reference=f"ref{i}",
                score=f"ref{i}",
                pitch_annotation=f"f0{i}",
                score_part=0,
                pitch_fmin=100,
                pitch_fmax=1000,
            )
            for i in range(2)
        ]
        sources = {
            p: "same"
            for p in (
                "benchmarks/modules/note/NoteDetectionBaselines.py",
                "benchmarks/modules/note/NoteBenchmarker.py",
                "benchmarks/modules/pitch/competitors/Attune.py",
                "benchmarks/modules/pitch/competitors/Crepe.py",
                "benchmarks/modules/pitch/PitchDetectorBase.py",
                "algorithms/Config.py",
                "app_logic/NoteData.py",
            )
        }
        sources[NOTEBOOK_PATH] = LEGACY_NOTEBOOK_SHA
        p = dict(
            sources=sources,
            cache_functions=LEGACY_FUNCTIONS,
            inputs={
                t[k]: t[k] + "-hash"
                for t in tracks
                for k in ("audio", "reference", "score", "pitch_annotation")
            },
            config=dict(
                onset_tolerance=0.05,
                pitch_tolerance=50,
                offset_ratio=0.2,
                offset_min_tolerance=0.05,
                workers=9,
                neural_workers=3,
                methods=["basic-pitch"],
                seed=0,
            ),
            versions={},
            python="same",
            platform="same",
            threads=1,
            device="CPU",
        )
        return (tracks, p)

    def test_refactored_layout_tracks_each_competitor_and_shared_code(self):
        tracks, provenance = self.fixture()
        sources = provenance["sources"]
        sources.pop("benchmarks/modules/note/NoteDetectionBaselines.py")
        sources["benchmarks/modules/pitch/PitchCache.py"] = "same"
        for name in ("NoteDetectorBase", "NoteCache"):
            sources[f"benchmarks/modules/note/{name}.py"] = "same"
        competitors = {
            "attune": "Attune",
            "attune-audio-only": "Attune",
            "basic-pitch": "BasicPitch",
            "crepe-notes": "CrepeNotes",
            "tony": "Tony",
        }
        for name in set(competitors.values()):
            sources[f"benchmarks/modules/note/competitors/{name}.py"] = "same"
        provenance["config"].update(
            audio_only_min_note=0.03, audio_only_fmin=32.0, audio_only_fmax=2093.0
        )
        provenance["cache_functions"] = NoteCache.function_fingerprints(
            Path(__file__).resolve().parents[1] / "NoteBenchmarker.py"
        )
        for method, competitor in competitors.items():
            task = {**tracks[0], "method": method}
            key = NoteCache.job_key(provenance, task)
            self.assertIsNotNone(key)
            for path in (
                f"benchmarks/modules/note/competitors/{competitor}.py",
                "benchmarks/modules/note/NoteDetectorBase.py",
                "benchmarks/modules/note/NoteCache.py",
            ):
                changed = copy.deepcopy(provenance)
                changed["sources"][path] = "changed"
                self.assertNotEqual(key, NoteCache.job_key(changed, task), path)
            foreign = "Tony" if competitor != "Tony" else "BasicPitch"
            changed = copy.deepcopy(provenance)
            changed["sources"][
                f"benchmarks/modules/note/competitors/{foreign}.py"
            ] = "changed"
            changed["config"]["workers"] = 100
            self.assertEqual(key, NoteCache.job_key(changed, task))
            changed["cache_functions"] = {}
            self.assertIsNone(NoteCache.job_key(changed, task))

    def test_scheduling_selection_git_and_unrelated_code_do_not_invalidate(self):
        tracks, p = self.fixture()
        t = {**tracks[0], "method": "basic-pitch"}
        key = NoteCache.job_key(p, t)
        changed = copy.deepcopy(p)
        changed["config"].update(
            workers=6, neural_workers=6, methods=["tony", "basic-pitch"], seed=8
        )
        changed["git_head"] = "new-commit"
        changed["sources"]["benchmarks/modules/note/tests/new_test.py"] = "new"
        changed["sources"]["benchmarks/modules/pitch/PitchNotebook.py"] = "new"
        changed["sources"][NOTEBOOK_PATH] = "new-orchestration-only"
        self.assertEqual(key, NoteCache.job_key(changed, t))
        changed["inputs"][tracks[1]["audio"]] = "other-track-edited"
        self.assertEqual(key, NoteCache.job_key(changed, t))

    def test_semantic_changes_invalidate(self):
        tracks, p = self.fixture()
        t = {**tracks[0], "method": "basic-pitch"}
        key = NoteCache.job_key(p, t)
        for section, name, value in [
            ("inputs", t["audio"], "new"),
            ("inputs", t["reference"], "new"),
            ("config", "pitch_tolerance", 25),
            ("versions", "tensorflow", "new"),
            ("sources", "benchmarks/modules/note/NoteDetectionBaselines.py", "new"),
            ("cache_functions", "_predict", "new"),
        ]:
            changed = copy.deepcopy(p)
            changed[section][name] = value
            self.assertNotEqual(key, NoteCache.job_key(changed, t), (section, name))

    def test_repeat_recovery_invalidates_only_attune(self):
        tracks, old = self.fixture()
        old["sources"]["benchmarks/modules/pitch/PitchCache.py"] = "same"
        old["config"]["audio_only_min_note"] = 0.03
        new = copy.deepcopy(old)
        new["cache_functions"]["_predict"] = REPEAT_RECOVERY_PREDICT_SHA
        for method in ("basic-pitch", "crepe-notes", "tony", "attune"):
            old["sources"]["benchmarks/modules/note/Tony.py"] = "same"
            new["sources"]["benchmarks/modules/note/Tony.py"] = "same"
            task = {**tracks[0], "method": method}
            before, after = (NoteCache.job_key(old, task), NoteCache.job_key(new, task))
            self.assertIsNotNone(before)
            if method == "attune":
                self.assertNotEqual(before, after)
            else:
                self.assertEqual(before, after)

    def test_shared_crepe_change_preserves_other_method_checkpoints(self):
        tracks, old = self.fixture()
        old["sources"]["benchmarks/modules/pitch/PitchCache.py"] = "same"
        old["sources"]["benchmarks/modules/note/Tony.py"] = "same"
        old["config"]["audio_only_min_note"] = 0.03
        old["cache_functions"] = {
            **LEGACY_FUNCTIONS,
            "_predict": REPEAT_RECOVERY_PREDICT_SHA,
        }
        new = copy.deepcopy(old)
        new["cache_functions"]["_predict"] = CREPE_CACHE_PREDICT_SHA
        for method in ("attune", "basic-pitch", "tony", "crepe-notes"):
            task = {**tracks[0], "method": method}
            before, after = (NoteCache.job_key(old, task), NoteCache.job_key(new, task))
            self.assertIsNotNone(before)
            if method == "crepe-notes":
                self.assertNotEqual(before, after)
            else:
                self.assertEqual(before, after)

    def test_force_methods_refreshes_attune_and_reuses_competitors(self):
        tracks, provenance = self.fixture()
        provenance["sources"]["benchmarks/modules/pitch/PitchCache.py"] = "same"
        provenance["config"]["audio_only_min_note"] = 0.03
        with tempfile.TemporaryDirectory() as directory:
            bench = NoteNotebook(NotebookConfig(methods=("attune", "basic-pitch")))
            bench.runs_root = Path(directory)
            calls = []

            class Pool:

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

                def imap_unordered(self, worker, jobs, chunksize):
                    for task, config, output in jobs:
                        calls.append(task["method"])
                        NoteCache.atomic_json(
                            Path(output) / "predictions" / f"{task['job_id']}.json",
                            {"intervals": [[0, 1]], "frequencies_hz": [440]},
                        )
                        yield {
                            **task,
                            "status": "ok",
                            "error": "",
                            "reference_notes": 1,
                            "estimated_notes": 1,
                            "precision": 1.0,
                            "recall": 1.0,
                            "precision_offset": 1.0,
                            "recall_offset": 1.0,
                        }

            class Context:

                def Pool(self, *args, **kwargs):
                    return Pool()

            with patch.object(
                bench, "select_tracks", return_value=tracks
            ), patch.object(bench, "provenance", return_value=provenance), patch.object(
                bench, "summarize", return_value=pd.DataFrame()
            ), patch.object(
                bench, "paired_comparisons", return_value=pd.DataFrame()
            ), patch(
                "benchmarks.modules.note.NoteBenchmarker.mp.get_context",
                return_value=Context(),
            ):
                bench.run_preliminary("coco")
                calls.clear()
                bench.config = NotebookConfig(
                    methods=("attune", "basic-pitch"), force_methods=("attune",)
                )
                rows = bench.run_preliminary("coco")
                self.assertEqual(calls, ["attune", "attune"])
                self.assertEqual(len(rows), 4)

    def test_only_audited_legacy_implementation_migrates(self):
        tracks, p = self.fixture()
        t = {**tracks[0], "method": "basic-pitch"}
        key = NoteCache.job_key(p, t)
        del p["cache_functions"]
        self.assertEqual(key, NoteCache.job_key(p, t))
        p["sources"][NOTEBOOK_PATH] = "unknown-old-code"
        self.assertIsNone(NoteCache.job_key(p, t))

    def test_interrupt_resume_and_repeat_do_not_redetect_completed_jobs(self):
        tracks, p = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            bench = NoteNotebook(NotebookConfig(methods=("basic-pitch",), workers=6))
            bench.runs_root = Path(directory)
            calls = []

            class Pool:
                interrupt = True

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

                def imap_unordered(self, worker, jobs, chunksize):
                    for task, config, output in jobs:
                        calls.append(task["track_id"])
                        NoteCache.atomic_json(
                            Path(output) / "predictions" / f"{task['job_id']}.json",
                            {"intervals": [[0, 1]], "frequencies_hz": [440]},
                        )
                        yield {
                            **task,
                            "status": "ok",
                            "error": "",
                            "reference_notes": 1,
                            "estimated_notes": 1,
                            "precision": 1.0,
                            "recall": 1.0,
                            "precision_offset": 1.0,
                            "recall_offset": 1.0,
                        }
                        if Pool.interrupt:
                            raise KeyboardInterrupt

            class Context:

                def Pool(self, *args, **kwargs):
                    return Pool()

            with patch.object(
                bench, "select_tracks", return_value=tracks
            ), patch.object(
                bench, "provenance", side_effect=lambda t: copy.deepcopy(p)
            ), patch.object(
                bench, "summarize", return_value=pd.DataFrame()
            ), patch.object(
                bench, "paired_comparisons", return_value=pd.DataFrame()
            ), patch(
                "benchmarks.modules.note.NoteBenchmarker.mp.get_context",
                return_value=Context(),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    bench.run_preliminary("coco")
                self.assertEqual(len(NoteCache.checkpoint_index(bench.runs_root)), 1)
                p["config"]["neural_workers"] = 6
                Pool.interrupt = False
                rows = bench.run_preliminary("coco")
                self.assertEqual(calls, ["0", "1"])
                self.assertEqual(len(rows), 2)
                with patch(
                    "benchmarks.modules.note.NoteBenchmarker.mp.get_context",
                    side_effect=AssertionError("No pool on cached rerun"),
                ):
                    bench.run_preliminary("coco")
                self.assertEqual(calls, ["0", "1"])


if __name__ == "__main__":
    unittest.main()
