"""Contract, spawn and cache coverage for the consolidated note module."""

import ast
import importlib
import multiprocessing
import tempfile
import unittest
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from algorithms.Config import Config
from app_logic.NoteData import Note, NoteData
from benchmarks.modules.note.NoteBenchmarker import NoteEvaluation, NoteRunner
from benchmarks.modules.note.NoteCache import NoteCache, LEGACY_FUNCTIONS
from benchmarks.modules.note.NoteDetectorBase import NoteDetectorBase
from benchmarks.modules.note.sweeps import NoteDetectionParamSweep


class NoteArchitectureTest(unittest.TestCase):
    root = Path(__file__).resolve().parents[1]

    def test_module_ownership(self):
        self.assertEqual(
            {p.stem for p in self.root.glob("*.py")} - {"__init__"},
            {"NoteBenchmarker", "NoteDetectorBase", "NoteNotebook", "NoteCache"},
        )
        for path in [
            *self.root.glob("*.py"),
            *(self.root / "competitors").glob("*.py"),
        ]:
            tree = ast.parse(path.read_text())
            self.assertFalse(
                any(
                    isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    for n in tree.body
                ),
                path,
            )
            if path.parent.name == "competitors" and path.stem != "__init__":
                module = importlib.import_module(
                    f"benchmarks.modules.note.competitors.{path.stem}"
                )
                self.assertTrue(
                    issubclass(getattr(module, path.stem), NoteDetectorBase)
                )
                self.assertEqual(
                    [n.name for n in tree.body if isinstance(n, ast.ClassDef)],
                    [path.stem],
                )

    def test_dispatch_uses_competitor_implementations(self):
        recording = SimpleNamespace(config=Config(), note_detector=None)
        for method in (
            "ruptures",
            "slope-window",
            "basic-pitch",
            "crepe-notes",
            "tony",
            "attune",
        ):
            detector = NoteDetectorBase.competitor(method)
            with patch.object(detector, "detect", return_value=method) as detect:
                self.assertEqual(
                    NoteDetectorBase(recording).detect(method, option=1), method
                )
                detect.assert_called_once_with(option=1)
        for method in (
            "attune",
            "attune-audio-only",
            "basic-pitch",
            "crepe-notes",
            "tony",
        ):
            detector = NoteDetectorBase.competitor(method)
            with patch.object(detector, "predict_task", return_value=method) as predict:
                task = {"method": method}
                self.assertEqual(NoteEvaluation._predict(task, "config"), method)
                predict.assert_called_once_with(task, "config")

    def test_spawned_class_worker(self):
        with ProcessPoolExecutor(
            max_workers=1, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            self.assertEqual(
                pool.submit(NoteRunner.parse_shard, "1/3").result(timeout=30), (1, 3)
            )

    def test_note_cache_roundtrip_and_legacy_duration(self):
        notes = NoteData()
        notes.write_note(
            Note(i=0, start_time=0.25, end_time=0.75, midi_num=[60.0, 64.0])
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "notes.json"
            NoteCache.save_note_data(notes, path, {"note_compute_time": 1.25})
            restored, metadata = NoteCache.load_note_data(path)
            self.assertEqual(restored.data[0.25].midi_num, [60.0, 64.0])
            self.assertEqual(metadata["note_compute_time"], 1.25)
        legacy = NoteCache._note_from_payload(
            dict(id=0, start_time=0.25, duration=0.5, midi_num=[60])
        )
        self.assertEqual(legacy.end_time, 0.75)

    def test_class_fingerprints_are_complete_and_nonempty(self):
        fingerprints = NoteCache.function_fingerprints(self.root / "NoteBenchmarker.py")
        self.assertEqual(set(fingerprints), set(LEGACY_FUNCTIONS))
        self.assertTrue(all(len(value) == 64 for value in fingerprints.values()))

    def test_batched_sweep_uses_ruptures_helpers(self):
        recording = SimpleNamespace()
        with patch.object(
            NoteDetectionParamSweep, "Ruptures", return_value=SimpleNamespace()
        ) as detector:
            self.assertEqual(
                NoteDetectionParamSweep._score_parameter_batch(
                    None, recording, [], [], 1.0, []
                ),
                [],
            )
            detector.assert_called_once_with(recording)
