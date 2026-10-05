from benchmarks.modules.note.competitors.CrepeNotes import CrepeNotes

"Focused checks for evaluation and reporting, independent of heavy models."
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from benchmarks.modules.note.NoteNotebook import NotebookConfig
from benchmarks.modules.note.NoteNotebook import NoteNotebook
from benchmarks.modules.note.NoteDetectorBase import NoteDetectorBase
from benchmarks.modules.note.NoteBenchmarker import NoteEvaluation


class NoteNotebookTest(unittest.TestCase):

    def test_crepe_finds_decoder_tools_outside_notebook_path(self):
        import os
        import shutil
        from benchmarks.modules.note.NoteDetectorBase import NoteDetectorBase

        with tempfile.TemporaryDirectory() as directory:
            binary_dir = Path(directory) / "bin"
            binary_dir.mkdir()
            for name in ("ffmpeg", "ffprobe"):
                binary = binary_dir / name
                binary.write_text("#!/bin/sh\nexit 0\n")
                binary.chmod(493)
            with patch.dict(os.environ, {"PATH": ""}), patch(
                "sys.prefix", directory
            ), patch.object(NoteDetectorBase, "_configure_external_model_environment"):
                CrepeNotes._configure_crepe_notes_environment()
                self.assertEqual(shutil.which("ffmpeg"), str(binary_dir / "ffmpeg"))
                self.assertEqual(shutil.which("ffprobe"), str(binary_dir / "ffprobe"))
                before = os.environ["PATH"]
                CrepeNotes._configure_crepe_notes_environment()
                self.assertEqual(os.environ["PATH"], before)

    def test_crepe_missing_decoder_fails_before_model_inference(self):
        from benchmarks.modules.note.NoteDetectorBase import NoteDetectorBase

        with patch("shutil.which", return_value=None), patch.object(
            Path, "is_file", return_value=False
        ), patch.object(NoteDetectorBase, "_configure_external_model_environment"):
            with self.assertRaisesRegex(FileNotFoundError, "requires ffmpeg"):
                CrepeNotes._configure_crepe_notes_environment()

    def test_bach10_part_frame_centers_and_fractional_median(self):
        from scipy.io import savemat

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example-GTNotes.mat"
            parts = np.empty((4, 1), dtype=object)
            for part in range(4):
                notes = np.empty((2, 1), dtype=object)
                notes[0, 0] = np.array(
                    [[1, 2, 3], [69.1 + part, 69.2 + part, 69.9 + part]]
                )
                notes[1, 0] = np.array([[4, 5, 6], [69.2 + part] * 3])
                parts[part, 0] = notes
            savemat(path, {"GTNotes": parts})
            iv, hz = NoteEvaluation.reference_notes(
                dict(dataset="bach10-original", reference=str(path), reference_part=2)
            )
            np.testing.assert_allclose(iv, [[0.023, 0.043], [0.053, 0.073]])
            np.testing.assert_allclose(hz, [440 * 2 ** ((71.2 - 69) / 12)] * 2)
            result = NoteEvaluation.score_predictions(iv, hz, iv, hz, NotebookConfig())
            self.assertEqual(result["f1"], 1)
            self.assertEqual(result["f1_offset"], 1)

    def test_urmp_uses_hz_and_duration_not_offset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Notes.txt"
            path.write_text("2.0 440.0 0.3\n")
            iv, hz = NoteEvaluation.reference_notes(
                {"dataset": "urmp", "reference": str(path)}
            )
            np.testing.assert_allclose(iv, [[2, 2.3]])
            np.testing.assert_allclose(hz, [440])

    def test_simultaneous_predictions_are_not_overwritten(self):
        iv, hz = NoteDetectorBase.event_arrays([(0, 1, 69), (0, 1, 81)])
        self.assertEqual(len(hz), 2)
        result = NoteEvaluation.score_predictions(
            iv[:1], hz[:1], iv, hz, NotebookConfig()
        )
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 1)

    def test_no_reference_latency_correction_and_offset_metric_is_distinct(self):
        iv, hz = NoteDetectorBase.event_arrays([(0, 1, 69)])
        shifted = NoteEvaluation.score_predictions(
            iv, hz, iv + 0.1, hz, NotebookConfig()
        )
        self.assertEqual(shifted["f1"], 0)
        shortened = NoteEvaluation.score_predictions(
            iv, hz, np.array([[0, 0.5]]), hz, NotebookConfig()
        )
        self.assertEqual(shortened["f1"], 1)
        self.assertEqual(shortened["f1_offset"], 0)

    def test_micro_summary_pools_counts_across_tracks_and_datasets(self):
        rows = []
        for dataset, n, tp in [("a", 100, 90), ("a", 10, 0), ("b", 10, 0)]:
            rows.append(
                dict(
                    dataset=dataset,
                    method="attune",
                    status="ok",
                    reference_notes=n,
                    estimated_notes=n,
                    precision=tp / n,
                    recall=tp / n,
                    precision_offset=tp / n,
                    recall_offset=tp / n,
                    audio_seconds=1,
                    cpu_seconds=1,
                )
            )
        frame = pd.DataFrame(rows)
        per_dataset = NoteNotebook.summarize(frame).set_index("dataset")
        self.assertAlmostEqual(per_dataset.loc["a", "f1"], 180 / 220)
        pooled = NoteNotebook.summarize(frame, by_dataset=False).iloc[0]
        self.assertEqual((pooled.tp, pooled.fp, pooled.fn), (90, 30, 30))
        self.assertEqual(pooled.f1, 0.75)
        self.assertEqual(pooled.f1_offset, 0.75)

    def test_empty_predictions_count_all_reference_notes_as_missed(self):
        row = dict(
            dataset="a",
            method="attune",
            status="ok",
            reference_notes=10,
            estimated_notes=0,
            precision=0.0,
            recall=0.0,
            precision_offset=0.0,
            recall_offset=0.0,
            audio_seconds=1,
            cpu_seconds=1,
        )
        result = NoteNotebook.summarize(pd.DataFrame([row])).iloc[0]
        self.assertEqual((result.tp, result.fp, result.fn, result.f1), (0, 0, 10, 0))
        row["status"] = "error"
        result = NoteNotebook.summarize(pd.DataFrame([row])).iloc[0]
        self.assertTrue(np.isnan(result.f1))
        self.assertFalse(result.complete)

    def test_failures_not_scored_as_zero_and_pairs_report_coverage(self):
        rows = []
        for method in ("attune", "basic-pitch"):
            for i in range(2):
                failed = method == "basic-pitch" and i == 1
                rows.append(
                    dict(
                        dataset="coco",
                        method=method,
                        track_id=str(i),
                        group=str(i),
                        status="error" if failed else "ok",
                        reference_notes=10,
                        estimated_notes=10,
                        precision=0.8,
                        recall=0.8,
                        precision_offset=0.6,
                        recall_offset=0.6,
                        f1=np.nan if failed else 0.8,
                        f1_offset=np.nan if failed else 0.6,
                        cpu_seconds=1,
                        audio_seconds=2,
                    )
                )
        data = pd.DataFrame(rows)
        summary = NoteNotebook.summarize(data).set_index("method")
        self.assertEqual(summary.loc["basic-pitch", "f1"], 0.8)
        self.assertEqual(summary.loc["basic-pitch", "failed"], 1)
        pair = NoteNotebook().paired_comparisons(data)
        self.assertFalse(pair.complete.any())
        self.assertTrue(pair.ci_low.isna().all())


if __name__ == "__main__":
    unittest.main()
