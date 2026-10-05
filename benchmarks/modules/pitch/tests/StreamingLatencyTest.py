"""Latency retention, cache invalidation, and exact pooling without detectors."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
from benchmarks.modules.pitch.PitchCache import PitchCache
from benchmarks.modules.pitch.PitchNotebook import PitchNotebook


class StreamingLatencyTest(unittest.TestCase):

    def save(self, root, track, values):
        row = {
            "dataset": "test",
            "track_id": track,
            "model": "attune",
            "_latency_samples": {
                "output_latency_ms": np.array(values, dtype=float),
                "call_cpu_ms": np.ones(len(values)),
                "estimate_times_seconds": np.arange(len(values), dtype=float),
            },
        }
        path = root / f"{track}.json"
        PitchCache._write_streaming_checkpoint(path, [row])
        return json.loads(path.read_text())[0]

    def test_lossless_samples_and_missing_or_corrupt_sidecars(self):
        with tempfile.TemporaryDirectory() as directory:
            row = self.save(Path(directory), "a", [1.23456789, 8.9])
            self.assertTrue(PitchCache.has_latency_samples(row))
            np.testing.assert_array_equal(
                PitchCache.read_latency_samples(row), [1.23456789, 8.9]
            )
            self.assertNotIn("_latency_samples", row)
            Path(row["latency_samples_path"]).write_bytes(b"broken")
            self.assertFalse(PitchCache.has_latency_samples(row))
            Path(row["latency_samples_path"]).unlink()
            self.assertFalse(PitchCache.has_latency_samples(row))
            self.assertFalse(
                PitchCache.has_latency_samples({"p95_output_latency_ms": 8.5})
            )

    def test_pool_updates_not_track_percentiles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = self.save(root, "a", [1.0] * 99)
            b = self.save(root, "b", [100.0])
            results = [SimpleNamespace(rows=pd.DataFrame([row])) for row in [a, b]]
            with patch("benchmarks.modules.pitch.PitchNotebook.display"):
                table = PitchNotebook().show_pooled_streaming_latency(*results)
                self.assertEqual(
                    table.loc["attune", "pooled_p95_output_latency_ms"], 1.0
                )
                self.assertEqual(table.loc["attune", "output_updates"], 100)
                with self.assertRaisesRegex(ValueError, "Duplicate"):
                    PitchNotebook().show_pooled_streaming_latency(
                        results[0], results[0]
                    )

    def test_resume_identical_checkpoint_from_different_suite_method_list(self):
        from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker
        from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio.wav"
            audio.write_bytes(b"test")
            example = PitchDetectorBase.PitchExample(
                track_id="a",
                dataset="test",
                audio_path=audio,
                ref_times=np.array([0.0]),
                ref_freqs=np.array([440.0]),
                fmin=200.0,
                fmax=600.0,
                estimate_cache_dir=root,
                stage_cache_path=root / "stage",
            )
            runner = PitchBenchmarker.PitchStreaming(
                PitchBenchmarker.Options(), PitchBenchmarker.StreamingConfig(workers=1)
            )
            old = root / "with_rmvpe" / "checkpoints"
            new = root / "without_rmvpe" / "checkpoints"
            path = runner._checkpoint_path(old, "attune", example)
            row = self.save(root, "a", [47.0])
            PitchCache._write_streaming_checkpoint(path, [row])
            with patch.object(runner, "_prepare_detectors") as prepare, patch.object(
                runner, "_run_example"
            ) as run:
                rows = runner.run([example], methods=["attune"], checkpoint_dir=new)
            run.assert_not_called()
            prepare.assert_not_called()
            self.assertEqual(len(rows), 1)
            Path(row["latency_samples_path"]).unlink()
            with patch.object(
                runner, "_prepare_detectors", return_value={}
            ), patch.object(runner, "_run_example", return_value=[]) as run:
                runner.run([example], methods=["attune"], checkpoint_dir=new)
            run.assert_called_once()
