"""Browser-policy and cache-reuse regressions without neural dependencies."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from benchmarks.modules.pitch.competitors.CrepeStreaming import CrepeStreaming
from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker
from benchmarks.modules.pitch.PitchCache import PitchCache
from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase
from benchmarks.modules.pitch.PitchNotebook import PitchNotebook


class CrepeStreamingTest(unittest.TestCase):
    def test_demo_gate_local_decoder_and_callback_resampling(self):
        detector = CrepeStreaming()
        activation = np.zeros(360)
        activation[100] = 0.5
        self.assertEqual(detector.decode(activation), 0)
        activation[100] = 0.51
        self.assertAlmostEqual(detector.decode(activation), 10 * 2 ** ((2000 + 1997.3794084376191) / 1200))
        frames = []

        def infer(frame):
            frames.append(frame.copy())
            return SimpleNamespace(numpy=lambda: activation[None])

        detector._infer = infer
        for sr in (44100, 48000):
            self.assertEqual(detector.buffer_size(sr), 4096)
            audio = np.arange(4096, dtype=np.float32)
            times, _ = detector.predict(audio, sr, 20, 5000)
            np.testing.assert_allclose(frames[-1][0], np.arange(1024) * sr / 16000, rtol=1e-7)
            np.testing.assert_array_equal(times, [0.032])
        self.assertEqual(detector.predict(np.zeros(4096), 44100, 20, 5000)[1][0], 0)

    def test_callbacks_are_nonoverlapping_and_wait_for_full_buffer(self):
        runner = PitchBenchmarker.PitchStreaming(PitchBenchmarker.Options())
        detector = CrepeStreaming()
        frames = []

        def predict(audio, *args):
            frames.append(audio.copy())
            return np.array([0.032]), np.array([440.0])

        detector.predict = predict
        audio = np.arange(4096 * 3 + 100, dtype=np.float32)
        example = SimpleNamespace(audio=lambda sr: (audio, sr), fmin=200, fmax=600)
        captured = {}

        def row(*args, **kwargs):
            captured.update(args=args, kwargs=kwargs)
            return {}

        with patch.object(runner, "_row", side_effect=row), patch.object(
            PitchDetectorBase, "measure", side_effect=lambda fn: (fn(), 0.001, 0.001)
        ):
            runner._run_rolling_example("crepe_tiny", detector, example)
        self.assertEqual(len(frames), 3)
        np.testing.assert_array_equal(frames[1], audio[4096:8192])
        np.testing.assert_allclose(captured["args"][3][0], np.arange(3) * 4096 / 44100 + 0.032)
        expected = 4096 / 44100 - 0.032
        np.testing.assert_allclose(captured["args"][6], expected + 0.001)
        self.assertAlmostEqual(captured["kwargs"]["algorithmic_lookahead"], expected)

    def test_expanded_membership_preserves_old_checkpoint_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "a.wav"
            audio.write_bytes(b"test")
            example = PitchDetectorBase.PitchExample(
                track_id="a", dataset="test", audio_path=audio,
                ref_times=np.array([0.0]), ref_freqs=np.array([440.0]),
                fmin=200, fmax=600, estimate_cache_dir=root, stage_cache_path=root / "stage",
            )
            runner = PitchBenchmarker.PitchStreaming(PitchBenchmarker.Options())
            for method in runner.METHODS:
                old = runner._checkpoint_path(root / "old/checkpoints", method, example)
                new = runner._checkpoint_path(root / "expanded/checkpoints", method, example)
                self.assertEqual(old.name, new.name)
                PitchCache._write_streaming_checkpoint(old, [{
                    "model": method, "dataset": "test", "track_id": "a",
                    "_latency_samples": {
                        "output_latency_ms": np.array([47.0]),
                        "call_cpu_ms": np.array([1.0]),
                        "estimate_times_seconds": np.array([0.0]),
                    },
                }])
            # Any inference pool creation means an existing baseline was rerun.
            with patch("benchmarks.modules.pitch.PitchBenchmarker.ProcessPoolExecutor", side_effect=AssertionError("cache miss")):
                rows = runner.run([example], checkpoint_dir=root / "expanded/checkpoints")
            self.assertEqual(set(rows.model), set(runner.METHODS))

    def test_paper_selection_is_forwarded_without_force(self):
        notebook = PitchNotebook()
        selected = (*PitchBenchmarker.PitchStreaming.METHODS, "crepe_tiny")
        with patch.object(notebook, "run_streaming_comparison") as run:
            notebook.run_paper_evaluation("Streaming", streaming_methods=selected)
        self.assertEqual(len(run.call_args_list), 3)
        for call in run.call_args_list:
            self.assertEqual(call.kwargs["suite_methods"], selected)
            self.assertNotIn("force", call.kwargs)


if __name__ == "__main__":
    unittest.main()
