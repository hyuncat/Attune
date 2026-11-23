"""Small numerical and cache tests; never run corpus experiments."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.modules.pitch.competitors.McLeod import McLeod, TartiniSession
from experiments.modules.pitch.PitchBenchmarker import PitchBenchmarker
from experiments.modules.pitch.PitchDetectorBase import PitchDetectorBase
from experiments.modules.pitch.PitchNotebook import PitchNotebook
from experiments.modules.pitch.tests.PitchMetricsTest import counted_rows


class McLeodTest(unittest.TestCase):
    @staticmethod
    def octave_fixture():
        t = np.arange(30 * 1024) / 44100
        return (.5 * np.sin(2 * np.pi * 440 * t)
                + np.where(t < .15, .1, .5) * np.sin(2 * np.pi * 220 * t))

    def test_native_nsdf_matches_direct_definition_and_lag_origin(self):
        frame = np.random.default_rng(0).normal(size=2048).astype(np.float32)
        expected = []
        for lag in range(1, 1025):
            a, b = frame[:-lag].astype(float), frame[lag:].astype(float)
            expected.append(2 * np.dot(a, b) / (np.dot(a, a) + np.dot(b, b)))
        np.testing.assert_allclose(McLeod.nsdf(frame), expected, atol=2e-7)
        np.testing.assert_array_equal(McLeod.nsdf(np.zeros(2048)), np.zeros(1024))

    def test_silence_transitions_and_native_grid(self):
        detector = McLeod()
        tone = .5 * np.sin(2 * np.pi * 440 * np.arange(12 * 1024) / 44100)
        audio = np.r_[np.zeros(4 * 1024), tone, np.zeros(4 * 1024), tone, np.zeros(71)]
        times, frequencies = detector.predict(audio, 44100, 20, 5000)
        np.testing.assert_array_equal(times, np.arange(32) * 1024 / 44100)
        np.testing.assert_array_equal(frequencies[:4], np.zeros(4))
        np.testing.assert_array_equal(frequencies[18:20], np.zeros(2))
        np.testing.assert_allclose(frequencies[6:16], 440, atol=.02)
        np.testing.assert_allclose(frequencies[22:], 440, atol=.02)
        self.assertEqual(len(detector.predict(np.zeros(1023), 44100, 20, 5000)[0]), 0)
        with self.assertRaises(ValueError):
            detector.predict(audio, 48000, 20, 5000)

    def test_offline_revises_octaves_live_outputs_remain_causal(self):
        detector = McLeod()
        audio = self.octave_fixture()
        live = []
        try:
            for start in range(0, len(audio), 1024):
                times, frequencies = detector.predict_streaming(audio[start:start+1024], 44100, 20, 5000)
                self.assertEqual(times[0], 0)
                live.append(frequencies[0])
            revised = detector._stream.frequencies()
            offline = detector.predict(audio, 44100, 20, 5000)[1]
            np.testing.assert_array_equal(offline, revised)
            self.assertGreater(live[0], 400)
            self.assertLess(offline[0], 230)
            # Prefix estimates cannot depend on the future suffix.
            first = live[:4]
            detector.reset_streaming()
            replay = [detector.predict_streaming(audio[i:i+1024], 44100, 20, 5000)[1][0]
                      for i in range(0, 4096, 1024)]
            np.testing.assert_array_equal(replay, first)
        finally:
            detector.close_streaming()

    def test_safety_patch_matches_unpatched_upstream_on_defined_inputs(self):
        # Build a second library from the archive with the safety guard disabled.
        # Both run the original C++ DSP, not a Python approximation as oracle.
        patched, reference = TartiniSession(), TartiniSession(safety_patches=False)
        try:
            for audio in (self.octave_fixture(),):
                for start in range(0, len(audio), 1024):
                    frame = audio[start:start+1024]
                    self.assertEqual(patched.push(frame), reference.push(frame))
                    np.testing.assert_array_equal(patched.frequencies(visible=False),
                                                  reference.frequencies(visible=False))
                    np.testing.assert_array_equal(patched.frequencies(), reference.frequencies())
        finally:
            patched.close()
            reference.close()

    def test_runner_resets_tracks_and_uses_native_hops_and_latency(self):
        runner = PitchBenchmarker.PitchStreaming(PitchBenchmarker.Options())
        detector = McLeod()
        audio = self.octave_fixture()
        example = SimpleNamespace(audio=lambda sr: (audio, sr), fmin=20, fmax=5000)
        captures = []
        def row(*args, **kwargs):
            captures.append((args, kwargs))
            return {}
        with patch.object(runner, "_row", side_effect=row), patch.object(
            PitchDetectorBase, "measure", side_effect=lambda fn: (fn(), .001, .001)
        ):
            for _ in range(2):
                runner._run_rolling_example("mcleod", detector, example)
        a, options = captures[0]
        np.testing.assert_array_equal(a[3][0], np.arange(30) * 1024 / 44100)
        np.testing.assert_array_equal(a[3][1], captures[1][0][3][1])
        np.testing.assert_allclose(a[6], 1024 / 44100 + .001)
        self.assertEqual(options["algorithmic_lookahead"], 1024 / 44100)
        self.assertIsNone(detector._stream)

    def test_old_mcleod_rows_are_stale_without_invalidating_peers(self):
        rows = pd.DataFrame([{"model": "mcleod"}, {"model": "praat"}])
        self.assertEqual(PitchNotebook._stale_cached_methods(rows, ["mcleod", "praat"]), ["mcleod"])
        rows.loc[rows.model == "mcleod", "adapter_version"] = McLeod.VERSION
        self.assertEqual(PitchNotebook._stale_cached_methods(rows, ["mcleod", "praat"]), [])

    def test_only_mcleod_checkpoint_identity_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "a.wav"
            audio.write_bytes(b"identity test only")
            example = PitchDetectorBase.PitchExample(
                track_id="a", dataset="test", audio_path=audio,
                ref_times=np.array([0.0]), ref_freqs=np.array([440.0]),
                fmin=20, fmax=5000, estimate_cache_dir=root, stage_cache_path=root / "stage",
            )
            runner = PitchBenchmarker.PitchStreaming(PitchBenchmarker.Options())
            for method in runner.METHODS:
                current = runner._checkpoint_path(root / "checkpoints", method, example)
                with patch.object(McLeod, "VERSION", "mpm_fft_v1"):
                    old = runner._checkpoint_path(root / "checkpoints", method, example)
                self.assertEqual(current == old, method != "mcleod", method)

    def test_new_offline_method_only_fills_missing_rows(self):
        def rows(method):
            return counted_rows(pd.DataFrame([{
                "model": method, "dataset": "test", "track_id": "a",
                "Raw Pitch Accuracy": 0.9,
            }]))

        with tempfile.TemporaryDirectory() as directory:
            notebook = PitchNotebook(PitchNotebook.NotebookConfig(methods=("peer", "mcleod")))
            notebook.runs_root = Path(directory)
            old = rows("peer")
            path = notebook.runs_root / "suite/rows.csv"
            path.parent.mkdir()
            old.to_csv(path, index=False)
            with patch.object(notebook, "_run_methods", return_value=rows("mcleod")) as run, patch(
                "experiments.modules.pitch.PitchNotebook.display"
            ):
                result = notebook.run_suite("suite", datasets=("test",), max_tracks=None)
            self.assertEqual(run.call_args.args[0], ["mcleod"])
            self.assertFalse(run.call_args.kwargs["force_reanalysis"])
            self.assertEqual(set(result.rows.model), {"peer", "mcleod"})
            pd.testing.assert_frame_equal(
                result.rows[result.rows.model == "peer"].reset_index(drop=True),
                old, check_dtype=False,
            )


if __name__ == "__main__":
    unittest.main()
