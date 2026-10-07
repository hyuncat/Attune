"""Shared CREPE frontend reuse without running the neural model."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
from benchmarks.modules.pitch.competitors.Crepe import Crepe
from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase


class CrepeCacheTest(unittest.TestCase):

    def test_pitch_run_preserves_native_tracks_for_notes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "take.wav"
            audio.write_bytes(b"audio-identity")
            example = PitchDetectorBase.PitchExample(
                "take",
                "test",
                audio,
                np.array([0.0, 0.01]),
                np.array([440.0, 440.0]),
                400.0,
                500.0,
                root,
                root / "stage",
            )
            detector = Crepe()
            detector._inference_threads = 8
            raw = (
                np.array([0.0, 0.01]),
                np.array([440.0, 880.0]),
                np.array([0.9, 0.1]),
            )
            with patch.object(
                PitchDetectorBase.PitchExample,
                "audio",
                return_value=(np.zeros(160), 16000),
            ), patch.object(detector, "predict_raw", return_value=raw) as predict:
                estimate = detector.estimate(example)
                np.testing.assert_array_equal(estimate.freqs, [440.0, 0.0])
                path = detector.raw_cache_path(detector.cache(example).path)
                with patch.object(
                    Crepe, "predict_raw", side_effect=AssertionError("inference")
                ):
                    result = Crepe().frontend(audio, path)
                np.testing.assert_array_equal(result[1], raw[1])
                np.testing.assert_array_equal(result[2], raw[2])
                self.assertTrue(result[-1])
                self.assertEqual(result[3], estimate.compute_seconds)
                self.assertEqual(predict.call_count, 1)
                self.assertEqual(estimate.metadata["inference_threads"], 8)
                cached = Crepe().estimate(example)
                self.assertTrue(cached.from_cache)
                self.assertEqual(cached.metadata["inference_threads"], 8)
                # Raw frontend reuse also preserves the timing provenance when
                # the thresholded pitch cache needs to be recreated.
                detector.cache(example).path.unlink()
                rebuilt = Crepe().estimate(example)
                self.assertEqual(rebuilt.metadata["inference_threads"], 8)

    def test_mismatch_disabled_force_and_corruption_recompute(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio, cache = (root / "audio", root / "raw.npz")
            audio.write_bytes(b"first")
            detector = Crepe()
            load = Mock(return_value=(np.zeros(160), 16000))
            raw = (np.array([0.0]), np.array([440.0]), np.array([0.1]))
            with patch.object(detector, "predict_raw", return_value=raw) as predict:
                detector.frontend(audio, cache, load_audio=load)
                self.assertTrue(detector.frontend(audio, cache, load_audio=load)[-1])
                audio.write_bytes(b"changed")
                self.assertFalse(detector.frontend(audio, cache, load_audio=load)[-1])
                detector.viterbi = False
                self.assertFalse(detector.frontend(audio, cache, load_audio=load)[-1])
                self.assertFalse(
                    detector.frontend(audio, cache, load_audio=load, force=True)[-1]
                )
                self.assertFalse(
                    detector.frontend(audio, cache, load_audio=load, use_cache=False)[
                        -1
                    ]
                )
                cache.write_bytes(b"broken")
                self.assertFalse(detector.frontend(audio, cache, load_audio=load)[-1])
                self.assertEqual(predict.call_count, 6)


if __name__ == "__main__":
    unittest.main()
