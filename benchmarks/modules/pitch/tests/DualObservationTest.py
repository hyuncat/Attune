from __future__ import annotations
from benchmarks.modules.pitch.competitors.PYIN import (
    PYINAdditionDetector,
    PYINPraatStyleVoicingSmoother,
)
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from algorithms.Config import Config, PYIN_GLOBAL_VOLUME_PERCENTILE
import algorithms.PitchSmoother as pitch_smoother_module
from benchmarks.modules.pitch.competitors.PYIN import PYIN
from algorithms.PitchDetector import PitchDetector
from algorithms.PitchSmoother import PitchSmoother, VoicingSmoother
from app_logic.JsonHandler import JsonHandler
from app_logic.user.ds.PitchData import Pitch, PitchData
from app_logic.user.ds.Recording import Recording
from benchmarks.modules.pitch.PitchCache import PitchCache
from benchmarks.modules.pitch.PitchNotebook import PitchNotebook


class DualObservationTest(unittest.TestCase):

    def setUp(self) -> None:
        self.config = Config(
            sr=8000,
            w1=512,
            h1=64,
            fmin=100.0,
            fmax=800.0,
            unv_thresh=0.985,
            min_volume=0.03,
            posthoc_unv_thresh=0.985,
            posthoc_min_volume=0.04,
        )
        function = pitch_smoother_module._viterbi_0110
        patcher = patch.object(
            pitch_smoother_module,
            "_viterbi_0110",
            getattr(function, "py_func", function),
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        reference_function = PYIN._viterbi_0110
        reference_patcher = patch.object(
            PYIN,
            "_viterbi_0110",
            getattr(reference_function, "py_func", reference_function),
        )
        reference_patcher.start()
        self.addCleanup(reference_patcher.stop)

    def frame(self, amplitude: float = 0.8) -> np.ndarray:
        time = np.arange(self.config.w1) / self.config.sr
        return np.asarray(
            amplitude * np.sin(2 * np.pi * 220.0 * time), dtype=np.float32
        )

    def test_detector_matches_ordinary_defensive_ablation_frontend(self) -> None:
        audio = np.concatenate((self.frame(), self.frame(0.2)))
        production = PitchDetector(config=self.config)
        observations, voiced = production.probabilities(audio)
        ordinary = PYINAdditionDetector(
            config=self.config, prominence=False, frame_length=self.config.w1
        )
        reference, reference_voiced = ordinary.probabilities(audio)
        np.testing.assert_array_equal(observations, reference)
        np.testing.assert_array_equal(voiced, reference_voiced)

    def test_detector_stores_only_one_canonical_observation(self) -> None:
        pitch = PitchDetector(config=self.config).detect_pitch(self.frame())
        self.assertTrue(pitch.candidate_pitches)
        self.assertFalse(hasattr(pitch, "posthoc_candidate_pitches"))
        self.assertFalse(hasattr(pitch, "posthoc_unvoiced_prob"))

    def test_completed_contour_matches_defensive_ablation(self) -> None:
        audio = np.tile(self.frame(), 8)
        production_pitches = PitchDetector(config=self.config).detect_pitches(audio)
        actual = VoicingSmoother(config=self.config).smooth(
            PitchSmoother(mode="joint", config=self.config).smooth(production_pitches)
        )
        reference_detector = PYINAdditionDetector(
            config=self.config, prominence=False, frame_length=self.config.w1
        )
        reference_pitches = reference_detector.detect_pitches(audio)
        expected = PYINPraatStyleVoicingSmoother(
            config=self.config,
            relative_floor=self.config.posthoc_min_volume,
            ceiling_percentile=PYIN_GLOBAL_VOLUME_PERCENTILE,
            unvoiced_threshold=self.config.posthoc_unv_thresh,
            switch_cost=0.0,
        ).smooth(reference_pitches)
        np.testing.assert_array_equal(
            [pitch.value for pitch in actual], [pitch.value for pitch in expected]
        )
        np.testing.assert_array_equal(
            [pitch.unvoiced_prob for pitch in actual],
            [pitch.unvoiced_prob for pitch in expected],
        )
        np.testing.assert_array_equal(
            [pitch.time for pitch in actual], [pitch.time for pitch in expected]
        )

    def test_preprocessing_only_centers_the_frame(self) -> None:
        detector = PitchDetector(config=self.config)
        samples = np.linspace(-0.4, 0.8, detector.FRAME_SIZE)
        expected = samples - np.mean(samples)
        actual, volume = detector.preprocess_audio(samples)
        np.testing.assert_allclose(actual, expected)
        self.assertAlmostEqual(volume, float(np.sqrt(np.mean(expected**2))))

    def test_causal_gate_suppresses_only_the_live_value(self) -> None:
        detector = PitchDetector(config=self.config)
        loud = detector.detect_pitch(self.frame(0.8))
        quiet = detector.detect_pitch(self.frame(0.001))
        self.assertNotEqual(loud.value, -1)
        self.assertEqual(quiet.value, -1)
        self.assertTrue(quiet.candidate_pitches)

    def test_posthoc_smoothing_uses_canonical_candidates(self) -> None:
        pitch = Pitch(
            time=0.0,
            candidates=[(70.0, 1.0)],
            volume=0.1,
            unvoiced_prob=0.0,
            live_distance=None,
            config=self.config,
        )
        smoothed = PitchSmoother(mode="joint", config=self.config).smooth([pitch])[0]
        self.assertAlmostEqual(smoothed.value, 70.0, places=1)
        self.assertEqual(smoothed.candidate_pitches, [(70.0, 1.0)])
        self.assertEqual(pitch.value, 70.0)

    def test_live_track_switches_to_posthoc_without_redetection(self) -> None:
        pitch = Pitch(
            time=0.0,
            candidates=[(70.0, 1.0)],
            volume=0.1,
            unvoiced_prob=0.0,
            live_distance=None,
            config=self.config,
        )
        pitch_data = PitchData(config=self.config)
        pitch_data.load([pitch])
        detector = SimpleNamespace(requires_smoothing=True, detect_pitches=Mock())
        recording = SimpleNamespace(
            pitches_smoothed=False,
            pitch_detector=detector,
            pitch_smoother=PitchSmoother(mode="joint", config=self.config),
            voicing_smoother=VoicingSmoother(config=self.config),
            pitch_data=pitch_data,
        )
        Recording.smooth_pitches(recording)
        detector.detect_pitches.assert_not_called()
        self.assertTrue(recording.pitches_smoothed)
        self.assertAlmostEqual(recording.pitch_data.data[0].value, 70.0, places=1)

    def test_posthoc_volume_floor_uses_full_track_percentile(self) -> None:
        volumes = np.arange(1.0, 101.0)
        smoother = VoicingSmoother(config=self.config)
        expected = (
            np.percentile(volumes, PYIN_GLOBAL_VOLUME_PERCENTILE)
            * self.config.posthoc_min_volume
        )
        self.assertAlmostEqual(smoother.volume_floor(volumes), expected)

    def test_canonical_observation_survives_compact_caches(self) -> None:
        pitch = PitchDetector(config=self.config).detect_pitch(self.frame())
        handler = JsonHandler()
        recording = SimpleNamespace(config=self.config)
        sidecar_payload = handler._pitch_to_payload(pitch)
        benchmark_payload = PitchCache._pitch_to_payload(pitch)
        sidecar_pitch = handler._pitch_from_payload(recording, sidecar_payload)
        benchmark_pitch = PitchCache._pitch_from_payload(benchmark_payload, self.config)
        self.assertEqual(len(sidecar_payload), 8)
        self.assertNotIn("posthoc_candidates", benchmark_payload)
        self.assertNotIn("posthoc_unvoiced_prob", benchmark_payload)
        for restored in (sidecar_pitch, benchmark_pitch):
            self.assertEqual(restored.candidate_pitches, pitch.candidate_pitches)
            self.assertEqual(restored.unvoiced_prob, pitch.unvoiced_prob)
            self.assertFalse(hasattr(restored, "posthoc_candidate_pitches"))


if __name__ == "__main__":
    unittest.main()
