from __future__ import annotations
import unittest
from unittest.mock import patch
import algorithms.PitchSmoother as pitch_smoother_module
from algorithms.Config import Config
from algorithms.PitchSmoother import PitchSmoother, VoicingSmoother
from app_logic.user.ds.PitchData import Pitch
from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker
from benchmarks.modules.pitch.competitors.PYIN import PYIN


class JointPyinSmootherTest(unittest.TestCase):

    def setUp(self) -> None:
        self.config = Config(
            sr=8000,
            w1=512,
            h1=64,
            fmin=100.0,
            fmax=800.0,
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

    def test_production_has_no_switch_cost_surface(self) -> None:
        self.assertFalse(hasattr(self.config, "posthoc_switch_cost"))
        self.assertNotIn("posthoc_switch_cost", Config.PITCH_DETECTION_FIELDS)
        self.assertFalse(hasattr(pitch_smoother_module, "_decode_voicing_path"))

    def pitch(
        self,
        frame: int,
        *,
        midi: float = 69.0,
        probability: float = 0.8,
        volume: float = 0.1,
        value: float | None = None,
    ) -> Pitch:
        return Pitch(
            time=frame * self.config.h1 / self.config.sr,
            candidates=[(midi, probability)],
            volume=volume,
            unvoiced_prob=1.0 - probability,
            live_distance=None,
            value=value,
            config=self.config,
        )

    def test_pitch_path_and_promoted_voicing_controller_are_separate(self) -> None:
        pitches = [
            self.pitch(0, probability=0.9),
            self.pitch(1, probability=0.0),
            self.pitch(2, probability=0.9),
        ]
        tracked = PitchSmoother(mode="joint", config=self.config).smooth(pitches)
        smoothed = VoicingSmoother(config=self.config).smooth(tracked)
        self.assertTrue(all((pitch.value != -1 for pitch in tracked)))
        self.assertNotEqual(smoothed[0].value, -1)
        self.assertEqual(smoothed[1].value, -1)
        self.assertNotEqual(smoothed[2].value, -1)

    def test_pitch_smoothing_uses_canonical_observations(self) -> None:
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

    def test_voicing_controller_applies_confidence_and_global_volume_gates(
        self,
    ) -> None:
        low_confidence = self.pitch(0, probability=0.0, volume=0.2, value=69.0)
        quiet_voiced = self.pitch(1, volume=0.001, value=69.0)
        loud_voiced = self.pitch(2, volume=0.2, value=69.0)
        smoothed = VoicingSmoother(config=self.config).smooth(
            [low_confidence, quiet_voiced, loud_voiced]
        )
        self.assertEqual(smoothed[0].value, -1)
        self.assertEqual(smoothed[1].value, -1)
        self.assertEqual(smoothed[2].value, 69.0)

    def test_registry_exposes_production_and_local_pyin_methods(self) -> None:
        names = {detector.name for detector in PitchBenchmarker.available_detectors()}
        self.assertIn("pyin", names)
        self.assertIn("attune", names)
        self.assertNotIn("attune_realtime", names)
        self.assertNotIn("attune_adhoc", names)
        self.assertNotIn("attune2_realtime", names)
        self.assertNotIn("attune2_adhoc", names)
        self.assertIn("basic_pitch", names)
        self.assertNotIn("pyin_smoothed", names)
        self.assertNotIn("pyin_decoupled_voicing", names)
        self.assertNotIn("pyin_praat_inspired_voicing", names)
        self.assertIsInstance(
            PitchBenchmarker.detector_for("pyin", PitchBenchmarker.Options()), PYIN
        )


if __name__ == "__main__":
    unittest.main()
