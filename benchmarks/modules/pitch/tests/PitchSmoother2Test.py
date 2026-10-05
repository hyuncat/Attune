"""Gap-bridging contracts independent of dataset accuracy."""

import unittest
from unittest.mock import patch

import numpy as np

from algorithms.Config import Config
from algorithms.PitchSmoother import VoicingSmoother, _viterbi_0110, PitchSmoother
from app_logic.user.ds.PitchData import Pitch


class PitchOnlySmootherTest(unittest.TestCase):
    def setUp(self):
        self.config = Config(sr=44100, w1=4096, h1=128, fmin=196.0, fmax=880.0)

    def frames(self, voiced):
        return [
            Pitch(
                time=i * 128 / 44100,
                candidates=[(69.0, 0.9), (81.0, 0.05)],
                volume=0.1,
                unvoiced_prob=0.05 if flag else 1.0,
                live_distance=None,
                config=self.config,
            )
            for i, flag in enumerate(voiced)
        ]

    def test_only_short_internal_gaps_are_bridged(self):
        smoother = PitchSmoother(
            mode="pitch_only", config=self.config, max_gap_seconds=0.05
        )
        voiced = np.array(
            [False, True] + [False] * 17 + [True] + [False] * 18 + [True, False]
        )
        expected = voiced.copy()
        expected[2:19] = True
        np.testing.assert_array_equal(smoother._tracking_mask(voiced), expected)

    def test_gap_observations_are_uniform_and_output_stays_unvoiced(self):
        smoother = PitchSmoother(
            mode="pitch_only", config=self.config, max_gap_seconds=0.05
        )
        frames = self.frames([True, False, False, True])
        frames[2] = None
        calls = []

        def decode(probability, transition, initial):
            calls.append(probability.copy())
            return _viterbi_0110(probability, transition, initial)

        with patch("algorithms.PitchSmoother._viterbi_0110", side_effect=decode):
            output = smoother.smooth(frames)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].shape[0], 4)
        np.testing.assert_allclose(calls[0][1:3], -np.log(smoother.n_pitch_bins))
        original, _ = smoother.observation_probabilities(frames)
        np.testing.assert_allclose(
            calls[0][[0, 3]], np.log(original[:, [0, 3]].T + np.finfo(float).tiny)
        )
        np.testing.assert_array_equal(
            [p is not None and p.value != -1 for p in output],
            VoicingSmoother(config=self.config).decode(frames),
        )
        self.assertEqual(frames[1].unvoiced_prob, 1.0)
        self.assertEqual(frames[1].candidate_pitches, [(69.0, 0.9), (81.0, 0.05)])

    def test_zero_gap_preserves_independent_runs(self):
        smoother = PitchSmoother(mode="pitch_only", config=self.config)
        frames = self.frames([True, True, False, True, True])
        expected = np.full(5, smoother.n_pitch_bins, dtype=np.uint16)
        expected[:2] = smoother.decode(frames[:2])
        expected[3:] = smoother.decode(frames[3:])
        np.testing.assert_array_equal(smoother.decode(frames), expected)

    def test_confidence_mixture_endpoints_and_normalization(self):
        plain = PitchSmoother(mode="pitch_only", config=self.config)
        soft = PitchSmoother(
            mode="pitch_only", config=self.config, confidence_emissions=True
        )
        frames = self.frames([True, True, True])
        for frame, unvoiced in zip(frames, [0.0, 0.75, 1.0]):
            frame.unvoiced_prob = unvoiced
        original, mass = plain.observation_probabilities(frames)
        mixed, mixed_mass = soft.observation_probabilities(frames)
        np.testing.assert_allclose(mixed[:, 0], original[:, 0])
        np.testing.assert_allclose(
            mixed[:, 1], 0.25 * original[:, 1] + 0.75 / soft.n_pitch_bins
        )
        np.testing.assert_allclose(mixed[:, 2], 1.0 / soft.n_pitch_bins)
        np.testing.assert_allclose(mixed.sum(axis=0), 1.0)
        np.testing.assert_array_equal(mass, mixed_mass)
        # Unlike simply scaling candidates, the mixture permits absent pitches.
        self.assertTrue(np.all(mixed[original[:, 1] == 0, 1] > 0))
        np.testing.assert_allclose(
            soft.observation_probabilities([None])[0], 1.0 / soft.n_pitch_bins
        )

    def test_confidence_preserves_voicing_and_raw_evidence(self):
        soft = PitchSmoother(
            mode="pitch_only",
            config=self.config,
            max_gap_seconds=0.05,
            confidence_emissions=True,
        )
        frames = self.frames([True, True, False, True, False])
        frames[1].unvoiced_prob = 0.7
        evidence = [
            (p.unvoiced_prob, list(p.candidate_pitches), p.value) for p in frames
        ]
        expected = VoicingSmoother(config=self.config).decode(frames)
        output = soft.smooth(frames)
        np.testing.assert_array_equal([p.value != -1 for p in output], expected)
        self.assertEqual(
            evidence, [(p.unvoiced_prob, p.candidate_pitches, p.value) for p in frames]
        )
        self.assertEqual(soft.smooth([]), [])

    def test_disabled_confidence_preserves_previous_path(self):
        frames = self.frames([True, False, True, True])
        default = PitchSmoother(
            mode="pitch_only", config=self.config, max_gap_seconds=0.05
        )
        disabled = PitchSmoother(
            mode="pitch_only",
            config=self.config,
            max_gap_seconds=0.05,
            confidence_emissions=False,
        )
        np.testing.assert_array_equal(default.decode(frames), disabled.decode(frames))

    def test_empty_silence_and_invalid_cutoffs(self):
        smoother = PitchSmoother(
            mode="pitch_only", config=self.config, max_gap_seconds=0.05
        )
        self.assertEqual(smoother.smooth([]), [])
        self.assertEqual(smoother.smooth([None]), [None])
        self.assertTrue(
            all(p.value == -1 for p in smoother.smooth(self.frames([False] * 4)))
        )
        for cutoff in [-1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                PitchSmoother(
                    mode="pitch_only", config=self.config, max_gap_seconds=cutoff
                )


if __name__ == "__main__":
    unittest.main()
