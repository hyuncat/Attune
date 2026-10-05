from __future__ import annotations
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.stats import beta as beta_distribution
from algorithms.Config import Config
from benchmarks.modules.pitch.competitors.PYIN import PYIN
from benchmarks.modules.pitch.competitors.PYIN import (
    PYIN,
    PYINPitchDetector,
    PYINPitchSmoother,
)


class PYINTest(unittest.TestCase):
    SR = 44100
    HOP_LENGTH = 128
    FMIN = 100.0
    FMAX = 1000.0

    def setUp(self) -> None:
        self.config = Config(
            sr=self.SR,
            w1=4096,
            h1=self.HOP_LENGTH,
            fmin=self.FMIN,
            fmax=self.FMAX,
            unv_thresh=0.5,
            min_volume=0.0,
        )
        function = PYIN._viterbi_0110
        patcher = patch.object(
            PYIN, "_viterbi_0110", getattr(function, "py_func", function)
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def reference_librosa():
        try:
            import librosa
        except Exception as exc:
            raise unittest.SkipTest(f"librosa unavailable: {exc}") from exc
        if librosa.__version__ != "0.11.0":
            raise unittest.SkipTest(
                f"exact parity test requires the requirements-pinned librosa 0.11.0; found {librosa.__version__}"
            )
        return librosa

    def audio(self) -> np.ndarray:
        rng = np.random.default_rng(871)
        sample = np.arange(self.SR // 4, dtype=np.float64)
        seconds = sample / self.SR
        audio = 0.25 * np.sin(2 * np.pi * (180 + 500 * seconds) * seconds)
        audio += 0.01 * rng.standard_normal(sample.size)
        audio[:1000] = 0
        return np.asarray(audio, dtype=np.float32)

    def test_default_geometry_preserves_librosa_physical_durations(self) -> None:
        detector = PYINPitchDetector(config=self.config)
        self.assertEqual(detector.frame_length, 4096)
        self.assertEqual(detector.HOP_SIZE, 128)
        self.assertAlmostEqual(detector.frame_length / detector.SR, 2048 / 22050)
        self.assertAlmostEqual(detector.HOP_SIZE / detector.SR, 128 / 44100)

    def test_frontend_and_joint_decode_match_librosa_0110(self) -> None:
        librosa = self.reference_librosa()
        audio = self.audio()
        reference_f0, reference_voiced, reference_voiced_probability = librosa.pyin(
            audio,
            fmin=self.FMIN,
            fmax=self.FMAX,
            sr=self.SR,
            frame_length=4096,
            hop_length=self.HOP_LENGTH,
            n_thresholds=100,
            beta_parameters=(2, 18),
            boltzmann_parameter=2,
            resolution=0.1,
            max_transition_rate=35.92,
            switch_prob=0.01,
            no_trough_prob=0.01,
            fill_na=None,
            center=True,
            pad_mode="constant",
        )
        detector = PYINPitchDetector(config=self.config)
        observations, voiced_probability = detector.probabilities(audio)
        from librosa.core import pitch as librosa_pitch

        framed_audio = librosa.util.frame(
            np.pad(audio, (4096 // 2, 4096 // 2)),
            frame_length=4096,
            hop_length=self.HOP_LENGTH,
        )
        yin_frames = librosa_pitch._cumulative_mean_normalized_difference(
            framed_audio, detector.min_period, detector.max_period
        )
        parabolic_shifts = librosa_pitch._parabolic_interpolation(yin_frames)
        thresholds = np.linspace(0.0, 1.0, 101)
        beta_probabilities = np.diff(beta_distribution.cdf(thresholds, 2, 18))
        reference_observations, helper_voiced_probability = getattr(
            librosa_pitch, "__pyin_helper"
        )(
            yin_frames,
            parabolic_shifts,
            self.SR,
            thresholds,
            2,
            beta_probabilities,
            0.01,
            detector.min_period,
            self.FMIN,
            detector.n_pitch_bins,
            detector.n_bins_per_semitone,
        )
        pitches = detector.detect_pitches(audio)
        smoother = PYINPitchSmoother(config=self.config)
        states = smoother.decode(pitches)
        actual_f0 = smoother.bin_freqs[states % smoother.n_pitch_bins]
        actual_voiced = states < smoother.n_pitch_bins
        self.assertEqual(observations.shape[1], reference_f0.size)
        np.testing.assert_array_equal(
            observations, reference_observations[0, : detector.n_pitch_bins]
        )
        np.testing.assert_array_equal(voiced_probability, helper_voiced_probability[0])
        np.testing.assert_array_equal(voiced_probability, reference_voiced_probability)
        np.testing.assert_array_equal(actual_voiced, reference_voiced)
        np.testing.assert_array_equal(actual_f0, reference_f0)
        np.testing.assert_array_equal(
            np.asarray([pitch.time for pitch in pitches]),
            np.arange(reference_f0.size) * self.HOP_LENGTH / self.SR,
        )

    def test_local_pyin_row_matches_public_librosa_exactly(self) -> None:
        librosa = self.reference_librosa()
        audio = self.audio()
        reference_f0, reference_voiced, _ = librosa.pyin(
            audio,
            fmin=self.FMIN,
            fmax=self.FMAX,
            sr=self.SR,
            frame_length=4096,
            hop_length=self.HOP_LENGTH,
            n_thresholds=100,
            beta_parameters=(2, 18),
            resolution=0.1,
            max_transition_rate=35.92,
            switch_prob=0.01,
            no_trough_prob=0.01,
            center=True,
            fill_na=np.nan,
        )
        reference_times = librosa.times_like(
            reference_f0, sr=self.SR, hop_length=self.HOP_LENGTH
        )
        reference_frequencies = np.where(reference_voiced, reference_f0, 0.0)
        actual_times, actual_frequencies = PYIN().predict(
            audio, self.SR, self.FMIN, self.FMAX
        )
        np.testing.assert_array_equal(actual_times, reference_times)
        np.testing.assert_array_equal(actual_frequencies, reference_frequencies)

    def test_pyin_wires_local_port_and_separate_cache(self) -> None:
        adapter = PYIN()
        source = Path("/tmp/example.pitch.pkl.xz")
        variant = adapter.variant_cache_path(source)
        self.assertNotEqual(variant, source)
        self.assertIn(adapter.CACHE_TAG, variant.name)
        recording = adapter.recording_for(adapter.config_for(self.FMIN, self.FMAX))
        self.assertIsInstance(recording.pitch_detector, PYINPitchDetector)
        self.assertIsInstance(recording.pitch_smoother, PYINPitchSmoother)
        self.assertEqual(recording.config.sr, self.SR)
        self.assertEqual(recording.config.h1, self.HOP_LENGTH)
        self.assertEqual(recording.pitch_detector.frame_length, 4096)

    def test_framewise_adapter_uses_raw_candidates_without_hmm(self) -> None:
        from unittest.mock import patch

        audio = self.audio()[:4096]
        adapter = PYIN()
        librosa = self.reference_librosa()
        f0, voiced, _ = librosa.pyin(
            audio,
            sr=self.SR,
            fmin=self.FMIN,
            fmax=self.FMAX,
            frame_length=4096,
            hop_length=self.HOP_LENGTH,
            center=False,
        )
        expected = float(f0[0]) if voiced[0] else 0.0
        with patch(
            "benchmarks.modules.pitch.competitors.PYIN.PYINPitchSmoother",
            side_effect=AssertionError("Streaming must not construct an HMM"),
        ):
            actual = adapter.predict_frame(
                audio, self.SR, self.FMIN, self.FMAX, self.HOP_LENGTH
            )
        self.assertAlmostEqual(actual, expected)

    def test_framewise_low_voiced_mass_uses_joint_states_not_half_cutoff(self):
        adapter = PYIN()
        detector = PYINPitchDetector(config=adapter.config_for(self.FMIN, self.FMAX))
        for mass, expected_voiced in [(0.1, True), (0.0001, False), (0.0, False)]:
            observations = np.zeros((detector.n_pitch_bins, 1))
            observations[12, 0] = mass
            with patch.object(
                PYINPitchDetector,
                "probabilities",
                return_value=(observations, np.array([mass])),
            ):
                actual = adapter.predict_frame(
                    np.zeros(4096), self.SR, self.FMIN, self.FMAX, self.HOP_LENGTH
                )
            expected = detector.bin_freqs[12] if expected_voiced else 0.0
            self.assertAlmostEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
