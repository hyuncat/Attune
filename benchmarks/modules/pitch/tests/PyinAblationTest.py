from __future__ import annotations
from benchmarks.modules.pitch.competitors.PYIN import (
    PYINAdditionDetector,
    PYINGlobalVolumeGateSmoother,
    PYINPraatStyleVoicingSmoother,
)
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import soundfile as sf
from benchmarks.modules.pitch.competitors.PYIN import PYIN
from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase
from benchmarks.modules.pitch.PitchNotebook import (
    POSTHOC_VARIANTS,
    PitchNotebook,
    STREAMING_VARIANTS,
)
from benchmarks.modules.pitch.competitors.Attune import AttuneRealtime


class PyinAblationStreamingTest(unittest.TestCase):
    SR = AttuneRealtime.DEFAULT_CONFIG.sr
    FMIN = 196.0
    FMAX = 880.0

    def setUp(self) -> None:
        function = PYIN._viterbi_0110
        patcher = patch.object(
            PYIN, "_viterbi_0110", getattr(function, "py_func", function)
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        duration = 0.15
        samples = np.arange(int(duration * self.SR), dtype=np.float64)
        audio = 0.2 * np.sin(2 * np.pi * 440.0 * samples / self.SR)
        audio_path = self.root / "tone.wav"
        sf.write(audio_path, audio, self.SR, subtype="FLOAT")
        times = np.arange(0.0, duration, 0.01, dtype=np.float64)
        self.example = PitchDetectorBase.PitchExample(
            track_id="tone",
            dataset="fixture",
            audio_path=audio_path,
            ref_times=times,
            ref_freqs=np.full(times.shape, 440.0, dtype=np.float64),
            fmin=self.FMIN,
            fmax=self.FMAX,
            estimate_cache_dir=self.root / "estimates",
            stage_cache_path=self.root / "stage.pitch.pkl.xz",
        )
        self.ablation = PitchNotebook.PyinAblation(
            PitchNotebook.PyinAblationConfig(
                profile="smoke",
                use_cache=False,
                cache_root=self.root,
                streaming_seconds=duration,
                streaming_workers=1,
            )
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_streaming_grid_matches_production_geometry(self) -> None:
        grid = self.ablation._streaming_grid(self.example)
        production = AttuneRealtime()
        config = production.config_for(self.FMIN, self.FMAX, sr=self.SR)
        detector = production.recording_for(config).pitch_detector
        self.assertEqual(grid.sample_rate, self.SR)
        self.assertEqual(grid.frame_size, detector.FRAME_SIZE)
        self.assertEqual(grid.integration_size, detector.INTEGRATION_SIZE)
        self.assertEqual(grid.hop_size, detector.HOP_SIZE)
        np.testing.assert_array_equal(
            grid.starts,
            np.arange(0, len(grid.audio) - detector.FRAME_SIZE + 1, detector.HOP_SIZE),
        )
        np.testing.assert_allclose(
            grid.times,
            0.5 * detector.INTEGRATION_SIZE / self.SR
            + np.arange(grid.starts.size, dtype=np.float64)
            * detector.HOP_SIZE
            / self.SR,
        )

    def test_all_variants_share_the_production_grid(self) -> None:
        grid = self.ablation._streaming_grid(self.example)
        results = [
            self.ablation._stream_variant(self.example, variant, grid)
            for variant in STREAMING_VARIANTS
        ]
        for result in results:
            np.testing.assert_array_equal(result.estimate.times, grid.times)
            self.assertEqual(result.sample_rate, grid.sample_rate)
            self.assertEqual(result.frame_size, grid.frame_size)
            self.assertEqual(result.integration_size, grid.integration_size)
            self.assertEqual(result.hop_size, grid.hop_size)
            self.assertEqual(len(result.estimate.freqs), len(grid.starts))
        row = self.ablation._streaming_row(
            self.example, STREAMING_VARIANTS[0], results[0]
        )
        self.assertGreater(row["algorithmic_lookahead_ms"], 0.0)
        self.assertGreater(row["mean_service_time_ms"], 0.0)
        self.assertGreaterEqual(row["median_queue_delay_ms"], 0.0)
        self.assertGreaterEqual(
            row["median_output_latency_ms"], row["algorithmic_lookahead_ms"]
        )

    def test_streaming_estimate_cache_round_trip(self) -> None:
        ablation = PitchNotebook.PyinAblation(
            PitchNotebook.PyinAblationConfig(
                profile="smoke",
                use_cache=True,
                cache_root=self.root,
                streaming_seconds=0.15,
                streaming_workers=1,
            )
        )
        grid = ablation._streaming_grid(self.example)
        fresh = ablation._stream_variant(self.example, STREAMING_VARIANTS[0], grid)
        cached = ablation._stream_variant(self.example, STREAMING_VARIANTS[0], grid)
        self.assertFalse(fresh.estimate.from_cache)
        self.assertTrue(cached.estimate.from_cache)
        np.testing.assert_array_equal(cached.estimate.times, grid.times)
        np.testing.assert_array_equal(cached.estimate.freqs, fresh.estimate.freqs)
        np.testing.assert_array_equal(cached.latencies, fresh.latencies)
        np.testing.assert_array_equal(cached.call_cpus, fresh.call_cpus)

    def test_threshold_sweep_reuses_one_cached_streaming_evidence_pass(self) -> None:
        ablation = PitchNotebook.PyinAblation(
            PitchNotebook.PyinAblationConfig(
                profile="smoke",
                use_cache=True,
                cache_root=self.root,
                streaming_seconds=0.15,
                streaming_workers=1,
            )
        )
        grid = ablation._streaming_grid(self.example)
        fresh = ablation._streaming_threshold_evidence(
            self.example, prominence=False, grid=grid
        )
        with patch.object(
            ablation,
            "_decode_streaming_frame",
            side_effect=AssertionError("threshold sweep reran pYIN"),
        ):
            cached = ablation._streaming_threshold_evidence(
                self.example, prominence=False, grid=grid
            )
        np.testing.assert_array_equal(cached.times, fresh.times)
        np.testing.assert_array_equal(cached.frequencies, fresh.frequencies)
        np.testing.assert_array_equal(cached.volumes, fresh.volumes)
        np.testing.assert_array_equal(
            cached.unvoiced_probabilities, fresh.unvoiced_probabilities
        )

    def test_threshold_sweep_scores_every_requested_causal_cell(self) -> None:
        rows = self.ablation._streaming_threshold_rows_for_example(
            self.example,
            thresholds=(0.85, 0.95),
            prominences=(False, True),
            volume_modes=(False, True),
            volume_floor_ratios=(0.02, 0.04),
        )
        frame = pd.DataFrame(rows)
        self.assertEqual(len(frame), 12)
        self.assertEqual(set(frame["unvoiced_threshold"]), {0.85, 0.95})
        self.assertEqual(set(frame["prominence"]), {False, True})
        self.assertEqual(set(frame["volume_gate"]), {False, True})
        self.assertEqual(
            set(frame.loc[frame["volume_gate"], "volume_floor_ratio"]), {0.02, 0.04}
        )
        self.assertTrue((frame["execution_mode"] == "Streaming threshold sweep").all())

    def test_detector_retains_centered_rms_without_changing_audio(self) -> None:
        config = AttuneRealtime().config_for(
            self.FMIN, self.FMAX, sr=self.SR, w1=4096, h1=128
        )
        detector = PYINAdditionDetector(config=config, prominence=False)
        frame = np.linspace(-0.25, 0.25, 4096, dtype=np.float64)
        pitch = detector.detect_pitch(frame)
        self.assertAlmostEqual(
            pitch.volume, float(np.sqrt(np.mean((frame - np.mean(frame)) ** 2)))
        )

    def test_global_volume_gate_runs_after_hmm(self) -> None:
        config = AttuneRealtime().config_for(
            self.FMIN, self.FMAX, sr=self.SR, w1=4096, h1=128
        )
        detector = PYINAdditionDetector(config=config, prominence=False)
        loud = detector.detect_pitch(
            0.2 * np.sin(2 * np.pi * 440.0 * np.arange(4096) / self.SR)
        )
        quiet = detector.detect_pitch(
            0.001 * np.sin(2 * np.pi * 440.0 * np.arange(4096) / self.SR)
        )
        smoother = PYINGlobalVolumeGateSmoother(
            config=config, relative_floor=0.1, ceiling_percentile=100.0
        )
        decoded = smoother.smooth([loud, quiet])
        self.assertNotEqual(decoded[0].value, -1)
        self.assertEqual(decoded[1].value, -1)

    def test_praat_style_controller_integrates_global_volume_floor(self) -> None:
        config = AttuneRealtime().config_for(
            self.FMIN, self.FMAX, sr=self.SR, w1=4096, h1=128
        )
        detector = PYINAdditionDetector(config=config, prominence=False)
        loud = detector.detect_pitch(
            0.2 * np.sin(2 * np.pi * 440.0 * np.arange(4096) / self.SR)
        )
        quiet = detector.detect_pitch(
            0.001 * np.sin(2 * np.pi * 440.0 * np.arange(4096) / self.SR)
        )
        smoother = PYINPraatStyleVoicingSmoother(
            config=config,
            relative_floor=0.1,
            ceiling_percentile=100.0,
            unvoiced_threshold=0.85,
            switch_cost=0.02,
        )
        decoded = smoother.smooth([loud, quiet])
        self.assertNotEqual(decoded[0].value, -1)
        self.assertEqual(decoded[1].value, -1)

    def test_experiment_has_two_posthoc_and_eight_streaming_cells(self) -> None:
        self.assertEqual(len(POSTHOC_VARIANTS), 2)
        self.assertEqual(len(STREAMING_VARIANTS), 8)
        self.assertEqual(
            {
                (v.prominence, v.volume_gate, v.unvoiced_gate)
                for v in STREAMING_VARIANTS
            },
            {
                (prominence, volume, unvoiced)
                for prominence in (False, True)
                for volume in (False, True)
                for unvoiced in (False, True)
            },
        )

    def test_streaming_unvoiced_gate_uses_config_global_threshold(self) -> None:
        config = AttuneRealtime().config_for(self.FMIN, self.FMAX, unv_thresh=0.85)
        gated = next(
            (
                variant
                for variant in STREAMING_VARIANTS
                if variant.unvoiced_gate
                and (not variant.volume_gate)
                and (not variant.prominence)
            )
        )
        below, _ = self.ablation._apply_streaming_gates(
            440.0,
            0.1,
            0.849,
            gated,
            config,
            volume_floor_ratio=0.02,
            running_volume_ceiling=0.0,
        )
        at_threshold, _ = self.ablation._apply_streaming_gates(
            440.0,
            0.1,
            0.85,
            gated,
            config,
            volume_floor_ratio=0.02,
            running_volume_ceiling=0.0,
        )
        self.assertEqual(below, 440.0)
        self.assertEqual(at_threshold, 0.0)

    def test_promotion_test_requires_paired_oa_and_rpa_superiority(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "dataset": "urmp",
                    "track_id": track,
                    "unvoiced_threshold": threshold,
                    "Overall Accuracy": score,
                    "Raw Pitch Accuracy": score,
                }
                for threshold, score in ((0.9, 0.8), (0.98, 0.95))
                for track in ("a", "b")
            ]
        )
        competitor = pd.DataFrame(
            [
                {
                    "dataset": "urmp",
                    "track_id": track,
                    "model": "praat",
                    "Overall Accuracy": 0.9,
                    "Raw Pitch Accuracy": 0.9,
                }
                for track in ("a", "b")
            ]
        )
        report = self.ablation.promotion_test(
            candidates,
            competitor,
            competitor="praat",
            candidate_columns=("unvoiced_threshold",),
        )
        winner = report.loc[report["unvoiced_threshold"] == 0.98].iloc[0]
        loser = report.loc[report["unvoiced_threshold"] == 0.9].iloc[0]
        self.assertTrue(bool(winner["Clear OA+RPA win"]))
        self.assertFalse(bool(loser["Clear OA+RPA win"]))
        self.assertGreater(winner["OA CI low"], 0.0)
        self.assertGreater(winner["RPA CI low"], 0.0)


if __name__ == "__main__":
    unittest.main()
