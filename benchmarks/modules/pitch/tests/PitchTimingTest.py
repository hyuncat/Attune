from __future__ import annotations
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
import numpy as np
import pandas as pd
from algorithms.Config import Config
from app_logic.user.ds.PitchData import Pitch, PitchData
from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker
from benchmarks.modules.pitch.tests.PitchMetricsTest import counted_rows
from benchmarks.modules.pitch.PitchCache import PitchCache
from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase
from benchmarks.modules.pitch.competitors.Attune import Attune


class _SleepingDetector(PitchDetectorBase):
    name = "sleeping"

    def predict(self, audio, sr, fmin, fmax):
        time.sleep(0.04)
        return (np.array([0.0, 0.01, 0.02]), np.array([100.0, 440.0, 600.0]))


class PitchTimingTest(unittest.TestCase):

    def test_streaming_latency_excludes_wall_wait_but_retains_backlog(self):
        from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker

        track = PitchBenchmarker.PitchStreaming._empty_track()
        PitchBenchmarker.PitchStreaming._append_frame(
            track, 0.02, 440.0, 0.1, 0.04, 5.0
        )
        PitchBenchmarker.PitchStreaming._append_frame(
            track, 0.04, 440.0, 0.12, 0.04, 7.0
        )
        np.testing.assert_allclose(track["latencies"], [0.12, 0.14])
        self.assertAlmostEqual(track["cpu"], 0.08)
        self.assertEqual(track["wall"], 12.0)
        np.testing.assert_allclose(track["cpus"], [0.04, 0.04])

    def test_external_estimates_are_range_gated_before_and_after_cache(self) -> None:
        detector = _SleepingDetector()
        with tempfile.TemporaryDirectory() as directory:
            example = SimpleNamespace(
                estimate_cache_dir=Path(directory),
                safe_id="track",
                fmin=200.0,
                fmax=500.0,
                audio=lambda sr: (np.zeros(16, dtype=np.float32), 16000),
            )
            fresh = detector.estimate(example, use_cache=True)
            cached = detector.estimate(example, use_cache=True)
        expected = np.array([0.0, 440.0, 0.0])
        np.testing.assert_array_equal(fresh.freqs, expected)
        np.testing.assert_array_equal(cached.freqs, expected)
        self.assertTrue(cached.from_cache)
        rejected = PitchDetectorBase.constrain_freqs_to_range(
            np.array([100.0, 300.0, 600.0]), 200.0, 500.0
        )
        np.testing.assert_array_equal(rejected, np.array([0.0, 300.0, 0.0]))

    def test_attune_estimates_are_range_gated_before_and_after_cache(self) -> None:
        detector = Attune()
        config = detector.config_for(200.0, 500.0)
        pitch_data = PitchData(config=config)
        frequencies = (100.0, 440.0, 600.0)
        pitch_data.data = [
            Pitch(
                time=index * 0.01,
                value=config.freq_to_midi(frequency),
                candidates=[(config.freq_to_midi(frequency), 1.0)],
                volume=1.0,
                unvoiced_prob=0.0,
                live_distance=None,
                config=config,
            )
            for index, frequency in enumerate(frequencies)
        ]
        timing = {
            "pitch_detector_compute_time": 0.1,
            "pitch_smoother_compute_time": 0.0,
            "pitch_compute_time": 0.1,
            "wall_pitch_detector_compute_time": 0.2,
            "wall_pitch_smoother_compute_time": 0.0,
            "wall_pitch_compute_time": 0.2,
            "compute_clock": PitchDetectorBase.COMPUTE_CLOCK,
        }
        stages = PitchCache.Stages(
            data={PitchCache.RAW: pitch_data, PitchCache.SMOOTHED: pitch_data},
            timing={PitchCache.RAW: timing, PitchCache.SMOOTHED: timing},
        )
        with tempfile.TemporaryDirectory() as directory:
            example = SimpleNamespace(
                fmin=200.0,
                fmax=500.0,
                stage_cache_path=Path(directory) / "track.pitch.pkl.xz",
            )
            recording = SimpleNamespace(audio_data=None)
            with patch.object(
                detector, "recording_for", return_value=recording
            ), patch.object(
                detector, "_audio_data", return_value=object()
            ), patch.object(
                detector, "detect_stages", return_value=stages
            ) as detect:
                fresh = detector.estimate(example, use_cache=True)
                cached = detector.estimate(example, use_cache=True)
        self.assertEqual(fresh.freqs[0], 0.0)
        self.assertEqual(fresh.freqs[2], 0.0)
        self.assertGreaterEqual(fresh.freqs[1], example.fmin)
        self.assertLessEqual(fresh.freqs[1], example.fmax)
        np.testing.assert_array_equal(cached.freqs, fresh.freqs)
        self.assertFalse(fresh.from_cache)
        self.assertTrue(cached.from_cache)
        self.assertEqual(detect.call_count, 1)
        self.assertEqual(fresh.metadata["range_gate_fmin"], 200.0)
        self.assertEqual(fresh.metadata["range_gate_fmax"], 500.0)

    def test_detector_compute_time_excludes_scheduler_wait(self) -> None:
        detector = _SleepingDetector()
        with tempfile.TemporaryDirectory() as directory:
            example = SimpleNamespace(
                estimate_cache_dir=Path(directory),
                safe_id="track",
                fmin=100.0,
                fmax=1000.0,
                audio=lambda sr: (np.zeros(16, dtype=np.float32), 16000),
            )
            estimate = detector.estimate(example, use_cache=False)
        wall = float(estimate.metadata["wall_pitch_compute_time"])
        self.assertEqual(
            estimate.metadata["compute_clock"], PitchDetectorBase.COMPUTE_CLOCK
        )
        self.assertGreaterEqual(wall, 0.035)
        self.assertLess(estimate.compute_seconds, wall * 0.5)

    def test_legacy_wall_clock_estimate_cache_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = PitchDetectorBase.Cache(Path(directory), "method", "track")
            np.savez(
                cache.path,
                times=np.array([0.0]),
                freqs=np.array([440.0]),
                compute_time=np.array(12.0),
            )
            self.assertFalse(cache.exists())
            self.assertIsNone(cache.read())
            cache.write(
                PitchDetectorBase.PitchEstimate.build(
                    [0.0],
                    [440.0],
                    0.1,
                    metadata={
                        "compute_clock": PitchDetectorBase.COMPUTE_CLOCK,
                        "wall_pitch_compute_time": 0.2,
                    },
                )
            )
            self.assertTrue(cache.exists())
            current = cache.read()
            self.assertIsNotNone(current)
            assert current is not None
            self.assertEqual(
                current.metadata["compute_clock"], PitchDetectorBase.COMPUTE_CLOCK
            )
            self.assertEqual(current.metadata["wall_pitch_compute_time"], 0.2)

    def test_attune_stage_timings_use_worker_cpu_clock(self) -> None:

        class Detector:

            @staticmethod
            def detect_pitches(audio, show_progress, verbose):
                time.sleep(0.03)
                return []

        class Smoother:

            @staticmethod
            def smooth(pitches, verbose):
                time.sleep(0.03)
                return []

        recording = SimpleNamespace(
            audio_data=SimpleNamespace(
                read_all=lambda: np.zeros(16, dtype=np.float32), t_origin=0.0
            ),
            pitch_detector=Detector(),
            pitch_smoother=Smoother(),
            voicing_smoother=Smoother(),
            config=Config(),
        )
        stages = Attune().detect_stages(recording, smooth=True)
        raw = stages.timing[PitchCache.RAW]
        smoothed = stages.timing[PitchCache.SMOOTHED]
        self.assertEqual(raw["compute_clock"], PitchDetectorBase.COMPUTE_CLOCK)
        self.assertEqual(smoothed["compute_clock"], PitchDetectorBase.COMPUTE_CLOCK)
        self.assertLess(
            raw["pitch_detector_compute_time"],
            raw["wall_pitch_detector_compute_time"] * 0.5,
        )
        self.assertLess(
            smoothed["pitch_smoother_compute_time"],
            smoothed["wall_pitch_smoother_compute_time"] * 0.5,
        )

    def test_attune_stage_cache_distinguishes_legacy_wall_timing(self) -> None:
        config = Config()
        with tempfile.TemporaryDirectory() as directory:
            cache = PitchCache(Path(directory) / "track.pitch.pkl.xz")
            stages = PitchCache.Stages(
                data={PitchCache.RAW: PitchData(config=config)},
                timing={
                    PitchCache.RAW: {
                        "pitch_detector_compute_time": 10.0,
                        "pitch_compute_time": 10.0,
                    }
                },
            )
            cache.write(stages)
            self.assertTrue(cache.has(PitchCache.RAW))
            self.assertFalse(
                cache.has_current_timing(
                    PitchCache.RAW, PitchDetectorBase.COMPUTE_CLOCK
                )
            )
            stages.timing[PitchCache.RAW].update(
                {
                    "compute_clock": PitchDetectorBase.COMPUTE_CLOCK,
                    "wall_pitch_detector_compute_time": 20.0,
                    "wall_pitch_compute_time": 20.0,
                }
            )
            cache.write(stages)
            self.assertTrue(
                cache.has_current_timing(
                    PitchCache.RAW, PitchDetectorBase.COMPUTE_CLOCK
                )
            )

    def test_summary_excludes_legacy_wall_timing_from_throughput(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "model": "method",
                    "Raw Pitch Accuracy": 1.0,
                    "audio_seconds": 10.0,
                    "pitch_compute_time": 2.0,
                    "compute_clock": PitchDetectorBase.COMPUTE_CLOCK,
                },
                {
                    "model": "method",
                    "Raw Pitch Accuracy": 1.0,
                    "audio_seconds": 10.0,
                    "pitch_compute_time": 100.0,
                    "compute_clock": PitchDetectorBase.LEGACY_COMPUTE_CLOCK,
                },
            ]
        )
        summary = PitchBenchmarker().summarize(counted_rows(rows), ["method"])
        self.assertEqual(summary.loc["method", PitchBenchmarker.REALTIME_COL], 5.0)


if __name__ == "__main__":
    unittest.main()
