"""Check pool construction without starting workers or pitch detection."""
import unittest
from pathlib import Path
from unittest.mock import patch
from experiments.modules.pitch.PitchBenchmarker import PitchBenchmarker
from experiments.modules.pitch.datasets.PitchDataset import PitchTrack


class PoolIntercepted(Exception):
    pass


class AttuneWorkersTest(unittest.TestCase):
    def test_offline_starts_eight_without_calibration_or_memory_limit(self):
        bench = PitchBenchmarker(PitchBenchmarker.Options())
        tracks = [PitchTrack(str(i), 'test', Path('unused'), Path('unused')) for i in range(26)]
        with patch.object(bench, 'is_cached', return_value=False):
            jobs, _ = bench.plan(['attune'], tracks, workers=1)
        self.assertGreaterEqual(len(jobs), 8)
        with patch.object(bench, 'memory_limited_workers', side_effect=AssertionError('RAM throttle')), patch(
                'experiments.modules.pitch.PitchBenchmarker.ProcessPoolExecutor',
                side_effect=PoolIntercepted) as pool:
            with self.assertRaises(PoolIntercepted):
                bench.run(jobs, workers=1, progress=False)
        self.assertEqual(pool.call_args.kwargs['max_workers'], 8)

    def test_streaming_starts_eight_or_remaining_tracks(self):
        runner = PitchBenchmarker.PitchStreaming(
            PitchBenchmarker.Options(), PitchBenchmarker.StreamingConfig(workers=1))
        for count in (3, 26):
            with self.subTest(count=count), patch.object(
                    PitchBenchmarker, 'memory_limited_workers', side_effect=AssertionError('RAM throttle')), patch(
                    'experiments.modules.pitch.PitchBenchmarker.ProcessPoolExecutor',
                    side_effect=PoolIntercepted) as pool:
                with self.assertRaises(PoolIntercepted):
                    runner.run([None] * count, methods=['attune'])
                self.assertEqual(pool.call_args.kwargs['max_workers'], min(8, count))

    def test_competitor_still_calibrates_and_uses_memory_limit(self):
        runner = PitchBenchmarker.PitchStreaming(
            PitchBenchmarker.Options(), PitchBenchmarker.StreamingConfig(workers=8))
        with patch.object(PitchBenchmarker, 'memory_limited_workers', return_value=2) as limit, patch(
                'experiments.modules.pitch.PitchBenchmarker.ProcessPoolExecutor',
                side_effect=PoolIntercepted) as pool:
            with self.assertRaises(PoolIntercepted):
                runner.run([None] * 26, methods=['praat'])
            limit.assert_called_once()
            self.assertEqual(pool.call_args.kwargs['max_workers'], 1)
