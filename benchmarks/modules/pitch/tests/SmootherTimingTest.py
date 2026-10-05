"""The same worker-local clock contract used by streaming benchmarks."""

import importlib
import multiprocessing
import time
import unittest
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch
from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase


def _timed_wait():
    result, cpu, wall = PitchDetectorBase.measure(lambda: time.sleep(0.2))
    return (cpu, wall)


class SmootherTimingTest(unittest.TestCase):

    def test_measure_uses_cpu_clock_and_thread_limit(self):
        module = importlib.import_module("benchmarks.modules.pitch.PitchDetectorBase")
        with patch.object(
            module.time, "process_time", side_effect=[10.0, 10.25]
        ), patch.object(
            module.time, "perf_counter", side_effect=[20.0, 23.0]
        ), patch.object(
            PitchDetectorBase, "single_threaded_numerics"
        ) as limit:
            result, cpu, wall = PitchDetectorBase.measure(lambda: "done")
        self.assertEqual(result, "done")
        self.assertEqual(cpu, 0.25)
        self.assertEqual(wall, 3.0)
        limit.return_value.__enter__.assert_called_once()
        limit.return_value.__exit__.assert_called_once()

    def test_parallel_wait_is_not_counted_as_cpu_work(self):
        with ProcessPoolExecutor(
            max_workers=2, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = [pool.submit(_timed_wait) for _ in range(2)]
            for future in futures:
                cpu, wall = future.result(timeout=60)
                self.assertGreaterEqual(wall, 0.18)
                self.assertLess(cpu, wall / 2)

    def test_attune_inherits_shared_measurement(self):
        from benchmarks.modules.pitch.competitors.Attune import Attune

        self.assertIs(Attune.measure.__func__, PitchDetectorBase.measure.__func__)
        self.assertEqual(Attune.COMPUTE_CLOCK, "process_cpu")


if __name__ == "__main__":
    unittest.main()
