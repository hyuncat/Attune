"""Resource policy and actual spawned-worker scheduling without detector datasets."""
import os
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker as B
from benchmarks.modules.pitch.datasets.PitchDataset import PitchTrack


def lightweight_job(options, job, *args):
    start = time.monotonic()
    time.sleep(0.15)
    return B.Outcome(
        "ok", job["method"],
        rows=[dict(pid=os.getpid(), start=start, end=time.monotonic(), from_cache=False,
                   tf_threads=int(os.environ.get("TF_NUM_INTRAOP_THREADS", "0")))],
        peak_memory_gib=B.peak_memory_gib(),
    )


class PitchSchedulingTest(unittest.TestCase):
    def test_resource_budget_and_pressure(self):
        # Same headroom as a busy 16 GiB desktop: old 2 GiB rule serializes;
        # a measured 0.3 GiB process safely fits twice with the 0.5 GiB floor.
        with patch("psutil.virtual_memory", return_value=SimpleNamespace(
            total=16 * 2**30, available=3.5 * 2**30
        )):
            self.assertEqual(B.memory_limited_workers(8), 1)
            self.assertEqual(B.memory_limited_workers(
                8, gib_per_worker=B.worker_memory_gib("attune", .3)), 2)
            self.assertEqual(B.memory_limited_workers(1, gib_per_worker=.5), 1)
        with patch("psutil.virtual_memory", return_value=SimpleNamespace(
            total=16 * 2**30, available=2 * 2**30
        )):
            with self.assertRaisesRegex(MemoryError, "resume"):
                B.memory_limited_workers(8)
        self.assertEqual(B.worker_memory_gib("spice"), 2)
        self.assertAlmostEqual(B.worker_memory_gib("spice", 1.5), 1.95)
        with patch("psutil.virtual_memory", side_effect=OSError):
            self.assertEqual(B.memory_limited_workers(8), 2)

    def test_plan_balances_small_jobs_without_losing_tracks(self):
        runner = B(B.Options())
        tracks = [PitchTrack(str(i), "test", Path("a"), Path("b")) for i in range(41)]
        with patch.object(runner, "is_cached", return_value=False):
            jobs, _ = runner.plan(["attune"], tracks, workers=2)
        self.assertEqual([t for job in jobs for t in job.tracks], tracks)
        self.assertLessEqual(max(len(j.tracks) for j in jobs), 4)
        self.assertEqual(len({j.key for j in jobs}), len(jobs))

    def test_calibration_then_real_parallel_processes(self):
        jobs = [B.Job("attune", i, (PitchTrack(str(i), "test", Path("a"), Path("b")),))
                for i in range(5)]
        with patch.object(B, "run_job", lightweight_job), patch(
            "psutil.virtual_memory", return_value=SimpleNamespace(
                total=16 * 2**30, available=5 * 2**30
            )
        ):
            rows, errors, skipped = B(B.Options()).run(jobs, workers=2, progress=False)
        self.assertFalse(errors)
        self.assertFalse(skipped)
        self.assertEqual(len(rows), 5)
        self.assertTrue((rows.iloc[1:].start >= rows.iloc[0].end).all())
        concurrent = rows.iloc[1:].to_dict("records")
        self.assertTrue(any(a["pid"] != b["pid"] and
                            max(a["start"], b["start"]) < min(a["end"], b["end"])
                            for a in concurrent for b in concurrent))

    def test_worker_reports_peak_memory(self):
        with patch.object(B, "_run_job", return_value=B.Outcome("ok", "attune")):
            result = B.run_job()
        self.assertGreater(result.peak_memory_gib, 0)

    def test_streaming_rechecks_memory_releases_pools_and_covers_tracks(self):
        from concurrent.futures import Future
        from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase
        import numpy as np

        pools = []

        class Pool:
            def __init__(self, max_workers, **kwargs):
                self.workers = max_workers
                self.closed = False
                pools.append(self)

            def submit(self, function, example, path):
                future = Future()
                future.set_result(([dict(model="praat", dataset="test",
                                         track_id=example["track_id"])], .3))
                return future

            def shutdown(self, **kwargs):
                self.closed = True

        examples = [PitchDetectorBase.PitchExample(
            track_id=str(i), dataset="test", audio_path=Path("a"),
            ref_times=np.array([0.]), ref_freqs=np.array([440.]),
            fmin=200., fmax=600., estimate_cache_dir=Path("cache"),
            stage_cache_path=Path("stage"),
        ) for i in range(20)]
        runner = B.PitchStreaming(B.Options(), B.StreamingConfig(workers=2))
        snapshots = [SimpleNamespace(total=16 * 2**30, available=g * 2**30)
                     for g in (5., 5., 3.)]
        with patch("benchmarks.modules.pitch.PitchBenchmarker.ProcessPoolExecutor", Pool), patch(
            "psutil.virtual_memory", side_effect=snapshots
        ):
            rows = runner.run(examples, methods=["praat"])
        self.assertEqual([p.workers for p in pools], [1, 2, 1])
        self.assertTrue(all(p.closed for p in pools))
        self.assertEqual(set(rows.track_id), {str(i) for i in range(20)})
        self.assertEqual(len(rows), 20)

    def test_small_jobs_reuse_only_one_model_per_worker(self):
        B._worker_detector.cache_clear()
        try:
            with patch.object(B, "detector_for", side_effect=lambda *args: object()) as load:
                first = B._worker_detector("spice", B.Options())
                self.assertIs(first, B._worker_detector("spice", B.Options()))
                B._worker_detector("praat", B.Options())
                self.assertIsNot(first, B._worker_detector("spice", B.Options()))
                self.assertEqual(load.call_count, 3)
        finally:
            B._worker_detector.cache_clear()

    def test_cached_results_do_not_underestimate_detector_memory(self):
        from concurrent.futures import Future
        counts = []

        class Pool:
            def __init__(self, max_workers, **kwargs):
                counts.append(max_workers)

            def submit(self, function, options, job, *args):
                future = Future()
                future.set_result(B.Outcome(
                    "ok", job["method"], rows=[dict(from_cache=True)], peak_memory_gib=.2
                ))
                return future

        jobs = [B.Job("spice", i, (PitchTrack(str(i), "test", Path("a"), Path("b")),))
                for i in range(5)]
        with patch("benchmarks.modules.pitch.PitchBenchmarker.ProcessPoolExecutor", Pool), patch.object(
            B, "_teardown"
        ), patch("psutil.virtual_memory", return_value=SimpleNamespace(
            total=16 * 2**30, available=5 * 2**30
        )):
            rows, _, _ = B(B.Options()).run(jobs, workers=2, progress=False)
        self.assertEqual(counts, [1, 1])
        self.assertEqual(len(rows), 5)

    def test_crepe_shares_cpu_budget_without_multiplying_model_copies(self):
        with patch.object(B, "default_pitch_workers", return_value=8):
            for processes, threads in ((1, 8), (2, 4), (3, 2), (4, 2), (8, 1)):
                self.assertEqual(B.inference_threads("crepe", processes, 8), threads)
                self.assertLessEqual(processes * threads, 8)
            self.assertEqual(B.inference_threads("crepe", 1, 2), 2)
            self.assertEqual(B.inference_threads("crepe", 1, 99), 8)
            self.assertEqual(B.inference_threads("spice", 1, 8), 1)
        runner = B(B.Options())
        tracks = [PitchTrack(str(i), "test", Path("a"), Path("b")) for i in range(8)]
        with patch.object(runner, "is_cached", return_value=False):
            jobs, _ = runner.plan(["crepe"], tracks, workers=8)
        self.assertEqual([len(job.tracks) for job in jobs], [1] * 8)

    def test_crepe_spawn_initializer_assigns_spare_threads(self):
        jobs = [B.Job("crepe", i, (PitchTrack(str(i), "test", Path("a"), Path("b")),))
                for i in range(3)]
        with patch.object(B, "run_job", lightweight_job), patch.object(
            B, "default_pitch_workers", return_value=8
        ), patch("psutil.virtual_memory", return_value=SimpleNamespace(
            total=16 * 2**30, available=5 * 2**30
        )):
            rows, errors, _ = B(B.Options()).run(jobs, workers=8, progress=False)
        self.assertFalse(errors)
        self.assertEqual(rows.iloc[0].tf_threads, 8)
        self.assertEqual(rows.iloc[1:].tf_threads.tolist(), [4, 4])
