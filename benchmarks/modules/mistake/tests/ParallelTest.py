"""Real spawned-process checks, without heavyweight model dependencies."""

import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def synthetic_case(task):
    from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker
    from threadpoolctl import threadpool_info
    import numpy as np

    np.dot(np.ones((8, 8)), np.ones((8, 8)))
    root = Path(task["ready"])
    (root / str(os.getpid())).touch()
    deadline = time.monotonic() + 20
    while len(list(root.iterdir())) < 2:
        if time.monotonic() > deadline:
            raise TimeoutError("Two CPU workers did not start")
        time.sleep(0.02)
    start, wall = (MistakeBenchmarker.cpu_seconds(), time.perf_counter())
    time.sleep(0.2)
    row = dict(
        input="detected",
        method="test",
        pid=os.getpid(),
        cpu=MistakeBenchmarker.cpu_seconds() - start,
        wall=time.perf_counter() - wall,
        threads=[p["num_threads"] for p in threadpool_info()],
    )
    request = (
        dict(
            audio=str(root), score_audio="", directory="", common={"method": "PolyTune"}
        )
        if task.get("neural")
        else None
    )
    return ([row], request)


class ParallelTest(unittest.TestCase):

    def test_staged_models_start_after_cpu_exit_and_load_once(self):
        self.check_staged_models(neural_workers=1, expected_models=1)

    def test_staged_auto_models_infer_concurrently_and_reuse_each_slot(self):
        with patch("os.cpu_count", return_value=10), patch(
            "psutil.virtual_memory", return_value=SimpleNamespace(available=4 * 2**30)
        ):
            self.check_staged_models(neural_workers=None, expected_models=2)

    def check_staged_models(self, neural_workers, expected_models):
        from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker

        results, clients = ([], [])
        barrier = threading.Barrier(expected_models)

        class FakeModel:

            def __init__(self, config, directory):
                self.calls = 0
                self.closed = False
                clients.append(self)

            def predict(self, audio, score, directory):
                for marker in Path(audio).iterdir():
                    try:
                        os.kill(int(marker.name), 0)
                    except ProcessLookupError:
                        continue
                    raise AssertionError(
                        "CPU worker still alive during model inference"
                    )
                self.calls += 1
                barrier.wait(timeout=20)
                return dict(input="audio", method="PolyTune")

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as tmp:
            ready = Path(tmp) / "ready"
            ready.mkdir()
            tasks = [
                dict(
                    key=i,
                    ready=str(ready),
                    neural=True,
                    jobs={
                        ("detected", "test"): ({}, ""),
                        ("audio", "PolyTune"): (
                            {"case": i},
                            str(Path(tmp) / f"{i}.json"),
                        ),
                    },
                )
                for i in range(4)
            ]
            with patch(
                "benchmarks.modules.mistake.MistakeBenchmarker.MistakeBenchmarker._case_worker",
                synthetic_case,
            ), patch(
                "benchmarks.modules.mistake.competitors.PolyTune.PolyTune.Client",
                FakeModel,
            ), patch(
                "benchmarks.modules.mistake.MistakeBenchmarker.MistakeBenchmarker.polytune_rows",
                side_effect=lambda task, payload: [payload],
            ):
                actual = MistakeBenchmarker.run_parallel(
                    tasks,
                    output=tmp,
                    polytune=None,
                    workers=2,
                    neural_workers=neural_workers,
                    polytune_last=True,
                    on_result=lambda key, batch: results.extend(batch),
                )
            self.assertEqual(len(list(Path(tmp).glob("*.json"))), 4)
        self.assertEqual(actual, expected_models)
        self.assertEqual(len(clients), expected_models)
        self.assertTrue(all((c.calls == 4 // expected_models for c in clients)))
        self.assertTrue(all((c.closed for c in clients)))
        self.assertEqual(
            [r["input"] for r in results], ["detected"] * 4 + ["audio"] * 4
        )

    def test_staged_cached_predictions_never_load_model(self):
        from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker

        with tempfile.TemporaryDirectory() as tmp:
            ready = Path(tmp) / "ready"
            ready.mkdir()
            tasks = [
                dict(key=i, ready=str(ready), jobs={("detected", "test"): ({}, "")})
                for i in range(2)
            ]
            with patch(
                "benchmarks.modules.mistake.MistakeBenchmarker.MistakeBenchmarker._case_worker",
                synthetic_case,
            ), patch(
                "benchmarks.modules.mistake.competitors.PolyTune.PolyTune.Client"
            ) as model:
                MistakeBenchmarker.run_parallel(
                    tasks,
                    output=tmp,
                    polytune=None,
                    workers=2,
                    neural_workers=1,
                    polytune_last=True,
                    on_result=lambda key, batch: None,
                )
            model.assert_not_called()

    def test_real_process_isolation_thread_limits_and_cpu_clock(self):
        from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker

        results = []
        with tempfile.TemporaryDirectory() as tmp:
            ready = Path(tmp) / "ready"
            ready.mkdir()
            tasks = [
                dict(key=i, ready=str(ready), jobs={("detected", "test"): ({}, "")})
                for i in range(2)
            ]
            with patch(
                "benchmarks.modules.mistake.MistakeBenchmarker.MistakeBenchmarker._case_worker",
                synthetic_case,
            ):
                MistakeBenchmarker.run_parallel(
                    tasks,
                    output=tmp,
                    polytune=None,
                    workers=2,
                    neural_workers=0,
                    on_result=lambda key, batch: results.extend(batch),
                )
        self.assertEqual(len({r["pid"] for r in results}), 2)
        self.assertTrue(all((r["cpu"] < r["wall"] / 2 for r in results)))
        self.assertTrue(
            all((r["threads"] and all((n == 1 for n in r["threads"])) for r in results))
        )

    def test_default_limits_reserve_cpu_and_bound_neural_memory(self):
        from types import SimpleNamespace
        from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker

        with patch("os.cpu_count", return_value=10), patch(
            "psutil.virtual_memory", return_value=SimpleNamespace(available=4 * 2**30)
        ):
            cpu, neural = MistakeBenchmarker.worker_limits()
        self.assertGreater(cpu, 1)
        self.assertEqual(neural, 1)
        self.assertLessEqual(cpu + neural, 9)
        self.assertEqual(MistakeBenchmarker.worker_limits(2, 2), (2, 2))

    def test_staged_limits_use_free_memory_cores_pending_and_device(self):
        from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker

        with patch("os.cpu_count", return_value=10), patch(
            "psutil.virtual_memory", return_value=SimpleNamespace(available=16 * 2**30)
        ):
            self.assertEqual(MistakeBenchmarker.staged_neural_limit(20), 9)
            self.assertEqual(MistakeBenchmarker.staged_neural_limit(2), 2)
            self.assertEqual(
                MistakeBenchmarker.staged_neural_limit(20, device="cuda"), 1
            )
        with patch(
            "psutil.virtual_memory", return_value=SimpleNamespace(available=4 * 2**30)
        ):
            self.assertEqual(MistakeBenchmarker.staged_neural_limit(20), 2)


if __name__ == "__main__":
    unittest.main()
