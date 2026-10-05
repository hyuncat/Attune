"""Real concurrency, model reuse, cache migration and one-line progress checks."""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def concurrent_cpu(task):
    from threadpoolctl import threadpool_info
    import numpy as np

    np.dot(np.ones((4, 4)), np.ones((4, 4)))
    root = Path(task["ready"])
    (root / str(os.getpid())).touch()
    deadline = time.monotonic() + 20
    while len(list(root.iterdir())) < 2:
        if time.monotonic() > deadline:
            raise TimeoutError("CPU jobs did not run concurrently")
        time.sleep(0.02)
    return dict(
        method="Attune",
        pid=os.getpid(),
        threads=[i["num_threads"] for i in threadpool_info()],
    )


def fake_neural_evaluate(task, client):
    return client.predict(task)


class NativeParallelTests(unittest.TestCase):

    def test_spawned_cpu_and_grouped_persistent_models(self):
        from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker

        clients, results = ([], [])
        barrier = threading.Barrier(2)

        class Model:

            def __init__(self, config, directory):
                self.method, self.calls, self.closed = (config.name, 0, False)
                if self.method == "LadderSym":
                    assert all((c.closed for c in clients if c.method == "PolyTune"))
                clients.append(self)

            def predict(self, task):
                for marker in Path(task["ready"]).iterdir():
                    try:
                        os.kill(int(marker.name), 0)
                    except ProcessLookupError:
                        continue
                    raise AssertionError("CPU processes must exit before neural stage")
                self.calls += 1
                barrier.wait(timeout=20)
                return dict(method=self.method)

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as tmp:
            ready = Path(tmp) / "ready"
            ready.mkdir()
            tasks = [
                dict(method=m, ready=str(ready), directory=tmp)
                for m, count in [("Attune", 2), ("PolyTune", 4), ("LadderSym", 4)]
                for _ in range(count)
            ]
            models = {
                m: SimpleNamespace(name=m, device="cpu")
                for m in ("PolyTune", "LadderSym")
            }
            with patch(
                "benchmarks.modules.mistake.MistakeBenchmarker.MistakeBenchmarker.native_cpu_worker",
                concurrent_cpu,
            ), patch(
                "benchmarks.modules.mistake.MistakeBenchmarker.MistakeBenchmarker.native_evaluate",
                fake_neural_evaluate,
            ), patch(
                "benchmarks.modules.mistake.competitors.PolyTune.PolyTune.Client", Model
            ), patch(
                "benchmarks.modules.mistake.competitors.LadderSym.LadderSym.Client",
                Model,
            ):
                counts = MistakeBenchmarker.native_run_tasks(
                    tasks,
                    output=tmp,
                    models=models,
                    workers=2,
                    neural_workers=2,
                    on_result=results.append,
                )
        self.assertEqual(
            counts, dict(cpu_workers=2, neural_workers={"PolyTune": 2, "LadderSym": 2})
        )
        self.assertEqual(
            [r["method"] for r in results],
            ["Attune"] * 2 + ["PolyTune"] * 4 + ["LadderSym"] * 4,
        )
        self.assertEqual(len({r["pid"] for r in results[:2]}), 2)
        self.assertTrue(
            all(
                (
                    r["threads"] and all((n == 1 for n in r["threads"]))
                    for r in results[:2]
                )
            )
        )
        self.assertEqual(len(clients), 4)
        self.assertTrue(all((c.closed and c.calls == 2 for c in clients)))

    def test_no_pending_jobs_load_no_models(self):
        from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker

        with patch(
            "benchmarks.modules.mistake.MistakeBenchmarker.MistakeBenchmarker.native_NativeNeuralPool"
        ) as pool:
            result = MistakeBenchmarker.native_run_tasks(
                [], output="unused", models={}, on_result=lambda p: None
            )
        pool.assert_not_called()
        self.assertEqual(result, dict(cpu_workers=0, neural_workers={}))

    def test_audited_serial_cache_migration_preserves_inference_checks(self):
        from benchmarks.modules.mistake.MistakeCache import MistakeCache
        from benchmarks.modules.mistake.MistakeCache import (
            NATIVE_SERIAL_INFERENCE_SHA256,
            NATIVE_SERIAL_RUNNER_SHA256,
        )

        current = dict(
            code={
                "NativeComparison.inference": NATIVE_SERIAL_INFERENCE_SHA256,
                "algorithms/Config.py": "same",
            },
            midi_range=[21, 108],
        )
        old = dict(
            code={
                "benchmarks/modules/mistake/MistakeBenchmarker.py": NATIVE_SERIAL_RUNNER_SHA256,
                "benchmarks/modules/mistake/tests/NativeComparisonTest.py": "oldtest",
                "algorithms/Config.py": "same",
            },
            midi_range=[21, 108],
        )
        self.assertTrue(MistakeCache.compatible_contract(old, current))
        current["code"]["algorithms/Config.py"] = "changed"
        self.assertFalse(MistakeCache.compatible_contract(old, current))

    def test_progress_36_evaluations_one_line_and_cached_resume(self):
        from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker
        from benchmarks.modules.mistake.datasets.NativeDatasets import save_json, digest

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.write_text("fixture")
            cases = [
                dict(
                    case_id=str(i),
                    piece=str(i),
                    files={"performance": str(source)},
                    hashes={"performance": digest(source)},
                )
                for i in range(12)
            ]
            manifest = root / "manifest.json"
            save_json(manifest, dict(dataset="coco", split="test", cases=cases))
            models = {
                m: SimpleNamespace(preflight=lambda: {})
                for m in ("PolyTune", "LadderSym")
            }

            def execute(tasks, **kwargs):
                if tasks:
                    kwargs["on_stage"]("PolyTune", 2)
                for task in tasks:
                    rows = MistakeBenchmarker.native_event_metrics([], [])
                    for row in rows:
                        row.update(
                            dataset="coco",
                            method=task["method"],
                            case_id=task["case"]["case_id"],
                        )
                    payload = dict(rows=rows)
                    save_json(Path(task["directory"]) / "result.json", payload)
                    kwargs["on_result"](payload)
                return {}

            with patch(
                "benchmarks.modules.mistake.MistakeBenchmarker.MistakeBenchmarker.native_models_for",
                return_value=models,
            ), patch(
                "benchmarks.modules.mistake.MistakeBenchmarker.MistakeBenchmarker.native_code_identity",
                return_value={},
            ), patch(
                "benchmarks.modules.mistake.MistakeBenchmarker.MistakeBenchmarker.native_run_tasks",
                side_effect=execute,
            ) as scheduler:
                stream = io.StringIO()
                with redirect_stdout(stream):
                    MistakeBenchmarker.native_run(manifest, root / "run")
                self.assertIn(
                    "[   0/36] starting: auto-sized CPU then auto-sized model workers",
                    stream.getvalue(),
                )
                self.assertIn(
                    "[   0/36] PolyTune: 2 persistent model workers", stream.getvalue()
                )
                self.assertIn("[   1/36] Attune: 0 detected", stream.getvalue())
                self.assertIn("[  13/36] PolyTune: 0 audio", stream.getvalue())
                self.assertEqual(
                    stream.getvalue().split("\r")[-1].strip(),
                    "[  36/36] LadderSym: 11 audio",
                )
                self.assertEqual(stream.getvalue().count("\n"), 1)
                self.assertEqual(len(scheduler.call_args.args[0]), 36)
                stream = io.StringIO()
                with redirect_stdout(stream):
                    MistakeBenchmarker.native_run(
                        manifest, root / "run", workers=2, neural_workers=1
                    )
                self.assertEqual(scheduler.call_args.args[0], [])
                self.assertEqual(
                    stream.getvalue().split("\r")[-1].strip(),
                    "[  36/36] cached: LadderSym 11 audio",
                )
                self.assertEqual(stream.getvalue().count("\n"), 1)


if __name__ == "__main__":
    unittest.main()
