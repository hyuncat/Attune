"""Execution-only scheduler: two model processes, two recording threads per model.

The original YourMT3Direct.predict implementation and checkpoint remain unchanged.
This module controls lifetime/concurrency, not model or note-decoding semantics.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import time

MODEL_PROCESSES = 2
RECORDINGS_PER_MODEL = 2


def serve(manifest):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock
    from unittest.mock import patch
    import YourMT3Direct as direct
    started = time.process_time()
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    jobs = json.loads(Path(manifest).read_text())
    model = direct.load_model()
    # Upstream lazily fills positional buffers on the first forward. Serialize
    # that first segment only, so later concurrent forwards read initialized buffers.
    original_inference = model.inference
    first_forward = Lock()
    warmed = False
    def inference(*args, **kwargs):
        nonlocal warmed
        if not warmed:
            with first_forward:
                if not warmed:
                    result = original_inference(*args, **kwargs)
                    warmed = True
                    return result
        return original_inference(*args, **kwargs)
    model.inference = inference
    startup_cpu = time.process_time() - started

    def run(job):
        cpu, wall = time.thread_time(), time.monotonic()
        output = Path(job['output'])
        temporary = output.with_suffix('.working')
        try:
            direct.predict(job['audio'], temporary)
            result = json.loads(temporary.read_text())
            # process_time inside predict includes the peer thread: replace it
            # with this single-threaded request's CPU plus its startup share.
            result.update(cpu_seconds=time.thread_time()-cpu + startup_cpu/len(jobs),
                          wall_seconds=time.monotonic()-wall,
                          startup_cpu_share=startup_cpu/len(jobs),
                          compute_clock='thread+amortized_model_startup',
                          model_pid=os.getpid(), model_identity=id(model),
                          started_monotonic=wall, finished_monotonic=time.monotonic())
        except Exception:
            import traceback
            result = {'error': traceback.format_exc()}
        temporary.write_text(json.dumps(result))
        temporary.replace(output)
        return job['job_id']

    # predict's numerical body stays untouched; reuse the loaded weights and do
    # not change PyTorch's global thread configuration from concurrent requests.
    with patch.object(direct, 'load_model', return_value=model), patch.object(torch, 'set_num_threads'):
        with ThreadPoolExecutor(max_workers=RECORDINGS_PER_MODEL) as pool:
            futures = [pool.submit(run, job) for job in jobs]
            for future in as_completed(futures):
                future.result()


def predictions(jobs, *, python=None, directory=None):
    """Yield each completed recording immediately; terminate children on interruption."""
    import tempfile
    from benchmarks.modules.note.experiments.YourMT3Direct import ASSETS
    python = python or ASSETS / 'venv/bin/python'
    with tempfile.TemporaryDirectory(prefix='attune-yourmt3-shared-', dir=directory) as folder:
        folder = Path(folder)
        processes = []
        logs = []
        outstanding = {}
        groups = [jobs[i::MODEL_PROCESSES] for i in range(min(MODEL_PROCESSES, len(jobs)))]
        try:
            for index, group in enumerate(groups):
                manifest = folder / f'jobs-{index}.json'
                records = []
                for payload in group:
                    task = payload[0]
                    output = folder / f'{task["job_id"]}.json'
                    records.append(dict(job_id=task['job_id'], audio=task['audio'], output=str(output)))
                    outstanding[task['job_id']] = (payload, output, index)
                manifest.write_text(json.dumps(records))
                # Persist shared-process diagnostics alongside the benchmark logs.
                log_path = Path(group[0][2]) / 'logs' / f'yourmt3-model-{index}.txt'
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log = log_path.open('w')
                logs.append(log)
                env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                       'OPENBLAS_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1',
                       'USE_TF': '0', 'WANDB_MODE': 'disabled', 'TORCHDYNAMO_DISABLE': '1'}
                processes.append(subprocess.Popen([str(python), str(Path(__file__).resolve()),
                                                    str(manifest)], env=env, stdout=log, stderr=log))
            while outstanding:
                completed = []
                for key, (payload, output, index) in list(outstanding.items()):
                    if output.exists():
                        result = json.loads(output.read_text())
                    elif processes[index].poll() is not None:
                        # Recheck after poll to avoid racing the final atomic rename.
                        if output.exists():
                            result = json.loads(output.read_text())
                        else:
                            result = {'error': f'YourMT3 model process exited ({processes[index].returncode}); see yourmt3-model-{index}.txt'}
                    else:
                        continue
                    completed.append(key)
                    yield payload, result
                for key in completed:
                    del outstanding[key]
                if outstanding:
                    time.sleep(0.1)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for log in logs:
                log.close()


if __name__ == '__main__':
    serve(sys.argv[1])
