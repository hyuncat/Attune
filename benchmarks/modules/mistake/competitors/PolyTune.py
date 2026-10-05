"""PolyTune: adapter, pinned setup, and isolated inference in one class."""

import sys
from pathlib import Path

_root = next((p for p in Path(__file__).resolve().parents if (p / "app.py").is_file()))
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
from benchmarks.modules.mistake.MistakeDetectorBase import MistakeDetectorBase
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from time import perf_counter
from benchmarks.paths import REPO_ROOT


@dataclass(frozen=True)
class PolyTune(MistakeDetectorBase):
    UPSTREAM_COMMIT = "d2055bb21759d457c8f21c1cf2e47c79af6248f5"
    CHECKPOINT_SHA256 = (
        "9ed49dcdf864cb4957facf6d0ecfed944ea0484a5a69bccf1a7b0ed0c222f207"
    )
    ASSETS = REPO_ROOT / "benchmarks/datasets/pretrained/polytune"
    WORKER = Path(__file__).resolve()
    repo: Path = ASSETS / "source"
    python: Path = ASSETS / "venv/bin/python"
    checkpoint: Path = ASSETS / "coco.ckpt"
    device: str = "cpu"
    timeout: int = 3600

    def preflight(self):
        for path in (self.repo / "inference_error.py", self.python, self.checkpoint):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing PolyTune asset: {path}. Run python -m benchmarks.modules.mistake.competitors.PolyTune setup (documented in the notebook)."
                )
        revision = subprocess.check_output(
            ["git", "-C", str(self.repo), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != PolyTune.UPSTREAM_COMMIT:
            raise ValueError(
                f"Expected PolyTune revision {PolyTune.UPSTREAM_COMMIT}, found {revision}"
            )
        if subprocess.check_output(
            [
                "git",
                "-C",
                str(self.repo),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            text=True,
        ).strip():
            raise ValueError(
                "PolyTune source has local modifications; use the pinned clean checkout"
            )
        checkpoint_hash = PolyTune.sha256(self.checkpoint)
        if checkpoint_hash != PolyTune.CHECKPOINT_SHA256:
            raise ValueError("Checkpoint is not the official CocoChorales-E checkpoint")
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "check.json"
            subprocess.run(
                [
                    str(self.python),
                    str(PolyTune.WORKER),
                    "worker",
                    "--repo",
                    str(self.repo),
                    "--check",
                    "--output",
                    str(out),
                ],
                check=True,
                timeout=120,
                cwd=self.repo,
                capture_output=True,
                text=True,
            )
            packages = json.loads(out.read_text())
        return dict(
            revision=revision,
            checkpoint_sha256=checkpoint_hash,
            device=self.device,
            packages=packages,
        )

    def predict(self, performance_audio, score_audio, directory):
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / "polytune.json"
        output.unlink(missing_ok=True)
        command = [
            str(self.python),
            str(PolyTune.WORKER),
            "worker",
            "--repo",
            str(self.repo),
            "--checkpoint",
            str(self.checkpoint),
            "--device",
            self.device,
            "--performance",
            str(Path(performance_audio).resolve()),
            "--score",
            str(Path(score_audio).resolve()),
            "--output",
            str(output),
        ]
        start = perf_counter()
        log = directory / "polytune.log"
        with log.open("w") as stream:
            result = subprocess.run(
                command,
                cwd=self.repo,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=self.timeout,
            )
        if result.returncode or not output.is_file():
            raise RuntimeError(f"PolyTune inference failed; see {log}")
        payload = json.loads(output.read_text())
        payload["wall_seconds"] = perf_counter() - start
        return payload

    @staticmethod
    def sha256(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    class Client:
        """One isolated, single-threaded model process reused by one pool slot."""

        def __init__(self, config, directory):
            import os
            from benchmarks.modules.mistake.MistakeDetectorBase import (
                MistakeDetectorBase,
            )

            self.config = config
            directory = Path(directory).resolve()
            directory.mkdir(parents=True, exist_ok=True)
            self.log = (directory / "model.log").open("a")
            env = dict(
                os.environ, **{name: "1" for name in MistakeDetectorBase.THREAD_ENV}
            )
            self.process = subprocess.Popen(
                [
                    str(config.python),
                    str(PolyTune.WORKER),
                    "worker",
                    "--repo",
                    str(config.repo),
                    "--checkpoint",
                    str(config.checkpoint),
                    "--device",
                    config.device,
                    "--serve",
                    "--output",
                    str(directory / "ready.json"),
                ],
                cwd=config.repo,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log,
                text=True,
                bufsize=1,
            )
            try:
                if not self._reply().get("ready"):
                    raise RuntimeError(
                        f"PolyTune startup failed: {directory / 'model.log'}"
                    )
            except BaseException:
                self.close()
                raise

        def _reply(self):
            import select

            if not select.select([self.process.stdout], [], [], self.config.timeout)[0]:
                self.process.kill()
                raise TimeoutError("Persistent PolyTune worker timed out")
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(
                    f"PolyTune process exited ({self.process.poll()}); see {self.log.name}"
                )
            return json.loads(line)

        def predict(self, performance_audio, score_audio, directory):
            directory = Path(directory).resolve()
            directory.mkdir(parents=True, exist_ok=True)
            output = directory / "polytune.json"
            output.unlink(missing_ok=True)
            request = dict(
                performance=str(Path(performance_audio).resolve()),
                score=str(Path(score_audio).resolve()),
                output=str(output),
                log=str(directory / "polytune.log"),
            )
            start = perf_counter()
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            reply = self._reply()
            if not reply.get("ok") or not output.exists():
                raise RuntimeError(
                    f"PolyTune failed: {reply.get('error')}; see {request['log']}"
                )
            result = json.loads(output.read_text())
            result["wall_seconds"] = perf_counter() - start
            return result

        def close(self):
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            for stream in (self.process.stdin, self.process.stdout, self.log):
                stream.close()

    @staticmethod
    def setup():
        import argparse
        from pathlib import Path
        import subprocess
        import sys
        import urllib.request

        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--python",
            default=sys.executable,
            help="Python 3.11 or 3.12 for the isolated venv",
        )
        args = parser.parse_args()
        PolyTune.ASSETS.mkdir(parents=True, exist_ok=True)
        repo = PolyTune.ASSETS / "source"
        if not repo.exists():
            subprocess.run(
                [
                    "git",
                    "clone",
                    "https://github.com/ben2002chou/Polytune.git",
                    str(repo),
                ],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(repo), "checkout", PolyTune.UPSTREAM_COMMIT],
                check=True,
            )
        revision = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != PolyTune.UPSTREAM_COMMIT:
            raise RuntimeError(
                "Existing source differs from the pinned revision; use a clean asset directory"
            )
        env = PolyTune.ASSETS / "venv"
        if not env.exists():
            subprocess.run([args.python, "-m", "venv", str(env)], check=True)
        requirements = [
            "setuptools==70.0.0",
            "numpy==1.26.4",
            "torch==2.3.0",
            "torchaudio==2.3.0",
            "torchvision==0.18.0",
            "transformers==4.40.1",
            "timm==0.9.16",
            "note-seq==0.0.5",
            "omegaconf==2.3.0",
            "einops==0.7.0",
            "librosa==0.10.1",
            "matplotlib==3.8.4",
            "scipy==1.13.0",
        ]
        subprocess.run(
            [str(env / "bin/python"), "-m", "pip", "install", *requirements], check=True
        )
        checkpoint = PolyTune.ASSETS / "coco.ckpt"
        if not checkpoint.exists():
            print(
                "Downloading official CocoChorales-E checkpoint (2.3 GB)...", flush=True
            )
            temporary = checkpoint.with_suffix(".part")
            urllib.request.urlretrieve(
                "https://huggingface.co/ben2002chou/Polytune/resolve/main/CocoChorales-E/last.ckpt",
                temporary,
            )
            if PolyTune.sha256(temporary) != PolyTune.CHECKPOINT_SHA256:
                raise RuntimeError(
                    "Checkpoint checksum mismatch; incomplete file retained as .part"
                )
            temporary.replace(checkpoint)
        if PolyTune.sha256(checkpoint) != PolyTune.CHECKPOINT_SHA256:
            raise RuntimeError("Existing checkpoint checksum mismatch")
        print(f"PolyTune ready under {PolyTune.ASSETS}")

    @staticmethod
    def worker_main():
        import argparse
        from contextlib import redirect_stdout, redirect_stderr
        import traceback
        import json
        import os
        from pathlib import Path
        import sys
        from time import perf_counter, process_time

        started_cpu = process_time()
        started_wall = perf_counter()
        parser = argparse.ArgumentParser()
        parser.add_argument("--repo", required=True)
        parser.add_argument("--checkpoint")
        parser.add_argument("--performance")
        parser.add_argument("--score")
        parser.add_argument("--output", required=True)
        parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
        parser.add_argument("--check", action="store_true")
        parser.add_argument("--serve", action="store_true")
        args = parser.parse_args()
        repo = Path(args.repo).resolve()
        cache = repo.parent / "runtime_cache"
        os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
        os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
        sys.path.insert(0, str(repo))
        sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
        from benchmarks.modules.mistake.MistakeDetectorBase import MistakeDetectorBase

        for name in MistakeDetectorBase.THREAD_ENV:
            os.environ[name] = "1"
        protocol = sys.stdout
        if args.serve:
            sys.stdout = sys.stderr
        import torch
        import librosa
        from librosa.core.audio import load
        from importlib.metadata import version
        from omegaconf import OmegaConf
        from transformers import T5Config
        from models.polytune import T5ForConditionalGeneration
        from inference_error import InferenceHandler

        output = Path(args.output)
        output.unlink(missing_ok=True)
        packages = {
            p: version(p)
            for p in (
                "torch",
                "torchaudio",
                "transformers",
                "note-seq",
                "timm",
                "librosa",
                "numpy",
                "scipy",
                "omegaconf",
                "setuptools",
            )
        }
        if args.check:
            output.write_text(json.dumps(packages))
            return
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        start = perf_counter()
        config = OmegaConf.load(Path(args.repo) / "config/model/polytune.yaml")
        model = T5ForConditionalGeneration(
            T5Config.from_dict(OmegaConf.to_container(config.config))
        )
        checkpoint = torch.load(
            args.checkpoint, map_location="cpu", weights_only=True, mmap=True
        )
        state = checkpoint["state_dict"]
        if not state or any((not key.startswith("model.") for key in state)):
            raise ValueError(
                "Unexpected official PolyTune checkpoint state_dict layout"
            )
        model.load_state_dict(
            {key[len("model.") :]: value for key, value in state.items()},
            strict=True,
            assign=True,
        )
        del checkpoint, state
        model.eval()
        from benchmarks.modules.mistake.competitors.PolyTune import PolyTune

        class CaptureHandler(InferenceHandler):
            _split_token_into_length = PolyTune.split_frames

            def _to_event(self, *a, **kw):
                self.result = super()._to_event(*a, **kw)
                return self.result

        handler = CaptureHandler(
            model=model, device=torch.device(args.device), contiguous_inference=True
        )
        setup_seconds = perf_counter() - started_wall
        setup_cpu = process_time() - started_cpu
        first = True

        def predict(performance_path, score_path, output):
            nonlocal first
            handler.result = None
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.unlink(missing_ok=True)
            midi_output = output.with_suffix(".mid")
            midi_output.unlink(missing_ok=True)
            start, cpu_start = (perf_counter(), process_time())
            performance, _ = librosa.load(performance_path, sr=16000)
            score, _ = librosa.load(score_path, sr=16000)
            with torch.inference_mode():
                handler.inference(
                    mistake_audio=performance,
                    score_audio=score,
                    audio_path=performance_path,
                    outpath=str(midi_output),
                    batch_size=1,
                    max_length=1024,
                    num_beams=1,
                )
            if handler.result is None or not midi_output.is_file():
                raise RuntimeError("PolyTune failed; inspect the worker log")
            labels = {1: "extra", 2: "missed", 3: "correct"}
            events = []
            for note in handler.result.notes:
                if note.instrument not in labels:
                    raise ValueError(
                        f"Unexpected PolyTune error class: {note.instrument}"
                    )
                events.append(
                    dict(
                        kind=labels[note.instrument],
                        onset=float(note.start_time),
                        end=float(note.end_time),
                        pitch=int(note.pitch),
                    )
                )
            cpu = process_time() - cpu_start
            payload = dict(
                events=events,
                inference_seconds=perf_counter() - start,
                cpu_seconds=cpu,
                execution_cpu_seconds=cpu + (setup_cpu if first else 0.0),
                setup_seconds=setup_seconds if first else 0.0,
                setup_cpu_seconds=setup_cpu if first else 0.0,
                model_load_cpu_seconds=setup_cpu,
                model_load_wall_seconds=setup_seconds,
                model_reused=not first,
                worker_pid=os.getpid(),
                compute_clock="process_time",
                timing_version="cpu_v1",
                packages=packages,
            )
            first = False
            temporary = output.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, indent=2))
            temporary.replace(output)
            return payload

        if not args.serve:
            predict(args.performance, args.score, args.output)
            return
        protocol.write(json.dumps(dict(ready=True, pid=os.getpid())) + "\n")
        protocol.flush()
        for line in sys.stdin:
            request = json.loads(line)
            if request.get("close"):
                break
            try:
                with Path(request["log"]).open("a") as log, redirect_stdout(
                    log
                ), redirect_stderr(log):
                    try:
                        predict(
                            request["performance"], request["score"], request["output"]
                        )
                    except BaseException:
                        traceback.print_exc()
                        raise
                reply = dict(ok=True)
            except Exception as exc:
                reply = dict(ok=False, error=repr(exc))
            protocol.write(json.dumps(reply) + "\n")
            protocol.flush()

    @staticmethod
    def split_frames(
        self,
        mistake_frames,
        score_frames,
        mistake_frame_times,
        score_frame_times,
        features,
        max_length=256,
        return_prompt_row=False,
    ):
        import numpy as np

        if return_prompt_row:
            raise ValueError("This adapter supports inference windows only")
        for frames, times in (
            (mistake_frames, mistake_frame_times),
            (score_frames, score_frame_times),
        ):
            if len(frames) != len(times):
                raise ValueError("Frame/time length mismatch")
        total = max(len(mistake_frames), len(score_frames))
        if not total:
            raise ValueError("Empty PolyTune audio inputs")
        batches, scores, times, score_times, lengths, score_lengths = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for start in range(0, total, max_length):
            score_start = start - max_length // 2
            left = max(0, -score_start)
            score_start = max(0, score_start)
            count = max(0, min(max_length, len(mistake_frames) - start))
            score_count = max(0, min(2 * max_length, len(score_frames) - score_start))
            batch = np.zeros((max_length, *mistake_frames.shape[1:]))
            score = np.zeros((2 * max_length, *score_frames.shape[1:]))
            time = np.zeros(max_length)
            score_time = np.zeros(2 * max_length)
            batch[:count] = mistake_frames[start : start + count]
            time[:count] = mistake_frame_times[start : start + count]
            copied = max(0, score_count - left)
            score[left : left + copied] = score_frames[
                score_start : score_start + copied
            ]
            score_time[left : left + copied] = score_frame_times[
                score_start : score_start + copied
            ]
            batches.append(batch)
            scores.append(score)
            times.append(time)
            score_times.append(score_time)
            lengths.append(count)
            score_lengths.append(score_count)
        return (
            np.stack(batches),
            np.stack(scores),
            np.stack(times),
            np.stack(score_times),
            lengths,
            score_lengths,
        )

    @staticmethod
    def cli():
        import sys

        action = sys.argv.pop(1) if len(sys.argv) > 1 else "help"
        if action == "setup":
            PolyTune.setup()
        elif action == "worker" and hasattr(PolyTune, "worker_main"):
            PolyTune.worker_main()
        else:
            raise SystemExit(
                "Usage: python -m benchmarks.modules.mistake.competitors.PolyTune setup [options]"
                + (" | worker [options]" if hasattr(PolyTune, "worker_main") else "")
            )


if __name__ == "__main__":
    PolyTune.cli()
