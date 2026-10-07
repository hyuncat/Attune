"""CREPE, in both shipped forms: the original TensorFlow package and the PyTorch port."""

from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
import tempfile
from pathlib import Path
import numpy as np
from typing import Any
from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase


class Crepe(PitchDetectorBase):
    """The reference TensorFlow CREPE, with its own Viterbi decoding."""

    name = "crepe"
    description = "CREPE CNN, TensorFlow reference implementation"
    install_hint = "pip install crepe tensorflow   (tensorflow-macos on Apple Silicon)"
    input_sr = 16000

    def __init__(
        self, model_capacity: str = "full", viterbi: bool = True, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.model_capacity = model_capacity
        self.viterbi = viterbi

    def ensure_available(self) -> None:
        try:
            self.quiet_tensorflow()
            import crepe
        except Exception as exc:
            raise self.unavailable(exc) from exc

    @staticmethod
    def raw_cache_path(estimate_path):
        return Path(estimate_path).with_suffix(".raw.npz")

    def frontend(
        self, audio_path, cache_path, *, use_cache=True, force=False, load_audio=None
    ):
        """Ungated F0/confidence, shared by pitch scoring and Crepe-Notes.

        Both consumers use the pitch benchmark's librosa mono/16 kHz input.
        This differs from crepe_notes' native-rate/resampy loading path.
        Never reconstruct confidence from legacy, thresholded pitch estimates.
        """
        versions = {}
        for package in ("crepe", "tensorflow", "librosa"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        identity = json.dumps(
            dict(
                schema=1,
                audio_sha256=hashlib.sha256(Path(audio_path).read_bytes()).hexdigest(),
                model_capacity=self.model_capacity,
                viterbi=self.viterbi,
                step_ms=int(round(self.step_seconds * 1000)),
                preprocessing="librosa-mono-16000",
                versions=versions,
            ),
            sort_keys=True,
        )
        path = Path(cache_path)
        if use_cache and (not force):
            try:
                with np.load(path, allow_pickle=False) as saved:
                    if (
                        str(saved["identity"].item()) == identity
                        and str(saved["compute_clock"].item()) == self.COMPUTE_CLOCK
                    ):
                        arrays = tuple(
                            (
                                saved[k].copy()
                                for k in ("times", "frequency", "confidence")
                            )
                        )
                        cpu, wall = (
                            float(saved["compute_time"]),
                            float(saved["wall_compute_time"]),
                        )
                        if (
                            arrays[0].ndim == 1
                            and arrays[0].size
                            and all(
                                (
                                    x.shape == arrays[0].shape and np.isfinite(x).all()
                                    for x in arrays
                                )
                            )
                            and np.isfinite(cpu)
                            and (cpu > 0)
                            and np.isfinite(wall)
                            and (wall > 0)
                        ):
                            self._frontend_inference_threads = (
                                int(saved["inference_threads"])
                                if "inference_threads" in saved else None
                            )
                            return (*arrays, cpu, wall, True)
            except (OSError, ValueError, KeyError, EOFError):
                pass
        if load_audio is None:
            import librosa

            audio, sr = librosa.load(str(audio_path), sr=self.input_sr, mono=True)
        else:
            audio, sr = load_audio()
        arrays, cpu, wall = self.measure(lambda: self.predict_raw(audio, sr))
        self._frontend_inference_threads = getattr(self, "_inference_threads", None)
        thread_metadata = (
            {"inference_threads": self._frontend_inference_threads}
            if self._frontend_inference_threads is not None else {}
        )
        if use_cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=path.parent, suffix=".npz", delete=False
            ) as tmp:
                temporary = Path(tmp.name)
            try:
                np.savez(
                    temporary,
                    times=arrays[0],
                    frequency=arrays[1],
                    confidence=arrays[2],
                    identity=identity,
                    compute_time=cpu,
                    wall_compute_time=wall,
                    compute_clock=self.COMPUTE_CLOCK,
                    **thread_metadata,
                )
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        return (*arrays, cpu, wall, False)

    def predict_raw(self, audio, sr):
        import crepe
        import tensorflow as tf

        self._inference_threads = (
            tf.config.threading.get_intra_op_parallelism_threads()
            or int(os.environ.get("TF_NUM_INTRAOP_THREADS", "0"))
        )
        times, freqs, confidence, _ = crepe.predict(
            audio,
            sr,
            model_capacity=self.model_capacity,
            viterbi=self.viterbi,
            step_size=int(round(self.step_seconds * 1000)),
            verbose=0,
        )
        return tuple(
            (np.asarray(x, dtype=np.float64) for x in (times, freqs, confidence))
        )

    def predict(self, audio, sr, fmin, fmax) -> Melody:
        times, freqs, confidence = self.predict_raw(audio, sr)
        threshold = 0.5 if self.confidence is None else self.confidence
        return (times, self.voiced_freqs(freqs, confidence >= threshold))

    def estimate(self, example, use_cache=True, force_reanalysis=False):
        if self.name != "crepe" or (
            example.degradation is not None and (not example.degradation.is_clean)
        ):
            return super().estimate(example, use_cache, force_reanalysis)
        cache = self.cache(example)
        fmin, fmax = self.clamp_range(example.fmin, example.fmax)
        if (
            use_cache
            and (not force_reanalysis)
            and ((cached := cache.read()) is not None)
        ):
            return self.constrain_estimate_to_range(cached, fmin, fmax)
        times, freqs, confidence, cpu, wall, hit = self.frontend(
            example.audio_path,
            self.raw_cache_path(cache.path),
            use_cache=use_cache,
            force=force_reanalysis,
            load_audio=lambda: example.audio(self.input_sr),
        )
        threshold = 0.5 if self.confidence is None else self.confidence
        estimate = PitchDetectorBase.PitchEstimate.build(
            times,
            self.voiced_freqs(freqs, confidence >= threshold),
            cpu,
            from_cache=hit,
            metadata={
                "compute_clock": self.COMPUTE_CLOCK,
                "wall_pitch_compute_time": wall,
                **({"inference_threads": self._frontend_inference_threads}
                   if self._frontend_inference_threads is not None else {}),
            },
        )
        if use_cache:
            cache.write(estimate)
        return self.constrain_estimate_to_range(estimate, fmin, fmax)


class TorchCrepe(Crepe):
    """The PyTorch port, which accepts a per-track search range CREPE proper ignores."""

    name = "torchcrepe"
    description = "CREPE CNN, PyTorch port (torchcrepe)"
    install_hint = "pip install torchcrepe torch"
    input_sr = 16000
    model_fmin = 32.7
    model_fmax = 1975.5

    def ensure_available(self) -> None:
        try:
            import torch
            import torchcrepe
        except Exception as exc:
            raise self.unavailable(exc) from exc

    def predict(self, audio, sr, fmin, fmax) -> Melody:
        import torch
        import torchcrepe

        threshold = 0.21 if self.confidence is None else self.confidence
        hop = int(round(self.step_seconds * sr))
        samples = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None]
        pitch, periodicity = torchcrepe.predict(
            samples,
            sr,
            hop_length=hop,
            fmin=fmin,
            fmax=fmax,
            model=self.model_capacity,
            return_periodicity=True,
            batch_size=512,
            device="cuda" if torch.cuda.is_available() else "cpu",
            pad=True,
        )
        pitch = pitch.squeeze(0).cpu().numpy()
        periodicity = periodicity.squeeze(0).cpu().numpy()
        times = np.arange(pitch.size, dtype=np.float64) * hop / sr
        return (times, self.voiced_freqs(pitch, periodicity >= threshold))
