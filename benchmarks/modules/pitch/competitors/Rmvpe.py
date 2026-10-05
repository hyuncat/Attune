"""RMVPE, the deep U-Net tracker vendored from the RVC forks."""

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Any, ClassVar
import numpy as np
from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase
from benchmarks.paths import DATASETS_ROOT


class Rmvpe(PitchDetectorBase):
    """Loaded pluggably: RMVPE has no canonical package, only per-fork vendoring.

    A checkpoint path and a dotted module exposing class ``RMVPE`` can come from
    the constructor or the environment; otherwise the vendored copy under
    ``benchmarks/datasets/rmvpe`` is found automatically. The call follows the
    RVC-standard ``infer_from_audio(audio16k, thred=...)``.
    """

    name = "rmvpe"
    description = "RMVPE deep U-Net (vendored)"
    install_hint = "vendor an RMVPE module exposing class RMVPE plus the rmvpe.pt checkpoint, then pass --rmvpe-checkpoint PATH --rmvpe-module DOTTED.PATH (or set RMVPE_CHECKPOINT / RMVPE_MODULE). e.g. RVC's infer/lib/rmvpe.py"
    input_sr = 16000
    HOP_SECONDS: ClassVar[float] = 0.01
    VENDORED_DIR: ClassVar[Path] = DATASETS_ROOT / "rmvpe"

    def __init__(
        self,
        checkpoint: str | None = None,
        module: str | None = None,
        threshold: float = 0.03,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.checkpoint = checkpoint or os.environ.get("RMVPE_CHECKPOINT")
        if not self.checkpoint and (self.VENDORED_DIR / "rmvpe.pt").exists():
            self.checkpoint = str(self.VENDORED_DIR / "rmvpe.pt")
        self.module = module or os.environ.get("RMVPE_MODULE")
        self.threshold = float(threshold)
        self._module_name: str | None = None

    def _candidate_modules(self) -> list[str]:
        candidates = [self.module] if self.module else []
        return [
            m
            for m in candidates + ["rmvpe", "infer.lib.rmvpe", "lib.rmvpe", "rvc.rmvpe"]
            if m
        ]

    def ensure_available(self) -> None:
        import importlib

        try:
            import torch
        except Exception as exc:
            raise self.unavailable(exc) from exc
        if self.VENDORED_DIR.is_dir() and str(self.VENDORED_DIR) not in sys.path:
            sys.path.insert(0, str(self.VENDORED_DIR))
        if not self.checkpoint or not Path(self.checkpoint).exists():
            raise self.Unavailable(
                self.name,
                f"checkpoint not found: {self.checkpoint!r}",
                self.install_hint,
            )
        for name in self._candidate_modules():
            try:
                if hasattr(importlib.import_module(name), "RMVPE"):
                    self._module_name = name
                    return
            except Exception:
                continue
        raise self.Unavailable(
            self.name,
            f"no module with class RMVPE in {self._candidate_modules()}",
            self.install_hint,
        )

    def _load(self):
        if self._model is None:
            import importlib
            import torch

            if self._module_name is None:
                self.ensure_available()
            model_class = getattr(importlib.import_module(self._module_name), "RMVPE")
            self._model = model_class(
                self.checkpoint,
                is_half=False,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        return self._model

    def predict(self, audio, sr, fmin, fmax) -> Melody:
        model = self._load()
        samples = np.ascontiguousarray(audio, dtype=np.float32)
        try:
            freqs = model.infer_from_audio(samples, thred=self.threshold)
        except TypeError:
            freqs = model.infer_from_audio(samples, self.threshold)
        freqs = np.asarray(freqs, dtype=np.float64).reshape(-1)
        times = np.arange(freqs.size, dtype=np.float64) * self.HOP_SECONDS
        return (times, freqs)
