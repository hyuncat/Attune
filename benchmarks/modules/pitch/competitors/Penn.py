"""FCNF0++ via the penn package."""

from __future__ import annotations
from typing import Any
import numpy as np
from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase


class Penn(PitchDetectorBase):
    """FCNF0++, clamped to penn's own representable range."""

    name = "penn"
    description = "FCNF0++ (penn), PyTorch"
    install_hint = "pip install penn torch"
    input_sr = 16000

    def ensure_available(self) -> None:
        try:
            import torch

            self._disable_mps(torch)
            import penn
        except Exception as exc:
            raise self.unavailable(exc) from exc

    @staticmethod
    def _disable_mps(torch: Any) -> None:
        """Stop penn's viterbi dep (torbi) from JIT-compiling its Metal backend."""
        try:
            torch.backends.mps.is_available = lambda: False
        except Exception:
            pass

    def predict(self, audio, sr, fmin, fmax) -> Melody:
        import penn
        import torch

        low = float(getattr(penn, "FMIN", 30.0))
        high = float(getattr(penn, "FMAX", 1984.0))
        fmin = float(min(max(fmin, low), high))
        fmax = float(max(min(fmax, high), low))
        if fmax <= fmin:
            fmin, fmax = (low, high)
        threshold = 0.065 if self.confidence is None else self.confidence
        samples = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None]
        pitch, periodicity = penn.from_audio(
            samples,
            sample_rate=sr,
            hopsize=self.step_seconds,
            fmin=fmin,
            fmax=fmax,
            batch_size=512,
            gpu=0 if torch.cuda.is_available() else None,
        )
        pitch = pitch.squeeze(0).cpu().numpy()
        periodicity = periodicity.squeeze(0).cpu().numpy()
        times = np.arange(pitch.size, dtype=np.float64) * self.step_seconds
        return (times, self.voiced_freqs(pitch, periodicity >= threshold))
