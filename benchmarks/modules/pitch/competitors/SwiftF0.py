"""SwiftF0, the CNN tracker from lars76's pitch-benchmark."""

from __future__ import annotations
import numpy as np
from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase


class SwiftF0(PitchDetectorBase):
    """Fixed G1..C7 model with the shared per-track output range gate.

    SwiftF0 always infers over its fixed trained range; its public ``fmin`` and
    ``fmax`` settings only change the voicing mask. The benchmark mirrors that
    behavior after inference so SwiftF0 receives the same admissible track range
    as methods whose APIs can use the range during candidate selection.
    """

    name = "swiftf0"
    description = "SwiftF0 CNN (swift-f0)"
    install_hint = "pip install swift-f0"
    input_sr = 16000
    model_fmin = 46.875
    model_fmax = 2093.75

    def ensure_available(self) -> None:
        try:
            import swift_f0
        except Exception as exc:
            raise self.unavailable(exc) from exc

    def _detector(self):
        if self._model is None:
            from swift_f0 import SwiftF0 as SwiftF0Model

            self._model = SwiftF0Model(
                confidence_threshold=(
                    0.9 if self.confidence is None else self.confidence
                ),
                fmin=self.model_fmin,
                fmax=self.model_fmax,
            )
        return self._model

    def predict(self, audio, sr, fmin, fmax) -> Melody:
        result = self._detector().detect_from_array(audio, sr)
        return (
            np.asarray(result.timestamps, dtype=np.float64),
            self.voiced_freqs(result.pitch_hz, result.voicing),
        )
