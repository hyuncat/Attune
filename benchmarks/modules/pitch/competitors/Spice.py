"""SPICE, Google's self-supervised pitch estimator, from TF Hub."""

from __future__ import annotations
from typing import ClassVar
from pathlib import Path
import numpy as np
from benchmarks.paths import DATASETS_ROOT
from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase


class Spice(PitchDetectorBase):
    """Outputs a unitless pitch index, mapped to Hz by the model card's constants."""

    name = "spice"
    description = "SPICE self-supervised pitch estimator (TF Hub)"
    install_hint = (
        "pip install tensorflow tensorflow_hub   (tensorflow-macos on Apple Silicon)"
    )
    input_sr = 16000
    HUB_URL: ClassVar[str] = "https://tfhub.dev/google/spice/2"
    LOCAL_MODEL: ClassVar[Path] = DATASETS_ROOT / "spice" / "2"
    DEFAULT_CONFIDENCE: ClassVar[float] = 0.35
    HOP: ClassVar[int] = 512

    @property
    def confidence_threshold(self) -> float:
        return self.DEFAULT_CONFIDENCE if self.confidence is None else self.confidence

    @property
    def cache_method(self) -> str:
        return f"{self.name}_confidence_{self.confidence_threshold!r}"

    def cache(self, example):
        return self.Cache(
            example.estimate_cache_dir, self.cache_method, example.safe_id
        )

    def has_cache(self, dataset, track) -> bool:
        return self.Cache(
            dataset.estimate_cache_dir(track), self.cache_method, track.safe_id
        ).exists()

    def ensure_available(self) -> None:
        try:
            import tensorflow
            import tensorflow_hub
        except Exception as exc:
            raise self.unavailable(exc) from exc
        self.quiet_tensorflow()

    @staticmethod
    def _to_hz(pitch_output: FloatArray) -> FloatArray:
        offset, slope, fmin, bins_per_octave = (25.58, 63.07, 10.0, 12.0)
        return fmin * 2.0 ** ((pitch_output * slope + offset) / bins_per_octave)

    def _load(self):
        if self._model is None:
            import tensorflow_hub as hub

            self.quiet_tensorflow()
            model = self.HUB_URL
            if (
                model == "https://tfhub.dev/google/spice/2"
                and (self.LOCAL_MODEL / "saved_model.pb").is_file()
            ):
                model = str(self.LOCAL_MODEL)
            self._model = hub.load(model)
        return self._model

    def predict(self, audio, sr, fmin, fmax) -> Melody:
        import tensorflow as tf

        output = self._load().signatures["serving_default"](
            tf.constant(audio, tf.float32)
        )
        pitch = np.asarray(output["pitch"]).reshape(-1)
        confidence = 1.0 - np.asarray(output["uncertainty"]).reshape(-1)
        threshold = self.confidence_threshold
        times = np.arange(pitch.size, dtype=np.float64) * self.HOP / sr
        return (times, self.voiced_freqs(self._to_hz(pitch), confidence >= threshold))
