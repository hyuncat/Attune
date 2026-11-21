"""Packaged CREPE tiny weights with the web demo's streaming input policy.

Reproduces https://marl.github.io/crepe/ callback geometry and decoding, but
uses the supported pip tiny weights rather than the older gh-pages export.
This is Python/TensorFlow CPU timing, not browser timing.
"""

import numpy as np

from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase


class CrepeStreaming(PitchDetectorBase):
    name = "crepe_tiny"
    description = "CREPE tiny, web-demo callback policy, TensorFlow CPU replay"
    install_hint = "pip install crepe tensorflow"
    VERSION = "tiny_web_callback_v1"
    MODEL_SR = 16000
    FRAME_SIZE = 1024
    CONFIDENCE = 0.5

    def __init__(self, *, rolling_10ms=False):
        super().__init__()
        self.rolling_10ms = rolling_10ms
        if rolling_10ms:
            self.VERSION = "tiny_rolling_64ms_10ms_v1"

    def buffer_size(self, sr):
        cls = type(self)
        if self.rolling_10ms:
            return int(np.ceil(0.064 * sr))
        size = 4
        while size < sr / cls.MODEL_SR * cls.FRAME_SIZE:
            size *= 2
        return size

    def ensure_available(self):
        if hasattr(self, "_infer"):
            return
        try:
            self.quiet_tensorflow()
            import crepe
            import tensorflow as tf

            with tf.device("/CPU:0"):
                model = crepe.core.build_and_load_model("tiny")

                @tf.function(input_signature=[tf.TensorSpec((1, 1024), tf.float32)])
                def infer(frame):
                    with tf.device("/CPU:0"):
                        centered = frame - tf.reduce_mean(frame)
                        scale = tf.norm(centered) / tf.sqrt(tf.constant(1024.0))
                        # The browser produces NaNs for digital silence; its
                        # confidence > 0.5 check then displays "no voice".
                        return model(tf.math.divide_no_nan(centered, scale), training=False)

                self._infer = infer
        except Exception as exc:
            raise self.unavailable(exc) from exc

    @staticmethod
    def decode(activation):
        activation = np.asarray(activation).reshape(360)
        if not np.isfinite(activation).all() or activation.max() <= 0.5:
            return 0.0
        center = int(activation.argmax())
        start, end = max(0, center - 4), min(360, center + 5)
        weights = activation[start:end]
        cents = np.arange(start, end) * 20.0 + 1997.3794084376191
        return float(10 * 2 ** (np.sum(weights * cents) / weights.sum() / 1200))

    def predict(self, audio, sr, fmin, fmax):
        if len(audio) != self.buffer_size(sr):
            raise ValueError("CREPE demo requires exactly one native callback buffer")
        # Match the demo's linear interpolation (integer decimation at 48 kHz),
        # including its unused tail of each nonoverlapping callback buffer.
        positions = np.arange(self.FRAME_SIZE) * (sr / self.MODEL_SR)
        frame = np.interp(positions, np.arange(len(audio)), audio).astype(np.float32)
        activation = self._infer(frame[None]).numpy()[0]
        frequency = 0.0 if np.std(frame) == 0 else self.decode(activation)
        return np.array([len(audio) / (2 * sr) if self.rolling_10ms else 0.032]), np.array([frequency])
