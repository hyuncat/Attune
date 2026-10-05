"""Monophonic projection of Basic Pitch's sub-semitone contour output."""

from __future__ import annotations
import os
from pathlib import Path
import numpy as np
from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase


class BasicPitch(PitchDetectorBase):
    """Strongest in-range contour bin, gated by Basic Pitch's note confidence.

    Basic Pitch is natively polyphonic.  This monophonic adapter selects one of
    its 264 contour bins (three bins per semitone) independently at every frame.
    The 88-bin note head is retained only as the model's standard 0.3 voicing
    gate; it never supplies the estimated frequency and no note-event decoding,
    duration filtering, or within-note flattening is applied.
    """

    name = "basic_pitch"
    description = "Basic Pitch contour head, projected to one sub-semitone F0 per frame"
    install_hint = "pip install basic-pitch==0.3.0 tensorflow"
    input_sr = 22050
    model_fmin = 27.5
    model_fmax = 4186.009044809578

    @staticmethod
    def _configure_environment() -> None:
        import logging
        import tempfile

        cache_dir = Path(tempfile.gettempdir()) / "attune-matplotlib"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        logging.getLogger("tensorflow").setLevel(logging.ERROR)

    @staticmethod
    def _model_path(default_path: Path) -> Path:
        """Pick the same serialized model the package's own runtime would."""
        import importlib.util

        for suffix, modules in (
            (".tflite", ("tensorflow", "tflite_runtime")),
            (".onnx", ("onnxruntime",)),
            (".mlpackage", ("coremltools",)),
        ):
            candidate = default_path.with_suffix(suffix)
            available = any((importlib.util.find_spec(m) is not None for m in modules))
            if candidate.exists() and available:
                return candidate
        return default_path

    def ensure_available(self) -> None:
        self._configure_environment()
        try:
            from basic_pitch import ICASSP_2022_MODEL_PATH
            from basic_pitch.inference import Model

            model_path = self._model_path(Path(ICASSP_2022_MODEL_PATH))
            if not model_path.exists():
                raise FileNotFoundError(model_path)
        except Exception as exc:
            raise self.unavailable(exc) from exc
        self.quiet_tensorflow()

    def _load(self):
        if self._model is None:
            from basic_pitch import ICASSP_2022_MODEL_PATH
            from basic_pitch.inference import Model

            self._model = Model(self._model_path(Path(ICASSP_2022_MODEL_PATH)))
        return self._model

    def predict(self, audio, sr, fmin, fmax) -> Melody:
        from basic_pitch.constants import (
            AUDIO_N_SAMPLES,
            AUDIO_SAMPLE_RATE,
            FFT_HOP,
            FREQ_BINS_CONTOURS,
            FREQ_BINS_NOTES,
        )
        from basic_pitch.inference import unwrap_output, window_audio_file
        from basic_pitch.note_creation import model_frames_to_time

        if sr != AUDIO_SAMPLE_RATE:
            raise ValueError(f"Basic Pitch expects {AUDIO_SAMPLE_RATE} Hz, got {sr}")
        overlapping_frames = 30
        overlap = overlapping_frames * FFT_HOP
        padded = np.concatenate(
            [
                np.zeros(overlap // 2, dtype=np.float32),
                np.asarray(audio, dtype=np.float32),
            ]
        )
        model = self._load()
        windows = {"note": [], "contour": []}
        for window, _ in window_audio_file(padded, AUDIO_N_SAMPLES - overlap):
            output = model.predict(np.expand_dims(window, axis=0))
            for head in windows:
                windows[head].append(np.asarray(output[head]))
        notes = np.asarray(
            unwrap_output(
                np.concatenate(windows["note"]), int(audio.size), overlapping_frames
            ),
            dtype=np.float64,
        )
        contours = np.asarray(
            unwrap_output(
                np.concatenate(windows["contour"]), int(audio.size), overlapping_frames
            ),
            dtype=np.float64,
        )
        if (
            notes.ndim != 2
            or contours.ndim != 2
            or notes.shape[0] == 0
            or (notes.shape[0] != contours.shape[0])
        ):
            return (np.array([0.0]), np.array([0.0]))
        note_freqs = np.asarray(FREQ_BINS_NOTES, dtype=np.float64)
        note_allowed = (note_freqs >= float(fmin)) & (note_freqs <= float(fmax))
        if not np.any(note_allowed):
            note_allowed = np.ones(note_freqs.shape, dtype=bool)
        note_activation = np.max(notes[:, note_allowed], axis=1)
        contour_freqs = np.asarray(FREQ_BINS_CONTOURS, dtype=np.float64)
        contour_allowed = (contour_freqs >= float(fmin)) & (
            contour_freqs <= float(fmax)
        )
        if not np.any(contour_allowed):
            contour_allowed = np.ones(contour_freqs.shape, dtype=bool)
        in_range_contours = contours[:, contour_allowed]
        contour_winner = np.argmax(in_range_contours, axis=1)
        frequencies = contour_freqs[np.flatnonzero(contour_allowed)[contour_winner]]
        threshold = 0.3 if self.confidence is None else float(self.confidence)
        freqs = self.voiced_freqs(frequencies, note_activation >= threshold)
        return (
            np.asarray(model_frames_to_time(contours.shape[0]), dtype=np.float64),
            freqs,
        )
