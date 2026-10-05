from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
from app_logic.NoteData import Note
from app_logic.NoteData import NoteData
from benchmarks.modules.note.NoteDetectorBase import NoteDetectorBase


class BasicPitch(NoteDetectorBase):
    """Basic Pitch transcription and optional backend setup."""

    _basic_pitch_unavailable_error: str | None = None

    def detect(
        self,
        onset_threshold: float = 0.5,
        frame_threshold: float = 0.3,
        minimum_note_length_ms: float | None = None,
        **_unused: Any,
    ) -> NoteData:
        """External Basic Pitch baseline, loaded lazily."""
        if self.__class__._basic_pitch_unavailable_error is not None:
            raise RuntimeError(
                f"Basic Pitch unavailable after a previous failed model load: {self.__class__._basic_pitch_unavailable_error}"
            )
        audio_path = self._audio_path_for_external_model()
        self._configure_basic_pitch_environment()
        if minimum_note_length_ms is not None:
            min_length = float(minimum_note_length_ms)
        else:
            min_seconds = getattr(self.config, "min_note_length", 0.03)
            try:
                score_note_data = self.recording.score_data.clipped_note_data(
                    channel=self.recording.active_instrument
                )
                min_seconds = score_note_data.get_min_note_length(
                    default=float(min_seconds), clean=True
                )
            except (AttributeError, KeyError, TypeError):
                pass
            min_length = 1000.0 * max(
                0.0, float(min_seconds) * self.config.min_note_length_factor
            )
        model_path = None
        try:
            from basic_pitch import ICASSP_2022_MODEL_PATH
            from basic_pitch.inference import AUDIO_SAMPLE_RATE
            from basic_pitch.inference import FFT_HOP
            from basic_pitch.inference import run_inference
            from basic_pitch.note_creation import model_output_to_notes

            model_path = self._basic_pitch_model_path(Path(ICASSP_2022_MODEL_PATH))
            model_output = run_inference(audio_path, model_path, debug_file=None)
            min_note_len = int(
                np.round(min_length / 1000.0 * (AUDIO_SAMPLE_RATE / FFT_HOP))
            )
            _, note_events = model_output_to_notes(
                model_output,
                onset_thresh=onset_threshold,
                frame_thresh=frame_threshold,
                min_note_len=min_note_len,
                min_freq=self.config.fmin,
                max_freq=self.config.fmax,
                include_pitch_bends=False,
            )
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            if model_path is not None:
                detail = f"{detail} (model={model_path})"
            self.__class__._basic_pitch_unavailable_error = detail
            raise RuntimeError(f"Basic Pitch unavailable: {detail}") from exc
        note_data = NoteData()
        origin = float(getattr(self.recording.audio_data, "t_origin", 0.0))
        for idx, event in enumerate(note_events):
            start, end, pitch, amplitude, *_ = event
            if end <= start:
                continue
            note_data.write_note(
                Note(
                    i=idx,
                    start_time=float(start) + origin,
                    end_time=float(end) + origin,
                    midi_num=[float(pitch)],
                    velocity=int(np.clip(round(float(amplitude) * 127), 1, 127)),
                )
            )
        return note_data

    @staticmethod
    def _configure_basic_pitch_environment() -> None:
        NoteDetectorBase._configure_external_model_environment()

    @classmethod
    def _basic_pitch_model_path(cls, default_model_path: Path) -> Path:
        for suffix, modules in (
            (".tflite", ("tensorflow", "tflite_runtime")),
            (".onnx", ("onnxruntime",)),
            (".mlpackage", ("coremltools",)),
        ):
            candidate = default_model_path.with_suffix(suffix)
            if candidate.exists() and any((cls._module_available(m) for m in modules)):
                return candidate
        return default_model_path

    @classmethod
    def predict_task(cls, task, config):
        import numpy as np

        recording, cfg, adapter, conditioning = cls.recording_for_task(task, config)
        timings = {}
        BasicPitch._configure_basic_pitch_environment()
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")
        import scipy.signal

        if not hasattr(scipy.signal, "gaussian"):
            scipy.signal.gaussian = scipy.signal.windows.gaussian
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import AUDIO_SAMPLE_RATE
        from basic_pitch.inference import FFT_HOP
        from basic_pitch.inference import run_inference
        from basic_pitch.note_creation import model_output_to_notes

        started = cls.cpu_seconds()
        output = run_inference(task["audio"], ICASSP_2022_MODEL_PATH, debug_file=None)
        timings["frontend_cpu_seconds"] = cls.cpu_seconds() - started
        started = cls.cpu_seconds()
        _, events = model_output_to_notes(
            output,
            onset_thresh=0.5,
            frame_thresh=0.3,
            min_note_len=int(np.round(0.1277 * AUDIO_SAMPLE_RATE / FFT_HOP)),
            min_freq=None,
            max_freq=None,
            multiple_pitch_bends=False,
            melodia_trick=True,
            midi_tempo=120,
        )
        timings["segmentation_cpu_seconds"] = cls.cpu_seconds() - started
        iv, hz = cls.event_arrays([(e[0], e[1], e[2]) for e in events])
        return (iv, hz, timings)
