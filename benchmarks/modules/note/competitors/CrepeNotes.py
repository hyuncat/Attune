from __future__ import annotations
import os
import tempfile
from pathlib import Path
from typing import Any
from app_logic.NoteData import NoteData
from benchmarks.modules.note.NoteDetectorBase import NoteDetectorBase


class CrepeNotes(NoteDetectorBase):
    """CREPE Notes transcription and decoder setup."""

    _crepe_notes_unavailable_error: str | None = None

    def detect(
        self,
        sensitivity: float = 0.001,
        min_duration: float = 0.03,
        min_velocity: int = 6,
        disable_splitting: bool = False,
        tuning_offset: float | bool = False,
        use_smoothing: bool = False,
        pitch_tracker: str = "crepe",
        detect_amplitude: bool = True,
        save_analysis_files: bool = False,
        **_unused: Any,
    ) -> NoteData:
        """External `crepe_notes` baseline.

        This intentionally calls the installed package instead of reimplementing
        the CREPE Notes paper over Attune PitchData. The package runs the chosen
        pitch tracker, applies its confidence-gradient postprocessor, writes MIDI,
        and we translate that MIDI back into Attune's NoteData.
        """
        if self.__class__._crepe_notes_unavailable_error is not None:
            raise RuntimeError(
                f"CREPE Notes unavailable after a previous failed import: {self.__class__._crepe_notes_unavailable_error}"
            )
        self._configure_crepe_notes_environment()
        try:
            from crepe_notes.crepe_notes import process
            from crepe_notes.crepe_notes import run_pitch_tracker
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            self.__class__._crepe_notes_unavailable_error = detail
            raise RuntimeError(f"CREPE Notes unavailable: {detail}") from exc
        origin = float(getattr(self.recording.audio_data, "t_origin", 0.0))
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "attune-crepe-notes.wav"
            self._write_external_audio(audio_path)
            try:
                frequency, confidence = run_pitch_tracker(
                    audio_path, tracker=pitch_tracker
                )
                midi_path = process(
                    frequency,
                    confidence,
                    audio_path,
                    output_label=f"{pitch_tracker}.transcription",
                    sensitivity=float(sensitivity),
                    use_smoothing=bool(use_smoothing),
                    min_duration=float(min_duration),
                    min_velocity=int(min_velocity),
                    disable_splitting=bool(disable_splitting),
                    tuning_offset=tuning_offset,
                    use_cwd=False,
                    detect_amplitude=bool(detect_amplitude),
                    save_analysis_files=bool(save_analysis_files),
                    pitch_tracker=pitch_tracker,
                )
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}"
                raise RuntimeError(f"CREPE Notes failed: {detail}") from exc
            return self._notedata_from_midi(Path(midi_path), origin=origin)

    @staticmethod
    def _configure_crepe_notes_environment() -> None:
        """Find decoder tools even when a GUI notebook omits Homebrew's PATH."""
        import shutil
        import sys

        NoteDetectorBase._configure_external_model_environment()
        for executable in ("ffmpeg", "ffprobe"):
            if shutil.which(executable):
                continue
            for directory in (
                Path(sys.prefix) / "bin",
                Path("/opt/homebrew/bin"),
                Path("/usr/local/bin"),
            ):
                candidate = directory / executable
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    os.environ["PATH"] = (
                        str(directory) + os.pathsep + os.environ.get("PATH", "")
                    )
                    break
            else:
                raise FileNotFoundError(
                    f"CREPE Notes requires {executable} for native onset detection; install FFmpeg or add its bin directory to the notebook PATH."
                )

    @classmethod
    def predict_task(cls, task, config):
        import tempfile
        from benchmarks.modules.note.NoteBenchmarker import NoteBenchmarker

        recording, cfg, adapter, conditioning = cls.recording_for_task(task, config)
        timings = {}
        CrepeNotes._configure_crepe_notes_environment()
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")
        from crepe_notes.crepe_notes import process
        import shutil

        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "input.wav"
            shutil.copyfile(task["audio"], audio)
            from benchmarks.modules.note.NoteCache import NoteCache

            frequency, confidence, timings = NoteCache.crepe_frontend(task, config)
            started = cls.cpu_seconds()
            midi_path = process(
                frequency,
                confidence,
                audio,
                use_cwd=False,
                save_analysis_files=False,
                pitch_tracker="crepe",
            )
            timings["segmentation_cpu_seconds"] = cls.cpu_seconds() - started
            notes = CrepeNotes._notedata_from_midi(Path(midi_path))
            iv, hz = NoteBenchmarker.notedata_to_intervals(notes, cfg)
        return (iv, hz, timings)
