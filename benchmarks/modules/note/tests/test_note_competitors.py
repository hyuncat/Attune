"""Exercise extracted prediction paths without loading neural model weights."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from algorithms.Config import Config
from app_logic.NoteData import Note, NoteData
from benchmarks.modules.note.NoteCache import NoteCache
from benchmarks.modules.note.NoteNotebook import NotebookConfig
from benchmarks.modules.note.competitors.Attune import Attune
from benchmarks.modules.note.competitors.BasicPitch import BasicPitch
from benchmarks.modules.note.competitors.CrepeNotes import CrepeNotes
from benchmarks.modules.note.competitors.Tony import Tony


class NoteCompetitorsTest(unittest.TestCase):
    @staticmethod
    def notes(pitch=60):
        notes = NoteData()
        notes.write_note(Note(i=0, start_time=0.0, end_time=1.0, midi_num=[pitch]))
        return notes

    def test_attune_uses_refined_notes_and_audio_only_skips_refinement(self):
        for conditioning in (True, False):
            raw, refined = self.notes(60), self.notes(62)
            recording = SimpleNamespace(
                pitch_data=SimpleNamespace(data=[]),
                transition_detector=SimpleNamespace(clear_transitions=Mock()),
                resize_score=Mock(),
                update_min_note_length=Mock(),
                note_detector=SimpleNamespace(detect_notes=Mock(return_value=raw)),
            )
            recording.align_score_and_refine = Mock(
                side_effect=lambda: setattr(recording, "note_data", refined)
            )
            with patch.object(
                Attune,
                "recording_for_task",
                return_value=(recording, Config(), None, conditioning),
            ), patch.object(
                NoteCache, "attune_frontend", return_value={"frontend_cpu_seconds": 2.0}
            ):
                intervals, pitches, timings = Attune.predict_task({}, NotebookConfig())
            self.assertEqual(intervals.tolist(), [[0.0, 1.0]])
            self.assertAlmostEqual(
                pitches[0], 440 * 2 ** (((62 if conditioning else 60) - 69) / 12)
            )
            self.assertEqual(timings["frontend_cpu_seconds"], 2.0)
            self.assertEqual(
                recording.align_score_and_refine.call_count, int(conditioning)
            )
            self.assertEqual("refinement_cpu_seconds" in timings, conditioning)

    def test_basic_pitch_preserves_polyphonic_events(self):
        inference = Mock(return_value="model output")
        decode = Mock(return_value=(None, [(0.0, 1.0, 60, 0.8), (0.0, 1.0, 64, 0.8)]))
        modules = {
            "tensorflow": SimpleNamespace(
                config=SimpleNamespace(set_visible_devices=Mock())
            ),
            "basic_pitch": SimpleNamespace(ICASSP_2022_MODEL_PATH="model"),
            "basic_pitch.inference": SimpleNamespace(
                AUDIO_SAMPLE_RATE=22050, FFT_HOP=256, run_inference=inference
            ),
            "basic_pitch.note_creation": SimpleNamespace(model_output_to_notes=decode),
        }
        with patch.dict(sys.modules, modules), patch.object(
            BasicPitch, "recording_for_task", return_value=(None, Config(), None, False)
        ), patch.object(BasicPitch, "_configure_basic_pitch_environment"):
            intervals, pitches, timing = BasicPitch.predict_task(
                {"audio": "input.wav"}, NotebookConfig()
            )
        self.assertEqual(intervals.tolist(), [[0.0, 1.0], [0.0, 1.0]])
        self.assertEqual(len(pitches), 2)
        inference.assert_called_once_with("input.wav", "model", debug_file=None)
        self.assertIsNone(decode.call_args.kwargs["min_freq"])
        self.assertIn("frontend_cpu_seconds", timing)
        self.assertIn("segmentation_cpu_seconds", timing)

    def test_crepe_uses_shared_frontend_and_native_segmentation(self):
        process = Mock(return_value="output.mid")
        modules = {
            "tensorflow": SimpleNamespace(
                config=SimpleNamespace(set_visible_devices=Mock())
            ),
            "crepe_notes.crepe_notes": SimpleNamespace(process=process),
        }
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "input.wav"
            audio.write_bytes(b"placeholder")
            with patch.dict(sys.modules, modules), patch.object(
                CrepeNotes,
                "recording_for_task",
                return_value=(None, Config(), None, False),
            ), patch.object(
                CrepeNotes, "_configure_crepe_notes_environment"
            ), patch.object(
                CrepeNotes, "_notedata_from_midi", return_value=self.notes()
            ), patch.object(
                NoteCache,
                "crepe_frontend",
                return_value=([440.0], [0.9], {"frontend_cpu_seconds": 2.0}),
            ) as frontend:
                intervals, pitches, timing = CrepeNotes.predict_task(
                    {"audio": str(audio)}, NotebookConfig()
                )
            frontend.assert_called_once()
        self.assertEqual(intervals.tolist(), [[0.0, 1.0]])
        self.assertEqual(process.call_args.args[:2], ([440.0], [0.9]))
        self.assertEqual(timing["frontend_cpu_seconds"], 2.0)

    def test_tony_uses_its_own_note_backend(self):
        recording = SimpleNamespace(config=Config(), note_detector=None)
        with patch.object(
            Tony,
            "recording_for_task",
            return_value=(recording, recording.config, None, False),
        ), patch.object(
            Tony, "_audio_path_for_external_model", return_value="input.wav"
        ), patch.object(
            Tony, "detect_notes", return_value=self.notes()
        ) as backend:
            intervals, pitches, timing = Tony.predict_task({}, NotebookConfig())
        backend.assert_called_once_with("input.wav")
        self.assertEqual(intervals.tolist(), [[0.0, 1.0]])
        self.assertEqual(timing, {})
