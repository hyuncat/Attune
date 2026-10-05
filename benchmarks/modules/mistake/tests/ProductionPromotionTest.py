"""Saved takes must not undo the promoted segmentation/alignment settings."""

from contextlib import redirect_stdout
from io import StringIO
import unittest

from algorithms.Config import Config
from algorithms.RepeatSplitter import RepeatSplitter
from app_logic.JsonHandler import JsonHandler
from app_logic.user.ds.Recording import Recording
from app_logic.user.ds.PitchData import Pitch
from app_logic.NoteData import Note


class ProductionPromotionTests(unittest.TestCase):
    def test_previous_confidence_gate_invalidates_saved_pitch_and_notes(self):
        old = Recording(
            config=Config(
                posthoc_unv_thresh=0.985, pitch_thresh=0.75, min_note_length_cap=0.1
            )
        )
        old.pitch_data.load(
            [
                Pitch(
                    time=0.0,
                    volume=0.2,
                    unvoiced_prob=0.0,
                    live_distance=None,
                    config=old.config,
                    candidates=[(69.0, 0.02)],
                    value=69.0,
                )
            ]
        )
        old.pitches_smoothed = True
        old.note_data.write_note(Note(0, 0.0, 0.4, [69.0]))
        handler = JsonHandler()
        payload = handler.to_cache_payload(recording=old)
        self.assertTrue(payload["pitch_data"]["pitches"])
        current = Recording()
        with redirect_stdout(StringIO()):
            handler.load_cache_payload(current, payload)
        self.assertEqual(current.config.posthoc_unv_thresh, 0.99)
        self.assertEqual(current.config.pitch_thresh, 0.60)
        self.assertEqual(current.config.min_note_length_cap, 0.20)
        self.assertEqual(current.pitch_data.frames_available(), 0)
        self.assertFalse(current.pitches_smoothed)
        self.assertEqual(current.note_data.times, [])
        self.assertIn("posthoc_unv_thresh", current.analysis_notice)

    def test_old_take_cannot_restore_old_analysis_defaults(self):
        old = Recording(
            config=Config(
                alignment_gamma_pitch=2.0,
                min_note_length_cap=0.0,
                min_note_length_factor=0.6,
                min_silence_duration_ms=10.0,
            )
        )
        handler = JsonHandler()
        payload = handler.to_cache_payload(recording=old)
        current = Recording()
        with redirect_stdout(StringIO()):
            handler.load_cache_payload(current, payload)
        defaults = Config()
        for name in (*Config.NOTE_SEGMENTATION_FIELDS, *Config.ALIGNMENT_FIELDS):
            self.assertEqual(getattr(current.config, name), getattr(defaults, name))
        self.assertIsInstance(current.repeat_splitter, RepeatSplitter)
        self.assertIs(current.repeat_splitter.config, current.config)
        self.assertEqual(current.note_data.times, [])
        self.assertIn("click Analyze", current.analysis_notice)
        self.assertIn("alignment_gamma_pitch 2 -> 4", current.analysis_notice)


if __name__ == "__main__":
    unittest.main()
