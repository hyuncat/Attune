"""Native audio timebase, performed range and safe cross-run neural reuse."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf
from algorithms.Config import Config
from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker
from benchmarks.modules.mistake.MistakeCache import MistakeCache
from benchmarks.modules.mistake.datasets.NativeDatasets import digest, save_json


class FrontendTests(unittest.TestCase):

    def test_resampling_preserves_duration_pitch_and_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "tone.wav"
            t = np.arange(16000) / 16000
            sf.write(wav, 0.3 * np.sin(2 * np.pi * 440 * t), 16000)
            original = digest(wav)
            audio, source_sr = MistakeBenchmarker.native_attune_audio(
                wav, Config(sr=44100)
            )
            self.assertEqual(source_sr, 16000)
            self.assertEqual(
                (audio.sr, audio.capacity, audio.end_index), (44100, 44100, 44100)
            )
            self.assertEqual(audio.get_length(), 1.0)
            self.assertEqual(np.argmax(abs(np.fft.rfft(audio.read_all()))), 440)
            self.assertEqual(digest(wav), original)
            sf.write(wav, audio.data, 44100)
            with patch("scipy.signal.resample_poly") as resample:
                MistakeBenchmarker.native_attune_audio(wav, Config(sr=44100))
                resample.assert_not_called()

    def test_range_includes_insertions_excludes_omissions(self):
        labels = {
            "correct": [dict(pitch=60), dict(pitch=72)],
            "extra": [dict(pitch=45), dict(pitch=84)],
            "missed": [dict(pitch=10), dict(pitch=120)],
        }
        with patch(
            "benchmarks.modules.mistake.MistakeBenchmarker.MistakeBenchmarker.native_midi_events",
            side_effect=lambda path, kind: labels[kind],
        ):
            self.assertEqual(
                MistakeBenchmarker.native_performed_range({k: k for k in labels}),
                (41, 88),
            )
        with patch(
            "benchmarks.modules.mistake.MistakeBenchmarker.MistakeBenchmarker.native_midi_events",
            return_value=[],
        ):
            with self.assertRaisesRegex(ValueError, "empty labels"):
                MistakeBenchmarker.native_performed_range(dict(correct="a", extra="b"))

    def test_native_substitution_uses_detected_time_deletion_keeps_score_time(self):
        from types import SimpleNamespace
        from app_logic.NoteData import Note, NoteData
        from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker

        reference = NoteData()
        wrong = Note(1, 2.273, 2.718, [64])
        omitted = Note(2, 4.0, 4.5, [60])
        reference.write_note(wrong)
        reference.write_note(omitted)
        played = Note(3, 2.019, 2.525, [65.1])
        mistakes = [
            SimpleNamespace(type="substitution", midi_note=wrong, user_note=played),
            SimpleNamespace(type="deletion", midi_note=omitted, user_note=None),
        ]
        events = MistakeBenchmarker.native_native_attune_events(
            mistakes, reference, [(played, wrong), (None, omitted)]
        )
        self.assertEqual(
            [(e["kind"], e["onset"], e["pitch"]) for e in events],
            [("missed", 2.019, 64.0), ("extra", 2.019, 65.1), ("missed", 4.0, 60.0)],
        )
        self.assertEqual(events[0]["end"], 2.525)
        self.assertEqual(wrong.start_time, 2.273)

    def test_native_half_semitone_and_short_segment_settings(self):
        from app_logic.NoteData import Note
        from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker

        config = Config(pitch_tolerance=0.5, min_note_length_cap=0.1)
        played = Note(1, 1.0, 1.4, [49.1])
        target = Note(2, 1.0, 1.4, [50])
        self.assertEqual(
            MistakeBenchmarker.label_pairs([(played, target)], config)[0].type,
            "substitution",
        )
        for score_min in (0.4, 0.15, 0.06):
            config.set_min_note_length(score_min)
            self.assertLessEqual(config.note_detection_min_seconds(), 0.1)
        played.midi_num = [49.6]
        self.assertEqual(MistakeBenchmarker.label_pairs([(played, target)], config), [])

    def test_reuse_validates_inputs_models_code_and_never_attune(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = dict(case_id="p/s", hashes={"performance": "abc"})
            contract = dict(
                device="cpu",
                packages={"numpy": "1"},
                ladder_contiguous_inference=True,
                models={"PolyTune": {"checkpoint": "abc"}},
                code={"worker.py": "abc", "NativeComparison.inference": "old"},
                manifest=dict(dataset="coco", cases=[case]),
            )
            save_json(root / "run.json", dict(contract=contract))
            save_json(
                root / "cases/p/s/polytune/result.json", dict(prediction={"events": []})
            )
            current = copy.deepcopy(contract)
            current["code"]["NativeComparison.inference"] = "new frontend"
            result = MistakeCache.reusable_audio_prediction(
                root, current, case, "PolyTune"
            )
            self.assertEqual(result["events"], [])
            self.assertIn("sha256", result["reused_from"])
            self.assertIsNone(
                MistakeCache.reusable_audio_prediction(root, current, case, "Attune")
            )
            for key, value in [
                ("device", "cuda"),
                ("code", {"worker.py": "changed"}),
                ("models", {"PolyTune": {"checkpoint": "different"}}),
            ]:
                changed = copy.deepcopy(current)
                changed[key] = value
                with self.assertRaisesRegex(ValueError, "Cannot reuse"):
                    MistakeCache.reusable_audio_prediction(
                        root, changed, case, "PolyTune"
                    )
            changed_case = dict(case_id="p/s", hashes={"performance": "different"})
            with self.assertRaisesRegex(ValueError, "assets changed"):
                MistakeCache.reusable_audio_prediction(
                    root, current, changed_case, "PolyTune"
                )


if __name__ == "__main__":
    unittest.main()
