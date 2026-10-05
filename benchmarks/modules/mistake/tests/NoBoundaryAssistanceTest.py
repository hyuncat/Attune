"""Truth durations cannot change detections or revive previously corrected caches."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from app_logic.NoteData import Note, NoteData
from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker
from benchmarks.modules.mistake.MistakeNotebook import MistakeNotebook


def notes(start, end):
    data = NoteData()
    data.write_note(Note(1, start, end, [60]))
    return data


class BoundaryIsolationTests(unittest.TestCase):

    def test_truth_duration_never_moves_detected_boundaries(self):
        for cached_flag in (True, None, False):
            with self.subTest(
                cached_flag=cached_flag
            ), tempfile.TemporaryDirectory() as tmp:
                cache = Path(tmp) / "notes.json"
                cache.write_text("{}")
                bench = MistakeBenchmarker()
                rec = MagicMock()
                rec.note_data = notes(0.12, 1.8)

                def detect(recording):
                    recording.note_data = notes(0.12, 1.8)
                    return (None, 0.01)

                with patch.object(
                    bench,
                    "load_note_data",
                    return_value=(
                        notes(0.12, 1.8),
                        {"trimmed_boundaries": cached_flag},
                    ),
                ), patch.object(
                    bench, "detect_recording_notes_timed", side_effect=detect
                ) as detector, patch.object(
                    bench, "save_note_data"
                ) as save, patch.object(
                    bench,
                    "_trim_boundary_notes",
                    side_effect=AssertionError("Truth touched extraction"),
                ):
                    bench.analyze_recording(
                        rec, note_cache_path=cache, trim_reference=notes(0.0, 0.3)
                    )
                    self.assertEqual(
                        (
                            rec.note_data.data[0.12].start_time,
                            rec.note_data.data[0.12].end_time,
                        ),
                        (0.12, 1.8),
                    )
                    self.assertEqual(
                        detector.call_count, 0 if cached_flag is False else 1
                    )
                    if cached_flag is not False:
                        self.assertIs(
                            save.call_args.kwargs["metadata"]["trimmed_boundaries"],
                            False,
                        )
                    rec.resize_score.assert_called_once()

    def test_historical_completed_run_cannot_silently_reappear(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "run.json").write_text(
                json.dumps({"status": "complete", "pipeline": "synth boundary trim"})
            )
            with self.assertRaisesRegex(ValueError, "truth-assisted boundaries"):
                MistakeNotebook.completed_results(dict(stage="audio", output=tmp))


if __name__ == "__main__":
    unittest.main()
