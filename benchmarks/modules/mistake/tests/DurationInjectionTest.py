"""Duration bounds and salience must survive the final timeline and MIDI export."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pretty_midi
from algorithms.Config import Config
from app_logic.NoteData import Note, NoteData
from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker
from benchmarks.modules.mistake.datasets.MistakeInjector import MistakeInjector
from benchmarks.modules.mistake.MistakeDetectorBase import MistakeDetectorBase


def score(durations):
    result = NoteData()
    onset = 0.103
    for i, duration in enumerate(durations):
        result.write_note(
            Note(i=i, start_time=onset, end_time=onset + duration, midi_num=[60 + i])
        )
        onset += duration
    return result


class DurationInjectionTest(unittest.TestCase):

    def inject(self, reference, code, seed=0, **kwargs):
        weights = [0.0] * 16
        weights[code] = 1.0
        injector = MistakeInjector(screwup_type_weights=weights, **kwargs)
        with patch.object(injector, "_choose_error_indices", return_value=(1.0, {1})):
            performed, history = injector.inject(reference, np.random.default_rng(seed))
        return (injector, performed, history)

    def test_bounds_and_salience_survive_export(self):
        for duration in [0.6, 0.601, 0.7, 1.0, 1.337, 4.0]:
            reference = score([1.0, duration, 1.0])
            for code in [4, 5]:
                for seed in range(15):
                    with self.subTest(duration=duration, code=code, seed=seed):
                        _, performed, history = self.inject(reference, code, seed)
                        if not history:
                            self.assertLess(duration, 0.61)
                            continue
                        seq = list(performed.data.values())
                        self.assertTrue(
                            all(
                                (
                                    a.end_time <= b.start_time
                                    for a, b in zip(seq, seq[1:])
                                )
                            )
                        )
                        with tempfile.TemporaryDirectory() as tmp:
                            path = Path(tmp) / "take.mid"
                            MistakeBenchmarker.notedata_to_pm(performed).write(
                                str(path)
                            )
                            actual = (
                                pretty_midi.PrettyMIDI(str(path))
                                .instruments[0]
                                .notes[1]
                            )
                        length = actual.end - actual.start
                        self.assertGreaterEqual(length, 0.5 * duration - 1e-09)
                        self.assertLessEqual(length, 1.5 * duration + 1e-09)
                        self.assertGreaterEqual(abs(length - duration), 0.3 - 1e-09)
                        self.assertAlmostEqual(history[0]["performed_duration"], length)
                        self.assertEqual(
                            history[0]["type"], "short" if code == 4 else "long"
                        )
                        truth, _ = MistakeDetectorBase.net_mistakes(
                            reference,
                            performed,
                            score_onsets={n.id: n.comparison_time for n in seq},
                        )
                        self.assertEqual(
                            [e["type"] for e in truth], [history[0]["type"]]
                        )
                        labels = MistakeBenchmarker.label_pairs(
                            [(seq[1], list(reference.data.values())[1])], Config()
                        )
                        self.assertEqual([e.type for e in labels], [history[0]["type"]])

    def test_ineligible_edits_are_skipped_without_switching_type(self):
        for code in [4, 5]:
            injector, performed, history = self.inject(score([1.0, 0.4, 1.0]), code)
            self.assertEqual(history, [])
            self.assertAlmostEqual(list(performed.data.values())[1].duration(), 0.4)
            self.assertIn("skipped_reason", injector.last_metadata["selected"][0])

    def test_long_error_keeps_duration_even_with_overlap_sampler_disabled(self):
        _, performed, history = self.inject(
            score([1.0, 1.0, 1.0]), 5, allow_overlap=False
        )
        seq = list(performed.data.values())
        self.assertGreaterEqual(seq[1].duration(), 1.3 - 1e-09)
        self.assertAlmostEqual(seq[2].start_time, seq[1].end_time)
        self.assertEqual(history[0]["type"], "long")

    def test_cumulative_drift_is_preserved(self):
        reference = score([1.0] * 8)
        weights = [0.0] * 16
        weights[5] = 1.0
        injector = MistakeInjector(screwup_type_weights=weights)
        with patch.object(
            injector, "_choose_error_indices", return_value=(1.0, {1, 2, 3, 4, 5})
        ):
            performed, history = injector.inject(reference, np.random.default_rng(8))
        self.assertEqual(len(history), 5)
        self.assertGreater(list(performed.data.values())[-1].timeline_delay, 1.49)

    def test_explicit_legacy_sampler_is_identified(self):
        injector, _, history = self.inject(
            score([1.0, 0.4, 1.0]), 5, duration_error_range_sec=(0.3, 0.6)
        )
        self.assertTrue(history)
        self.assertEqual(
            injector.last_metadata["duration_error_policy"], "legacy_absolute"
        )

    def test_invalid_bounds(self):
        for factors in [(0, 1.5), (1.0, 1.5), (0.5, 1.0), (0.5, float("nan"))]:
            with self.assertRaises(ValueError):
                MistakeInjector(duration_factor_range=factors)
        for floor in [0, -1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                MistakeInjector(duration_error_min_sec=floor)


if __name__ == "__main__":
    unittest.main()
