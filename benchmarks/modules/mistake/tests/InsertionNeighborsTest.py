"""Final-neighbor safeguards for synthetic insertion cases."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
from benchmarks.modules.mistake.datasets.MistakeInjector import MistakeInjector
from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker
from benchmarks.modules.mistake.MistakeDetectorBase import MistakeDetectorBase
from benchmarks.modules.mistake.tests.CompetitorComparisonTest import notes
from benchmarks.modules.note.NoteBenchmarker import OneInstrumentScoreData


class InsertionNeighborTests(unittest.TestCase):

    def assert_distinct(self, performed, metadata):
        seq = [performed.data[t] for t in performed.times]
        for i in metadata["inserted_note_indices"]:
            for j in (i - 1, i + 1):
                if 0 <= j < len(seq):
                    self.assertNotIn(seq[i].midi_num[0], seq[j].midi_num)
        return seq

    def forced(self, pitches, indices, codes, sampled):
        source = notes([(i, i + 1, p) for i, p in enumerate(pitches)])
        injector = MistakeInjector(timing_std_ms=0.0, duration_std=0.0)
        rng = Mock(wraps=np.random.default_rng(7))
        rng.choice.side_effect = codes
        with patch.object(
            injector, "_choose_error_indices", return_value=(1.0, set(indices))
        ), patch.object(injector, "_sample_changed_pitch", side_effect=sampled):
            performed, truth = injector.inject(source, rng)
        self.assert_distinct(performed, injector.last_metadata)
        return (injector, performed, truth)

    def test_insertion_after_host_cannot_copy_next_note(self):
        injector, performed, _ = self.forced([60, 61, 65], [0], [3], [61])
        self.assertEqual(len(injector.last_metadata["insertion_pitch_repairs"]), 1)
        self.assertEqual(
            [
                n.midi_num[0]
                for n in performed.data.values()
                if n.source_score_id is not None
            ],
            [60, 61, 65],
        )

    def test_prefix_insertion_cannot_copy_previous_note(self):
        injector, _, _ = self.forced([60, 61, 65], [1], [2], [60])
        self.assertEqual(len(injector.last_metadata["insertion_pitch_repairs"]), 1)

    def test_deletion_exposes_a_new_neighbor(self):
        injector, _, _ = self.forced([60, 62, 61, 65], [0, 1], [3, 0], [61])
        self.assertEqual(len(injector.last_metadata["insertion_pitch_repairs"]), 1)

    def test_substitution_changes_next_neighbor(self):
        injector, _, _ = self.forced([60, 62, 65], [0, 1], [3, 1], [61, 61])
        self.assertEqual(len(injector.last_metadata["insertion_pitch_repairs"]), 1)

    def test_adjacent_insertions(self):
        self.forced([60, 62, 65], [0, 1], [3, 2], [61, 61])

    def test_midi_extremes_seed_reproducibility_and_roundtrip(self):
        source = notes(
            [
                (i * 0.5, (i + 1) * 0.5, p)
                for i, p in enumerate([0, 1, 2, 127, 126, 125, 60, 61, 60, 62])
            ]
        )
        for code in [2, 3]:
            weights = np.zeros(16)
            weights[code] = 1
            for seed in range(30):
                injector = MistakeInjector(
                    mistake_rate=1.0,
                    protect_boundary_notes=False,
                    screwup_type_weights=weights,
                    timing_std_ms=0.0,
                    duration_std=0.0,
                )
                performed, _ = injector.inject(source, np.random.default_rng(seed))
                self.assert_distinct(performed, injector.last_metadata)
                payload = dict(
                    injector=injector.last_metadata,
                    performance_timeline=MistakeBenchmarker.performance_timeline(
                        performed
                    ),
                )
                again, _ = injector.inject(source, np.random.default_rng(seed))
                self.assertEqual(
                    MistakeBenchmarker.performance_timeline(again),
                    payload["performance_timeline"],
                )
                if seed == 0:
                    with tempfile.TemporaryDirectory() as tmp:
                        midi = Path(tmp) / "performance.mid"
                        MistakeBenchmarker.notedata_to_pm(performed).write(str(midi))
                        loaded = OneInstrumentScoreData(midi).note_data
                        MistakeDetectorBase.timeline_score_onsets(payload, loaded)
                        self.assert_distinct(loaded, payload["injector"])

    def test_serialized_validation_rejects_a_collision(self):
        performed = notes([(0, 1, 60), (1, 2, 60), (2, 3, 62)])
        payload = dict(
            injector=dict(
                timeline_protocol="monophonic_edits_v3",
                insertion_pitch_policy="distinct_from_final_neighbors_v1",
                inserted_note_indices=[1],
            ),
            performance_timeline=MistakeBenchmarker.performance_timeline(performed),
        )
        with self.assertRaisesRegex(ValueError, "matches a performed neighbor"):
            MistakeDetectorBase.timeline_score_onsets(payload, performed)


if __name__ == "__main__":
    unittest.main()
