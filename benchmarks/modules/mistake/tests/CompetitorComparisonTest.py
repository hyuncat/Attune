"""Net-error accounting and representation-invariant scoring contracts."""

import unittest
from app_logic.NoteData import Note, NoteData
from app_logic.Alignment import Mistake
from benchmarks.modules.mistake.MistakeDetectorBase import MistakeDetectorBase
from benchmarks.modules.mistake.MistakeBenchmarker import (
    COMPETITOR_AUDIO_METHODS,
    COMPETITOR_METHODS,
    MistakeBenchmarker,
)


def notes(spec):
    data = NoteData()
    for i, (start, end, pitch) in enumerate(spec):
        data.write_note(Note(i, start, end, [pitch]))
    return data


class NetMistakesTest(unittest.TestCase):

    def test_deleted_then_wrong_replacement_is_one_substitution(self):
        score = notes([(0, 0.4, 60), (1, 1.4, 62)])
        final = notes([(0, 0.4, 60), (1, 1.4, 65)])
        truth, _ = MistakeDetectorBase.net_mistakes(score, final)
        self.assertEqual([e["type"] for e in truth], ["substitution"])
        s, u = (score.data[1], final.data[1])
        for predictions in (
            [Mistake("substitution", u, s)],
            [Mistake("deletion", None, s), Mistake("insertion", u, None)],
        ):
            counts = MistakeBenchmarker.score_symbolic(
                predictions, truth, canonical=True
            )
            self.assertEqual(counts["deletion"], (1, 0, 0))
            self.assertEqual(counts["insertion"], (1, 0, 0))

    def test_correct_replacement_cancels_pitch_error(self):
        score = notes([(0, 0.4, 60)])
        self.assertEqual(
            MistakeDetectorBase.net_mistakes(score, notes([(0, 0.4, 60)]))[0], []
        )

    def test_duration_recount_uses_final_span(self):
        score = notes([(0, 1, 60)])
        self.assertEqual(
            MistakeDetectorBase.net_mistakes(score, notes([(0, 0.9, 60)]))[0], []
        )
        self.assertEqual(
            MistakeDetectorBase.net_mistakes(score, notes([(0, 0.5, 60)]))[0][0][
                "type"
            ],
            "short",
        )

    def test_extra_near_correct_note_remains_extra(self):
        score = notes([(0, 0.4, 60)])
        final = notes([(0, 0.4, 65), (0.02, 0.42, 60)])
        truth, pairs = MistakeDetectorBase.net_mistakes(score, final)
        self.assertEqual([e["type"] for e in truth], ["insertion"])
        self.assertEqual(pairs[0]["performed_note_id"], 1)

    def test_replacement_outside_gate_is_missed_and_extra(self):
        truth, _ = MistakeDetectorBase.net_mistakes(
            notes([(0, 0.4, 60)]), notes([(0.3, 0.7, 65)])
        )
        self.assertEqual({e["type"] for e in truth}, {"deletion", "insertion"})

    def test_empty_inputs(self):
        self.assertEqual(
            MistakeDetectorBase.net_mistakes(notes([]), notes([])), ([], [])
        )
        self.assertEqual(
            MistakeDetectorBase.net_mistakes(notes([(0, 0.4, 60)]), notes([]))[0][0][
                "type"
            ],
            "deletion",
        )
        self.assertEqual(
            MistakeDetectorBase.net_mistakes(notes([]), notes([(0, 0.4, 60)]))[0][0][
                "type"
            ],
            "insertion",
        )


class AudioMistakeMetricsTest(unittest.TestCase):

    def test_substitution_equals_missed_plus_extra(self):
        from benchmarks.modules.mistake.MistakeDetectorBase import MistakeDetectorBase

        score = notes([(0, 0.4, 60)])
        final = notes([(0, 0.4, 65)])
        truth, _ = MistakeDetectorBase.net_mistakes(score, final)
        events = MistakeDetectorBase.truth_events(truth, score, final)
        predicted = MistakeDetectorBase.predicted_events(
            [Mistake("substitution", final.data[0], score.data[0])], score
        )
        self.assertEqual(
            MistakeDetectorBase.score_events(predicted, events)["audio_pitch"],
            (2, 0, 0),
        )
        self.assertEqual(
            MistakeDetectorBase.score_events(
                [
                    dict(kind="extra", onset=0, pitch=65),
                    dict(kind="missed", onset=0, pitch=60),
                ],
                events,
            )["audio_pitch"],
            (2, 0, 0),
        )

    def test_pitch_and_class_are_required(self):
        from benchmarks.modules.mistake.MistakeDetectorBase import MistakeDetectorBase

        truth = [dict(kind="extra", onset=0, pitch=65)]
        for prediction in [
            dict(kind="extra", onset=0, pitch=60),
            dict(kind="missed", onset=0, pitch=65),
        ]:
            self.assertEqual(
                MistakeDetectorBase.score_events([prediction], truth)["audio_pitch"],
                (0, 1, 1),
            )

    def test_duplicate_predictions_count_as_false_positive(self):
        from benchmarks.modules.mistake.MistakeDetectorBase import MistakeDetectorBase

        event = dict(kind="extra", onset=0, pitch=65)
        self.assertEqual(
            MistakeDetectorBase.score_events([event, event], [event])["audio_pitch"],
            (1, 1, 0),
        )

    def test_matching_maximizes_cardinality(self):
        from benchmarks.modules.mistake.MistakeDetectorBase import MistakeDetectorBase

        make = lambda t: dict(kind="extra", onset=t, pitch=60)
        self.assertEqual(
            MistakeDetectorBase.score_events(
                [make(0.05), make(-0.05)], [make(0), make(0.15)], 0.11
            )["audio_pitch"],
            (2, 0, 0),
        )

    def test_absent_error_class_and_empty_outputs(self):
        from benchmarks.modules.mistake.MistakeDetectorBase import MistakeDetectorBase

        self.assertEqual(
            MistakeDetectorBase.score_events([], [])["audio_pitch"], (0, 0, 0)
        )
        self.assertEqual(
            MistakeDetectorBase.score_events(
                [], [dict(kind="missed", onset=0, pitch=60)]
            )["audio_pitch"],
            (0, 0, 1),
        )


class FittedScoreIdentityTest(unittest.TestCase):

    def test_missing_and_aliased_ids_map_by_verified_score_order(self):
        from benchmarks.modules.mistake.MistakeDetectorBase import MistakeDetectorBase

        reference = notes([(0, 0.5, 60), (1, 1.5, 60)])
        reference.data[0].id, reference.data[1].id = (5, 8)
        fitted = notes([(2, 3, 60), (4, 5, 60)])
        fitted.data[2].id, fitted.data[4].id = (8, 6)
        mistakes = [
            Mistake("deletion", None, fitted.data[2]),
            Mistake("deletion", None, fitted.data[4]),
        ]
        mapped = MistakeDetectorBase.with_reference_score_ids(
            mistakes, reference, fitted
        )
        self.assertEqual([m.midi_note.id for m in mapped], [5, 8])
        self.assertEqual(
            [
                e["onset"]
                for e in MistakeDetectorBase.predicted_events(mapped, reference)
            ],
            [0, 1],
        )
        self.assertEqual([m.midi_note.start_time for m in mapped], [2, 4])
        self.assertEqual([m.midi_note.id for m in mistakes], [8, 6])
        self.assertEqual(
            MistakeBenchmarker.score_symbolic(
                mapped,
                [
                    dict(type="deletion", score_note_id=5, time=0),
                    dict(type="deletion", score_note_id=8, time=1),
                ],
                canonical=True,
            )["deletion"],
            (2, 0, 0),
        )

    def test_scoring_fix_preserves_only_unaffected_competitor_checkpoints(self):
        import tempfile
        from pathlib import Path
        from copy import deepcopy
        from benchmarks.modules.mistake.MistakeCache import (
            MistakeCache,
            SCORER_ID_UPGRADE,
            SCORER_PATH,
        )

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "score.mid"
            source.write_bytes(b"fixture")
            metadata = dict(
                sources=[str(source)],
                source_metadata={},
                tolerances=[0.1],
                code={SCORER_PATH: SCORER_ID_UPGRADE[0]},
                packages={},
            )
            old = MistakeCache(tmp, metadata)
            methods = [
                "Attune (Checker 3)",
                "Attune (no refinement)",
                "Parangonar DualDTW",
                "Parangonar Automatic",
                "PolyTune",
            ]
            for method in methods:
                kind = "audio" if method == "PolyTune" else "detected"
                job = old.job(str(source), 0, 0.25, kind, method)
                metrics = {"audio_pitch", "audio_missed", "audio_extra"}
                if method != "PolyTune":
                    metrics |= {
                        "substitution",
                        "deletion",
                        "insertion",
                        "short",
                        "long",
                        "pitch",
                        "duration",
                        "legacy_five_type",
                    }
                old.save(
                    job,
                    [
                        dict(
                            source=str(source),
                            seed=0,
                            rate=0.25,
                            input=kind,
                            method=method,
                            case_id="case",
                            tolerance=0.1,
                            metric=m,
                        )
                        for m in metrics
                    ],
                )
            current = deepcopy(metadata)
            current["code"][SCORER_PATH] = SCORER_ID_UPGRADE[1]
            store = MistakeCache(tmp, current)
            for method in methods:
                kind = "audio" if method == "PolyTune" else "detected"
                restored = store.load(store.job(str(source), 0, 0.25, kind, method))
                self.assertEqual(restored is None, method.startswith("Attune"))

    def test_changed_score_or_stale_object_fails_instead_of_guessing(self):
        from benchmarks.modules.mistake.MistakeDetectorBase import MistakeDetectorBase

        reference = notes([(0, 1, 60)])
        wrong = notes([(0, 1, 61)])
        with self.assertRaisesRegex(ValueError, "pitch sequence"):
            MistakeDetectorBase.with_reference_score_ids([], reference, wrong)
        stale = notes([(2, 3, 60)]).data[2]
        with self.assertRaisesRegex(ValueError, "outside"):
            MistakeDetectorBase.with_reference_score_ids(
                [Mistake("deletion", None, stale)], reference, reference
            )


class PolyTuneAdapterTest(unittest.TestCase):

    def test_missing_fresh_output_cannot_reuse_old_predictions(self):
        import json
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from benchmarks.modules.mistake.competitors.PolyTune import PolyTune

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "polytune.json"
            output.write_text(json.dumps({"events": []}))
            runner = PolyTune(repo=root)
            with patch(
                "benchmarks.modules.mistake.competitors.PolyTune.subprocess.run",
                return_value=SimpleNamespace(returncode=0),
            ):
                with self.assertRaisesRegex(RuntimeError, "inference failed"):
                    runner.predict(root / "played.wav", root / "score.wav", root)
            self.assertFalse(output.exists())


class CocoSourceTest(unittest.TestCase):

    def test_source_instrument_is_preserved(self):
        import tempfile
        from pathlib import Path
        import pretty_midi
        from benchmarks.modules.mistake.datasets.MistakeCases import source_program

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flute.mid"
            midi = pretty_midi.PrettyMIDI()
            flute = pretty_midi.Instrument(73)
            flute.notes = [pretty_midi.Note(90, 72, 0, 1)]
            midi.instruments = [flute]
            midi.write(str(path))
            self.assertEqual(source_program(path), 73)
            horn = pretty_midi.Instrument(60)
            horn.notes = [pretty_midi.Note(90, 60, 0, 1)]
            midi.instruments.append(horn)
            midi.write(str(path))
            with self.assertRaisesRegex(ValueError, "stem"):
                source_program(path)


class InjectedPitchGridTest(unittest.TestCase):

    def test_grid_and_nonzero_offsets_at_range_edges(self):
        import numpy as np
        from benchmarks.modules.mistake.datasets.MistakeInjector import MistakeInjector

        injector = MistakeInjector()
        rng = np.random.default_rng(0)
        for original in (0, 127, 60, 60.3, 60.8):
            for _ in range(100):
                pitch = injector._sample_changed_pitch(original, rng)
                self.assertIsInstance(pitch, int)
                self.assertTrue(0 <= pitch <= 127)
                self.assertNotEqual(pitch, round(original))

    def test_insertion_and_substitution_paths_use_grid(self):
        import numpy as np
        from benchmarks.modules.mistake.datasets.MistakeInjector import MistakeInjector

        reference = notes([(0, 0.4, 60.3), (1, 1.4, 64.2), (2, 2.4, 67.1)])
        for code in (1, 2, 3):
            weights = [0.0] * 16
            weights[code] = 1.0
            injector = MistakeInjector(
                mistake_rate=1.0,
                screwup_type_weights=weights,
                protect_boundary_notes=False,
                timing_std_ms=0.0,
                duration_std=0.0,
            )
            performed, truth = injector.inject(reference, np.random.default_rng(3))
            ids = {e["score_note_id"] for e in truth if e["type"] == "substitution"}
            changed = [
                n
                for n in performed.data.values()
                if code == 1 and n.source_score_id in ids or n.source_score_id is None
            ]
            self.assertTrue(changed)
            self.assertTrue(all((float(n.midi_num[0]).is_integer() for n in changed)))


class MonophonicInsertionTest(unittest.TestCase):

    def test_insertions_shift_suffix_and_net_truth_ignores_that_delay(self):
        import numpy as np
        from unittest.mock import patch
        from benchmarks.modules.mistake.datasets.MistakeInjector import MistakeInjector

        reference = notes([(0, 1, 60), (1, 2, 64), (2, 3, 67)])
        weights = [0.0] * 16
        weights[3] = 1.0
        injector = MistakeInjector(
            screwup_type_weights=weights, timing_std_ms=0.0, duration_std=0.0
        )
        with patch.object(injector, "_choose_error_indices", return_value=(1.0, {0})):
            performed, history = injector.inject(reference, np.random.default_rng(0))
        seq = list(performed.data.values())
        self.assertEqual([n.start_time for n in seq], [0, 1, 2, 3])
        self.assertTrue(all((a.end_time <= b.start_time for a, b in zip(seq, seq[1:]))))
        self.assertEqual(history[0]["time"], seq[1].start_time)
        truth, _ = MistakeDetectorBase.net_mistakes(
            reference, performed, score_onsets={n.id: n.comparison_time for n in seq}
        )
        self.assertEqual([e["type"] for e in truth], ["insertion"])

    def test_adjacent_insertions_with_same_nominal_onset_relink(self):
        import numpy as np
        from unittest.mock import Mock, patch
        from benchmarks.modules.mistake.datasets.MistakeInjector import MistakeInjector

        reference = notes([(0, 1, 60), (1, 2, 64), (2, 3, 67)])
        injector = MistakeInjector(timing_std_ms=0.0, duration_std=0.0)
        rng = Mock(wraps=np.random.default_rng(0))
        rng.choice.side_effect = [3, 2]
        with patch.object(
            injector, "_choose_error_indices", return_value=(1.0, {0, 1})
        ), patch.object(injector, "_sample_changed_pitch", side_effect=[61, 65]):
            performed, history = injector.inject(reference, rng)
        seq = list(performed.data.values())
        self.assertTrue(all((a.end_time <= b.start_time for a, b in zip(seq, seq[1:]))))
        self.assertEqual([e["time"] for e in history], [1.0, 2.0])
        self.assertFalse(any(("_performed_index" in e for e in history)))

    def test_insertion_then_deletion_cancel_in_final_recount(self):
        import numpy as np
        from unittest.mock import Mock, patch
        from benchmarks.modules.mistake.datasets.MistakeInjector import MistakeInjector

        reference = notes([(0, 1, 60), (1, 2, 64), (2, 3, 67), (3, 4, 69)])
        injector = MistakeInjector(timing_std_ms=0.0, duration_std=0.0)
        rng = Mock(wraps=np.random.default_rng(0))
        rng.choice.side_effect = [3, 0, 1]
        with patch.object(
            injector, "_choose_error_indices", return_value=(1.0, {0, 1, 2})
        ), patch.object(injector, "_sample_changed_pitch", side_effect=[64, 71]):
            performed, history = injector.inject(reference, rng)
        seq = list(performed.data.values())
        self.assertEqual([n.start_time for n in seq], [0, 1, 2, 3])
        self.assertTrue(all((a.end_time <= b.start_time for a, b in zip(seq, seq[1:]))))
        truth, _ = MistakeDetectorBase.net_mistakes(
            reference, performed, score_onsets={n.id: n.comparison_time for n in seq}
        )
        self.assertEqual([e["type"] for e in truth], ["substitution"])
        self.assertEqual(truth[0]["score_note_id"], list(reference.data.values())[2].id)

    def test_deletions_advance_suffix_without_false_net_errors(self):
        import numpy as np
        from unittest.mock import patch
        from benchmarks.modules.mistake.datasets.MistakeInjector import MistakeInjector
        from benchmarks.modules.mistake.MistakeDetectorBase import MistakeDetectorBase

        reference = notes([(0, 1, 60), (1, 2, 64), (2.2, 3.2, 67)])
        weights = [0.0] * 16
        weights[0] = 1.0
        injector = MistakeInjector(screwup_type_weights=weights)
        with patch.object(injector, "_choose_error_indices", return_value=(1.0, {1})):
            performed, history = injector.inject(reference, np.random.default_rng(0))
        seq = list(performed.data.values())
        self.assertAlmostEqual(seq[-1].start_time, 1.2)
        self.assertAlmostEqual(seq[-1].duration(), 1.0)
        entries = [
            dict(
                id=n.id,
                onset=n.start_time,
                pitch=n.midi_num[0],
                score_onset=n.comparison_time,
            )
            for n in seq
        ]
        for n in seq:
            n.id += 100
        mapping = MistakeDetectorBase.timeline_score_onsets(
            dict(injector=injector.last_metadata, performance_timeline=entries),
            performed,
        )
        truth, _ = MistakeDetectorBase.net_mistakes(
            reference, performed, score_onsets=mapping
        )
        self.assertEqual([e["type"] for e in truth], ["deletion"])
        self.assertEqual(truth[0]["time"], 1.0)

    def test_long_edits_are_also_monophonic_without_false_suffix_errors(self):
        import numpy as np
        from unittest.mock import patch
        from benchmarks.modules.mistake.datasets.MistakeInjector import MistakeInjector

        reference = notes([(0, 1, 60), (1, 2, 64), (2, 3, 67)])
        weights = [0.0] * 16
        weights[5] = 1.0
        injector = MistakeInjector(screwup_type_weights=weights)
        with patch.object(injector, "_choose_error_indices", return_value=(1.0, {0})):
            performed, _ = injector.inject(reference, np.random.default_rng(0))
        seq = list(performed.data.values())
        self.assertTrue(all((a.end_time <= b.start_time for a, b in zip(seq, seq[1:]))))
        truth, _ = MistakeDetectorBase.net_mistakes(
            reference, performed, score_onsets={n.id: n.comparison_time for n in seq}
        )
        self.assertEqual([e["type"] for e in truth], ["long"])


class MidiSerializationTest(unittest.TestCase):

    def test_subtick_note_does_not_hang_until_repeated_pitch(self):
        import contextlib
        import io
        import tempfile
        from pathlib import Path
        import numpy as np
        import pretty_midi
        from benchmarks.modules.mistake.datasets.MistakeInjector import MistakeInjector
        from benchmarks.modules.mistake.MistakeDetectorBase import MistakeDetectorBase
        from benchmarks.modules.note.NoteBenchmarker import OneInstrumentScoreData

        reference = notes([(0.1, 0.1001, 60), (0.5, 1.0, 62), (1.0, 1.5, 60)])
        injector = MistakeInjector(
            mistake_rate=0.0, timing_std_ms=0.0, duration_std=0.0
        )
        performed, _ = injector.inject(reference, np.random.default_rng(0))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "short.mid"
            MistakeBenchmarker.notedata_to_pm(performed).write(str(path))
            raw = sorted(
                pretty_midi.PrettyMIDI(str(path)).instruments[0].notes,
                key=lambda n: n.start,
            )
            self.assertEqual(len(raw), 3)
            self.assertLess(raw[0].end, 0.11)
            self.assertTrue(all((n.end > n.start for n in raw)))
            self.assertTrue(all((a.end <= b.start for a, b in zip(raw, raw[1:]))))
            with contextlib.redirect_stdout(io.StringIO()):
                parsed = OneInstrumentScoreData(path).note_data
            mapping = MistakeDetectorBase.timeline_score_onsets(
                dict(
                    injector=injector.last_metadata,
                    performance_timeline=MistakeBenchmarker.performance_timeline(
                        performed
                    ),
                ),
                parsed,
            )
            self.assertEqual(len(mapping), 3)


class PolyTuneWindowTest(unittest.TestCase):

    def test_either_recording_can_end_first(self):
        import numpy as np
        from benchmarks.modules.mistake.competitors.PolyTune import PolyTune

        for played, score in [(3344, 3060), (3060, 3344), (30, 10)]:
            result = PolyTune.split_frames(
                None,
                np.ones((played, 2)),
                np.ones((score, 2)),
                np.arange(played),
                np.arange(score),
                {},
            )
            self.assertTrue(all((n >= 0 for n in result[4] + result[5])))
            self.assertEqual(result[0].shape[1:], (256, 2))
            self.assertEqual(result[1].shape[1:], (512, 2))
            if played > score + 128:
                self.assertFalse(result[1][-1].any())
            if score > played + 256:
                self.assertFalse(result[0][-1].any())


class ResumableComparisonTest(unittest.TestCase):

    def test_failure_preserves_methods_and_next_run_only_computes_missing(self):
        import contextlib
        import io
        import json
        import sys
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from benchmarks.modules.mistake.MistakeBenchmarker import (
            COMPETITOR_AUDIO_METHODS,
            COMPETITOR_METHODS,
            MistakeBenchmarker,
        )
        from benchmarks.modules.mistake.MistakeCache import MistakeCache
        import pandas as pd

        attempts = []
        completed_units = len(MistakeCache.units(COMPETITOR_METHODS)) - len(
            COMPETITOR_AUDIO_METHODS
        )
        completed_rows = completed_units * 11 * 3

        def evaluate(source, seed, rate, output, methods, tolerances, **kwargs):
            cached = kwargs["cached"]
            attempts.append(set(cached))
            print("pipeline noise")
            print("diagnostic warning", file=sys.stderr)
            for kind, method in MistakeCache.units(methods):
                if (kind, method) in cached:
                    continue
                if len(attempts) == 1 and method == "PolyTune":
                    raise RuntimeError("simulated neural failure")
                metrics = {"audio_pitch", "audio_missed", "audio_extra"}
                if method not in COMPETITOR_AUDIO_METHODS:
                    metrics |= {
                        "substitution",
                        "deletion",
                        "insertion",
                        "short",
                        "long",
                        "pitch",
                        "duration",
                        "legacy_five_type",
                    }
                batch = [
                    dict(
                        case_id="case",
                        source=source,
                        seed=seed,
                        rate=rate,
                        input=kind,
                        method=method,
                        tolerance=t,
                        metric=m,
                        tp=1,
                        fp=0,
                        fn=0,
                        precision=1.0,
                        recall=1.0,
                        f1=1.0,
                        score_notes=10,
                        seconds=0.1,
                    )
                    for t in tolerances
                    for m in metrics
                ]
                kwargs["on_result"](batch)

        stream, errors = (io.StringIO(), io.StringIO())
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            MistakeBenchmarker, "preflight", return_value={}
        ), patch.object(MistakeBenchmarker, "evaluate_case", side_effect=evaluate):
            source = Path(tmp) / "source.mid"
            source.write_bytes(b"fixture")
            output = Path(tmp) / "results"
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(errors):
                with self.assertRaisesRegex(RuntimeError, "simulated neural failure"):
                    MistakeBenchmarker.run_comparison(
                        [source], output, rates=(0.25,), parallel=False
                    )
                self.assertEqual(
                    len(list((output / "checkpoints").glob("*.json"))), completed_units
                )
                self.assertEqual(len(pd.read_csv(output / "rows.csv")), completed_rows)
                self.assertEqual(
                    json.loads((output / "run.json").read_text())["status"], "failed"
                )
                rows = MistakeBenchmarker.run_comparison(
                    [source], output, rates=(0.25,), parallel=False
                )
                cached_rows = MistakeBenchmarker.run_comparison(
                    [source], output, rates=(0.25,), parallel=False
                )
            self.assertEqual(len(attempts), 2)
            self.assertEqual(len(attempts[1]), completed_units)
            self.assertEqual(
                len(rows), completed_rows + 9 * len(COMPETITOR_AUDIO_METHODS)
            )
            self.assertEqual(
                len(cached_rows), completed_rows + 9 * len(COMPETITOR_AUDIO_METHODS)
            )
            self.assertEqual(
                len(list((output / "checkpoints").glob("*.json"))),
                completed_units + len(COMPETITOR_AUDIO_METHODS),
            )
            self.assertFalse(
                rows.duplicated(
                    ["case_id", "method", "input", "metric", "tolerance"]
                ).any()
            )
            self.assertEqual(
                json.loads((output / "run.json").read_text())["status"], "complete"
            )
        self.assertEqual(stream.getvalue().count("\n"), 3)
        self.assertNotIn("noise", stream.getvalue())
        self.assertEqual(errors.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
