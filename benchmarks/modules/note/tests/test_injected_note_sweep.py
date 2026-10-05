import unittest
from types import SimpleNamespace
import pandas as pd
from benchmarks.modules.note.sweeps.InjectedNoteSweep import Axes
from benchmarks.modules.note.sweeps.InjectedNoteSweep import effective_seconds
from benchmarks.modules.note.sweeps.InjectedNoteSweep import grid
from benchmarks.modules.note.sweeps.InjectedNoteSweep import parameter_jobs
from benchmarks.modules.note.sweeps.InjectedNoteSweep import summarize


class InjectedNoteSweepTest(unittest.TestCase):

    def test_parallel_groups_cover_grid_once_and_keep_duration_variants_together(self):
        variants = grid()
        groups = parameter_jobs(variants)
        self.assertEqual(len(groups), 20)
        self.assertTrue(all((len(chunk) == 35 for chunk in groups.values())))
        self.assertEqual(
            sorted((v["variant"] for chunk in groups.values() for v in chunk)),
            sorted((v["variant"] for v in variants)),
        )
        for chunk in groups.values():
            self.assertEqual(
                len({(v["pitch_step"], v["silence_ms"]) for v in chunk}), 1
            )

    def test_grid_and_baseline(self):
        variants = grid()
        self.assertEqual(len(variants), 700)
        self.assertEqual(len({v["variant"] for v in variants}), 700)
        self.assertTrue(variants[0]["baseline"])
        self.assertEqual(sum((v["baseline"] for v in variants)), 1)
        with self.assertRaises(ValueError):
            grid(Axes(cap_ms=(0.0,)))
        with self.assertRaises(ValueError):
            grid(Axes(score_factor=(None,)))

    def test_cap_preserves_short_score_notes(self):
        self.assertEqual(effective_seconds(1.0, 0.5, 50.0), 0.05)
        self.assertEqual(effective_seconds(0.04, 0.5, 50.0), 0.02)
        self.assertEqual(effective_seconds(1.0, 0.5, None), 0.5)

    def test_short_notes_survive_absolute_cap(self):
        from algorithms.Config import Config
        from algorithms.NoteDetector import NoteDetector
        from app_logic.user.ds.PitchData import Pitch
        from app_logic.user.ds.PitchData import PitchData

        cfg = Config(
            sr=1000,
            h1=1,
            min_note_length=1.0,
            min_note_length_factor=0.6,
            min_note_length_cap=0.0,
            min_silence_duration_ms=3.0,
            pitch_thresh=0.75,
        )
        data = PitchData(cfg)
        data.data = [
            Pitch(i / 1000, 0.1, 0.0, 0.0, cfg, candidates=[(pitch, 1.0)], value=pitch)
            for i in range(120)
            for pitch in [60.0 if i < 60 else 64.0]
        ]
        detector = NoteDetector(SimpleNamespace(pitch_data=data, config=cfg))
        self.assertEqual(len(detector.detect_notes(data.data).times), 0)
        cfg.min_note_length_factor = (
            effective_seconds(1.0, 0.5, 50.0) / cfg.min_note_length
        )
        notes = detector.detect_notes(data.data)
        self.assertEqual(len(notes.times), 2)
        self.assertEqual([n.midi_num[0] for n in notes.data.values()], [60.0, 64.0])

    def test_promoted_cap_and_fast_score_rule(self):
        from algorithms.Config import Config

        cfg = Config(min_note_length=1.0)
        self.assertEqual(cfg.note_detection_min_seconds(), 0.1)
        self.assertEqual(cfg.note_detection_min_frames(), 35)
        cfg.set_min_note_length(0.04)
        self.assertEqual(cfg.note_detection_min_seconds(), 0.02)
        self.assertEqual(cfg.note_detection_min_frames(), 7)
        self.assertEqual(cfg.min_silence_duration_ms, 40.0)
        self.assertIn("min_note_length_cap", cfg.note_segmentation_config())

    def test_production_detector_caps_runs_and_change_points(self):
        from algorithms.Config import Config
        from algorithms.NoteDetector import NoteDetector
        from app_logic.user.ds.PitchData import Pitch
        from app_logic.user.ds.PitchData import PitchData

        cfg = Config(sr=1000, h1=1, min_note_length=1.0)
        data = PitchData(cfg)
        data.data = [
            Pitch(i / 1000, 0.1, 0.0, 0.0, cfg, candidates=[(pitch, 1.0)], value=pitch)
            for i in range(240)
            for pitch in [60.0 if i < 120 else 64.0]
        ]
        detector = NoteDetector(SimpleNamespace(pitch_data=data, config=cfg))
        notes = detector.detect_notes(data.data)
        self.assertEqual([n.midi_num[0] for n in notes.data.values()], [60.0, 64.0])
        cfg.min_note_length_cap = 0.0
        self.assertEqual(len(detector.detect_notes(data.data).times), 0)

    def test_summary_pools_counts_and_filters_clean_regressions(self):
        rows = []
        for variant, baseline, clean_fp in [("base", True, 0), ("candidate", False, 1)]:
            for rate, counts in [
                (0.25, (1, 0, 0)),
                (0.25, (0, 9, 1)),
                (0.0, (0, clean_fp, 0)),
            ]:
                row = dict(
                    variant=variant,
                    baseline=baseline,
                    cap_ms=None,
                    score_factor=0.6,
                    pitch_step=0.75,
                    silence_ms=10.0,
                    rate=rate,
                    score_notes=10,
                )
                for prefix in (
                    "note50",
                    "note100",
                    "note200",
                    "note_offsets",
                    "audio_pitch50",
                    "audio_pitch100",
                    "audio_pitch200",
                    "audio_extra100",
                    "audio_missed100",
                ):
                    row.update(
                        {f"{prefix}_{k}": v for k, v in zip(("tp", "fp", "fn"), counts)}
                    )
                rows.append(row)
        summary = summarize(pd.DataFrame(rows)).set_index("variant")
        self.assertAlmostEqual(
            summary.loc["base", "injected_audio_pitch100_f1"], 100 / 6
        )
        self.assertTrue(summary.loc["base", "clean_safe"])
        self.assertFalse(summary.loc["candidate", "clean_safe"])


if __name__ == "__main__":
    unittest.main()
