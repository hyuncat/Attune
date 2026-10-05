"""Range coverage at score boundaries and final injected extremes."""

from pathlib import Path
import tempfile
from unittest.mock import patch
from types import SimpleNamespace
import unittest
import numpy as np
from algorithms.Config import Config
from algorithms.PitchDetector import PitchDetector
from app_logic.user.ds.Recording import Recording
from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker
from benchmarks.modules.mistake.tests.CompetitorComparisonTest import notes
from benchmarks.modules.pitch.competitors.Attune import Attune


class PitchRangeTests(unittest.TestCase):

    def test_app_and_benchmark_pad_low_register_and_single_pitch_scores(self):
        for low, high in [(28, 43), (71, 80), (60, 60)]:
            with self.subTest(low=low, high=high):
                host = SimpleNamespace(
                    active_instrument=0,
                    score_data=SimpleNamespace(
                        note_midi_range=lambda channel: (low, high)
                    ),
                )
                cfg = Recording._default_config_from_score(host)
                self.assertAlmostEqual(cfg.freq_to_midi(cfg.fmin), low - 4)
                self.assertAlmostEqual(cfg.freq_to_midi(cfg.fmax), high + 4)
                benchmark = MistakeBenchmarker().config_for_performance(
                    notes([(0, 1, low), (1, 2, high)])
                )
                np.testing.assert_allclose(
                    [benchmark.fmin, benchmark.fmax], [cfg.fmin, cfg.fmax]
                )
                PitchDetector(config=cfg)

    def test_injected_extremes_keep_margin_beyond_score_only_bounds(self):
        performed = notes([(0, 1, 49), (1, 2, 83)])
        cfg = MistakeBenchmarker().config_for_performance(performed)
        self.assertAlmostEqual(cfg.freq_to_midi(cfg.fmin), 45)
        self.assertAlmostEqual(cfg.freq_to_midi(cfg.fmax), 87)

    def test_score_endpoint_tones_have_candidates_inside_padded_range(self):
        for low, high in [(28, 43), (71, 80)]:
            cfg = MistakeBenchmarker().config_for_performance(
                notes([(0, 1, low), (1, 2, high)])
            )
            detector = PitchDetector(config=cfg)
            for pitch in [low, high]:
                with self.subTest(pitch=pitch):
                    t = np.arange(cfg.w1) / cfg.sr
                    frame = np.sin(2 * np.pi * cfg.midi_to_freq(pitch) * t)[:, None]
                    observations, _, _ = detector._probabilities_from_frames(frame)
                    correct = abs(detector.bin_midis - pitch) <= 0.5
                    self.assertGreater(observations[correct].sum(), 0.1)

    def test_only_mistake_loader_rejects_unknown_or_changed_range(self):
        bench = MistakeBenchmarker()
        rec = SimpleNamespace(config=Config(fmin=150, fmax=1000))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pitches.pkl.xz"
            path.touch()
            with patch.object(
                bench.attune, "load_or_detect_pitches", return_value={}
            ) as load:
                bench.load_mistake_pitches(rec, path)
                self.assertFalse(load.call_args.kwargs["use_cache"])
                bench.load_mistake_pitches(rec, path)
                self.assertTrue(load.call_args.kwargs["use_cache"])
                rec.config.fmax = 1200
                bench.load_mistake_pitches(rec, path)
                self.assertFalse(load.call_args.kwargs["use_cache"])
                bench.load_mistake_pitches(rec, path, use_cache=False)
                self.assertFalse(load.call_args.kwargs["use_cache"])
        cfg = Config()
        np.testing.assert_allclose(
            Attune.range_from_midi([60, 72]),
            [cfg.midi_to_freq(60), cfg.midi_to_freq(72)],
        )


if __name__ == "__main__":
    unittest.main()
