"""Promotion matches the accepted experiment and invalidates old pitch stages."""

import ast
import json
import lzma
import pickle
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
from algorithms.Config import Config
from algorithms.PitchSmoother import VoicingSmoother, PitchSmoother
from app_logic.user.ds.PitchData import Pitch
from app_logic.user.ds.Recording import Recording
from benchmarks.modules.pitch.PitchCache import PitchCache
from benchmarks.modules.pitch.competitors.Attune import Attune


class ProductionPitchSmootherTest(unittest.TestCase):

    def setUp(self):
        self.config = Config(sr=44100, w1=4096, h1=128, fmin=196.0, fmax=880.0)

    def test_app_and_benchmark_use_promoted_configuration(self):
        for recording in [
            Recording(config=self.config),
            Attune().recording_for(self.config),
        ]:
            smoother = recording.pitch_smoother
            self.assertIsInstance(smoother, PitchSmoother)
            self.assertEqual(smoother.max_gap_seconds, 0.05)
            self.assertTrue(smoother.confidence_emissions)

    def test_two_stage_production_matches_complete_experiment(self):
        frames = [
            Pitch(
                time=i * 128 / 44100,
                candidates=[(69.0, 0.15), (81.0, 0.1)],
                volume=0.1,
                unvoiced_prob=0.75 if i % 7 else 1.0,
                live_distance=None,
                config=self.config,
            )
            for i in range(50)
        ]
        frames[10] = None
        expected = PitchSmoother(
            mode="pitch_only",
            config=self.config,
            max_gap_seconds=0.05,
            confidence_emissions=True,
        ).smooth(frames)
        tracked = PitchSmoother(config=self.config).smooth(frames)
        for raw, output in zip(frames, tracked):
            if raw is not None:
                self.assertEqual(raw.unvoiced_prob, output.unvoiced_prob)
        actual = VoicingSmoother(config=self.config).smooth(tracked)
        np.testing.assert_equal(
            [(p.value, p.unvoiced_prob) if p else (-1.0, 1.0) for p in actual],
            [(p.value, p.unvoiced_prob) if p else (-1.0, 1.0) for p in expected],
        )

    def test_previous_cache_version_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.pitch.pkl.xz"
            for version in (1, 12):
                with self.subTest(version=version):
                    with lzma.open(path, "wb") as handle:
                        pickle.dump(
                            {
                                "version": version,
                                "stages": {"smoothed": {"pitches": []}},
                            },
                            handle,
                        )
                    self.assertIsNone(
                        PitchCache(path).read(PitchCache.SMOOTHED, self.config)
                    )

    def test_unsupported_bins_are_hidden_only_after_decoding(self):
        smoother = PitchSmoother(config=self.config)
        bins = [100, 105, 110, 115]
        frames = [
            Pitch(
                time=i * 0.003,
                candidates=[(smoother.bin_midis[b], 0.8)],
                volume=0.1,
                unvoiced_prob=0.2,
                live_distance=None,
                config=self.config,
            )
            for i, b in enumerate(bins)
        ]
        frames[1].candidate_pitches.append((smoother.bin_midis[106], 0.0))
        path = np.array([100, 106, 110, 115], dtype=np.uint16)
        with patch.object(smoother, "decode", return_value=path) as decode:
            tracked = smoother.smooth(frames)
        decode.assert_called_once_with(frames)
        final = VoicingSmoother(config=self.config).smooth(tracked)
        self.assertEqual(final[1].value, -1.0)
        self.assertEqual(final[1].unvoiced_prob, 1.0)
        for i in (0, 2, 3):
            self.assertEqual(final[i].value, smoother.bin_midis[path[i]])
        self.assertEqual(frames[1].unvoiced_prob, 0.2)
        self.assertEqual(frames[1].value, smoother.bin_midis[105])
        self.assertEqual(final[1].candidate_pitches, frames[1].candidate_pitches)

    def test_support_uses_same_quantized_bin_as_observations(self):
        smoother = PitchSmoother(config=self.config)
        midi = smoother.bin_midis[100]
        frame = Pitch(
            time=0.0,
            candidates=[(midi + 0.02, 0.01)],
            volume=0.1,
            unvoiced_prob=0.5,
            live_distance=None,
            config=self.config,
        )
        with patch.object(smoother, "decode", return_value=np.array([100])):
            result = smoother.smooth([frame])[0]
        self.assertEqual(result.value, midi)
        self.assertEqual(result.unvoiced_prob, 0.5)

    def test_v13_preserves_raw_but_invalidates_smoothed_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v13.pitch.pkl.xz"
            with lzma.open(path, "wb") as handle:
                pickle.dump(
                    {
                        "version": 13,
                        "stages": {"raw": {"pitches": []}, "smoothed": {"pitches": []}},
                    },
                    handle,
                )
            self.assertIsNotNone(PitchCache(path).read(PitchCache.RAW, self.config))
            self.assertIsNone(PitchCache(path).read(PitchCache.SMOOTHED, self.config))

    def test_notebook_reuses_cached_methods_by_default(self):
        root = Path(__file__).resolve().parents[4]
        notebook = json.loads((root / "benchmarks/notebooks/pitch.ipynb").read_text())
        calls = []
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            if source.startswith("%"):
                continue
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr.startswith("run_")
                ):
                    options = {k.arg: k.value for k in node.keywords}
                    if "force" in options:
                        self.assertIs(ast.literal_eval(options["force"]), False)
                    self.assertNotIn("methods", options)
                    calls.append(node.func.attr)
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and (node.func.id == "NotebookConfig")
                ):
                    options = {k.arg: k.value for k in node.keywords}
                    self.assertIs(ast.literal_eval(options["use_cache"]), True)
                    self.assertIs(ast.literal_eval(options["force_rerun"]), False)
        self.assertGreaterEqual(len(calls), 9)


if __name__ == "__main__":
    unittest.main()
