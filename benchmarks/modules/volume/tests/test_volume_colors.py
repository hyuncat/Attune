"""Run: at-venv/bin/python -m unittest discover -s benchmarks/modules/volume/tests"""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest

import numpy as np
from app_logic.user.VolumeLevel import RelativeLevelStats, VolumeColorScale


class VolumeColorsTest(unittest.TestCase):
    def test_skewed_distribution_center_endpoints_and_gain(self):
        levels = [-60, -59, -58, -57, -20]
        scale = VolumeColorScale(levels)
        self.assertAlmostEqual(scale.reference_db, -50.8)
        self.assertEqual(scale.fraction_db(scale.reference_db), .5)
        self.assertEqual(scale.fraction_db(-100), 0)
        self.assertEqual(scale.fraction_db(0), 1)
        # The quieter extreme need not reach purple in an asymmetric take.
        self.assertGreater(scale.fraction_db(-60), 0)
        self.assertAlmostEqual(scale.fraction_db(-20), 1)
        self.assertAlmostEqual(scale.half_range_db, 30.8)
        self.assertAlmostEqual(scale.fraction_db(-50.8 - 6) +
                               scale.fraction_db(-50.8 + 6), 1)
        self.assertAlmostEqual(scale.fraction_db(-40) - scale.fraction_db(-46),
                               scale.fraction_db(-46) - scale.fraction_db(-52))
        shifted = VolumeColorScale(np.array(levels) + 12)
        probes = np.linspace(-65, -15, 101)
        values = [scale.fraction_db(x) for x in probes]
        self.assertTrue(np.all(np.diff(values) >= 0))
        np.testing.assert_allclose(values, [shifted.fraction_db(x + 12) for x in probes])

    def test_ties_constant_and_empty(self):
        scale = VolumeColorScale([-40, -40, -20, -20, -20])
        self.assertEqual(scale.fraction_db(-40), 0)
        self.assertAlmostEqual(scale.fraction_db(-20), 5 / 6)
        self.assertEqual(scale.fraction_db(-28), .5)
        self.assertEqual(VolumeColorScale([-30] * 4).fraction_db(-30), .5)
        self.assertIsNone(VolumeColorScale([]).fraction_db(-30))
        for volume in (0, -1, float('nan'), float('inf')):
            self.assertIsNone(scale.fraction(volume))

    def test_written_voiced_frames_live_append_overwrite_and_trim(self):
        def frame(db, voiced=True):
            return SimpleNamespace(volume=10 ** (db / 20), value=60 if voiced else -1,
                                   unvoiced_prob=0 if voiced else 1)
        pd = SimpleNamespace(data=[frame(-60), frame(-20), frame(0, False), None], end=3)
        pd.frames_available = lambda: pd.end
        pd.is_voiced_pitch = lambda p: p is not None and p.value != -1
        stats = RelativeLevelStats()
        stats.sync(pd)
        self.assertEqual(stats.count, 2)
        self.assertAlmostEqual(stats.reference_db, -40)
        self.assertEqual(stats.color_scale.fraction_db(stats.reference_db), .5)
        pd.data[3] = frame(-10)
        pd.end = 4
        stats.sync(pd)
        self.assertEqual(stats.count, 3)
        pd.data[3] = frame(-40)  # Practice's stalled last slot.
        stats.sync(pd)
        self.assertAlmostEqual(stats.reference_db, -40)
        pd.end = 1
        stats.sync(pd)
        self.assertEqual(stats.count, 1)
        self.assertEqual(stats.color_scale.fraction_db(-60), .5)

    def test_live_colors_freeze_and_review_recolors(self):
        def frame(db):
            return SimpleNamespace(volume=10 ** (db / 20), value=60, unvoiced_prob=0)
        pd = SimpleNamespace(data=[frame(-40), frame(-28)])
        pd.frames_available = lambda: len(pd.data)
        pd.is_voiced_pitch = lambda p: p is not None and p.value != -1
        stats = RelativeLevelStats()
        stats.sync(pd)
        frozen = dict(stats.live_fractions)
        self.assertEqual(frozen[id(pd.data[0])], .5)
        self.assertEqual(frozen[id(pd.data[1])], .75)
        pd.data.append(frame(0))
        stats.sync(pd)
        for key, value in frozen.items():
            self.assertEqual(stats.live_fractions[key], value)
        self.assertEqual(stats.live_fractions[id(pd.data[-1])], 1)
        self.assertNotEqual(stats.color_scale.fraction_db(-40), frozen[id(pd.data[0])])
        # Starting the view late / batching must yield the same frozen colors.
        batched = RelativeLevelStats()
        batched.sync(pd)
        self.assertEqual(stats.live_fractions, batched.live_fractions)
        pd.data[-1] = frame(-50)
        stats.sync(pd)
        for key, value in frozen.items():
            self.assertEqual(stats.live_fractions[key], value)

    def test_web_python_parity(self):
        root = Path(__file__).resolve().parents[4]
        cases = [[-60, -59, -58, -57, -20], [-40, -40, -20, -20, -20], [-30], []]
        # Use exactly the same RMS->dB round-trip as real pitch frames.
        frames = [[[i, [[60, 1]], 10 ** (db / 20), 0, None, None, False, 60]
                   for i, db in enumerate(levels)] for levels in cases]
        probes = [-100, -60, -59, -50.8, -40, -30, -28, -20, 0]
        script = '''
import { volumeColorScale, volumeFractionDb } from './web/src/colors.js';
const input = JSON.parse(process.argv[1]);
console.log(JSON.stringify(input.frames.map(frames => {
  const scale = volumeColorScale(frames);
  return input.probes.map(db => volumeFractionDb(db, scale));
})));
'''
        result = subprocess.run(['node', '--input-type=module', '-e', script,
                                 json.dumps(dict(frames=frames, probes=probes))],
                                cwd=root, text=True, capture_output=True, check=True)
        actual = json.loads(result.stdout)
        for track, values in zip(frames, actual):
            scale = VolumeColorScale([20 * np.log10(f[2]) for f in track])
            expected = [scale.fraction_db(db) for db in probes]
            if not track:
                self.assertEqual(values, expected)
            else:
                np.testing.assert_allclose(values, expected, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
