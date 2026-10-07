"""Regression: Yang targets average half cycles equally within each event."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import numpy as np

from benchmarks.modules.vibrato.datasets.YangDataset import YangDataset
from benchmarks.modules.vibrato.VibratoBenchmarker import VibratoBenchmarker
from benchmarks.modules.vibrato.VibratoDetectorBase import VibratoEstimate


class EventYangParameters(unittest.TestCase):
    def test_changing_half_cycles_have_event_means(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            area, extrema = root/'area.csv', root/'extrema.csv'
            area.write_text('0,0,.6\n')
            extrema.write_text('.1\n.2\n.4\n.5\n')
            recording = YangDataset.Recording(root/'test.wav', area, extrema, 'Violin', 'fixture')
            times = np.arange(70)/100
            pitch = np.interp(times, [.1,.2,.4,.5], [60,60.4,59.6,60.8])
            example, = YangDataset._examples_from_annotations(recording, times, pitch, pitch_stage='fixture')
        target = example.score_mask
        # Equal half-cycle weighting, not weighting by half-cycle duration.
        np.testing.assert_allclose(example.rate_hz[target], (5+2.5+5)/3)
        np.testing.assert_allclose(example.width_cents[target], (40+80+120)/3)
        self.assertTrue(example.is_vibrato[[5,55]].all())
        self.assertTrue(np.isfinite(example.rate_hz[[5,55]]).all())
        bench = VibratoBenchmarker()
        method = SimpleNamespace(name='fixture', description='fixture', scores_center=False)
        exact = VibratoEstimate(example.rate_hz.copy(), example.width_cents.copy(), example.is_vibrato)
        correct = bench._score(method, example, exact, 0.)
        self.assertAlmostEqual(correct['aggregate_f1'], 1.)
        self.assertAlmostEqual(correct['aggregate_soft_f1'], 1.)


if __name__ == '__main__':
    unittest.main()
