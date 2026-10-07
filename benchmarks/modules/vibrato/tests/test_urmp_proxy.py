"""Reference-unit and missing-data checks for the provisional URMP benchmark."""
import unittest
import numpy as np
from benchmarks.modules.vibrato.datasets.URMPDataset import ReferenceConfig, reference_parameters


class URMPProxyTests(unittest.TestCase):
    def setUp(self):
        self.times = np.arange(200)/100.
        self.config = ReferenceConfig()

    def test_known_rate_and_peak_to_peak_units(self):
        pitch = 69 + .25*np.sin(2*np.pi*5*self.times)
        result = reference_parameters(self.times, pitch, self.config)
        self.assertTrue(result['is_vibrato'])
        self.assertAlmostEqual(result['rate_hz'], 5, delta=.03)
        self.assertAlmostEqual(result['width_cents'], 50, delta=.5)

    def test_straight_tone_and_out_of_band_are_negative(self):
        for pitch in (np.full(200,69.), 69+.25*np.sin(2*np.pi*1.5*self.times),
                      69+.25*np.sin(2*np.pi*12*self.times), 69+self.times):
            self.assertFalse(reference_parameters(self.times,pitch,self.config)['is_vibrato'])

    def test_missing_run_is_excluded_not_negative(self):
        pitch=69+.25*np.sin(2*np.pi*5*self.times)
        pitch[50:54]=np.nan
        self.assertIsNone(reference_parameters(self.times,pitch,self.config))
        self.assertIsNone(reference_parameters(self.times[:1],pitch[:1],self.config))
        self.assertIsNone(reference_parameters(self.times,np.full(200,np.nan),self.config))

if __name__ == '__main__':
    unittest.main()
