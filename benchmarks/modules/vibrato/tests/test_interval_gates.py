"""Synthetic duration/cycle checks; never runs a dataset benchmark."""
from types import SimpleNamespace
import unittest
import numpy as np
from benchmarks.modules.vibrato.VibratoDetectorBase import VibratoEstimate
from benchmarks.modules.vibrato.tests.IntervalGates import IntervalGate, apply_interval_gate

class IntervalGateChecks(unittest.TestCase):
    def apply(self, rates, mask, gate, dt=.01, pitch=None):
        n=len(rates)
        example=SimpleNamespace(times=np.arange(n)*dt, pitch_midi=np.full(n,60.) if pitch is None else pitch,
                                metadata={'analysis_note_bounds':[(0,.03,60),(.03,n*dt,60)]})
        estimate=VibratoEstimate(np.asarray(rates,float),np.full(n,40.),np.asarray(mask,bool))
        result=apply_interval_gate(example,estimate,gate)
        np.testing.assert_array_equal(result.rate_hz,estimate.rate_hz)
        np.testing.assert_array_equal(estimate.detected,mask)
        return result.detected

    def test_exact_cycle_and_variable_rate(self):
        self.assertTrue(self.apply([4]*10+[6]*10,[True]*20,IntervalGate('cycle',min_cycles=1)).all())
        self.assertFalse(self.apply([5]*19,[True]*19,IntervalGate('cycle',min_cycles=1)).any())

    def test_duration_frame_support_and_last_run(self):
        self.assertTrue(self.apply([5]*25,[True]*25,IntervalGate('duration',min_seconds=.25)).all())
        self.assertFalse(self.apply([5]*24,[True]*24,IntervalGate('duration',min_seconds=.25)).any())
        self.assertTrue(self.apply([5]*40,[False]*15+[True]*25,IntervalGate('duration',min_seconds=.25))[-25:].all())

    def test_no_bridging_but_no_note_reset(self):
        self.assertFalse(self.apply([5]*41,[True]*19+[False]+[True]*21,IntervalGate('duration',min_seconds=.25)).any())
        self.assertTrue(self.apply([5]*25,[True]*25,IntervalGate('duration',min_seconds=.25)).all())

    def test_invalid_rate_and_unvoiced_break_run(self):
        rates=[5]*41;rates[20]=np.nan
        self.assertFalse(self.apply(rates,[True]*41,IntervalGate('duration',min_seconds=.25)).any())
        pitch=np.full(41,60.);pitch[20]=np.nan
        self.assertFalse(self.apply([5]*41,[True]*41,IntervalGate('duration',min_seconds=.25),pitch=pitch).any())

    def test_empty_and_invalid_minimum(self):
        self.assertFalse(self.apply([5]*30,[False]*30,IntervalGate('cycle',min_cycles=1)).any())
        with self.assertRaises(ValueError): IntervalGate('bad',min_cycles=-1)
        with self.assertRaises(ValueError): IntervalGate('bad',min_seconds=float('nan'))

if __name__=='__main__':unittest.main()
