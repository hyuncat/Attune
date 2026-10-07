"""Synthetic gate checks; no dataset scoring or inference."""
import unittest

class FrameGateChecks(unittest.TestCase):
    def test_frame_gates(self):
        # Synthetic plumbing checks only; no audio inference or benchmark scoring.
        from types import SimpleNamespace
        import numpy as np
        from benchmarks.modules.vibrato.VibratoDetectorBase import VibratoEstimate
        from benchmarks.modules.vibrato.tests.FrameGates import (
            DEFAULT_GATES, FrameGate, apply_gate, load_score_examples, run_frame_gates,
        )

        _example = SimpleNamespace(times=np.arange(8), pitch_midi=np.array([60., 60., 60., 60., 60., 60., np.nan, 60.]))
        _estimate = VibratoEstimate(
            rate_hz=np.array([4., 9., 3.99, 6., 6., 6., 6., 6.]),
            width_cents=np.array([20., 20., 40., 19.99, 40., 40., 40., 40.]),
            detected=np.array([True, True, True, True, False, True, True, True]),
            quality=np.array([.30, .5, .9, .9, .9, .299, .9, np.nan]),
        )
        _gate = FrameGate('check', 4., 9., 20., .30)
        _gated = apply_gate(_example, _estimate, _gate)
        np.testing.assert_array_equal(_gated.detected, [True, True, False, False, False, False, False, False])
        np.testing.assert_array_equal(_estimate.detected, [True, True, True, True, False, True, True, True])
        np.testing.assert_array_equal(_gated.rate_hz, _estimate.rate_hz)
        np.testing.assert_array_equal(_gated.width_cents, _estimate.width_cents)
        # Labels are never read by the gate.
        _example.is_vibrato = np.ones(8, dtype=bool)
        np.testing.assert_array_equal(apply_gate(_example, _estimate, _gate).detected, _gated.detected)
        # Quality is optional for rate/width-only gates; missing quality must not pass a quality gate.
        _no_quality = VibratoEstimate(_estimate.rate_hz, _estimate.width_cents, _estimate.detected)
        assert apply_gate(_example, _no_quality, DEFAULT_GATES[1]).detected[0]
        try:
            apply_gate(_example, _no_quality, _gate)
        except ValueError:
            pass
        else:
            raise AssertionError('Missing quality should fail explicitly')
        _empty = VibratoEstimate(np.zeros(8), np.zeros(8), np.zeros(8, dtype=bool))
        assert not apply_gate(_example, _empty, _gate).detected.any()
        print('Frame-gate checks passed: thresholds, voicing, quality, immutable curves, and no label dependence.')


if __name__ == "__main__":
    unittest.main()
