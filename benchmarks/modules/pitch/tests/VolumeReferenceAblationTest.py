"""Run with python -m unittest benchmarks.modules.pitch.tests.VolumeReferenceAblationTest."""
import unittest
from types import SimpleNamespace
import numpy as np
from algorithms.Config import Config
from algorithms.PitchSmoother import VoicingSmoother
from benchmarks.modules.pitch.sweeps.VolumeReferenceAblation import volume_thresholds, controller_for


class VolumeReferenceAblationTest(unittest.TestCase):
    def test_low_percentile_cutoffs_and_ties(self):
        config = Config()
        volumes = np.arange(101, dtype=float)
        for q in range(1, 5):
            floors = volume_thresholds(volumes, f'percentile_{q}', config)
            np.testing.assert_allclose(floors, q)
            pitches = [SimpleNamespace(volume=v, unvoiced_prob=0.1,
                                       candidate_pitches=[(60, 1)]) for v in volumes]
            mask = controller_for(f'percentile_{q}')(config=config).decode(pitches)
            np.testing.assert_array_equal(mask, volumes >= q)
        # Quietest-percentile gating must not reject a quota when RMS ties.
        pitches = [SimpleNamespace(volume=0., unvoiced_prob=0.1,
                                   candidate_pitches=[(60, 1)]) for _ in range(100)]
        self.assertTrue(controller_for('percentile_4')(config=config).decode(pitches).all())

    def test_equivalence_requires_recording_specific_alpha(self):
        config = Config()
        config.min_volume = .03  # Historical comparison coefficient, not a default assertion.
        volumes = np.r_[np.ones(100), 10.0]
        current = volume_thresholds(volumes, 'current_p95', config)
        proposed = volume_thresholds(volumes, 'global_max_live_alpha', config)
        np.testing.assert_allclose(current, .04)
        np.testing.assert_allclose(proposed, .3)
        self.assertNotEqual(current[0], proposed[0])
        config.min_volume = config.posthoc_min_volume * np.percentile(volumes, 95) / max(volumes)
        np.testing.assert_allclose(volume_thresholds(volumes, 'global_max_live_alpha', config), current)

    def test_running_max_has_no_future_peak_leakage(self):
        config = Config()
        config.min_volume = .03
        floors = volume_thresholds([1, 2, 100], 'running_max_live_alpha', config)
        np.testing.assert_allclose(floors, [.03, .06, 3])

    def test_controller_changes_only_volume_in_both_passes(self):
        config = Config()
        pitches = [SimpleNamespace(volume=v, unvoiced_prob=u, candidate_pitches=c)
                   for v, u, c in [(1, .1, [(60, 1)]), (.02, .1, [(60, 1)]),
                                    (10, .99, [(60, 1)]), (1, .1, []), (.1, .988, [(60, 1)])]]
        pitches.insert(2, None)
        for threshold in [config.posthoc_unv_thresh, config.pitch_tracking_unv_thresh]:
            for rule in ['global_max_live_alpha', 'running_max_live_alpha', 'global_max_post_alpha']:
                present = [p for p in pitches if p is not None]
                floors = iter(volume_thresholds([p.volume for p in present], rule, config))
                expected = [False if p is None else bool(p.volume >= next(floors) and p.candidate_pitches and p.unvoiced_prob < threshold) for p in pitches]
                actual = controller_for(rule)(config=config).decode(pitches, unvoiced_threshold=threshold)
                np.testing.assert_array_equal(actual, expected)
        self.assertIs(controller_for('current_p95'), VoicingSmoother)

    def test_summary_pairs_tracks_instead_of_row_positions(self):
        import pandas as pd
        from benchmarks.modules.pitch.PitchBenchmarker import COUNT_COLUMNS
        from benchmarks.modules.pitch.sweeps.VolumeReferenceAblation import summarize
        rows = []
        for track, variant, oa in [('b', 'global_max_live_alpha', .7),
                                   ('a', 'current_p95', .9),
                                   ('b', 'current_p95', .8),
                                   ('a', 'global_max_live_alpha', .95)]:
            row = dict.fromkeys(COUNT_COLUMNS, 100)
            row.update(dataset='coco', track_id=track, variant=variant,
                       overall_correct_frames=100 * oa, **{'Overall Accuracy': oa})
            rows.append(row)
        summary = summarize(pd.DataFrame(rows))
        proposed = summary[(summary.Group == 'All') & (summary.variant == 'global_max_live_alpha')].iloc[0]
        self.assertAlmostEqual(proposed['Median paired OA delta (pp)'], -2.5)
        self.assertAlmostEqual(proposed['Pooled OA delta (pp)'], -2.5)
