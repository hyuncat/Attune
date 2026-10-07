"""Track checkpoints survive failures and preserve the existing notebook reports."""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from benchmarks.modules.vibrato.VibratoBenchmarker import VibratoBenchmarker
from benchmarks.modules.vibrato.datasets.SyntheticDataset import SyntheticDataset
from benchmarks.modules.vibrato.tests.VibratoBenchmarkerTest import _PreparedFixtureEstimator


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        example = SyntheticDataset.build()[0]
        self.examples = [replace(example, case_id=name, metadata={}) for name in ('a', 'b')]
        self.bench = VibratoBenchmarker()

    def run_cached(self, directory, **kwargs):
        return self.bench.run(self.examples, [_PreparedFixtureEstimator('fixture')],
                              cache_dir=Path(directory), strict=True, **kwargs)

    def test_interrupted_run_resumes_and_corrupt_cache_is_rebuilt(self):
        original = VibratoBenchmarker._compute_detector_group_job
        calls = []

        def interrupt(bench, estimator, group, strict, **kwargs):
            calls.append(group[0].case_id)
            if group[0].case_id == 'b':
                raise KeyboardInterrupt()
            return original(bench, estimator, group, strict, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(VibratoBenchmarker, '_compute_detector_group_job', side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    self.run_cached(directory)
            self.assertEqual(len(list(Path(directory).glob('*.pkl.xz'))), 1)
            with patch.object(VibratoBenchmarker, '_compute_detector_group_job', wraps=original) as compute:
                raw = self.run_cached(directory)
                self.assertEqual(compute.call_count, 1)
                self.assertEqual(compute.call_args.args[2][0].case_id, 'b')
            with patch.object(VibratoBenchmarker, '_compute_detector_group_job', side_effect=AssertionError('cache miss')):
                replay = self.run_cached(directory)
            pd.testing.assert_frame_equal(raw, replay)
            next(Path(directory).glob('*.pkl.xz')).write_bytes(b'truncated')
            with patch.object(VibratoBenchmarker, '_compute_detector_group_job', wraps=original) as compute:
                self.run_cached(directory)
                self.assertEqual(compute.call_count, 1)

    def test_spawned_workers_persist_reusable_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = self.run_cached(directory, workers=2)
            self.assertEqual(len(list(Path(directory).glob('*.pkl.xz'))), 2)
            with patch.object(VibratoBenchmarker, '_compute_detector_group_job', side_effect=AssertionError('cache miss')):
                replay = self.run_cached(directory)
            pd.testing.assert_frame_equal(raw, replay)

    def test_changed_settings_and_inputs_invalidate(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_cached(directory)
            original = VibratoBenchmarker._compute_detector_group_job
            self.bench.rate_accuracy_tolerance_hz = 0.75
            with patch.object(VibratoBenchmarker, '_compute_detector_group_job', wraps=original) as compute:
                self.run_cached(directory)
                self.assertEqual(compute.call_count, 2)
            self.examples[0] = replace(self.examples[0], pitch_midi=self.examples[0].pitch_midi + 1)
            with patch.object(VibratoBenchmarker, '_compute_detector_group_job', wraps=original) as compute:
                self.run_cached(directory)
                self.assertEqual(compute.call_count, 1)

    def test_failed_jobs_are_not_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(_PreparedFixtureEstimator, 'estimate', side_effect=ValueError('temporary failure')):
                raw = self.bench.run(self.examples, [_PreparedFixtureEstimator('fixture')], cache_dir=Path(directory))
            self.assertTrue(raw.error.astype(bool).all())
            self.assertFalse(list(Path(directory).glob('*.pkl.xz')))
            self.run_cached(directory)
            self.assertEqual(len(list(Path(directory).glob('*.pkl.xz'))), 2)

    def test_coco_cli_writes_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(VibratoBenchmarker, '_load_examples', return_value=(self.examples, 'fixture')), \
                 patch.object(VibratoBenchmarker, '_selected_detectors', return_value=[_PreparedFixtureEstimator('fixture')]):
                status = VibratoBenchmarker.main(['--coco', '--output-dir', directory, '--workers', '1', '--quiet', '--no-console-summary'])
            self.assertEqual(status, 0)
            manifest = json.loads((Path(directory) / 'run_config.json').read_text())
            self.assertTrue(manifest['dataset']['coco_track_selection_policy'])
            self.assertTrue(manifest['dataset']['coco_profile_parameter_sampler'])



if __name__ == '__main__':
    unittest.main()
