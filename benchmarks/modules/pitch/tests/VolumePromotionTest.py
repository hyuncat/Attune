"""Promotion safety checks: synthetic frames and mocked inference only."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from algorithms.Config import Config
from algorithms.PitchDetector import PitchDetector
from benchmarks.modules.pitch.PitchCache import PitchCache
from benchmarks.modules.pitch.PitchNotebook import PitchNotebook
from benchmarks.modules.pitch.tests import PitchNotebookTest as fixtures


class VolumePromotionTest(unittest.TestCase):
    def test_live_gate_matches_causal_percentile_and_resets(self):
        detector = PitchDetector(config=Config())
        volumes = np.r_[100., np.ones(100), .1, 0., .01]
        with patch.object(detector, '_candidates', return_value=[(60., 1.)]), patch.object(
                detector, '_live_value', return_value=(60., .01)):
            for i, volume in enumerate(volumes):
                pitch = detector._pitch_from_observations(None, np.array([.99]),
                                                          np.array([volume]), 0, i)
                expected = 60. if volume >= .04 * np.percentile(volumes[:i + 1], 95) else -1.
                self.assertEqual(pitch.value, expected)
        detector.load_config(Config())
        self.assertEqual(detector._volume_history, [])
        self.assertEqual(Config().min_volume, Config().posthoc_min_volume)

    def test_offline_only_attune_runs_then_reuses(self):
        with tempfile.TemporaryDirectory() as directory:
            notebook = PitchNotebook(PitchNotebook.NotebookConfig(
                methods=('attune', 'praat'), attune_only_refresh=True))
            notebook.runs_root = Path(directory)
            path = notebook.runs_root / 'suite' / 'rows.csv'
            path.parent.mkdir()
            old = fixtures.PitchNotebookTest.rows({'attune': .2, 'praat': .7})
            old['pitch_cache_version'] = 15
            old.to_csv(path, index=False)
            fresh = fixtures.PitchNotebookTest.rows({'attune': .9})
            fresh['pitch_cache_version'] = PitchCache.VERSION
            with patch.object(notebook, '_recover_frozen_rows', side_effect=lambda p,r,*a,**k:r), patch.object(
                    notebook, '_run_methods', return_value=fresh) as run, patch(
                    'benchmarks.modules.pitch.PitchNotebook.display'):
                result = notebook.run_suite('suite', ('dataset',), None)
                self.assertEqual(run.call_args.args[0], ['attune'])
                pd.testing.assert_frame_equal(result.rows[result.rows.model.eq('praat')].reset_index(drop=True),
                                              old[old.model.eq('praat')].reset_index(drop=True))
                run.reset_mock()
                notebook.run_suite('suite', ('dataset',), None)
                run.assert_not_called()

    def test_missing_competitor_stops_before_inference(self):
        notebook = PitchNotebook(PitchNotebook.NotebookConfig(attune_only_refresh=True))
        with self.assertRaisesRegex(RuntimeError, 'without running competitor'):
            notebook._require_frozen_competitors(None, ['attune', 'praat'])

    def test_streaming_guard_requires_latency_sidecars(self):
        notebook = PitchNotebook(PitchNotebook.NotebookConfig(attune_only_refresh=True))
        rows = fixtures.PitchNotebookTest.rows({'attune': .5, 'praat': .7})
        with self.assertRaises(RuntimeError):
            notebook._require_frozen_competitors(rows, ['attune', 'praat'], streaming=True)

    def test_checkpoint_version_only_changes_attune(self):
        from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker
        from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / 'audio.wav'
            audio.touch()
            example = PitchDetectorBase.PitchExample(
                track_id='a', dataset='test', audio_path=audio,
                ref_times=np.array([0.]), ref_freqs=np.array([440.]),
                fmin=200, fmax=600, estimate_cache_dir=root, stage_cache_path=root / 'stage')
            runner = PitchBenchmarker.PitchStreaming(PitchBenchmarker.Options())
            methods = ('attune', 'praat', 'pyin_framewise', 'swiftf0', 'spice')
            before = {m: runner._checkpoint_path(root, m, example) for m in methods}
            with patch.object(PitchCache, 'VERSION', PitchCache.VERSION + 1):
                after = {m: runner._checkpoint_path(root, m, example) for m in methods}
            self.assertNotEqual(before.pop('attune'), after.pop('attune'))
            self.assertEqual(before, after)
    def test_streaming_refresh_runs_only_attune(self):
        from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker
        with tempfile.TemporaryDirectory() as directory:
            notebook = PitchNotebook(PitchNotebook.NotebookConfig(attune_only_refresh=True))
            notebook.runs_root = Path(directory)
            old = fixtures.PitchNotebookTest.rows({'attune': .2, 'praat': .7})
            old['pitch_cache_version'] = 15
            fresh = fixtures.PitchNotebookTest.rows({'attune': .9})
            fresh['pitch_cache_version'] = PitchCache.VERSION
            for frame in (old, fresh):
                frame["instrument"] = "violin"
                for column, value in [('frame_size_samples', 4096), ('integration_size_samples', 2048),
                                      ('hop_size_samples', 128), ('sample_rate_hz', 44100)]:
                    frame[column] = value
            with patch.object(notebook, '_read_run', return_value=old), patch.object(
                    notebook, '_recover_frozen_rows', side_effect=lambda p,r,*a,**k:r), patch.object(
                    PitchCache, 'has_latency_samples', return_value=True), patch.object(
                    PitchBenchmarker, 'tracks', return_value=[]), patch.object(
                    PitchBenchmarker.PitchStreaming, 'run', return_value=fresh) as run, patch(
                    'benchmarks.modules.pitch.PitchNotebook.display'):
                result = notebook.run_streaming_comparison(suite_methods=('attune', 'praat'))
            self.assertEqual(run.call_args.kwargs['methods'], ['attune'])
            self.assertFalse(run.call_args.kwargs['force'])
            pd.testing.assert_frame_equal(result.rows[result.rows.model.eq('praat')].reset_index(drop=True),
                                          old[old.model.eq('praat')].reset_index(drop=True))
