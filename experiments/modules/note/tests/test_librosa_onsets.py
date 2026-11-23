"""Small onset/adapter and no-legacy-inference checks; no corpus runs."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from experiments.modules.note.NoteBenchmarker import NoteEvaluation, NotebookConfig
from experiments.modules.note.NoteCache import NoteCache
from experiments.modules.note.NoteDetectorBase import NoteDetectorBase
from experiments.modules.note.competitors.LibrosaFlux import LibrosaFlux
from experiments.modules.note.competitors.LibrosaSuperFlux import LibrosaSuperFlux


class LibrosaOnsetsTest(unittest.TestCase):
    def test_voiced_runs_onsets_offsets_and_silence(self):
        f0 = np.array([440., 440., 440., 440., np.nan, 220., 220.])
        iv, hz = LibrosaFlux.decode(np.array([.2, .45]), f0, np.isfinite(f0), .1, .67)
        np.testing.assert_allclose(iv, [[0, .2], [.2, .4], [.5, .67]])
        np.testing.assert_allclose(hz, [440., 440., 220.])
        iv, hz = LibrosaFlux.decode(np.array([]), f0, np.zeros(7, bool), .1, .7)
        self.assertEqual(iv.shape, (0, 2))
        self.assertEqual(len(hz), 0)

    def test_dispatch_and_real_onset_extraction_on_tiny_audio(self):
        sr = 44100
        y = np.zeros(sr)
        rng = np.random.default_rng(1)
        y[11025:11466] = rng.normal(size=441)
        for name, cls in [('librosa-flux', LibrosaFlux), ('librosa-superflux', LibrosaSuperFlux)]:
            self.assertIs(NoteDetectorBase.competitor(name), cls)
            onsets = cls.onset_times(y, sr)
            self.assertTrue(np.any(np.abs(onsets - .25) < .05), onsets)
            self.assertEqual(len(cls.onset_times(np.zeros(sr), sr)), 0)

    def test_cache_only_miss_cannot_start_workers(self):
        config = NotebookConfig(methods=('tony', 'librosa-flux'), cache_only_methods=('tony',))
        runner = NoteEvaluation(config)
        with tempfile.TemporaryDirectory() as directory:
            runner.runs_root = Path(directory)
            with patch.object(runner, 'select_tracks', return_value=[dict(dataset='coco', track_id='tiny')]), patch.object(
                runner, 'provenance', return_value={}
            ), patch.object(NoteCache, 'checkpoint_index', return_value={}), patch.object(
                NoteCache, 'job_key', return_value='missing'
            ), patch('multiprocessing.get_context', side_effect=AssertionError('No workers allowed')):
                with self.assertRaisesRegex(RuntimeError, 'Cache-only method tony'):
                    runner.run_preliminary('coco')
        with self.assertRaises(ValueError):
            NotebookConfig(force=True, cache_only_methods=('tony',))

    def test_notebook_discovers_caches_without_forcing_methods(self):
        path = Path(__file__).resolve().parents[3] / 'notebooks/note.ipynb'
        notebook = json.loads(path.read_text())
        source = ''.join(notebook['cells'][1]['source'])
        self.assertNotIn("cache_only_methods=", source)
        self.assertNotIn("BASELINE_PREDICTION_RUNS", source)
        self.assertIn("plan_preliminary", source)
        self.assertIn('force_methods=()', source)
        self.assertIn('librosa-superflux', NotebookConfig().methods)
