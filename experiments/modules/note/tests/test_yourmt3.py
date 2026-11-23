"""YourMT3+ integration contracts; these tests never execute a neural model."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from experiments.modules.note.NoteBenchmarker import NotebookConfig
from experiments.modules.note.NoteDetectorBase import NoteDetectorBase
from experiments.modules.note.NoteCache import NoteCache
from experiments.modules.note.competitors.YourMT3 import YourMT3
from experiments.modules.note.tests import test_note_result_cache as cache_tests


class YourMT3Test(unittest.TestCase):
    def test_default_registration(self):
        self.assertIn('yourmt3', NotebookConfig().methods)
        self.assertIs(NoteDetectorBase.competitor('yourmt3'), YourMT3)

    def test_subprocess_keeps_polyphony_and_child_timing(self):
        def fake_run(command, **kwargs):
            self.assertEqual(kwargs['env']['USE_TF'], '0')
            self.assertEqual(kwargs['env']['OMP_NUM_THREADS'], '1')
            Path(command[command.index('--output') + 1]).write_text(json.dumps(dict(
                intervals=[[0, 1], [0, 2]], frequencies_hz=[440, 660], cpu_seconds=2.5)))
        with patch('experiments.modules.note.competitors.YourMT3.subprocess.run', side_effect=fake_run):
            intervals, hz, timings = YourMT3.predict_task({'audio': 'unused.wav'}, NotebookConfig())
        np.testing.assert_array_equal(intervals, [[0, 1], [0, 2]])
        np.testing.assert_array_equal(hz, [440, 660])
        self.assertEqual(timings['yourmt3_runtime_cpu_seconds'], 2.5)

    def test_new_model_changes_invalidate_only_its_cache(self):
        tracks, provenance = cache_tests.NoteResultCacheTest().fixture()
        sources = provenance['sources']
        sources.pop('experiments/modules/note/NoteDetectionBaselines.py')
        for path in ('NoteDetectorBase', 'NoteCache', 'competitors/BasicPitch',
                     'competitors/YourMT3', 'experiments/YourMT3Direct'):
            sources[f'experiments/modules/note/{path}.py'] = 'same'
        provenance['yourmt3_runtime'] = {'checkpoint': 'official', 'versions': 'pinned'}
        task = {**tracks[0], 'method': 'yourmt3'}
        old_task = {**tracks[0], 'method': 'basic-pitch'}
        new_key = NoteCache.job_key(provenance, task)
        old_key = NoteCache.job_key(provenance, old_task)
        self.assertIsNotNone(new_key)
        changed = copy.deepcopy(provenance)
        changed['config']['methods'] += ['yourmt3']
        self.assertEqual(old_key, NoteCache.job_key(changed, old_task))
        for dependency in ('competitors/YourMT3', 'experiments/YourMT3Direct'):
            changed = copy.deepcopy(provenance)
            changed['sources'][f'experiments/modules/note/{dependency}.py'] = 'changed'
            self.assertNotEqual(new_key, NoteCache.job_key(changed, task))
            self.assertEqual(old_key, NoteCache.job_key(changed, old_task))
        changed = copy.deepcopy(provenance)
        changed['yourmt3_runtime']['checkpoint'] = 'changed'
        self.assertNotEqual(new_key, NoteCache.job_key(changed, task))
        self.assertEqual(old_key, NoteCache.job_key(changed, old_task))
        changed.pop('yourmt3_runtime')
        self.assertIsNone(NoteCache.job_key(changed, task))
