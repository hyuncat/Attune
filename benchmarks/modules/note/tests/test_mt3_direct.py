"""Cheap dispatch/config/subprocess contracts; no model or corpus inference."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from benchmarks.modules.note.NoteBenchmarker import NotebookConfig
from benchmarks.modules.note.NoteDetectorBase import NoteDetectorBase
from benchmarks.modules.note.competitors.MT3 import MT3


class MT3DirectTest(unittest.TestCase):
    def test_default_registration(self):
        self.assertIn('mt3', NotebookConfig().methods)
        self.assertEqual(NotebookConfig(methods=('mt3',)).methods, ('mt3',))
        self.assertIs(NoteDetectorBase.competitor('mt3'), MT3)

    def test_subprocess_retains_polyphony_and_reports_child_cpu(self):
        def fake_run(command, **kwargs):
            self.assertEqual(kwargs['env']['JAX_PLATFORMS'], 'cpu')
            self.assertIn('predict', command)
            out = Path(command[command.index('--output') + 1])
            out.write_text(json.dumps(dict(intervals=[[0., 1.], [0., 2.]],
                                           frequencies_hz=[440., 660.], cpu_seconds=1.25)))
        with patch.object(Path, 'exists', return_value=True), patch(
            'benchmarks.modules.note.competitors.MT3.subprocess.run', side_effect=fake_run
        ):
            intervals, hz, timings = MT3.predict_task({'audio': '/tmp/example.wav'}, NotebookConfig())
        np.testing.assert_allclose(intervals, [[0., 1.], [0., 2.]])
        np.testing.assert_allclose(hz, [440., 660.])
        self.assertEqual(timings['mt3_runtime_cpu_seconds'], 1.25)

    def test_notebook_is_bounded_by_default(self):
        root = Path(__file__).resolve().parents[4]
        notebook = json.loads((root / 'benchmarks/notebooks/mt3-port.ipynb').read_text())
        code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
        compile(code, 'mt3-port.ipynb', 'exec')
        self.assertIn('RUN_CLIP = False', code)
        self.assertNotIn('run_preliminary', code)
