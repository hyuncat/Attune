"""Check the notebook facade and paired suite wiring without audio inference."""
import ast
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from benchmarks.modules.vibrato.VibratoBenchmarker import VibratoBenchmarker
from benchmarks.modules.vibrato.VibratoNotebook import VibratoNotebook
from benchmarks.modules.vibrato.competitors.Attune import Attune, GatedAttune
from benchmarks.modules.vibrato.VibratoDetectorBase import VibratoEstimate
from benchmarks.modules.vibrato.datasets.SyntheticDataset import SyntheticDataset
from benchmarks.modules.vibrato.tests.FrameGates import FrameGate, apply_gate


class NotebookTest(unittest.TestCase):
    def test_core_has_no_floating_functions(self):
        root = Path(__file__).resolve().parents[1]
        for name in ('VibratoBenchmarker', 'VibratoNotebook', 'VibratoDetectorBase'):
            tree = ast.parse((root / f'{name}.py').read_text())
            self.assertFalse(any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                                 for n in tree.body), name)

    def test_suite_dispatch_and_saved_manifest(self):
        example = SyntheticDataset.build(replicates=1)[0]
        example.metadata.update(analysis_note_bounds=[(0., 1., 60.)],
                                note_boundary_source='fixture')
        for suite in ('yang', 'coco'):
            with self.subTest(suite=suite), tempfile.TemporaryDirectory() as directory:
                with patch.object(VibratoNotebook, f'{suite}_examples', return_value=[example]) as loader, \
                     patch.object(VibratoBenchmarker, 'run', return_value=pd.DataFrame()) as run, \
                     patch.object(VibratoBenchmarker, 'write_reports'), \
                     patch.object(VibratoBenchmarker, 'summarize', return_value=pd.DataFrame()):
                    _, output = VibratoNotebook.run_suite(suite, root=directory, workers=1)
                loader.assert_called_once()
                methods = run.call_args.args[1]
                self.assertEqual(sum(d.name == 'attune' for d in methods), 1)
                self.assertTrue(any(isinstance(d, GatedAttune) for d in methods))
                self.assertNotIn('herrera_bonada_yang_window', [d.name for d in methods])
                self.assertTrue((output / 'run_config.json').is_file())
                self.assertTrue((output / 'note_bounds.json').is_file())
                self.assertEqual(run.call_args.kwargs['cache_dir'], output / 'checkpoints')

    def test_gate_matches_existing_protocol(self):
        example = SyntheticDataset.build(replicates=1)[0]
        size = len(example.times)
        estimate = VibratoEstimate(np.resize([3., 4., 9., 10., np.nan], size),
                                   np.resize([19., 20., 21.], size), np.ones(size, dtype=bool))
        expected = apply_gate(example, estimate, FrameGate(**GatedAttune.frame_gate()))
        with patch.object(Attune, 'estimate', return_value=estimate):
            actual = GatedAttune().estimate(example)
        np.testing.assert_array_equal(actual.detected, expected.detected)
        np.testing.assert_array_equal(actual.rate_hz, expected.rate_hz)
        self.assertEqual(actual.metadata, expected.metadata)
