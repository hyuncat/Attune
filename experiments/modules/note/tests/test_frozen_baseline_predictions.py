"""Pinned baseline predictions survive frontend changes without model calls."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from experiments.modules.note.NoteCache import NoteCache
from experiments.modules.note.NoteBenchmarker import NoteEvaluation, NotebookConfig


class FrozenBaselinePredictionTest(unittest.TestCase):
    def test_rescores_saved_notes_and_rejects_changed_audio(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            audio = root / 'audio.wav'
            audio.write_bytes(b'unchanged audio')
            source, output = root / 'old', root / 'new'
            for run in (source, output):
                for name in ('checkpoints', 'predictions'):
                    (run / name).mkdir(parents=True)
            metadata = {'inputs': {str(audio): hashlib.sha256(audio.read_bytes()).hexdigest()}}
            (source / 'metadata.json').write_text(json.dumps(metadata))
            task = dict(dataset='coco', track_id='test', method='basic-pitch',
                        job_id='job', audio=str(audio))
            row = {**task, 'status': 'ok', 'cpu_seconds': 12., 'f1': 0.}
            (source / 'checkpoints/job.json').write_text(json.dumps(row))
            (source / 'predictions/job.json').write_text(json.dumps(
                {'intervals': [[0., 1.]], 'frequencies_hz': [440.]}))
            index = NoteCache.frozen_prediction_index((source,))
            candidate = index['coco', 'test', 'basic-pitch']
            with patch.object(NoteEvaluation, '_predict', side_effect=AssertionError('No model calls')), patch.object(
                NoteEvaluation, 'reference_notes', return_value=(np.array([[0., 1.]]), np.array([440.]))
            ):
                result = NoteCache.rescore_frozen_prediction(candidate, metadata, task, NotebookConfig(), output)
                self.assertEqual(result['f1'], 1.)
                self.assertEqual(result['cpu_seconds'], 12.)
                self.assertEqual(result['execution_cpu_seconds'], 0.)
                self.assertIsNone(result['cache_key'])
                self.assertTrue(result['inference_provenance'].endswith('old/metadata.json'))
                audio.write_bytes(b'changed audio')
                with self.assertRaisesRegex(ValueError, 'audio changed'):
                    NoteCache.rescore_frozen_prediction(candidate, metadata, task, NotebookConfig(), output)
                with self.assertRaisesRegex(ValueError, 'cannot be used for Attune'):
                    NoteCache.rescore_frozen_prediction(candidate, metadata, {**task, 'method': 'attune'}, NotebookConfig(), output)


if __name__ == '__main__':
    unittest.main()
