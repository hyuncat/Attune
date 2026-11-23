"""Model slot reuse, tail scheduling, and independent source selection."""
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.modules.mistake.MistakeBenchmarker import MistakeBenchmarker
from experiments.modules.mistake.datasets.NativeDatasets import digest


class AudioSchedulingTests(unittest.TestCase):
    def test_model_families_run_sequentially_with_full_parallel_pool(self):
        clients, results = [], []
        barriers = {m: threading.Barrier(2) for m in ('PolyTune', 'LadderSym')}
        lock = threading.Lock()

        class Model:
            def __init__(self, config, directory, *, cpu_threads=1):
                self.method, self.closed, self.threads = config.name, False, cpu_threads
                with lock:
                    assert all(c.closed or c.method == self.method for c in clients)
                    if self.method == 'LadderSym':
                        assert len(results) == 4
                    clients.append(self)

            def predict(self, task):
                barriers[self.method].wait(timeout=10)
                return task

            def close(self):
                self.closed = True

        tasks = [dict(method=m, index=i) for m in ('LadderSym', 'PolyTune') for i in range(4)]
        with tempfile.TemporaryDirectory() as tmp, patch('os.cpu_count', return_value=10), patch(
            'experiments.modules.mistake.competitors.PolyTune.PolyTune.Client', Model
        ), patch('experiments.modules.mistake.competitors.LadderSym.LadderSym.Client', Model), patch.object(
            MistakeBenchmarker, 'native_evaluate', side_effect=lambda task, client: client.predict(task)
        ):
            stats = MistakeBenchmarker.run_neural_jobs(
                [(t['method'], (t,), None) for t in tasks], output=tmp,
                polytune=SimpleNamespace(name='PolyTune'), laddersym=SimpleNamespace(name='LadderSym'),
                workers=2, native=True, on_result=lambda key, payload: results.append(payload),
            )
        self.assertEqual([r['method'] for r in results], ['PolyTune'] * 4 + ['LadderSym'] * 4)
        self.assertEqual(len(clients), 4)
        self.assertTrue(all(c.closed and c.threads == 4 for c in clients))
        self.assertEqual(stats['neural_workers'], {'PolyTune': 2, 'LadderSym': 2})

    def test_distinct_piece_cohort_is_deterministic_and_rejects_shortfall(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            midi = root / 'fixture.mid'
            midi.write_bytes(b'fixture')
            cases = [dict(track_id=f'{piece}/{instrument}', group=str(piece), instrument=instrument,
                          files={'score': str(midi)}, hashes={'score': digest(midi)})
                     for piece in range(10) for instrument in ('flute', 'violin')]
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps(dict(cases=cases)))
            selected = MistakeBenchmarker.audio_cohort(manifest, 8, distinct_pieces=True)
            self.assertEqual(len({c['group'] for c in selected}), 8)
            manifest.write_text(json.dumps(dict(cases=list(reversed(cases)))))
            self.assertEqual(selected, MistakeBenchmarker.audio_cohort(manifest, 8, distinct_pieces=True))
            with self.assertRaisesRegex(ValueError, 'Require 11'):
                MistakeBenchmarker.audio_cohort(manifest, 11, distinct_pieces=True)


if __name__ == '__main__':
    unittest.main()
