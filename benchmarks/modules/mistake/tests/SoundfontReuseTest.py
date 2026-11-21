"""Reuse neural inference across output folders, retaining input validation."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from benchmarks.modules.mistake.MistakeCache import MistakeCache
from benchmarks.modules.mistake.datasets.NativeDatasets import digest


class SoundfontReuseTest(unittest.TestCase):
    def test_frontend_change_reuses_but_changed_input_or_model_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            files = {}
            for name in ('performance', 'score_audio', 'score_midi'):
                p = root / name
                p.write_bytes(name.encode())
                files[name] = str(p)
            case = dict(case_id='case', files=files, hashes={k: digest(v) for k,v in files.items()})
            contract = dict(source={'device': 'cpu'}, models={'PolyTune': 'weights'},
                            intervention={'cases': [case]}, code={
                                'benchmarks/modules/mistake/competitors/PolyTune.py': 'adapter'})
            (root / 'run.json').write_text(json.dumps({'contract': contract}))
            checkpoint = root / 'cases/case/polytune/result.json'
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text(json.dumps({'prediction': {'events': [], 'cpu_seconds': 4.}}))
            current = copy.deepcopy(contract)
            current['defaults'] = {'posthoc_unv_thresh': .9875}
            prediction = MistakeCache.reusable_soundfont_prediction(root, current, case, 'PolyTune')
            self.assertTrue(prediction['inference_cache_hit'])
            self.assertEqual(prediction['cpu_seconds'], 4.)
            self.assertIsNone(MistakeCache.reusable_soundfont_prediction(root, current, case, 'Attune'))
            current['models']['PolyTune'] = 'different weights'
            with self.assertRaisesRegex(ValueError, 'model identity changed'):
                MistakeCache.reusable_soundfont_prediction(root, current, case, 'PolyTune')
            current['models']['PolyTune'] = 'weights'
            Path(files['performance']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed performance'):
                MistakeCache.reusable_soundfont_prediction(root, current, case, 'PolyTune')


if __name__ == '__main__':
    unittest.main()
