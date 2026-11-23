"""Official MT3, in a separate CPU JAX runtime; no unofficial model port."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from experiments.modules.note.NoteDetectorBase import NoteDetectorBase
from experiments.modules.note.experiments.MT3Direct import ASSETS, ROOT


class MT3(NoteDetectorBase):
    def detect(self, **kwargs):
        raise NotImplementedError(
            "MT3 uses the audio-only notebook task interface; select mt3 in NotebookConfig.methods"
        )

    @staticmethod
    def runtime_provenance():
        import hashlib
        python = ASSETS / 'venv/bin/python'
        paths = [ASSETS / 'sources.json', ASSETS / 'checkpoint/provenance.json',
                 ASSETS / 'checkpoint/params.npz']
        if not python.exists() or not all(p.exists() for p in paths):
            raise RuntimeError('Set up official MT3 with experiments/experiments/mt3-port.ipynb first')
        manifest = json.loads((ASSETS / 'sources.json').read_text())
        paths.extend(ASSETS / 'runtime' / name for name in manifest['files'])
        versions = subprocess.check_output([str(python), '-m', 'pip', 'freeze'], text=True)
        return dict(files={str(p.relative_to(ASSETS)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                    versions=versions, device='cpu')

    @classmethod
    def predict_task(cls, task, config):
        python = ASSETS / 'venv/bin/python'
        if not python.exists() or not (ASSETS / 'checkpoint/params.npz').exists():
            raise RuntimeError('Run mt3-port.ipynb setup before requesting MT3 inference')
        with tempfile.TemporaryDirectory(prefix='attune-mt3-') as folder:
            output = Path(folder) / 'notes.json'
            env = {**os.environ, 'JAX_PLATFORMS': 'cpu', 'OMP_NUM_THREADS': '1',
                   'TF_NUM_INTRAOP_THREADS': '1', 'TF_NUM_INTEROP_THREADS': '1'}
            subprocess.run([str(python), str(ROOT / 'experiments/modules/note/experiments/MT3Direct.py'),
                            'predict', '--audio', task['audio'], '--output', str(output)],
                           check=True, env=env, stdout=subprocess.DEVNULL)
            result = json.loads(output.read_text())
        return (np.asarray(result['intervals'], float).reshape(-1, 2),
                np.asarray(result['frequencies_hz'], float),
                {'mt3_runtime_cpu_seconds': result['cpu_seconds']})
