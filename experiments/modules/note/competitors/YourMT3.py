"""Authors' YourMT3+ YPTF.MoE+Multi (noPS), isolated CPU runtime."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from experiments.modules.note.NoteDetectorBase import NoteDetectorBase
from experiments.modules.note.experiments.YourMT3Direct import (
    ASSETS, SOURCE, ROOT, REVISION, CHECKPOINT_SHA256, MODEL_ARGS, sha256,
)


class YourMT3(NoteDetectorBase):
    def detect(self, **kwargs):
        raise NotImplementedError('Select yourmt3 in the note notebook audio-only task interface')

    @staticmethod
    def runtime_provenance():
        python = ASSETS / 'venv/bin/python'
        checkpoint = ASSETS / 'model.ckpt'
        if not python.exists() or not checkpoint.exists() or not (SOURCE / 'model_helper.py').exists():
            raise RuntimeError('Run YourMT3Direct.py setup before requesting yourmt3')
        digest = sha256(checkpoint)
        if digest != CHECKPOINT_SHA256:
            raise ValueError('YourMT3+ checkpoint SHA-256 mismatch')
        paths = [SOURCE / 'model_helper.py', *sorted((SOURCE / 'amt/src').rglob('*.py'))]
        return dict(revision=REVISION, variant='YPTF.MoE+Multi (noPS)',
                    checkpoint_sha256=digest, args=MODEL_ARGS,
                    files={str(p.relative_to(SOURCE)): sha256(p) for p in paths},
                    versions=subprocess.check_output([str(python), '-m', 'pip', 'freeze'], text=True),
                    device='cpu', precision='float32', batch_size=1)

    @classmethod
    def predict_task(cls, task, config):
        python = ASSETS / 'venv/bin/python'
        with tempfile.TemporaryDirectory(prefix='attune-yourmt3-') as folder:
            output = Path(folder) / 'notes.json'
            env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                   'USE_TF': '0', 'WANDB_MODE': 'disabled', 'TORCHDYNAMO_DISABLE': '1'}
            subprocess.run([str(python), str(ROOT / 'experiments/modules/note/experiments/YourMT3Direct.py'),
                            'predict', '--audio', task['audio'], '--output', str(output)],
                           check=True, env=env, stdout=subprocess.DEVNULL)
            result = json.loads(output.read_text())
        return (np.asarray(result['intervals'], float).reshape(-1, 2),
                np.asarray(result['frequencies_hz'], float),
                {'yourmt3_runtime_cpu_seconds': result['cpu_seconds']})
