"""Materialize the exact frozen COCO-E2 evaluation or tuning performances.

Run `python experiments/datasets/coco-e2/materialize.py benchmark --check`
(or `sweeps`). Omit --check to copy verified assets into this directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'app.py').is_file())
sys.path.insert(0, str(ROOT))
from experiments.paths import relocated_path

HERE = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def materialize(cohort, *, check=False, regenerate=False):
    """Copy frozen assets, or regenerate missing cases and require identical hashes."""
    cases = json.loads((HERE / f'{cohort}-tracks.json').read_text())['cases']
    for case in cases:
        sources = {key: ROOT / value['path'] for key, value in case['files'].items()}
        if regenerate and any(not p.is_file() for p in sources.values()):
            from experiments.modules.mistake.datasets.MistakeCases import prepare_case
            score = sources['score']
            if not score.is_file():
                raise FileNotFoundError(f'Extract the frozen CocoChorales source first: {score}')
            _, asset = prepare_case(score, case['seed'], case['rate'],
                                    HERE / cohort / '_generated',
                                    source_info=case['spec']['source_info'],
                                    prepare_audio=True, prepare_pitches=False)
            sources.update({key: relocated_path(asset[key]) for key in ('audio', 'midi', 'truth')})
        for key, source in sources.items():
            expected = case['files'][key]['sha256']
            if not source.is_file():
                raise FileNotFoundError(f'Missing frozen asset: {source}. Restore the experiment cache or use --regenerate.')
            if sha256(source) != expected:
                raise ValueError(f'Frozen {cohort} asset differs: {source}')
            if not check:
                destination = HERE / cohort / case['case_id'] / (key + ''.join(source.suffixes))
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    if sha256(destination) != expected:
                        raise ValueError(f'Refusing to overwrite a different asset: {destination}')
                else:
                    shutil.copy2(source, destination)
    print(f'{cohort}: verified {len(cases)} performances' + ('' if check else f'; materialized under {HERE / cohort}'))
    return len(cases)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cohort', choices=('benchmark', 'sweeps'))
    parser.add_argument('--check', action='store_true', help='Verify without copying')
    parser.add_argument('--regenerate', action='store_true', help='Generate missing assets; reject non-identical outputs')
    args = parser.parse_args()
    materialize(args.cohort, check=args.check, regenerate=args.regenerate)
