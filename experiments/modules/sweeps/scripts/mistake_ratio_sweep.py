"""Coco-E ratio/rho experiment; production defaults and old reports stay immutable.

Implement alpha*rho through the existing configurable gap intercepts, so the
same detector and repeat-recovery deletion fees see the experimental cost.
"""
from fractions import Fraction
from math import gcd, isfinite, lcm
from pathlib import Path
import os
import shutil

from experiments.modules.sweeps import AppendixSweeps as base
from experiments.modules.sweeps.MixedCohort import selected_frontend

RHO_VALUES = (.25, .5, 1., 1.25, 2., 3., 5.)
MODEL = 'pitch_scaled_shared_gap_rho_v1'


def unique_ratios(axes=None):
    axes = base.MISTAKE_AXES if axes is None else axes
    ratios = set()
    for a in axes['alignment_gamma_pitch']:
        for b in axes['alignment_gamma_time']:
            if not isfinite(a) or not isfinite(b) or a < 0 or b < 0:
                raise ValueError('Weights must be finite and nonnegative')
            if a == b == 0:
                continue
            x, y = Fraction(str(a)), Fraction(str(b))
            denominator = lcm(x.denominator, y.denominator)
            x, y = int(x * denominator), int(y * denominator)
            divisor = gcd(x, y)
            ratios.add((x // divisor, y // divisor))
    return sorted(ratios, key=lambda pair: float('inf') if pair[1] == 0 else pair[0]/pair[1])


def variants(ratios=None, rhos=RHO_VALUES):
    ratios = unique_ratios() if ratios is None else unique_ratios_from_pairs(ratios)
    if not rhos or any(not isfinite(r) or r < 0 for r in rhos):
        raise ValueError('Supply finite nonnegative rho values')
    result = []
    for alpha, beta in ratios:
        # At alpha=0 every rho gives the identical cost; evaluate only once.
        for rho in ([0.] if alpha == 0 else sorted(set(rhos))):
            params = dict(alignment_gamma_pitch=float(alpha), alignment_gamma_time=float(beta),
                          ins_cost=float(alpha*rho), del_cost=float(alpha*rho))
            result.append(dict(variant=base.digest(dict(model=MODEL, params=params))[:16],
                               params=params, ratio=f'{alpha}:{beta}', rho=float(rho), model=MODEL))
    return result


def unique_ratios_from_pairs(pairs):
    if not pairs:
        raise ValueError('Supply at least one ratio')
    result = set()
    for a, b in pairs:
        result.update(unique_ratios(dict(alignment_gamma_pitch=[a], alignment_gamma_time=[b])))
    if not result:
        raise ValueError('The 0:0 ratio is undefined')
    return sorted(result, key=lambda pair: float('inf') if pair[1] == 0 else pair[0]/pair[1])


def parameter_table(candidates):
    import pandas as pd
    return pd.DataFrame([{'Pitch:Timing': v['ratio'], 'Rho (semitones)': v['rho'],
                          'Pitch Weight': v['params']['alignment_gamma_pitch'],
                          'Timing Weight': v['params']['alignment_gamma_time'],
                          'Gap intercept (alpha*rho)': v['params']['ins_cost']}
                         for v in candidates])


def prepare(output, candidates, seeds=(0, 1)):
    """Resolve exact old frontend contract; never search for a plausible winner."""
    from algorithms.Config import Config
    from experiments.modules.note.NoteBenchmarker import NoteBenchmarker
    from dataclasses import asdict
    output = Path(output).resolve()
    saved = base.read(output / 'cohort.json')
    items = [i for i in base.inputs(saved) if i.get('dataset', 'coco') == 'coco']
    frontend, upstream = selected_frontend(output)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Supply distinct injection seeds')
    source_contract = dict(cohort=saved, items=items, frontend=frontend, upstream=upstream,
        defaults=asdict(Config()), seeds=list(seeds), identity=base.code_identity(),
        soundfont_sha256=base.file_hash(NoteBenchmarker.SOUNDFONT_PATH),
        protocol='E2 rate .25; equal five edit types; one clean control; event gate .1 s/.5 semitone')
    source = output / 'mistake' / base.digest(source_contract)[:16]
    if not (source/'contract.json').exists() or base.read(source/'contract.json') != source_contract:
        raise RuntimeError('Complete the linked Coco-E frontend first; no compatible source contract exists')
    # Verify all 585 initial merged-note checkpoints before scheduling any work.
    checkpoints = []
    for item in items:
        for seed, rate in [(0, 0.)] + [(s, .25) for s in seeds]:
            relative = Path('frontend') / item['track_id'] / f'{seed}_{rate}'
            folder = source / relative
            provenance = base.read(folder/'provenance.json')
            path = folder/'notes.pkl'
            if base.file_hash(path) != provenance['notes_sha256']:
                raise ValueError(f'Frontend cache changed: {path}')
            checkpoints.append(dict(relative=str(relative), sha256=provenance['notes_sha256']))
    contract = dict(source_contract_sha256=base.digest(source_contract),
                    source=str(source), checkpoints=checkpoints,
                    model=MODEL, runner_sha256=base.file_hash(__file__),
                    normalization='coprime nonnegative integer pitch:time weights',
                    rho_policy='shared semitone penalty; alpha=0 evaluated once',
                    variants=candidates, items=items, frontend=frontend, seeds=list(seeds))
    directory = output / 'mistake_ratio_rho' / base.digest(contract)[:16]
    base.contract(directory, contract)
    # Link immutable initial notes into an isolated run; never reuse old predictions.
    for entry in checkpoints:
        relative = Path(entry['relative'])
        for name in ('notes.pkl', 'provenance.json'):
            src, dst = source/relative/name, directory/relative/name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
    return directory, contract


def run(output=base.DEFAULT_OUTPUT, *, ratios=None, rhos=RHO_VALUES, workers=8, seeds=(0, 1)):
    import pandas as pd
    candidates = variants(ratios, rhos)
    directory, contract = prepare(output, candidates, seeds)
    tasks = [(item, str(directory), candidates, contract['frontend'], tuple(seeds))
             for item in contract['items']]
    batches = base.parallel(base._mistake, tasks, workers)
    summary = base.finish(directory, batches, candidates, contract, 'mistake')
    labels = parameter_table(candidates)
    labels = labels.drop(columns='Gap intercept (alpha*rho)')
    # The alpha=0 row uses rho=0 because rho has no effect at this endpoint.
    def label(frame):
        frame = frame.copy()
        frame['Rho (semitones)'] = [c/a if a else 0. for a,c in zip(frame['Pitch Weight'],frame.ins_cost)]
        frame['Pitch:Timing'] = [f'{int(a)}:{int(b)}' for a,b in zip(frame['Pitch Weight'],frame['Timing Weight'])]
        return frame.rename(columns={'ins_cost':'Insertion intercept', 'del_cost':'Deletion intercept'})
    summary = label(summary)
    for filename in ('rows.csv', 'summary.csv', 'by_dataset.csv'):
        label(pd.read_csv(directory/filename)).to_csv(directory/filename,index=False)
    best = base.read(directory/'best.json')
    chosen = next(v for v in candidates if v['variant']==best['variant'])
    best.update(model=MODEL, rho=chosen['rho'], ratio=chosen['ratio'])
    base.atomic(directory/'best.json',best)
    return dict(summary=summary, directory=directory, best=best)
