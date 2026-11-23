"""Read-only, pooled-injection reports from completed mistake prediction caches.

Deliberately imports no pipeline runners: incomplete caches raise rather than
triggering audio rendering, detection, or alignment.
"""
import hashlib
import json
from pathlib import Path

import pandas as pd
from experiments.paths import relocated_path


def read(path):
    return json.loads(relocated_path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def summarize(source, variants, items, seeds=(0, 1)):
    source = Path(source)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Specify distinct injection seeds')
    ids = [item['track_id'] for item in items]
    if len(ids) != 195 or len(set(ids)) != 195:
        raise ValueError('Expected exactly 195 distinct source tracks')
    if len({v['variant'] for v in variants}) != len(variants):
        raise ValueError('Duplicate candidate variants')
    rows = []
    for variant in variants:
        for track_id in ids:
            path = source / 'predictions' / track_id / (variant['variant'] + '.json')
            payload = read(path)
            row = payload['row']
            if row['track_id'] != track_id or row['variant'] != variant['variant']:
                raise ValueError(f'Prediction identity mismatch: {path}')
            if any(row[k] != value for k, value in variant['params'].items()):
                raise ValueError(f'Prediction parameters mismatch: {path}')
            cases = [c for c in payload['cases'] if c['seed'] in seeds and c['rate'] == .25]
            if len(cases) != len(seeds) or {c['seed'] for c in cases} != set(seeds):
                raise ValueError(f'Expected one injected case per seed {seeds}: {path}')
            case = {k: sum(c[k] for c in cases) for k in ('tp', 'fp', 'fn')}
            rows.append(dict(track_id=track_id, seeds=str(tuple(seeds)), variant=variant['variant'],
                             alpha=row['alignment_gamma_pitch'], beta=row['alignment_gamma_time'],
                             rho=row.get('ins_cost', 5.),
                             **{k: case[k] for k in ('tp', 'fp', 'fn')},
                             artifact=str(path)))
    rows = pd.DataFrame(rows)
    summary = rows.groupby(['alpha', 'beta', 'rho'], as_index=False)[['tp', 'fp', 'fn']].sum()
    summary['Precision'] = 100 * summary.tp / (summary.tp + summary.fp).replace(0, float('nan'))
    summary['Recall'] = 100 * summary.tp / (summary.tp + summary.fn).replace(0, float('nan'))
    summary['Mistake-Event F1'] = 100 * 2 * summary.tp / (2 * summary.tp + summary.fp + summary.fn).replace(0, float('nan'))
    summary[['Precision', 'Recall', 'Mistake-Event F1']] = summary[['Precision', 'Recall', 'Mistake-Event F1']].fillna(100.)
    summary['Source Tracks'] = 195
    summary['Performances'] = 195 * len(seeds)
    summary = summary.sort_values(['Mistake-Event F1', 'alpha', 'beta', 'rho'], ascending=[False, True, True, True]).reset_index(drop=True)
    return dict(rows=rows, summary=summary, source=source)


def load_reports(output, *, seeds=(0, 1), penalty_run='ac91b8b2c996636c'):
    """Filter the completed revised cohort, preserving every source artifact."""
    output = Path(output)
    active = read(output / 'mistake_tuning/active.json')
    source = relocated_path(active['directory'])
    contract = read(source / 'contract.json')
    if digest(contract) != active['contract_sha256']:
        raise ValueError('Active weight contract has changed')
    if not set(seeds).issubset(contract['seeds']):
        raise ValueError('Requested seed was not cached')
    grid = read(source / 'grids/61de3a099d7ad099/contract.json')
    if {k: v for k, v in grid.items() if k != 'variants'} != contract:
        raise ValueError('Weight grid does not match active contract')
    if len(grid['variants']) != 34:
        raise ValueError('Expected the completed 34-candidate weight grid')
    penalty_source = output / 'mistake_fixed_rho' / penalty_run
    penalty = read(penalty_source / 'contract.json')
    if (relocated_path(penalty['source']) != source or penalty['source_contract_sha256'] != digest(contract)
            or penalty['items'] != contract['items'] or penalty['frontend'] != contract['frontend']):
        raise ValueError('Penalty sweep does not match the active cohort/frontend')
    weights = summarize(source, grid['variants'], contract['items'], seeds)
    penalties = summarize(penalty_source, penalty['variants'], contract['items'], seeds)
    combined = pd.concat([weights['summary'], penalties['summary']], ignore_index=True)
    duplicate = combined[combined.duplicated(['alpha', 'beta', 'rho'], keep=False)]
    if any(len(group[['tp', 'fp', 'fn']].drop_duplicates()) != 1
           for _, group in duplicate.groupby(['alpha', 'beta', 'rho'])):
        raise ValueError('Shared candidate has inconsistent cached counts')
    combined = combined.drop_duplicates(['alpha', 'beta', 'rho']).sort_values(
        ['Mistake-Event F1', 'alpha', 'beta', 'rho'], ascending=[False, True, True, True]).reset_index(drop=True)
    return dict(weights=weights, penalties=penalties, summary=combined, seeds=tuple(seeds))
