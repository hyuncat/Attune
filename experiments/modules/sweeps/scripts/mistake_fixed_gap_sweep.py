"""Fixed-gap evidence and a cheap conditional rho sweep, preserving old runs."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
import importlib
import os
import shutil

from algorithms.Config import Config
from experiments.modules.sweeps import AppendixSweeps as base
from experiments.modules.sweeps.MixedCohort import selected_frontend

RHO_VALUES = (3., 4., 5., 6., 7.)
MODEL = 'fixed_shared_gap_penalty_v1'
DEFAULT_WORKERS = min(10, os.cpu_count() or 1)


def variants(rhos=RHO_VALUES):
    penalties=base.grid(dict(rho=tuple(sorted(set(rhos)))))
    result=[]
    for v in penalties:
        params=dict(alignment_gamma_pitch=6., alignment_gamma_time=2.,
                    ins_cost=v['params']['rho'], del_cost=v['params']['rho'])
        result.append(dict(variant=base.digest(dict(model=MODEL,params=params))[:16],params=params))
    return result


def historical_source_run(output=base.DEFAULT_OUTPUT, seeds=(0, 1)):
    from experiments.modules.note.NoteBenchmarker import NoteBenchmarker
    output = Path(output).resolve()
    saved = base.read(output/'cohort.json')
    items = [i for i in base.inputs(saved) if i.get('dataset','coco')=='coco']
    # This is historical evidence, with the exact pre-promotion defaults.
    source=output/'mistake/b4b353b06b399ec4'
    expected=base.read(source/'contract.json')
    identity=base.code_identity()
    promotion=base.read(Path(__file__).with_name('production_promotion_2026_11_17.json'))
    for path,change in promotion.items():
        if identity['code'][path] == change['after']:
            identity['code'][path]=change['before']
    if identity!=expected['identity'] or items!=expected['items'] or saved!=expected['cohort']:
        raise ValueError('Historical sweep inputs/code changed beyond the audited defaults promotion')
    if list(seeds)!=expected['seeds'] or base.file_hash(NoteBenchmarker.SOUNDFONT_PATH)!=expected['soundfont_sha256']:
        raise ValueError('Historical sweep seeds/soundfont changed')
    for stage in ('pitch','note'):
        if base.file_hash(output/stage/'best.json')!=expected['upstream'][stage+'_best_sha256']:
            raise ValueError('Historical frontend winner changed')
    return source, expected


def source_run(output=base.DEFAULT_OUTPUT, seeds=(0,1)):
    """Use the revised cohort once its weight sweep has completed."""
    historical, expected = historical_source_run(output,seeds)
    pointer=Path(output)/'mistake_tuning/active.json'
    if not pointer.exists():
        return historical,expected
    active=base.read(pointer)
    from experiments.modules.sweeps.scripts import mistake_training_cohort as cohort
    if active['replacement_sha256'] != base.digest(cohort.selection(output)) or active['historical_contract_sha256'] != base.digest(expected):
        raise ValueError('Revised mistake cohort is stale; rerun refresh_weights')
    directory=Path(active['directory'])
    contract=base.read(directory/'contract.json')
    if active['contract_sha256']!=base.digest(contract) or contract['cohort_runner_sha256']!=base.file_hash(cohort.__file__):
        raise ValueError('Revised cohort contract changed; rerun refresh_weights')
    return directory,contract


def refresh_weights(output=base.DEFAULT_OUTPUT, workers=DEFAULT_WORKERS):
    """Reuse 194 sources; compute the replacement source for all original weights."""
    from experiments.modules.sweeps.scripts import mistake_training_cohort as cohort
    source,old=historical_source_run(output)
    selected=cohort.selection(output)
    replacement=cohort.replacement_item(output)
    items=[replacement if i['track_id']==cohort.OLD_TRACK else i for i in old['items']]
    if len(items)!=195 or len({i['track_id'] for i in items})!=195 or any(i['track_id']==cohort.OLD_TRACK for i in items):
        raise ValueError('Expected one replacement in the 195-track mistake cohort')
    variants=base.grid(base.MISTAKE_AXES)
    contract=dict(old,items=items,replacement=selected,
                  historical_contract_sha256=base.digest(old),
                  cohort_runner_sha256=base.file_hash(cohort.__file__))
    directory=Path(output)/'mistake_tuning'/base.digest(contract)[:16]
    base.contract(directory,contract)
    for item in items:
        if item['track_id']==replacement['track_id']:continue
        for folder in (source/'frontend'/item['track_id']).iterdir():
            provenance=base.read(folder/'provenance.json')
            if base.file_hash(folder/'notes.pkl')!=provenance['notes_sha256']:
                raise ValueError(f'Changed frontend: {folder}')
            for name in ('notes.pkl','provenance.json'):
                _link(folder/name,directory/'frontend'/item['track_id']/folder.name/name)
        for v in variants:
            payload=base.read(source/'predictions'/item['track_id']/(v['variant']+'.json'))
            dst=directory/'predictions'/item['track_id']/(v['variant']+'.json')
            if not dst.exists():
                payload['row']['artifact']=str(dst)
                base.atomic(dst,payload)
    tasks=[(i,str(directory),variants,contract['frontend'],tuple(contract['seeds'])) for i in items]
    # Only tasks without completed predictions require a worker process.
    batches=[];pending=[]
    for task in tasks:
        paths=[directory/'predictions'/task[0]['track_id']/(v['variant']+'.json') for v in variants]
        if all(p.exists() for p in paths):batches.append([base.read(p)['row'] for p in paths])
        else:pending.append(task)
    if pending:batches+=base.parallel(_mistake,pending,min(workers,len(pending)))
    report=directory/'grids'/base.digest(variants)[:16]
    meta=dict(contract,variants=variants);base.contract(report,meta)
    base.finish(report,batches,variants,meta,'mistake')
    base.atomic(Path(output)/'mistake_tuning/active.json',dict(directory=str(directory),
        contract_sha256=base.digest(contract),historical_contract_sha256=base.digest(old),
        replacement_sha256=base.digest(selected)))
    return previous_results(output)


def previous_results(output=base.DEFAULT_OUTPUT):
    """Read the complete original 34-candidate sweep, without scheduling jobs."""
    import pandas as pd
    source, contract = source_run(output)
    original = base.grid(base.MISTAKE_AXES)
    report = source/'grids'/base.digest(original)[:16]
    if base.read(report/'contract.json') != dict(contract, variants=original):
        raise ValueError('Historical sweep contract changed')
    rows = pd.read_csv(report/'rows.csv')
    keys = ['Pitch Weight','Timing Weight']
    expected_ids = {i['track_id'] for i in contract['items']}
    if (rows.duplicated(keys+['track_id']).any() or len(rows)!=len(original)*len(expected_ids)
            or any(set(g.track_id)!=expected_ids for _,g in rows.groupby(keys))):
        raise ValueError('Historical weight sweep is incomplete')
    summary = pd.read_csv(report/'summary.csv')
    counts=rows.groupby(keys)[['tp','fp','fn','Clean False Positives']].sum().sort_index()
    if not counts.equals(summary.set_index(keys)[list(counts)].sort_index()):
        raise ValueError('Historical summary does not match per-track counts')
    return dict(summary=summary, rows=rows, directory=report, source=source, contract=contract)


def _link(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists():
        try: os.link(src,dst)
        except OSError: shutil.copy2(src,dst)


def prepare(output=base.DEFAULT_OUTPUT, rhos=RHO_VALUES):
    old=previous_results(output)
    source, inherited=old['source'],old['contract']
    candidates=variants(rhos)
    checkpoints=[]
    for item in inherited['items']:
        for seed,rate in [(0,0.),(0,.25),(1,.25)]:
            relative=Path('frontend')/item['track_id']/f'{seed}_{rate}'
            provenance=base.read(source/relative/'provenance.json')
            sha=base.file_hash(source/relative/'notes.pkl')
            if sha!=provenance['notes_sha256']:
                raise ValueError(f'Changed frontend cache: {relative}')
            checkpoints.append(dict(relative=str(relative),sha256=sha))
    contract=dict(source=str(source), source_contract_sha256=base.digest(inherited),
                  runner_sha256=base.file_hash(__file__), model=MODEL, variants=candidates,
                  checkpoints=checkpoints, items=inherited['items'], frontend=inherited['frontend'],
                  seeds=inherited['seeds'], scope='conditional rho sweep at fixed 6:2; not a joint weight/rho search')
    directory=Path(output)/'mistake_fixed_rho'/base.digest(contract)[:16]
    base.contract(directory,contract)
    for entry in checkpoints:
        relative=Path(entry['relative'])
        for name in ('notes.pkl','provenance.json'):
            _link(source/relative/name,directory/relative/name)
    # The exact 6:2, rho=5 result is already completed. Relabel its cache identity only.
    baseline=next((v for v in candidates if v['params']['ins_cost']==5.),None)
    if baseline:
        original_id=base.grid(dict(alignment_gamma_pitch=(6.,),alignment_gamma_time=(2.,)))[0]['variant']
        for item in inherited['items']:
            old_path=source/'predictions'/item['track_id']/(original_id+'.json')
            payload=base.read(old_path)
            row=payload['row']
            if row['track_id']!=item['track_id'] or row['alignment_gamma_pitch']!=6 or row['alignment_gamma_time']!=2:
                raise ValueError(f'Invalid baseline prediction: {old_path}')
            target=directory/'predictions'/item['track_id']/(baseline['variant']+'.json')
            if not target.exists():
                payload['row']=dict(row,variant=baseline['variant'],artifact=str(target),**baseline['params'])
                payload['reused_from']=dict(path=str(old_path),sha256=base.file_hash(old_path))
                base.atomic(target,payload)
    return directory,contract


def _mistake(task):
    """Same scoring worker, but parse each score once per task, cloning per use.

    Every candidate still gets independent mutable score state. This avoids
    repeated music21/MIDI parsing and temporary MIDI normalization on every case.
    """
    item,directory,candidates,frontend,seeds=task
    paths=[Path(directory)/'predictions'/item['track_id']/(v['variant']+'.json') for v in candidates]
    if all(p.exists() for p in paths):
        rows=[base.read(p)['row'] for p in paths]
        base.cached_progress(item,candidates)
        return rows
    module=importlib.import_module('experiments.modules.note.NoteBenchmarker')
    original=module.OneInstrumentScoreData
    templates={}
    def fresh_score(path, collapse_simultaneous=True):
        key=(str(path),collapse_simultaneous)
        if key not in templates:
            templates[key]=original(path,collapse_simultaneous=collapse_simultaneous)
        return deepcopy(templates[key])
    # Preserve the original live gates/default alignment during frontend construction.
    # Candidate weights still override these during alignment.
    frontend=dict(frontend,unv_thresh=0.985,min_volume=0.04,
                  alignment_gamma_pitch=4.,alignment_gamma_time=1.)
    task=(item,directory,candidates,frontend,seeds)
    started=perf_counter()
    with patch.object(module,'OneInstrumentScoreData',side_effect=fresh_score):
        rows=base._mistake(task)
    base.atomic(Path(task[1])/'timings'/(task[0]['track_id']+'.json'),
                dict(seconds=perf_counter()-started,score_parses=len(templates)))
    return rows


def run(output=base.DEFAULT_OUTPUT, *, rhos=RHO_VALUES, workers=DEFAULT_WORKERS):
    import pandas as pd
    directory,contract=prepare(output,rhos)
    # Start larger tasks first to reduce the final long-track tail.
    items=sorted(contract['items'],key=lambda i:sum(p.stat().st_size for p in
                 (directory/'frontend'/i['track_id']).glob('*/notes.pkl')),reverse=True)
    tasks=[(item,str(directory),contract['variants'],contract['frontend'],tuple(contract['seeds'])) for item in items]
    started=perf_counter()
    batches=base.parallel(_mistake,tasks,min(workers,len(tasks)))
    summary=base.finish(directory,batches,contract['variants'],contract,'mistake')
    for name in ('rows.csv','summary.csv','by_dataset.csv'):
        frame=pd.read_csv(directory/name)
        frame=frame.rename(columns={'ins_cost':'Fixed penalty rho','del_cost':'Deletion penalty'})
        frame.to_csv(directory/name,index=False)
    summary=summary.rename(columns={'ins_cost':'Fixed penalty rho','del_cost':'Deletion penalty'})
    base.atomic(directory/'runtime.json',dict(wall_seconds=perf_counter()-started,workers=workers,
                scheduling='largest cached frontends first; one native thread per process'))
    return dict(summary=summary,directory=directory)
