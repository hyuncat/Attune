"""Refresh only Attune on frozen evaluation inputs; retain competitor evidence."""
from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import json
import hashlib
import pandas as pd
from algorithms.Config import Config
from benchmarks.notebooks.scripts.evaluation_holdout import ROOT, read, write, sha


def _worker(task):
    from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker as M
    mode, payload, checkpoint = task
    checkpoint=Path(checkpoint)
    if checkpoint.exists():return read(checkpoint)
    if mode=='E':
        rows=M.native_cpu_worker(payload)['rows']
    else:
        from unittest.mock import patch
        import importlib
        module=importlib.import_module('benchmarks.modules.mistake.MistakeBenchmarker')
        payload=dict(payload)
        asset=payload.pop('frozen_asset')
        def frozen_case(midi,seed,rate,output,**kwargs):
            return frozen_e2_case(asset,midi,seed,rate,output)
        with patch.object(module,'prepare_case',side_effect=frozen_case):
            rows=M.evaluate_case(**payload)
    write(checkpoint,rows)
    return rows


def refresh(source, *, mode, workers=4, dry_run=False):
    """New output per defaults/code/source contract; never schedule a competitor."""
    from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker as M
    source=Path(source)
    old=read(source/'run.json')
    if old['status']!='complete':raise ValueError(f'Incomplete source run: {source}')
    frame=pd.read_csv(source/'rows.csv',low_memory=False)
    competitors=frame.loc[~frame.method.str.startswith('Attune')].copy()
    if competitors.empty:raise ValueError('Missing frozen competitor results')
    # Require the complete original case/condition coverage before doing any work.
    keys=['track_id','seed','rate','metric','tolerance'] if mode=='symbolic' else (
        ['case_id','protocol','metric','tolerance'] if mode=='E' else ['source','seed','rate','metric','tolerance'])
    for method,g in competitors.loc[competitors.metric.eq('audio_pitch')].groupby('method'):
        attune=frame.loc[frame.method.str.startswith('Attune') & frame.metric.eq('audio_pitch')]
        if set(map(tuple,g[keys].to_numpy())) != set(map(tuple,attune[keys].to_numpy())):
            raise ValueError(f'Incomplete frozen competitor coverage: {method}')
    contract=dict(defaults=asdict(Config()),code=M.native_code_identity(),
                  helper_sha256=sha(__file__),source=str(source.resolve()),
                  source_rows_sha256=sha(source/'rows.csv'),source_run_sha256=sha(source/'run.json'),mode=mode)
    tag=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()[:16]
    output=ROOT/'benchmarks/results'/('evaluation_defaults_'+tag)
    if dry_run:
        return dict(output=str(output), mode=mode, competitors=sorted(competitors.method.unique()),
                    cached_competitor_rows=len(competitors), defaults=asdict(Config()))
    if (output/'refresh.json').exists():
        assert read(output/'refresh.json')==contract
        if (output/'run.json').exists() and read(output/'run.json')['status']=='complete':
            if sha(output/'rows.csv')!=read(output/'run.json')['rows_sha256']:
                raise ValueError('Refreshed report changed')
            return output
    write(output/'refresh.json',contract)
    write(output/'run.json',dict(status='running'))
    if mode=='symbolic':
        inner=output/'attune'
        write(inner/'manifest.json',read(source/'manifest.json'))
        prior=old['contract']
        selection=read(source/'manifest.json')['selection']
        M.run_isolated(prior['dataset'],inner,root=selection['root'],
                       per_instrument=selection['per_instrument'],seed=selection['seed'],
                       injection_seeds=prior['injection_seeds'],tolerances=prior['tolerances'],
                       methods=['Attune (no refinement)'],workers=workers)
        fresh=pd.read_csv(inner/'rows.csv')
        metadata=read(inner/'run.json')
        metadata['contract']['methods']=old['contract']['methods']
        write(output/'manifest.json',read(inner/'manifest.json'))
    else:
        tasks=[]
        if mode=='E':
            manifest=read(source/'manifest.json');write(output/'manifest.json',manifest)
            baseline=old['contract']['source']
            for case in manifest['cases']:
                # Use the exact previously rendered WAV/MIDI paths and hashes.
                for name,path in case['files'].items():
                    if sha(path)!=case['hashes'][name]:raise ValueError(f'Changed frozen asset: {path}')
                payload=dict(case=case,method='Attune',dataset='coco',
                    directory=str(output/'cases'/case['case_id']/'attune'),
                    midi_range=baseline['midi_range'],tolerances=baseline['tolerances'],range_policy=baseline['range_policy'])
                tasks.append((mode,payload,str(output/'checkpoints'/(case['case_id']+'.json'))))
        elif mode=='E2':
            assets={}
            for path in (source/'cases').glob('*/manifest.json'):
                asset=read(path);spec=asset['spec']
                assets[(spec['source'],spec['seed'],spec['rate'])]=asset
            for source_midi in old['sources']:
                for seed in old['seeds']:
                    for rate in old['rates']:
                        key=hashlib.sha256(f'{source_midi}|{seed}|{rate}'.encode()).hexdigest()[:20]
                        payload=dict(midi=source_midi,seed=seed,rate=rate,output=str(output),
                            methods=['Attune (repeat only)'],tolerances=old['tolerances'],
                            source_info=old['source_metadata'][source_midi],input_kinds=('detected',),
                            frozen_asset=assets[(source_midi,seed,rate)])
                        tasks.append((mode,payload,str(output/'checkpoints'/(key+'.json'))))
        else:raise ValueError(mode)
        with M.single_thread_environment(), ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
            print(f'{mode}: refreshing/resuming {len(tasks)} Attune cases; all competitors cached.',flush=True)
            batches=[]
            for index,batch in enumerate(pool.map(_worker,tasks),1):
                batches.append(batch)
                if index%10==0 or index==len(tasks):print(f'{mode}: {index}/{len(tasks)} Attune cases ready',flush=True)
        fresh=pd.DataFrame([r for batch in batches for r in batch])
        metadata=dict(old)
        if mode=='E':metadata['contract']=dict(old['contract'],defaults=asdict(Config()),code=M.native_code_identity())
        else:
            write(output/'production_defaults.json',asdict(Config()))
            metadata['code']={p:sha(ROOT/p) for p in old['code']}
            metadata['execution_code']=dict(refresh_sha256=sha(__file__))
    # Each method keeps its original identity/provenance; these are combined reports.
    metadata['frozen_competitors']=dict(source=str(source),rows_sha256=contract['source_rows_sha256'])
    metadata['attune_defaults']=asdict(Config())
    old_attune=frame.loc[frame.method.str.startswith('Attune') & frame.metric.eq('audio_pitch')]
    current=fresh.loc[fresh.metric.eq('audio_pitch')]
    if set(map(tuple,current[keys].to_numpy())) != set(map(tuple,old_attune[keys].to_numpy())):
        raise ValueError('Incomplete refreshed Attune coverage')
    rows=pd.concat([fresh,competitors],ignore_index=True)
    rows.to_csv(output/'rows.csv',index=False)
    metadata.update(status='complete',rows_sha256=sha(output/'rows.csv'))
    write(output/'run.json',metadata)
    return output


def frozen_e2_case(asset, midi, seed, rate, output):
    """Use the evaluated performance exactly; never reinject or rerender it."""
    import shutil
    from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker
    spec=asset['spec']
    if (str(midi),seed,rate)!=(spec['source'],spec['seed'],spec['rate']):
        raise ValueError('Frozen case identity mismatch')
    if sha(midi)!=spec['source_hash']:
        raise ValueError('Frozen score changed')
    for key in ('audio','midi','truth'):
        if not Path(asset[key]).is_file():raise FileNotFoundError(asset[key])
    directory=Path(output)/'cases'/asset['case_id']
    directory.mkdir(parents=True,exist_ok=True)
    item=dict(asset)
    # New mutable caches are separate; compatible stages can seed them.
    for key in ('pitch_data','note_data'):
        destination=directory/Path(asset[key]).name
        if key=='pitch_data' and not destination.exists() and Path(asset[key]).exists():
            shutil.copy2(asset[key],destination)
            sidecar=Path(str(asset[key])+'.mistake-range.json')
            if sidecar.exists():shutil.copy2(sidecar,str(destination)+'.mistake-range.json')
        item[key]=str(destination)
    write(directory/'manifest.json',item)
    write(directory/'frozen_assets.json',{k:dict(path=item[k],sha256=sha(item[k])) for k in ('audio','midi','truth')})
    return MistakeBenchmarker(),item
