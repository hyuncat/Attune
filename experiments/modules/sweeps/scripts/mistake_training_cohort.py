"""One frozen, same-instrument replacement for mistake tuning only."""
from dataclasses import asdict
from pathlib import Path
import re
from experiments.modules.sweeps import AppendixSweeps as base

OLD_TRACK = 'test__brass_track222654__2_horn'
SEED = 20261117


def selection(output=base.DEFAULT_OUTPUT):
    from experiments.modules.pitch.datasets.CocoChorales import CocoChorales
    output=Path(output)
    source=base.read(output/'cohort.json')
    coco=CocoChorales(root=source['selection']['root'],split='test')
    evaluation=base.ROOT/'experiments/results/caches'
    e_path=evaluation/'coco_e_isolated_260_tracks_annotations_v2_2026-10-07/manifest.json'
    e2_path=evaluation/'coco_e2_isolated_260_tracks_annotations_v2_2026-10-07/manifest.json'
    native={c['group'] for c in base.read(e_path)['cases']}
    excluded={i['group'] for i in source['items'] if i['dataset']=='coco'}
    excluded|={c['group'] for c in base.read(e2_path)['cases']}
    # Include the actual main pitch evaluation selection, independently of E2.
    import pandas as pd
    pitch_path=evaluation/'pitch/notebook_runs/paper_offline_coco/seed_0__per_instrument_20/rows.csv'
    excluded|=set(pd.read_csv(pitch_path,usecols=['track']).track)
    spec=dict(old_track=OLD_TRACK,seed=SEED,instrument='horn',
              source_cohort_sha256=base.file_hash(output/'cohort.json'),
              evaluation_hashes={str(p):base.file_hash(p) for p in (e_path,e2_path,pitch_path)},
              policy='random within horn; exclude all current tuning pieces and evaluated Coco/E2 pieces; also exclude native-E numeric IDs')
    path=output/'mistake_tuning/replacement.json'
    if path.exists():
        saved=base.read(path)
        if saved['spec']!=spec:raise ValueError('Replacement selection inputs changed; review the frozen replacement')
        return saved
    pool=[r for r in coco.read_manifest('test') if r.instrument=='horn' and r.track not in excluded
          and re.search(r'track(\d+)',r.track)[1] not in native]
    chosen=coco.sample_records_per_instrument(pool,1,SEED)[0]
    saved=dict(spec=spec,record=asdict(chosen),track_id=chosen.track_id,eligible_tracks=len(pool))
    base.atomic(path,saved)
    return saved


def replacement_item(output=base.DEFAULT_OUTPUT):
    """Called only by the manual sweep run; extract this single stem if needed."""
    saved=selection(output)
    cohort=base.read(Path(output)/'cohort.json')
    from experiments.modules.pitch.datasets.CocoChorales import CocoChorales
    coco=CocoChorales(root=cohort['selection']['root'],split='test')
    record=CocoChorales.Stem(**saved['record'])
    coco.materialize_records([record])
    item=base.inputs(dict(selection=cohort['selection'],records=[saved['record']]))[0]
    return dict(item,dataset='coco',group=record.track,
                note_reference=dict(dataset='coco',reference=item['paths']['midi']))
