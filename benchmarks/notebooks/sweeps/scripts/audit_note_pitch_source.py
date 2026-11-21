import sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / 'app.py').is_file())))
import json
from copy import deepcopy
import numpy as np
from unittest.mock import patch
from benchmarks.modules.sweeps import AppendixSweeps as base
from benchmarks.notebooks.sweeps.scripts import note_method_sweeps as runner

OUT=(base.ROOT / 'benchmarks/results/sweeps/appendix_195_v1')/'diagnostics/note_pitch_source_audit'
SOURCE=(base.ROOT / 'benchmarks/results/sweeps/appendix_195_v1')/'note_ruptures/f4f130905b95cf55'

def median_value(self, pitches):
    return [float(np.median([p.value for p in pitches]))]

def audit(task):
    from benchmarks.modules.note.NoteBenchmarker import NoteEvaluation
    item, pitch_path, pitch_params, baseline_variant = task
    output=OUT/(item['track_id']+'.json')
    pitches=base.blob(pitch_path)
    ref=NoteEvaluation.reference_notes(dict(dataset='coco',reference=item['paths']['midi']))
    saved=base.read(SOURCE/'checkpoints'/item['track_id']/(baseline_variant['variant']+'.json'))
    rows=[dict(mode='legacy_adapter',**{k:saved[k] for k in ['track_id','instrument','tp','fp','fn','tp_offset','fp_offset','fn_offset','f1','f1_offset','reference_notes','estimated_notes','status']})]
    for mode in ['smoothed_adapter','production_same_settings','production_defaults']:
        params=runner.BASELINE if mode!='production_defaults' else {}
        rec=base._recording(item,dict(pitch_params,**params),deepcopy(pitches))
        if mode=='smoothed_adapter':
            with patch.object(runner.CappedRuptures,'_pelt_segment_pitch',median_value):
                rec.note_data=runner.detect_notes(rec,'kernelcpd-linear')
        else:
            rec.detect_notes()
        rec.align_score_and_refine()
        metrics=base._note_metrics(ref,rec.note_data,rec.config)
        rows.append(dict(mode=mode,track_id=item['track_id'],instrument=item['instrument'],**metrics))
    base.atomic(output,rows)
    return rows

def main():
    import pandas as pd
    contract=base.read(SOURCE/'contract.json')
    variant=next(v for v in runner.variants_for() if v['method']=='kernelcpd-linear')
    items=contract['items']
    # All modes use the exact same 195 cached pitch tracks, score and scorer.
    tasks=[(i,contract['pitch_winner']['artifacts'][i['track_id']]['path'],contract['pitch_winner']['params'],variant) for i in items]
    rows=base.parallel(audit,tasks,workers=8)
    frame=pd.DataFrame([r for batch in rows for r in batch])
    summary=frame.groupby('mode')[['tp','fp','fn','tp_offset','fp_offset','fn_offset']].sum()
    for suffix in ['','_offset']:
        summary['f1'+suffix]=2*summary['tp'+suffix]/(2*summary['tp'+suffix]+summary['fp'+suffix]+summary['fn'+suffix])
    OUT.mkdir(parents=True,exist_ok=True)
    frame.to_csv(OUT/'rows.csv',index=False);summary.to_csv(OUT/'summary.csv')
    base.atomic(OUT/'protocol.json',dict(source=str(SOURCE),tracks=len(items),baseline=runner.BASELINE,pitch_params=contract['pitch_winner']['params'],changed='candidate median vs smoothed value median; production consolidation; production defaults',full_parameter_sweep=False))
    print(summary.to_string())
if __name__=='__main__': main()
