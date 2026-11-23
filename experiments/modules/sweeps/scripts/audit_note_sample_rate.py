import sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / 'app.py').is_file())))
from copy import deepcopy
import numpy as np
from experiments.modules.sweeps import AppendixSweeps as base
from experiments.modules.sweeps.scripts import note_method_sweeps as runner
OUT=(base.ROOT / 'experiments/results/caches/sweeps/appendix_195_v1')/'diagnostics/note_sample_rate_audit'
SOURCE=(base.ROOT / 'experiments/results/caches/sweeps/appendix_195_v1')/'note_ruptures/f4f130905b95cf55'

def audit(task):
    import librosa
    from algorithms.Config import Config
    from experiments.modules.note.NoteBenchmarker import OneInstrumentScoreData,NoteEvaluation
    from experiments.modules.pitch.competitors.Attune import Attune
    item,params=task
    notes=OneInstrumentScoreData(item['paths']['midi'])
    low,high=Config.padded_midi_range(min(notes.midi_numbers),max(notes.midi_numbers))
    adapter=Attune();cfg=adapter.config_for(*adapter.range_from_midi([low,high]),**params)
    rec=adapter.recording_for(cfg,score_data=notes)
    samples,_=librosa.load(item['paths']['audio'],sr=cfg.sr,mono=True)
    rec.audio_data=adapter._audio_data_from_samples(samples,cfg)
    rec.pitch_data=adapter.detect_stages(rec).data['smoothed']
    pitches=rec.pitch_data.data
    reference=NoteEvaluation.reference_notes(dict(dataset='coco',reference=item['paths']['midi']))
    old=base.read((base.ROOT / 'experiments/results/caches/sweeps/appendix_195_v1')/'diagnostics/note_pitch_source_audit'/(item['track_id']+'.json'))
    rows=[dict(row,sample_rate=16000) for row in old if row['mode'] in ['production_same_settings','production_defaults']]
    for mode in ['production_same_settings','production_defaults']:
        overrides=runner.BASELINE if mode=='production_same_settings' else {}
        from dataclasses import replace
        config=replace(cfg,**overrides)
        r=adapter.recording_for(config,score_data=OneInstrumentScoreData(item['paths']['midi']))
        r.pitch_data=adapter._pitch_data(deepcopy(pitches),config,0.)
        r.pitch_data.end_index=len(pitches)
        r.detect_notes();r.align_score_and_refine()
        metrics=base._note_metrics(reference,r.note_data,config)
        rows.append(dict(mode=mode,sample_rate=44100,track_id=item['track_id'],instrument=item['instrument'],**metrics))
    base.atomic(OUT/(item['track_id']+'.json'),rows)
    return rows

def main():
    import pandas as pd
    source=base.read(SOURCE/'contract.json');groups={}
    for item in sorted(source['items'],key=lambda i:i['track_id']):groups.setdefault(item['instrument'],item)
    rows=base.parallel(audit,[(i,source['pitch_winner']['params']) for i in groups.values()],workers=8)
    f=pd.DataFrame([r for batch in rows for r in batch]);g=f.groupby(['mode','sample_rate'])[['tp_offset','fp_offset','fn_offset']].sum()
    g['f1']=2*g.tp_offset/(2*g.tp_offset+g.fp_offset+g.fn_offset)
    OUT.mkdir(parents=True,exist_ok=True);f.to_csv(OUT/'rows.csv',index=False);g.to_csv(OUT/'summary.csv')
    base.atomic(OUT/'protocol.json',dict(tracks=len(groups),selection='first track_id per instrument; no metric selection',pitch_params=source['pitch_winner']['params'],sample_rates=[16000,44100],window_samples=4096,hop_samples=128,full_parameter_sweep=False))
    print(g.to_string())
if __name__=='__main__':main()
