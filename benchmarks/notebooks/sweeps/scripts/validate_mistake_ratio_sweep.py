"""Synthetic checks only; does not run any corpus alignment candidates."""
import sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p/'app.py').is_file())))
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
from algorithms.Config import Config
from algorithms.MistakeDetector import MistakeDetector
from app_logic.NoteData import Note
from benchmarks.modules.sweeps import AppendixSweeps as base
from benchmarks.notebooks.sweeps.scripts import mistake_ratio_sweep as runner


def main():
    candidates=runner.variants()
    assert len(runner.unique_ratios())==14 and len(candidates)==92
    assert len({v['variant'] for v in candidates})==92
    assert runner.unique_ratios_from_pairs([(6,2),(3,1),(0,2),(0,1)])==[(0,1),(3,1)]
    assert len([v for v in candidates if v['ratio']=='0:1'])==1
    note=Note(0,0.,.5,[60.])
    for v in candidates:
        d=MistakeDetector(config=Config(**v['params']))
        expected=v['params']['alignment_gamma_pitch']*v['rho']+v['params']['alignment_gamma_time']*.5
        assert d.get_insertion_cost(note)==expected==d.get_deletion_cost(note)
    baseline=next(v for v in candidates if v['ratio']=='4:1' and v['rho']==1.25)
    old=MistakeDetector(config=Config())
    new=MistakeDetector(config=Config(**baseline['params']))
    assert old.get_insertion_cost(note)==new.get_insertion_cost(note)
    # End-to-end report serialization, with synthetic worker rows only.
    with TemporaryDirectory() as tmp:
        directory=Path(tmp)
        def prepare(output, variants, seeds):
            return directory,dict(items=[dict(track_id='synthetic',dataset='coco')],frontend={},seeds=list(seeds))
        def parallel(fn,tasks,workers):
            assert len(tasks)==1
            rows=[]
            for v in tasks[0][2]:
                path=directory/(v['variant']+'.json')
                base.atomic(path,{'synthetic':True})
                rows.append(dict(track_id='synthetic',dataset='coco',instrument='violin',
                    variant=v['variant'],artifact=str(path),tp=1,fp=0,fn=0,clean_fp=0,score=1.,**v['params']))
            return [rows]
        with patch.object(runner,'prepare',side_effect=prepare),patch.object(base,'parallel',side_effect=parallel):
            results=runner.run(directory,ratios=[(3,1),(6,2),(0,1)],rhos=[1.,2.])
        assert len(results['summary'])==3
        assert {'Pitch:Timing','Rho (semitones)','Insertion intercept'} <= set(results['summary'])
        assert base.read(directory/'best.json')['model']==runner.MODEL
    notebook=json.loads((base.ROOT/'benchmarks/notebooks/sweeps/mistake.ipynb').read_text())
    for c in notebook['cells']:
        if c['cell_type']=='code':
            compile(''.join(c['source']),'mistake.ipynb','exec')
            assert c['execution_count'] is None and c['outputs']==[]
    print('Ratio deduplication, scaled gap costs, baseline equivalence, reporting and notebook checks passed.')


if __name__=='__main__':
    main()
