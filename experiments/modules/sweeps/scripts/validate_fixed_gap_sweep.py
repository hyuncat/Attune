"""Bounded cached-track parity check; never launches the full penalty sweep."""
import sys
from pathlib import Path
sys.path.insert(0,str(next(p for p in Path(__file__).resolve().parents if (p/'app.py').is_file())))
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch
from experiments.modules.sweeps.scripts import mistake_fixed_gap_sweep as runner
from experiments.modules.sweeps import AppendixSweeps as base


def main():
    old=runner.previous_results()
    assert len(old['rows'])==195*34
    candidates=runner.variants()
    assert len(candidates)==5
    assert all(v['params']['ins_cost']==v['params']['del_cost'] for v in candidates)
    # A typical cached source, all three conditions, two penalties.
    item=old['contract']['items'][0]
    candidates=runner.variants((3.,5.))
    from algorithms.PitchDetector import PitchDetector
    with TemporaryDirectory() as temporary, patch.object(PitchDetector,'detect_pitches',side_effect=AssertionError('Unexpected pitch inference')):
        root=Path(temporary)
        for name in ('standard','optimized'):
            for folder in (old['source']/'frontend'/item['track_id']).iterdir():
                dst=root/name/'frontend'/item['track_id']/folder.name
                runner._link(folder/'notes.pkl',dst/'notes.pkl')
        tasks=[(item,str(root/name),candidates,old['contract']['frontend'],(0,1)) for name in ('standard','optimized')]
        start=perf_counter();a=base._mistake(tasks[0]);standard=perf_counter()-start
        start=perf_counter();b=runner._mistake(tasks[1]);optimized=perf_counter()-start
        for x,y in zip(a,b):
            assert {k:v for k,v in x.items() if k!='artifact'}=={k:v for k,v in y.items() if k!='artifact'}
            assert base.read(x['artifact'])['cases']==base.read(y['artifact'])['cases']
        timing=base.read(root/'optimized'/'timings'/(item['track_id']+'.json'))
        assert timing['score_parses']==1
        with patch.object(base,'blob',side_effect=AssertionError('Cache not reused')):
            assert runner._mistake(tasks[1])==b
        print(f'Exact event/prediction parity; standard {standard:.3f}s, optimized {optimized:.3f}s; score parses 1.')
    import json
    notebook=json.loads((base.ROOT/'experiments/sweeps/mistake.ipynb').read_text())
    for cell in notebook['cells']:
        if cell['cell_type']=='code':compile(''.join(cell['source']),'mistake.ipynb','exec')
    print('Historical coverage, resume, notebook and fixed-penalty grid checks passed. Full sweep not run.')


if __name__=='__main__':
    main()
