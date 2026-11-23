"""Bounded checks for balanced ranking, cache migration and linked Coco-E tuning."""
import sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / 'app.py').is_file())))
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import numpy as np
import pandas as pd
from experiments.modules.sweeps import AppendixSweeps as base
from experiments.modules.sweeps import MixedCohort as mixed
from experiments.modules.sweeps.scripts import note_method_sweeps as note


def main():
    # A large Coco corpus must have exactly the same influence as each small corpus.
    rows = [dict(variant='a', dataset=d, tp=tp, fp=fp, fn=fn, clean_fp=0)
            for d, n, tp, fp, fn in [('coco', 195, 1, 0, 0),
                                      ('bach10-original', 4, 0, 1, 1),
                                      ('urmp', 13, 0, 1, 1)] for _ in range(n)]
    score = base.summarize_scores(pd.DataFrame(rows), 'mistake').iloc[0]
    assert np.isclose(score.score, 1 / 3)
    assert score.tp == 195
    onset = pd.DataFrame(rows).assign(status='ok', reference_notes=lambda x: x.tp+x.fn,
                                      estimated_notes=lambda x: x.tp+x.fp)
    for suffix in ['', '_offset']:
        for col in ['tp', 'fp', 'fn']:
            onset[col+suffix] = onset[col]
    for col in ['precision', 'recall', 'f1', 'precision_offset', 'recall_offset', 'f1_offset']:
        onset[col] = onset.tp / (onset.tp+onset.fp)
    assert np.isclose(base.summarize_scores(onset, 'note').iloc[0].score, 1/3)

    manifest = mixed.materialize()
    assert manifest == mixed.materialize()  # Frozen restart, no resampling.
    items = base.inputs(manifest)
    assert len(items) == 212 and note.method_selection_items(items) == items
    from experiments.modules.pitch.datasets.Bach10 import Bach10
    from experiments.modules.pitch.datasets.URMP import URMP
    for dataset, cls in [('bach10-original', Bach10), ('urmp', URMP)]:
        expected = {t.track_id for t in cls(per_instrument=1, seed=20261116).tracks()}
        assert {i['source_track_id'] for i in items if i['dataset']==dataset} == expected
    for i in items:
        if i['dataset'] != 'coco':
            times, f0 = mixed.pitch_reference(i)
            assert len(times) == len(f0) > 0

    # Verify migrated completed rows are usable without calling any pitch detector.
    directory = base.DEFAULT_OUTPUT / 'pitch'
    meta = dict(cohort=manifest, items=items, variants=base.grid(base.PITCH_AXES), identity=base.code_identity())
    mixed.reuse_pitch(manifest['reuse_pitch_from'], directory, meta)
    coco = next(i for i in items if i['dataset']=='coco')
    from algorithms.PitchDetector import PitchDetector
    with patch.object(PitchDetector, 'detect_pitches', side_effect=AssertionError('Unexpected pitch inference')):
        legacy_variants = base.read(Path(manifest['reuse_pitch_from']) / 'contract.json')['variants']
        cached = base._pitch((coco, str(directory), legacy_variants, manifest['selection']['root']))
    assert len(cached) == len(legacy_variants) and {r['dataset'] for r in cached} == {'coco'}
    old = base.read(Path(manifest['reuse_pitch_from']) / coco['track_id'] / 'rows.json')
    assert [r['overall_correct_frames'] for r in cached] == [r['overall_correct_frames'] for r in old]
    bad = dict(meta, identity=dict(meta['identity'], packages={}))
    try:
        mixed.reuse_pitch(manifest['reuse_pitch_from'], directory, bad)
    except ValueError:
        pass
    else:
        raise AssertionError('Accepted incompatible source packages')

    # Production median-spread merging must not chain 60 -> 60.4 -> 60.8,
    # and it must not merge across distinct voiced runs.
    from types import SimpleNamespace
    recording = base._recording(coco, {})
    adapter = note.CappedRuptures(recording)
    run = [SimpleNamespace(value=p, time=j*.01) for j,p in enumerate([60.,60.,60.4,60.4,60.8,60.8])]
    second = [SimpleNamespace(value=60.8, time=1+j*.01) for j in range(2)]
    merged = adapter._notes_from_run_breakpoints([run,second], [[2,4,6],[2]], False)
    notes = [merged.read_note(i=i) for i in range(len(merged.times))]
    assert len(notes)==3 and np.isclose(notes[0].midi_num[0],60.2)
    assert notes[1].end_time < notes[2].start_time

    # No guessed frontend when note tuning has not completed.
    with TemporaryDirectory() as temporary:
        try:
            mixed.selected_frontend(temporary)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Missing upstream winner accepted')

    # Completed winners resolve exact params; replacing upstream pitch rejects the note pointer.
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        pitch = dict(params={'posthoc_min_volume': .02}, artifacts={})
        winner = dict(method='kernelcpd-linear', params={'pitch_thresh': .6}, artifacts={})
        source = {'fixture': 'completed'}
        base.atomic(root/'pitch/best.json', pitch)
        base.atomic(root/'report/best.json', winner)
        base.atomic(root/'note/best.json', winner)
        base.atomic(root/'note/selection.json', dict(source_sha256=base.digest(source),
            best_path=str(root/'report/best.json'), best_sha256=base.file_hash(root/'report/best.json'),
            pitch_best_sha256=base.file_hash(root/'pitch/best.json')))
        with patch.object(note, 'prepare', return_value=([], pitch, root, source)):
            settings, _ = mixed.selected_frontend(root)
            assert settings == dict(posthoc_min_volume=.02, pitch_thresh=.6, note_method='kernelcpd-linear')
            base.atomic(root/'pitch/best.json', dict(pitch, params={'posthoc_min_volume': .04}))
            try:
                mixed.selected_frontend(root)
            except ValueError:
                pass
            else:
                raise AssertionError('Stale note winner accepted')

    # Real cached E2 fixture, selected gate + note params, no rendering or raw inference.
    frontend = dict(posthoc_min_volume=.02, posthoc_unv_thresh=.9875,
                    min_silence_duration_ms=80., min_note_length_factor=.75,
                    min_note_length_cap=.2, pitch_thresh=.6, note_method='kernelcpd-linear')
    with TemporaryDirectory() as temporary, \
            patch.object(PitchDetector, 'detect_pitches', side_effect=AssertionError('Raw E2 cache ignored')), \
            patch('experiments.modules.mistake.datasets.MistakeCases.prepare_case',
                  side_effect=AssertionError('Existing E2 audio ignored')):
        variants = base.grid(dict(alignment_gamma_pitch=(4.,), alignment_gamma_time=(1.,)))
        result = base._mistake((coco, temporary, variants, frontend, (0,)))
        assert len(result)==1 and result[0]['dataset']=='coco'
        for file in Path(temporary).glob('frontend/*/*/provenance.json'):
            cfg = base.read(file)['config']
            assert all(cfg[k]==v for k,v in frontend.items() if k!='note_method')

    # Dispatch stays Coco-only; frontend is loaded from the linked winners.
    with TemporaryDirectory() as temporary:
        root=Path(temporary)
        base.atomic(root/'cohort.json',manifest)
        def fake_parallel(fn, tasks, workers):
            assert len(tasks)==195 and all(t[0]['dataset']=='coco' for t in tasks)
            assert all(t[3]==frontend for t in tasks)
            return []
        with patch.object(mixed,'selected_frontend',return_value=(frontend,{'test':'upstream'})), \
                patch.object(base,'parallel',side_effect=fake_parallel), \
                patch.object(base,'finish',return_value='ok'):
            assert base.run_mistake(root,axes=dict(alignment_gamma_pitch=(4.,),alignment_gamma_time=(1.,)))=='ok'
    for name in ['pitch','note','mistake']:
        notebook=json.loads((base.ROOT/f'experiments/sweeps/{name}.ipynb').read_text())
        for c in notebook['cells']:
            if c['cell_type']=='code':
                compile(''.join(c['source']),name,'exec')
                assert not c['outputs'] and c['execution_count'] is None
    print('Balanced sweep checks passed (no corpus sweep or fresh pitch inference).')


if __name__=='__main__':
    main()
