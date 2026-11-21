"""Full-cohort dispatch and cache migration checks without running a sweep."""
import sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / 'app.py').is_file())))
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
from benchmarks.notebooks.sweeps.scripts import note_method_sweeps as note
from benchmarks.modules.sweeps import AppendixSweeps as base


def main():
    items = [dict(track_id=f'coco-{i}-{j}', dataset='coco', instrument=str(i))
             for i in range(13) for j in range(15)]
    items += [dict(track_id=f'{d}-{i}', dataset=d, instrument=str(i))
              for d, n in [('bach10-original', 4), ('urmp', 13)] for i in range(n)]
    assert note.method_selection_items(items) == items
    assert len(note.method_selection_items(items[:195])) == 195
    try:
        note.method_selection_items(items[1:])
    except ValueError:
        pass
    else:
        raise AssertionError('Incomplete Coco cohort accepted')
    # Frozen pitch evidence survives only the exact audited defaults promotion.
    promotion = base.read(Path(note.__file__).with_name('production_promotion_2026_11_17.json'))
    old_identity = dict(code={p: c['before'] for p, c in promotion.items()}, packages={'numpy': 'fixed'})
    identity = dict(code={p: c['after'] for p, c in promotion.items()}, packages={'numpy': 'fixed'})
    upstream = dict(items=items, identity=old_identity)
    winner = dict(contract_sha256=base.digest(upstream), artifacts={i['track_id']: {} for i in items})
    note.validate_pitch_upstream(upstream, identity, items, winner)
    note.validate_pitch_upstream(upstream, old_identity, items, winner)
    for bad_identity, bad_items, bad_winner in (
        (dict(identity, packages={}), items, winner),
        (dict(identity, code=dict(identity['code'], unknown='changed')), items, winner),
        (identity, items[:-1], winner),
        (identity, items, dict(winner, contract_sha256='stale')),
        (identity, items, dict(winner, artifacts={})),
    ):
        try:
            note.validate_pitch_upstream(upstream, bad_identity, bad_items, bad_winner)
        except ValueError:
            pass
        else:
            raise AssertionError('Incompatible pitch provenance accepted')
    variants = note.variants_for()
    with TemporaryDirectory() as tmp:
        root = Path(tmp) / 'new'
        source = dict(runner_sha256='new', rbf_cost_sha256='same', evaluation_stage=note.EVALUATION_STAGE)
        prior = dict(source, runner_sha256='992183e5c81823432aa1e1e023d1099d4e66b50bcea9ce7a60abd4abadcf8b94')
        old = root.parent / base.digest(prior)[:16]
        base.contract(old, prior)
        for v in variants:
            directory = old / 'checkpoints' / items[0]['track_id']
            for suffix in ('.initial.pkl', '.refined.pkl'):
                base.atomic(directory / (v['variant'] + suffix), ['cached notes'], True)
            base.atomic(directory / (v['variant'] + '.json'), dict(
                method=v['method'], artifact='old', initial_artifact='old', **v['params']))
        base.atomic(old / 'best.json', {'old_winner': True})
        note.reuse_previous_checkpoints(root, source, items, variants)
        note.reuse_previous_checkpoints(root, source, items, variants)  # idempotent resume
        for v in variants:
            row = base.read(root / 'checkpoints' / items[0]['track_id'] / (v['variant'] + '.json'))
            assert base.blob(row['artifact']) == ['cached notes']
            assert base.blob(row['initial_artifact']) == ['cached notes']
        assert not (root / 'best.json').exists()
        assert (old / 'best.json').exists()
        # Changed inference dependencies cannot import these results.
        changed = root.parent / 'changed'
        note.reuse_previous_checkpoints(changed, dict(source, rbf_cost_sha256='different'), items, variants)
        assert not changed.exists()
        pitch = dict(params={}, artifacts={i['track_id']: {'path': 'unused'} for i in items})
        def parallel(fn, tasks, workers):
            assert len(tasks) == 212 and workers == 8
            assert sum(len(task[2]) for task in tasks) == 1272
            assert [t[0] for t in tasks] == items
            return []
        def finish(directory, batches, specs, selected):
            assert selected == items
            base.atomic(directory / 'best.json', {'method': 'kernelcpd-linear'})
            return None, None
        with patch.object(note, 'prepare', return_value=(items, pitch, root, source)), \
             patch.object(base, 'parallel', side_effect=parallel), \
             patch.object(note, '_finish', side_effect=finish):
            result = note.run(root, stage='methods', workers=8)
        assert len(base.read(result['directory'] / 'contract.json')['track_ids']) == 212
        assert len(base.read(root / 'method_selection.json')['track_ids']) == 212
    notebook = json.loads(Path('benchmarks/notebooks/sweeps/note.ipynb').read_text())
    for c in notebook['cells']:
        if c['cell_type'] == 'code':
            compile(''.join(c['source']), c['id'], 'exec')
    print('Full 212-track dispatch, 1,272 runs, all-method cache reuse, incompatible-cache rejection, and notebook syntax passed.')


if __name__ == '__main__':
    main()
