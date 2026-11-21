"""Bounded checks for RBF equivalence, partial reporting, and detached launch."""
import sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / 'app.py').is_file())))
import os
import tempfile
from time import perf_counter
from unittest.mock import patch, Mock
import numpy as np
import ruptures as rpt
from ruptures.costs import CostRbf
from benchmarks.notebooks.sweeps.scripts.fast_rbf import PrefixCostRbf
from benchmarks.notebooks.sweeps.scripts import note_background as bg
from benchmarks.notebooks.sweeps.scripts import note_method_sweeps as runner
from benchmarks.modules.sweeps import AppendixSweeps as base


def main():
    rng = np.random.default_rng(7)
    signals = [np.full((200, 1), 60.), rng.normal(size=(200, 2)),
               np.repeat([60., 62., 65., 60.], 100).reshape(-1, 1) + rng.normal(0, .05, (400, 1))]
    for signal in signals:
        original, fast = CostRbf().fit(signal), PrefixCostRbf().fit(signal)
        for _ in range(300):
            a, b = sorted(rng.choice(len(signal) + 1, 2, replace=False))
            np.testing.assert_allclose(fast.error(a, b), original.error(a, b), atol=1e-8, rtol=1e-8)
        for penalty in (.5, 5., 20.):
            expected = rpt.Pelt(model='rbf', min_size=10, jump=1).fit(signal).predict(pen=penalty)
            actual = rpt.Pelt(custom_cost=PrefixCostRbf(), min_size=10, jump=1).fit(signal).predict(pen=penalty)
            assert actual == expected, (actual, expected)
    signal = rng.normal(size=(1500, 1))
    timings = []
    for cls in (CostRbf, PrefixCostRbf):
        cost = cls().fit(signal)
        start = perf_counter()
        for end in range(750, 1501):
            cost.error(0, end)
        timings.append(perf_counter() - start)
    print(f'Segment-query microbenchmark: original {timings[0]:.3f}s; prefix {timings[1]:.3f}s')

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # A launch should return a handle immediately, and a second launch reattach.
        process = Mock(pid=os.getpid())
        with patch.object(runner, 'prepare', return_value=([], {}, root, {})), \
             patch.object(bg.subprocess, 'Popen', return_value=process) as launch:
            job = bg.start(root, workers=2)
            assert bg.start(root, workers=2) == job
            assert launch.call_count == 1
            assert launch.call_args.kwargs['start_new_session'] is True
        # Provisional tuning must not read an unfinished winner, and must keep all195 tracks.
        items = [dict(track_id=f'{i}-{j}', instrument=str(i)) for j in range(15) for i in range(13)]
        pitch = dict(params={}, artifacts={i['track_id']: {'path': 'unused'} for i in items})
        source = {}
        def finish(directory, batches, variants, selected_items):
            assert len(selected_items) == 195
            assert {v['method'] for v in variants} == {'kernelcpd-linear'}
            base.atomic(directory / 'best.json', {'method': 'kernelcpd-linear'})
            return None, None
        with patch.object(runner, 'prepare', return_value=(items, pitch, root, source)), \
             patch.object(runner, 'reuse_previous_checkpoints'), \
             patch.object(runner, 'winning_method_variants', side_effect=AssertionError('Requires winner')), \
             patch.object(base, 'parallel', return_value=[]) as parallel, \
             patch.object(runner, '_finish', side_effect=finish):
            result = runner.run(root, stage='parameters', selected_method='kernelcpd-linear',
                                axes={k: (v,) for k, v in runner.BASELINE.items()}, workers=6)
            assert 'parameters_provisional' in str(result['directory'])
            assert len(parallel.call_args.args[1]) == 195
            assert parallel.call_args.kwargs['workers'] == 6
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        items = [dict(track_id=f'{i}-{j}', instrument=str(i), paths={'audio': 'unused'})
                 for j in range(2) for i in range(13)]
        base.contract(root, {'items': items})
        variants = runner.variants_for()
        job = dict(root=str(root), directory=str(root / 'job'),
                   methods=list(runner.METHOD_CONFIGS), pid=os.getpid())
        empty = bg.leaderboard(job, {})
        assert len(empty) == 6 and empty['Completed Tracks'].sum() == 0
        for v, item in ((variants[0], items[0]), (variants[1], items[0]), (variants[1], items[1])):
            row = dict(track_id=item['track_id'], method=v['method'], variant=v['variant'],
                       compute_seconds=1., status='ok', reference_notes=5, estimated_notes=4, **v['params'],
                       **{k + suffix: value for suffix in ('', '_offset')
                          for k, value in [('tp', 3), ('fp', 1), ('fn', 2), ('precision', .75), ('recall', .6)]})
            base.atomic(root / 'checkpoints' / item['track_id'] / (v['variant'] + '.json'), row)
        with patch('soundfile.info', return_value=Mock(duration=2.)):
            partial = bg.leaderboard(job, {})
        assert len(partial) == 6 and partial['Completed Tracks'].sum() == 3
        assert not {'variant', 'score'} & set(partial)
        assert partial['Transcription F1'].notna().sum() == 2
        np.testing.assert_allclose(partial['Transcription F1'].dropna(), 2 / 3)
        np.testing.assert_allclose(partial['Audio (s) / Compute (s)'].dropna(), 2.)
    print('Empty and partial leaderboard counts, metrics, and speed reporting passed.')
    print('RBF cost/segmentation parity, detached reattachment, and independent provisional tuning passed.')


if __name__ == '__main__':
    main()
