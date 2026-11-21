"""Detached, resumable method comparison and read-only partial reports."""
import os
from pathlib import Path
import subprocess
import sys
import traceback

from benchmarks.modules.sweeps import AppendixSweeps as base
from benchmarks.notebooks.sweeps.scripts import note_method_sweeps as runner


def _alive(pid):
    try:
        reaped, _ = os.waitpid(pid, os.WNOHANG)
        if reaped:
            return False
    except ChildProcessError:
        pass  # A reattached job may have been launched by a previous kernel.
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def start(output, *, methods=None, workers=8):
    """Return immediately; rerunning attaches to the same unfinished job."""
    output = Path(output).resolve()
    methods = tuple(runner.METHOD_CONFIGS) if methods is None else tuple(methods)
    variants = runner.variants_for(methods)
    _, _, root, _ = runner.prepare(output)
    directory = root / 'background' / base.digest(variants)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    launch_path = directory / 'launch.json'
    state_path = directory / 'state.json'
    if launch_path.exists():
        launch = base.read(launch_path)
        state = base.read(state_path) if state_path.exists() else {}
        if state.get('status') == 'complete' or (state.get('status') != 'failed' and _alive(launch['pid'])):
            return launch
    state_path.unlink(missing_ok=True)
    repo = next(p for p in Path(__file__).resolve().parents if (p / 'app.py').is_file())
    log = directory / 'worker.log'
    with log.open('ab') as stream:
        process = subprocess.Popen(
            [sys.executable, '-m', __name__, '--output', str(output),
             '--directory', str(directory), '--workers', str(workers),
             '--methods', *methods], cwd=repo, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True, stdin=subprocess.DEVNULL,
        )
    launch = dict(pid=process.pid, root=str(root), directory=str(directory),
                  methods=list(methods), log=str(log))
    base.atomic(launch_path, launch)
    return launch


def leaderboard(job, manifest):
    """Read completed checkpoints; never wait for workers or run inference."""
    import pandas as pd
    from benchmarks.notebooks.sweeps.scripts.note_sweep_reporting import speed_table
    import soundfile as sf
    root = Path(job['root'])
    source = base.read(root / 'contract.json')
    items = runner.method_selection_items(source['items'])
    variants = runner.variants_for(job['methods'])
    rows = []
    for item in items:
        for variant in variants:
            path = root / 'checkpoints' / item['track_id'] / (variant['variant'] + '.json')
            if path.exists():
                rows.append(base.read(path))
    counts = {v['method']: sum(r['method'] == v['method'] for r in rows) for v in variants}
    total = len(items)
    table = pd.DataFrame([dict(Method=runner.METHOD_LABELS[v['method']],
                               **v['params'], **{'Completed Tracks': counts[v['method']],
                               'Total Tracks': total,
                               'Complete': counts[v['method']] == total}) for v in variants])
    table = base.result_table(table)
    if rows:
        frame = pd.DataFrame(rows)
        summary = base.summarize_scores(frame, 'note')
        specs = {v['variant']: v for v in variants}
        summary['Method'] = summary['variant'].map(lambda v: runner.METHOD_LABELS[specs[v]['method']])
        summary = base.result_table(summary.drop(columns=['variant', 'score'], errors='ignore'), 'note')
        # Parameter columns already exist in the preview table.
        metrics = [c for c in summary if c not in table or c == 'Method']
        table = table.merge(summary[metrics], on='Method', how='left', validate='one_to_one')
        audio = {}
        for item in items:
            if item['track_id'] in set(frame.track_id):
                audio[item['track_id']] = sf.info(item['paths']['audio']).duration
        display_rows = base.result_table(frame.drop(columns=['variant', 'score'], errors='ignore'))
        display_rows['Method'] = frame.method.map(runner.METHOD_LABELS)
        speeds, keys = speed_table(display_rows, audio)
        table = table.merge(speeds, on=keys, how='left', validate='one_to_one')
        table = table.sort_values('Transcription F1', ascending=False, na_position='last')
    for name in ('Onset F1', 'Transcription F1', 'Audio (s) / Compute (s)'):
        if name not in table:
            table[name] = float('nan')
    state_path = Path(job['directory']) / 'state.json'
    state = base.read(state_path) if state_path.exists() else {}
    status = state.get('status', 'running' if _alive(job['pid']) else 'stopped; rerun launch cell to resume')
    print(f"{len(rows)}/{total * len(variants)} runs complete | {status}")
    print('Provisional: methods with different completed-track counts are not directly comparable.')
    if state.get('error'):
        print(state['error'])
    if state.get('result_directory'):
        print('Final reports:', state['result_directory'])
    return table.reset_index(drop=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--directory', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--methods', nargs='+', required=True)
    args = parser.parse_args()
    directory = Path(args.directory)
    base.atomic(directory / 'state.json', dict(status='running'))
    try:
        result = runner.run(args.output, stage='methods', methods=args.methods, workers=args.workers)
        from benchmarks.notebooks.sweeps.scripts.note_sweep_reporting import add_speed
        add_speed(result, base.read(Path(args.output) / 'cohort.json'))
        base.atomic(directory / 'state.json', dict(status='complete',
                    result_directory=str(result['directory']), selected_method=result['selected_method']))
    except BaseException:
        base.atomic(directory / 'state.json', dict(status='failed', error=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
