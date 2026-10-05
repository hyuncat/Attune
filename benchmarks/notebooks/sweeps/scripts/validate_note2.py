"""Validate the two frozen note2 settings on chorales excluded from tuning."""
from pathlib import Path
import sys
import argparse
import json
import hashlib
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import multiprocessing as mp
from contextlib import redirect_stdout
from contextlib import redirect_stderr
ROOT = next((p for p in Path(__file__).resolve().parents if (p / 'app.py').exists()))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
from app_logic.user.ds.Recording import Recording
from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker
from benchmarks.modules.pitch.datasets.CocoChorales import CocoChorales
from benchmarks.modules.note.NoteBenchmarker import CocoNoteBenchmarker
from benchmarks.modules.mistake.MistakeCache import MistakeCache
from benchmarks.modules.note.sweeps.InjectedNoteSweep import load_cases
from benchmarks.modules.note.sweeps.InjectedNoteSweep import _case_rows
from benchmarks.modules.note.sweeps.InjectedNoteSweep import _init_worker
from benchmarks.modules.note.sweeps.InjectedNoteSweep import summarize
from benchmarks.modules.note.sweeps.InjectedNoteSweep import digest
Recording.recompute_vibrato = lambda self, note_aware=True: None

def compare_case(task):
    case, variants, output = task
    path = Path(output) / 'logs' / f'{case['case_id']}.log'
    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open('w') as stream, redirect_stdout(stream), redirect_stderr(stream):
        return _case_rows(case, variants)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    output = ROOT / 'benchmarks/results/sweeps/note2_validation'
    corpus = output / 'corpus'
    output.mkdir(parents=True, exist_ok=True)
    tuning = pd.read_csv(ROOT / 'benchmarks/results/mistake_competitors_coco_extended/sample.csv')
    excluded = set(tuning['group'])
    selected = {}
    dataset_root = ROOT / 'benchmarks/datasets/cocochorales_tiny'
    coco = CocoChorales(root=dataset_root, split='test')
    locator = CocoNoteBenchmarker(root=dataset_root)
    records = sorted(coco.read_manifest('test'), key=lambda r: hashlib.sha256(f'1:{r.track_id}'.encode()).hexdigest())
    for record in records:
        if record.track in excluded or record.instrument in selected or record.instrument not in set(tuning.instrument):
            continue
        path = locator.local_midi_path(record)
        if path is None:
            continue
        selected[record.instrument] = dict(source=str(path.resolve()), dataset='coco', split=record.split, track_id=record.track_id, group=record.track, ensemble=record.ensemble, instrument=record.instrument, stem=record.stem, selection_seed=1)
    if len(selected) != len(tuning.instrument.unique()):
        raise ValueError('Could not select a fresh stem for every instrument')
    sample = pd.DataFrame(selected.values()).sort_values('instrument')
    assert set(sample['group']).isdisjoint(excluded)
    sample.to_csv(output / 'sample.csv', index=False)
    print(f'{len(sample)} fresh stems, no chorale overlap with tuning; two fixed settings only.', flush=True)
    source_metadata = sample.drop(columns='selection_seed').set_index('source').to_dict(orient='index')
    MistakeBenchmarker.run_comparison(sample.source.tolist(), corpus, seeds=(0, 1), rates=(0.0, 0.25), methods=('Attune (no refinement)',), input_kinds=('detected',), workers=args.workers, tolerances=(0.05, 0.1, 0.2), source_metadata=source_metadata)
    cases, promoted = load_cases(corpus)
    former = dict(cap_ms=None, score_factor=0.6, pitch_step=0.75, silence_ms=10.0)
    variants = [dict(variant=digest(p)[:16], baseline=label == 'promoted', setting=label, **p) for label, p in [('former', former), ('promoted', promoted)]]
    all_rows = []
    with MistakeBenchmarker.single_thread_environment(), ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'), initializer=_init_worker) as pool:
        futures = [pool.submit(compare_case, (c, variants, str(output))) for c in cases]
        try:
            for count, future in enumerate(as_completed(futures), 1):
                all_rows.extend(future.result())
                print(f'\r{count}/{len(cases)} validation cases complete', end='', flush=True)
        finally:
            for future in futures:
                future.cancel()
            print()
    rows = pd.DataFrame(all_rows)
    rows.to_csv(output / 'rows.csv', index=False)
    summary = summarize(rows)
    summary['setting'] = summary.variant.map({v['variant']: v['setting'] for v in variants})
    summary.to_csv(output / 'summary.csv', index=False)
    MistakeCache.atomic_json(output / 'validation.json', dict(status='complete', excluded_chorales=sorted(excluded), new_chorales=sorted(set(sample['group'])), variants=variants, cases=len(cases), protocol='Frozen note2 promotion versus former settings; no repeat correction or vibrato annotation. New chorales, same synthesis/injection distribution.'))
    print(summary[['setting', 'injected_audio_pitch100_f1', 'injected_note100_f1', 'clean_audio_pitch100_fp']].to_string(index=False))
if __name__ == '__main__':
    main()
