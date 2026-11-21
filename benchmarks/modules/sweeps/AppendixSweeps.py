"""Linked pitch/note/alignment tuning with equal dataset weighting.

Workers live in an importable module for spawn-based Jupyter multiprocessing.
Only local, self-generated pickle caches are read. Never load untrusted caches.
"""
from __future__ import annotations

import hashlib
import gzip
import json
import multiprocessing as mp
import os
import pickle
import re
import queue
from contextlib import redirect_stdout, redirect_stderr
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / 'benchmarks/results/sweeps/balanced_212_v3'
PITCH_AXES = dict(posthoc_min_volume=(0.0, 0.02, 0.04, 0.05, 0.06, 0.08),
                  posthoc_unv_thresh=(0.95, 0.97, 0.98, 0.985, 0.9875, 0.99))
NOTE_AXES = dict(min_silence_duration_ms=(10., 20., 40., 60., 80.),
                 min_note_length_factor=(0.25, 0.5, 0.75),
                 min_note_length_cap=(0.05, 0.1, 0.2),
                 pitch_thresh=(0.25, 0.5, 0.6, 0.75, 1.0))
MISTAKE_AXES = dict(alignment_gamma_pitch=(0., 1., 2., 3., 4., 6., 8.),
                    alignment_gamma_time=(0., 0.5, 1., 2., 4.))


PARAMETER_LABELS = {
    'posthoc_min_volume': 'Min Volume',
    'posthoc_unv_thresh': 'Unv Thresh',
    'min_silence_duration_ms': 'Min Pause (ms)',
    'min_note_length_factor': 'Min Note Factor',
    'min_note_length_cap': 'Min Note Cap (s)',
    'pitch_thresh': 'Min Pitch Distance (semitones)',
    'alignment_gamma_pitch': 'Pitch Weight',
    'alignment_gamma_time': 'Timing Weight',
}
METRIC_LABELS = {
    'pitch': 'Overall Accuracy',
    'note': 'Transcription F1',
    'mistake': 'Mistake Event F1',
}
_progress_queue = None


def parameter_text(params):
    return ', '.join(f'{PARAMETER_LABELS.get(k, k)}={v:g}' for k, v in params.items())


def result_table(frame, stage=None):
    """Readable public tables; cache identifiers remain internal."""
    table = frame.drop(columns=['variant', 'artifact'], errors='ignore').copy()
    if stage == 'note':
        table = table.rename(columns={
            'f1': 'Onset F1', 'f1_offset': 'Transcription F1',
            'precision': 'Onset Precision', 'recall': 'Onset Recall',
            'precision_offset': 'Transcription Precision',
            'recall_offset': 'Transcription Recall',
        })
    if stage is not None and 'score' in table:
        metric = METRIC_LABELS[stage]
        if metric in table:
            table = table.drop(columns='score')
        else:
            table = table.rename(columns={'score': metric})
    return table.rename(columns={**PARAMETER_LABELS, 'clean_fp': 'Clean False Positives'}).reset_index(drop=True)


class StatusLine:
    """One display_id in notebooks, one carriage-return line in terminals."""
    def __init__(self):
        self.handle = None
        self.width = 0
        try:
            from IPython import get_ipython
            self.notebook = getattr(get_ipython(), 'kernel', None) is not None
        except ImportError:
            self.notebook = False

    def update(self, message):
        if self.notebook:
            from IPython.display import display
            content = {'text/plain': message}
            if self.handle is None:
                self.handle = display(content, raw=True, display_id=True)
            else:
                self.handle.update(content, raw=True)
        else:
            print('\r' + message.ljust(self.width), end='', flush=True)
            self.width = max(self.width, len(message))

    def close(self):
        if not self.notebook:
            print(flush=True)


def progress(item, variant, *, done=False, phase='Testing'):
    if _progress_queue is not None:
        _progress_queue.put((item['track_id'], variant['params'], done, phase))


def cached_progress(item, variants):
    for variant in variants:
        progress(item, variant, done=True, phase='Cached')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value, binary=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f'.{os.getpid()}.tmp')
    try:
        if binary:
            temp.write_bytes(gzip.compress(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL), compresslevel=1))
        else:
            temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def blob(path):
    return pickle.loads(gzip.decompress(Path(path).read_bytes()))


def read(path):
    return json.loads(Path(path).read_text())


def grid(axes):
    import math
    if not axes or any(not values for values in axes.values()):
        raise ValueError('Every axis must contain candidates')
    if any(not math.isfinite(v) or v < 0 for values in axes.values() for v in values):
        raise ValueError('Candidates must be finite and nonnegative')
    rows = [dict(zip(axes, values)) for values in product(*axes.values())]
    return [dict(params=p, variant=digest(p)[:16]) for p in rows
            if not ('alignment_gamma_pitch' in p and
                    p['alignment_gamma_pitch'] == p['alignment_gamma_time'] == 0)]


def initialize_worker(events=None):
    global _progress_queue
    _progress_queue = events
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[key] = '1'
    from threadpoolctl import threadpool_limits
    global _limits
    _limits = threadpool_limits(limits=1)


def _quiet_task(function, task):
    # Child prints/progress bars must not create competing notebook output lines.
    # Exceptions still propagate through Future.result() with their traceback.
    with open(os.devnull, 'w') as stream, redirect_stdout(stream), redirect_stderr(stream):
        return function(task)


def parallel(function, tasks, workers=8, *, status=None):
    initialize_worker()
    context = mp.get_context('spawn')
    events = context.Queue()
    own_status = status is None
    status = status or StatusLine()
    sweep = function.__name__ in ('_pitch', '_note', '_mistake')
    total = sum(len(task[2]) for task in tasks) if sweep else len(tasks)
    completed = 0
    result = []
    status.update(f'0/{total} | Starting {workers} workers')

    def render_event(event):
        nonlocal completed
        track, params, done, phase = event
        completed += int(done)
        current = completed if done else min(completed + 1, total)
        status.update(f'{current}/{total} | {track} | {parameter_text(params)} | {phase}')

    def drain():
        while True:
            try:
                event = events.get_nowait()
            except queue.Empty:
                break
            render_event(event)

    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                                 initializer=initialize_worker, initargs=(events,)) as pool:
            pending = {pool.submit(_quiet_task, function, task): task for task in tasks}
            while pending:
                ready, _ = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
                drain()
                for future in ready:
                    task = pending.pop(future)
                    result.append(future.result())
                    if not sweep:
                        completed += 1
                        if function.__name__ == '_export_pitch':
                            label = f"{task[0]['track_id']} | {parameter_text(task[2]['params'])} | Caching winner"
                        elif function.__name__ == '_materialize':
                            label = f"Materializing {len(task[1])} tracks"
                        else:
                            label = function.__name__
                        status.update(f'{completed}/{total} | {label}')
            # Drain buffered final events before joining the workers, so their
            # queue feeder threads cannot block shutdown on a full pipe.
            while sweep and completed < total:
                render_event(events.get(timeout=10))
        drain()
        if sweep and completed != total:
            raise RuntimeError(f'Incomplete progress accounting: {completed}/{total}')
    finally:
        events.close()
        events.join_thread()
        if own_status:
            status.close()
    return result


def code_identity():
    from importlib.metadata import version
    paths = [p for folder in ('algorithms', 'app_logic', 'benchmarks/modules')
             for p in (ROOT / folder).rglob('*.py') if '__pycache__' not in p.parts]
    return dict(code={str(p.relative_to(ROOT)): file_hash(p) for p in sorted(paths)},
                packages={p: version(p) for p in
                          ('numpy', 'scipy', 'librosa', 'numba', 'ruptures', 'music21', 'mir_eval')})


def cohort(output=DEFAULT_OUTPUT, *, root=None, split='test', seed=20261116,
           exclusion_files=None):
    """Freeze 15/instrument, excluding whole source pieces found in prior results.

    This is a new development cohort drawn from the locally retained partition;
    it does not turn the already consulted test set into a pristine holdout.
    A train partition can be supplied instead. No fallback to overlapping tracks.
    """
    from benchmarks.modules.pitch.datasets.CocoChorales import CocoChorales
    output = Path(output).resolve()
    coco = CocoChorales(root=root, split=split)
    selection = dict(root=str(coco.root.resolve()), split=split, seed=seed, count=195)
    path = output / 'cohort.json'
    if path.exists():
        saved = read(path)
        if saved['selection'] != selection:
            raise ValueError('Cohort selection changed; use a new output directory')
        return saved
    records = coco.load_or_build_manifest(split)
    if not records:
        raise FileNotFoundError(f'No {split} stems at {coco.root}; supply the dataset root')
    # Exclude any instrument from a used piece, including historical tuning runs.
    pattern = re.compile(r'(?:string|brass|woodwind|random)_track\d+')
    excluded, sources = set(), {}
    print('Auditing prior result CSV/JSON files; the first cohort freeze may take several minutes.', flush=True)
    files = list(exclusion_files) if exclusion_files is not None else [
        p for ext in ('*.csv', '*.json') for p in (ROOT / 'benchmarks/results').rglob(ext)
        if output not in p.parents]
    for name in sorted(map(Path, files)):
        matches = set()
        with name.open(errors='replace') as stream:
            for line in stream:
                matches.update(pattern.findall(line))
        if matches:
            excluded.update(matches)
            sources[str(name.resolve())] = file_hash(name)
    # Reserve the standard 20-per-instrument seed-0 test cohort even if its
    # reports have been moved out of this checkout.
    reserved = coco.read_manifest('test')
    if reserved:
        excluded.update(r.track for r in coco.sample_records_per_instrument(reserved, 20, 0))
    candidates = [r for r in records if r.track not in excluded]
    chosen = coco.sample_records_per_instrument(candidates, 15, seed)
    counts = Counter(r.instrument for r in chosen)
    if len(counts) != 13 or set(counts.values()) != {15} or len(chosen) != 195:
        raise ValueError(f'Need exactly 13 instruments × 15 tracks: {counts}')
    saved = dict(selection=selection, records=[asdict(r) for r in chosen],
                 excluded_pieces=sorted(excluded), exclusion_sources=sources,
                 instrument_counts=dict(counts),
                 policy='seeded per instrument; source-piece exclusion; development only')
    atomic(path, saved)
    return saved


def materialize(output=DEFAULT_OUTPUT, workers=8, **selection):
    from benchmarks.modules.sweeps.MixedCohort import materialize as mixed_materialize
    return mixed_materialize(output, workers=workers, **selection)


def materialize_legacy(output, workers=8, **selection):
    saved = cohort(output, **selection)
    groups = {}
    for row in saved['records']:
        groups.setdefault(row['shard'], []).append(row)
    parallel(_materialize, [(saved['selection']['root'], rows) for rows in groups.values()], workers)
    return saved


def _materialize(task):
    from benchmarks.modules.pitch.datasets.CocoChorales import CocoChorales
    root, rows = task
    coco = CocoChorales(root=root)
    coco.materialize_records([CocoChorales.Stem(**r) for r in rows])
    return len(rows)


def inputs(saved):
    from benchmarks.modules.pitch.datasets.CocoChorales import CocoChorales
    from benchmarks.modules.note.NoteBenchmarker import CocoNoteBenchmarker
    if 'items' in saved:
        from benchmarks.modules.sweeps.MixedCohort import validate_items
        return validate_items(saved['items'])
    coco = CocoChorales(root=saved['selection']['root'])
    locator = CocoNoteBenchmarker(root=coco.root)
    items = []
    for row in saved['records']:
        record = CocoChorales.Stem(**row)
        audio, midi = coco.local_wav_path(record), locator.local_midi_path(record)
        if audio is None or midi is None:
            raise FileNotFoundError(f'Materialize cohort first: {record.track_id}')
        paths = dict(audio=str(audio.resolve()), midi=str(midi.resolve()),
                     f0=str(coco.f0_path_for_record(record).resolve()))
        items.append(dict(track_id=record.track_id, instrument=record.instrument,
                          record=row, paths=paths,
                          hashes={k: file_hash(p) for k, p in paths.items()}))
    return items


def contract(directory, value):
    directory = Path(directory)
    path = directory / 'contract.json'
    if path.exists() and read(path) != value:
        raise ValueError(f'Inputs, code or grid changed at {directory}; choose a new run tag')
    atomic(path, value)


# Native-rate caches must NOT be upgraded as reporting-only changes now that
# the frontend uses the benchmark's 44.1 kHz analysis grid.
_PITCH_REPORTING_PREDECESSORS = set()


def pitch_contract(directory, value):
    """Migrate only the reviewed reporting update, preserving expensive evidence.

    All other code, dependencies, input hashes, cohort and grid must match.
    Do not use this exception for arbitrary algorithm changes.
    """
    from copy import deepcopy
    from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker
    import pandas as pd
    directory = Path(directory)
    path = directory / 'contract.json'
    if path.exists():
        old = read(path)
        helper = 'benchmarks/modules/sweeps/AppendixSweeps.py'
        previous_hash = old.get('identity', {}).get('code', {}).get(helper)
        normalized = deepcopy(old)
        if previous_hash in _PITCH_REPORTING_PREDECESSORS:
            normalized['identity']['code'][helper] = value['identity']['code'][helper]
        if old != value and normalized == value:
            # A reporting migration is cache-only: require every original row,
            # including frame counts, before accepting the old evidence.
            for item in value['items']:
                checkpoint = directory / item['track_id'] / 'rows.json'
                rows = read(checkpoint)
                expected = {v['variant']: v['params'] for v in value['variants']}
                if (len(rows) != len(expected)
                        or {r['variant'] for r in rows} != set(expected)
                        or any(r['track_id'] != item['track_id']
                               or r['instrument'] != item['instrument']
                               or any(r[k] != v for k, v in expected[r['variant']].items())
                               for r in rows)
                        or not PitchBenchmarker.current_scores(pd.DataFrame(rows))):
                    raise ValueError(f'Incompatible cached pitch rows: {checkpoint}')
            archive = directory / 'report_history' / digest(old)[:16]
            for name in ('contract.json', 'rows.csv', 'summary.csv', 'best.json'):
                source = directory / name
                if source.exists():
                    target = archive / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        target.write_bytes(source.read_bytes())
            atomic(directory / 'report_migration.json', dict(
                reason='Reporting-only upgrade; recompute pooled pitch metrics from saved frame counts',
                previous_contract=digest(old), current_contract=digest(value),
                preserved_reports=str(archive), tracks=len(value['items']),
            ))
            atomic(path, value)
    contract(directory, value)


def load_winner(output, stage):
    path = Path(output) / stage / 'best.json'
    if not path.exists():
        raise RuntimeError(f'Complete {stage}.ipynb first: {path}')
    winner = read(path)
    for artifact in winner['artifacts'].values():
        if file_hash(artifact['path']) != artifact['sha256']:
            raise ValueError(f'Winner cache changed: {artifact["path"]}')
    return winner


def _recording(item, overrides, pitches=None):
    from benchmarks.modules.note.NoteBenchmarker import OneInstrumentScoreData
    from benchmarks.modules.pitch.competitors.Attune import Attune
    from algorithms.Config import Config
    score = OneInstrumentScoreData(item['paths']['midi'])
    low, high = Config.padded_midi_range(min(score.midi_numbers), max(score.midi_numbers))
    # Match the offline pitch/note benchmark's fixed 44.1 kHz analysis grid.
    # Native 16 kHz with the same 4096/128 sample window/hop changes the detector.
    config = Attune().config_for(*Attune.range_from_midi([low, high]), **overrides)
    rec = Attune().recording_for(config, score_data=score)
    if pitches is not None:
        rec.pitch_data = Attune._pitch_data(pitches, config, 0.)
        rec.pitch_data.end_index = len(pitches)
        for pitch in pitches:
            if pitch is not None:
                pitch.config = config
    return rec


def _pitch(task):
    from copy import deepcopy
    from algorithms.PitchDetector import PitchDetector
    from algorithms.PitchSmoother import PitchSmoother, VoicingSmoother
    from app_logic.user.ds.AudioData import AudioData
    from benchmarks.modules.pitch.competitors.Attune import Attune
    from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker
    from benchmarks.modules.pitch.datasets.CocoChorales import CocoChorales
    item, directory, variants, root = task
    directory = Path(directory) / item['track_id']
    checkpoint = directory / 'rows.json'
    cached = {r['variant']: r for r in read(checkpoint)} if checkpoint.exists() else {}
    for v in variants:
        if v['variant'] in cached and any(cached[v['variant']][k] != value for k, value in v['params'].items()):
            raise ValueError('Cached pitch parameters do not match variant')
    if all(v['variant'] in cached for v in variants):
        cached_progress(item, variants)
        return [cached[v['variant']] for v in variants]
    progress(item, variants[0], phase='Preparing pitch candidates')
    rec = _recording(item, {})
    raw_path = directory / 'raw.pkl'
    if raw_path.exists():
        raw = blob(raw_path)
    else:
        import librosa
        samples, _ = librosa.load(item['paths']['audio'], sr=rec.config.sr, mono=True)
        raw = PitchDetector(config=rec.config).detect_pitches(samples, show_progress=False, verbose=False)
        atomic(raw_path, raw, True)
    if item.get('dataset', 'coco') == 'coco':
        coco = CocoChorales(root=root)
        track = coco.records_to_tracks([CocoChorales.Stem(**item['record'])])[0]
        ref_t, ref_f = coco.reference(track)
    else:
        from benchmarks.modules.sweeps.MixedCohort import pitch_reference
        ref_t, ref_f = pitch_reference(item)
    rows = []
    for v in variants:
        if v['variant'] in cached:
            rows.append(cached[v['variant']])
            progress(item, v, done=True, phase='Cached')
            continue
        progress(item, v)
        path = directory / (v['variant'] + '.pkl')
        cfg = replace(rec.config, **v['params'])
        if path.exists():
            pitches = blob(path)
        else:
            # Volume affects HMM tracking regions as well as the final gate.
            # Confidence only affects the final gate; reuse HMM per volume.
            tracking = directory / ('tracking_' + digest(dict(volume=cfg.posthoc_min_volume))[:16] + '.pkl')
            if tracking.exists():
                tracked = blob(tracking)
            else:
                tracked = PitchSmoother(config=cfg).smooth(deepcopy(raw), verbose=False)
                atomic(tracking, tracked, True)
            pitches = VoicingSmoother(config=cfg).smooth(tracked)
        times, hz = Attune.melody(Attune._pitch_data(pitches, cfg, 0.), cfg)
        metrics = PitchBenchmarker.score_frames(ref_t, ref_f, times, hz)
        rows.append(dict(track_id=item['track_id'], instrument=item['instrument'], dataset=item.get('dataset', 'coco'),
                         variant=v['variant'], score=metrics['Overall Accuracy'],
                         artifact=str(path), **metrics, **v['params']))
        progress(item, v, done=True, phase='Tested')
    atomic(checkpoint, rows)
    return rows


def _note_metrics(reference, detected, config):
    from benchmarks.modules.note.NoteBenchmarker import (
        NoteBenchmarker, NoteEvaluation, NotebookConfig,
    )
    ref_iv, ref_hz = reference
    est_iv, est_hz = NoteBenchmarker.notedata_to_intervals(detected, config)
    metrics = NoteEvaluation.score_predictions(
        ref_iv, ref_hz, est_iv, est_hz, NotebookConfig(),
    )
    metrics.update(status='ok', reference_notes=len(ref_hz), estimated_notes=len(est_hz))
    metrics.update(NoteEvaluation.note_event_counts(metrics))
    return metrics


def f1(tp, fp, fn):
    return 2 * tp / max(1, 2 * tp + fp + fn)


def _note(task):
    from copy import deepcopy
    from benchmarks.modules.note.NoteBenchmarker import OneInstrumentScoreData
    item, directory, variants, pitch_params, pitch_path = task
    directory = Path(directory) / item['track_id']
    checkpoint = directory / 'rows.json'
    if checkpoint.exists():
        cached_progress(item, variants)
        return read(checkpoint)
    pitches = blob(pitch_path)
    from benchmarks.modules.note.NoteBenchmarker import NoteEvaluation
    reference = NoteEvaluation.reference_notes(dict(dataset='coco', reference=item['paths']['midi']))
    rows = []
    for v in variants:
        progress(item, v)
        path = directory / (v['variant'] + '.pkl')
        rec = _recording(item, dict(pitch_params, **v['params']), deepcopy(pitches))
        if path.exists():
            notes = blob(path)
        else:
            rec.detect_notes()
            notes = rec.note_data
            atomic(path, notes, True)
        metrics = _note_metrics(reference, notes, rec.config)
        rows.append(dict(track_id=item['track_id'], instrument=item['instrument'], dataset=item.get('dataset', 'coco'),
                         variant=v['variant'], score=metrics['f1_offset'],
                         artifact=str(path), **metrics, **v['params']))
        progress(item, v, done=True, phase='Tested')
    atomic(checkpoint, rows)
    return rows


def summarize_scores(frame, stage):
    if 'dataset' in frame and frame.dataset.nunique() > 1:
        import pandas as pd
        parts = [summarize_scores(group.drop(columns='dataset'), stage).assign(dataset=name)
                 for name, group in frame.groupby('dataset')]
        per_dataset = pd.concat(parts, ignore_index=True)
        counts = {'tp', 'fp', 'fn', 'tp_offset', 'fp_offset', 'fn_offset', 'clean_fp'}
        numeric = per_dataset.select_dtypes('number').columns
        metrics = [c for c in numeric if c not in counts]
        result = per_dataset.groupby('variant')[metrics].mean()
        for col in counts & set(per_dataset):
            result[col] = per_dataset.groupby('variant')[col].sum()
        return result.reset_index()
    if stage == 'pitch':
        from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker
        summary = PitchBenchmarker.pool_scores(frame, by='variant').reset_index()
        summary['score'] = summary['Overall Accuracy']
        return summary
    if stage == 'note':
        from benchmarks.modules.note.NoteBenchmarker import NoteEvaluation
        # Delegate aggregation too: this is the same pooled-count report used
        # by NoteNotebook, including both onset-only and offset-aware metrics.
        prepared = frame.assign(method=frame.variant, dataset='coco',
                                audio_seconds=0., cpu_seconds=0.)
        summary = NoteEvaluation.summarize(prepared, by_dataset=False)
        columns = ['method'] + [key + suffix for suffix in ('', '_offset')
                                for key in ('tp', 'fp', 'fn', 'precision', 'recall', 'f1')]
        summary = summary[columns].rename(columns={'method': 'variant'})
        summary['score'] = summary.f1_offset
        return summary
    if stage == 'mistake':
        # Match MistakeBenchmarker.summarize / the E2 notebook: pooled
        # audio_pitch counts at 100 ms / 50 cents on injected cases only.
        # Clean-control false alarms are reported separately.
        summary = frame.groupby('variant')[['tp', 'fp', 'fn', 'clean_fp']].sum().reset_index()
        denominator = 2 * summary.tp + summary.fp + summary.fn
        summary['score'] = 2 * summary.tp / denominator.replace(0, float('nan'))
        return summary
    raise ValueError(f'Unknown sweep stage: {stage}')


def finish(directory, batches, variants, provenance, stage):
    import pandas as pd
    frame = pd.DataFrame([r for batch in batches for r in batch])
    if frame.empty or frame.duplicated(['track_id', 'variant']).any():
        raise ValueError('Empty or duplicate sweep rows')
    expected_ids = {i['track_id'] for i in provenance['items']}
    if (set(frame.variant) != {v['variant'] for v in variants}
            or any(set(g.track_id) != expected_ids for _, g in frame.groupby('variant'))):
        raise ValueError('Incomplete candidate coverage; no winner can be published')
    if 'dataset' in frame:
        per_dataset = pd.concat([summarize_scores(g.drop(columns='dataset'), stage).assign(dataset=d)
                                 for d, g in frame.groupby('dataset')], ignore_index=True)
        specs = {v['variant']: v['params'] for v in variants}
        for name in variants[0]['params']:
            per_dataset[name] = per_dataset.variant.map(lambda v: specs[v][name])
        result_table(per_dataset, stage).to_csv(Path(directory) / 'by_dataset.csv', index=False)
    summary = summarize_scores(frame, stage)
    summary = summary.sort_values(['score', 'variant'], ascending=[False, True], kind='stable')
    params = {v['variant']: v['params'] for v in variants}
    for name in variants[0]['params']:
        summary[name] = summary.variant.map(lambda v: params[v][name])
    directory = Path(directory)
    result_table(frame, stage).to_csv(directory / 'rows.csv', index=False)
    result_table(summary, stage).to_csv(directory / 'summary.csv', index=False)
    selected = summary.iloc[0].variant
    artifacts = {r.track_id: dict(path=r.artifact, sha256=file_hash(r.artifact))
                 for r in frame[frame.variant == selected].itertuples()}
    best = dict(stage=stage, variant=selected, params=params[selected],
                score=float(summary.iloc[0].score), contract_sha256=digest(provenance),
                artifacts=artifacts, interpretation='development selection, not held-out evaluation')
    atomic(directory / 'best.json', best)
    return result_table(summary, stage)


def run_pitch(output=DEFAULT_OUTPUT, *, axes=None, workers=8):
    output = Path(output).resolve()
    saved = read(output / 'cohort.json')
    items = inputs(saved)
    variants = grid(PITCH_AXES if axes is None else axes)
    meta = dict(cohort=saved, items=items, variants=variants, identity=code_identity())
    directory = output / 'pitch'
    from benchmarks.modules.sweeps.MixedCohort import extend_pitch_contract
    extend_pitch_contract(directory, meta)
    pitch_contract(directory, meta)
    if saved.get('reuse_pitch_from'):
        from benchmarks.modules.sweeps.MixedCohort import reuse_pitch
        reuse_pitch(saved['reuse_pitch_from'], directory, meta)
    status = StatusLine()
    batches = parallel(_pitch, [(i, str(directory), variants, saved['selection']['root']) for i in items], workers, status=status)
    import pandas as pd
    frame = pd.DataFrame([r for batch in batches for r in batch])
    scores = summarize_scores(frame, 'pitch').set_index('variant')['score']
    selected = sorted(scores.index, key=lambda v: (-scores[v], v))[0]
    winner = next(v for v in variants if v['variant'] == selected)
    try:
        parallel(_export_pitch, [(i, str(directory), winner) for i in items], workers, status=status)
    finally:
        status.close()
    return finish(directory, batches, variants, meta, 'pitch')


def run_note(output=DEFAULT_OUTPUT, *, axes=None, workers=8):
    output = Path(output).resolve()
    saved = read(output / 'cohort.json')
    items = inputs(saved)
    winner = load_winner(output, 'pitch')
    variants = grid(NOTE_AXES if axes is None else axes)
    meta = dict(cohort=saved, items=items, variants=variants, pitch_winner=winner, identity=code_identity())
    # Refuse an upstream run produced by different code or cohort.
    upstream = read(output / 'pitch/contract.json')
    if upstream['identity'] != meta['identity'] or upstream['items'] != items:
        raise ValueError('Pitch inputs/code changed; use a new run directory')
    directory = output / 'note'
    contract(directory, meta)
    batches = parallel(_note, [(i, str(directory), variants, winner['params'],
                              winner['artifacts'][i['track_id']]['path']) for i in items], workers)
    return finish(directory, batches, variants, meta, 'note')


def _mistake(task):
    from copy import deepcopy
    from algorithms.Config import Config
    from app_logic.user.ds.AudioData import AudioData
    from benchmarks.modules.mistake.datasets.MistakeCases import prepare_case
    from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker
    from benchmarks.modules.mistake.MistakeDetectorBase import MistakeDetectorBase as Events
    from benchmarks.modules.note.NoteBenchmarker import OneInstrumentScoreData
    from benchmarks.modules.pitch.competitors.Attune import Attune
    item, directory, variants, frontend, seeds = task
    directory = Path(directory)
    rows = []
    # One clean control, plus independent E2 injections of the same source.
    conditions = [(0, 0.)] + [(seed, .25) for seed in seeds]
    cases = []
    for seed, rate in conditions:
        progress(item, variants[0], phase=f'Preparing E2 frontend (seed={seed}, rate={rate:g})')
        folder = directory / 'frontend' / item['track_id'] / f'{seed}_{rate}'
        path = folder / 'notes.pkl'
        if path.exists():
            cached = blob(path)
        else:
            from benchmarks.modules.sweeps.MixedCohort import reusable_e2, restore_e2_raw
            reusable = reusable_e2(item, seed, rate)
            if reusable is not None:
                bench, asset = MistakeBenchmarker(), reusable['asset']
            else:
                bench, asset = prepare_case(item['paths']['midi'], seed, rate,
                    directory / 'e2', source_info=dict(dataset='coco', instrument=item['instrument'],
                    track_id=item['track_id']), prepare_audio=True, prepare_pitches=False)
            reference = OneInstrumentScoreData(item['paths']['midi']).note_data
            performed = OneInstrumentScoreData(asset['midi']).note_data
            cfg = bench.config_for_performance(performed)
            cfg = replace(cfg, **{k: v for k, v in frontend.items() if k != 'note_method'})
            audio = AudioData(audio_filepath=asset['audio'], config=cfg)
            cfg.sr = audio.sr
            rec = bench.attune.recording_for(cfg, score_data=OneInstrumentScoreData(item['paths']['midi']))
            rec.audio_data = audio
            # Stage checkpoint survives a later segmentation failure.
            if not restore_e2_raw(rec, reusable):
                bench.attune.load_or_detect_pitches(rec, folder / 'pitch.pkl.xz')
            rec.pitch_data.end_index = len(rec.pitch_data.data)
            from benchmarks.notebooks.sweeps.scripts.note_method_sweeps import detect_notes
            rec.note_data = detect_notes(rec, frontend.get('note_method', 'kernelcpd-linear'))
            truth_payload = read(asset['truth'])
            score_onsets = Events.timeline_score_onsets(truth_payload, performed)
            net, _ = Events.net_mistakes(reference, performed, pitch_tolerance=cfg.pitch_tolerance,
                                        duration_tolerance=cfg.timing_tolerance, score_onsets=score_onsets)
            truth = Events.truth_events(net, reference, performed)
            if rate == 0 and truth:
                raise ValueError('E2 clean control has net pitch errors')
            cached = dict(config=asdict(cfg), pitches=rec.pitch_data.data, notes=rec.note_data,
                          truth=truth, source=asset, rate=rate, seed=seed)
            atomic(path, cached, True)
            atomic(folder / 'provenance.json', dict(source=asset,
                   hashes={k: file_hash(asset[k]) for k in ('audio', 'midi', 'truth')},
                   config=asdict(cfg), truth=truth, notes_sha256=file_hash(path)))
        cases.append(cached)
    for v in variants:
        progress(item, v)
        path = directory / 'predictions' / item['track_id'] / (v['variant'] + '.json')
        if path.exists():
            rows.append(read(path)['row'])
            progress(item, v, done=True, phase='Cached')
            continue
        totals = [0, 0, 0]
        clean_fp = 0
        details = []
        for cached in cases:
            cfg = Config(**dict(cached['config'], **v['params']))
            reference = OneInstrumentScoreData(item['paths']['midi']).note_data
            rec = Attune().recording_for(cfg, score_data=OneInstrumentScoreData(item['paths']['midi']))
            rec.pitch_data = Attune._pitch_data(deepcopy(cached['pitches']), cfg, 0.)
            rec.pitch_data.end_index = len(rec.pitch_data.data)
            rec.note_data = deepcopy(cached['notes'])
            rec.align_score_and_refine()
            predicted = Events.predicted_events(Events.with_reference_score_ids(
                MistakeBenchmarker.label_pairs(rec.alignment.pairs, cfg), reference,
                rec.score_data.note_data), reference)
            tp, fp, fn = Events.score_events(predicted, cached['truth'], .1)['audio_pitch']
            if cached['rate'] > 0:
                totals = [a + b for a, b in zip(totals, (tp, fp, fn))]
            else:
                clean_fp += fp
            details.append(dict(seed=cached['seed'], rate=cached['rate'], tp=tp, fp=fp, fn=fn,
                                predicted=predicted, truth=cached['truth']))
        tp, fp, fn = totals
        row = dict(track_id=item['track_id'], instrument=item['instrument'], dataset=item.get('dataset', 'coco'),
                   variant=v['variant'], tp=tp, fp=fp, fn=fn, clean_fp=clean_fp,
                   score=f1(tp, fp, fn), artifact=str(path), **v['params'])
        atomic(path, dict(row=row, cases=details))
        rows.append(row)
        progress(item, v, done=True, phase='Tested')
    return rows


def run_mistake(output=DEFAULT_OUTPUT, *, axes=None, workers=8, frontend=None, seeds=(0, 1)):
    """Coco-only E2 frontend selected by completed mixed-dataset upstream sweeps."""
    from algorithms.Config import Config
    from benchmarks.modules.note.NoteBenchmarker import NoteBenchmarker
    output = Path(output).resolve()
    saved = read(output / 'cohort.json')
    items = [i for i in inputs(saved) if i.get('dataset', 'coco') == 'coco']
    variants = grid(MISTAKE_AXES if axes is None else axes)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Provide distinct injection seeds')
    from benchmarks.modules.sweeps.MixedCohort import selected_frontend
    selected, upstream = selected_frontend(output)
    if frontend is not None and frontend != selected:
        raise ValueError('Frontend must match the completed pitch and note winners')
    frontend = selected
    if any(k.startswith('alignment_') for k in frontend):
        raise ValueError('Pass alignment weights through axes, not frontend')
    base = dict(cohort=saved, items=items, frontend=frontend, upstream=upstream, defaults=asdict(Config()),
                seeds=list(seeds), identity=code_identity(),
                soundfont_sha256=file_hash(NoteBenchmarker.SOUNDFONT_PATH),
                protocol='E2 rate .25; equal five edit types; one clean control; event gate .1 s/.5 semitone')
    directory = output / 'mistake' / digest(base)[:16]
    contract(directory, base)
    meta = dict(base, variants=variants)
    # Separate reports for each grid, shared frontend and per-variant checkpoints.
    report = directory / 'grids' / digest(variants)[:16]
    contract(report, meta)
    batches = parallel(_mistake, [(i, str(directory), variants, frontend, tuple(seeds)) for i in items], workers)
    return finish(report, batches, variants, meta, 'mistake')


def _export_pitch(task):
    from algorithms.PitchSmoother import VoicingSmoother
    item, directory, variant = task
    directory = Path(directory) / item['track_id']
    path = directory / (variant['variant'] + '.pkl')
    if not path.exists():
        cfg = _recording(item, variant['params']).config
        tracking = directory / ('tracking_' + digest(dict(volume=cfg.posthoc_min_volume))[:16] + '.pkl')
        atomic(path, VoicingSmoother(config=cfg).smooth(blob(tracking)), True)
    return str(path)
