"""Fresh frontend, fixed production HMM, cheap final-gate sweep on paper tracks.

Ordinary pitch caches are neither read nor written. This experiment persists
pre-gate arrays in its own directory, so changing the final grid only rescores.
The HMM's region-selection threshold stays at the production baseline: this is
not a sweep of the threshold used internally by PitchSmoother.decode.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BASELINE = ROOT / "benchmarks/results/pitch/notebook_runs/paired_analysis/offline/4f615702b8c6/paired_rows.csv"
DEFAULT_OUTPUT = ROOT / "benchmarks/results/pitch/final_voicing_449_2026-10-08"
DEFAULT_THRESHOLDS = (0.50, 0.70, 0.80, 0.85, 0.90, 0.925, 0.95, 0.96,
                      0.97, 0.975, 0.98, 0.9825, 0.985, 0.9875, 0.99)


def recommended_workers():
    """Leave two cores and desktop memory free; one numerical thread per job."""
    import psutil
    memory = psutil.virtual_memory()
    reserve = max(2 * 1024**3, int(memory.total * 0.15))
    memory_workers = max(1, int((memory.available - reserve) / (1.5 * 1024**3)))
    return max(1, min((os.cpu_count() or 2) - 2, memory_workers))


def _initialize_worker():
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[name] = '1'
    from threadpoolctl import threadpool_limits
    global _thread_limit
    _thread_limit = threadpool_limits(limits=1)


def _fingerprint():
    paths = [Path(__file__), ROOT / 'algorithms/Config.py',
             ROOT / 'algorithms/PitchDetector.py', ROOT / 'algorithms/PitchSmoother.py',
             ROOT / 'benchmarks/modules/pitch/competitors/Attune.py',
             ROOT / 'benchmarks/modules/pitch/PitchBenchmarker.py']
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def _plan(baseline):
    import pandas as pd
    from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker
    frame = pd.read_csv(baseline)
    frame = frame[frame.model.eq('attune')].copy()
    if frame.duplicated(['dataset', 'track_id']).any():
        raise ValueError('Duplicate baseline tracks')
    if frame.groupby('dataset').size().to_dict() != {'bach10-original': 40, 'coco': 260, 'urmp': 149}:
        raise ValueError('Expected the frozen 449-track offline cohort')
    jobs = []
    for name, group in frame.groupby('dataset'):
        options = PitchBenchmarker.Options(datasets=(name,), per_instrument=20 if name == 'coco' else None, seed=0)
        dataset = PitchBenchmarker.dataset_for(name, options)
        tracks = {t.track_id: t for t in dataset.tracks()}
        if set(tracks) != set(group.track_id):
            raise ValueError(f'{name}: selection differs from frozen paper cohort')
        jobs.extend((tracks[row['track_id']], row) for row in group.to_dict('records'))
    return jobs


def score_grid(evidence, thresholds, fmin, fmax):
    """Only array masking and the ordinary 10 ms frame scorer per threshold."""
    import numpy as np
    from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker
    from benchmarks.modules.pitch.PitchDetectorBase import PitchDetectorBase
    rows = []
    for threshold in thresholds:
        keep = evidence['eligible'] & (evidence['unvoiced'] < threshold)
        hz = PitchDetectorBase.constrain_freqs_to_range(
            np.where(keep, evidence['hz'], 0.0), fmin, fmax)
        rows.append({'threshold': threshold, **PitchBenchmarker.score_frames(
            evidence['ref_times'], evidence['ref_freqs'], evidence['times'], hz)})
    return rows


def _track(track, baseline, output, thresholds, fingerprint):
    from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker
    with PitchBenchmarker.silence_runtime():
        return _track_quiet(track, baseline, Path(output), thresholds, fingerprint)


def _track_quiet(track, baseline, output, thresholds, fingerprint):
    from dataclasses import replace
    import numpy as np
    from algorithms.PitchDetector import PitchDetector
    from algorithms.PitchSmoother import PitchSmoother, VoicingSmoother
    from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker, COUNT_COLUMNS
    from benchmarks.modules.pitch.competitors.Attune import Attune
    dataset = PitchBenchmarker.dataset_for(track.dataset, PitchBenchmarker.Options())
    example = dataset.example(track)
    if not np.allclose([example.fmin, example.fmax], [baseline['fmin'], baseline['fmax']], rtol=0, atol=1e-10):
        raise ValueError(f'Changed pitch range: {track.track_id}')
    from algorithms.Config import PYIN_TRACKING_UNV_THRESH
    config = replace(Attune().config_for(example.fmin, example.fmax),
                     posthoc_unv_thresh=PYIN_TRACKING_UNV_THRESH)
    source = example.audio_path.resolve()
    stat = source.stat()
    identity = dict(dataset=track.dataset, track=track.track_id, source=str(source),
                    size=stat.st_size, mtime=stat.st_mtime_ns, code=fingerprint,
                    config=repr(config), baseline=baseline['reference_hash'],
                    reference=hashlib.sha256(example.ref_times.tobytes() + example.ref_freqs.tobytes()).hexdigest())
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = output / 'evidence' / f'{key}.npz'
    reused = path.exists()
    if reused:
        with np.load(path, allow_pickle=False) as cached:
            evidence = {k: cached[k] for k in cached.files}
    else:
        # Disable only the live value gate. Candidate probabilities and RMS stay
        # intact; production HMM receives its ordinary, fixed configuration.
        audio, _ = example.audio(config.sr)
        raw = PitchDetector(config=replace(config, unv_thresh=1.0, min_volume=0.0)).detect_pitches(
            audio, show_progress=False, verbose=False)
        tracked = PitchSmoother(config=config).smooth(raw, show_progress=False, verbose=False)
        present = [p for p in tracked if p is not None]
        times = np.asarray([p.time for p in present])
        bins_per_semitone = int(np.ceil(1 / PitchSmoother.RESOLUTION))
        midi = np.asarray([p.value for p in present])
        bins = np.round((midi - config.freq_to_midi(config.fmin)) * bins_per_semitone)
        hz = config.fmin * 2.0 ** (bins / (12 * bins_per_semitone))
        volume = np.asarray([max(0.0, p.volume) for p in present])
        floor = VoicingSmoother(config=config).volume_floor(volume)
        # Independent control: apply the real production controller to the
        # same tracked path, then its normal melody conversion and range gate.
        controlled = VoicingSmoother(config=config).smooth(tracked)
        control_times, control_hz = Attune.melody(Attune._pitch_data(controlled, config, 0.0), config)
        control_hz = Attune.constrain_freqs_to_range(control_hz, example.fmin, example.fmax)
        production = PitchBenchmarker.score_frames(example.ref_times, example.ref_freqs, control_times, control_hz)
        evidence = dict(times=times, hz=hz,
                        production_counts=np.asarray([production[k] for k in COUNT_COLUMNS], dtype=np.int64),
                        unvoiced=np.asarray([p.unvoiced_prob for p in present]),
                        eligible=(midi != -1) & (volume >= floor) & np.asarray([bool(p.candidate_pitches) for p in present]),
                        ref_times=example.ref_times, ref_freqs=example.ref_freqs)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as handle:
            np.savez_compressed(handle, **evidence)
        temporary.replace(path)
    scored = score_grid(evidence, thresholds, example.fmin, example.fmax)
    control = next(r for r in scored if r['threshold'] == config.posthoc_unv_thresh)
    if control['reference_hash'] != baseline['reference_hash']:
        raise RuntimeError(f'{track.dataset}/{track.track_id}: reference changed')
    for k, expected in zip(COUNT_COLUMNS, evidence['production_counts']):
        if control[k] != int(expected):
            raise RuntimeError(f'{track.dataset}/{track.track_id}: mask differs from fresh production controller: {k}')
    paper_matches = all(int(baseline[k]) == control[k] for k in COUNT_COLUMNS)
    for row in scored:
        row.update(dataset=track.dataset, track_id=track.track_id,
                   instrument=baseline['instrument'], source_piece=baseline['source_piece'],
                   paper_baseline_counts_match=paper_matches)
        row.update({f'paper_{k}': int(baseline[k]) for k in COUNT_COLUMNS})
    return scored, reused


def summarize(frame):
    import pandas as pd
    from benchmarks.modules.pitch.PitchBenchmarker import METRIC_COUNTS, COUNT_COLUMNS
    parts = []
    for label, selected in [('All', frame), ('Real audio', frame[frame.dataset.ne('coco')]),
                            ('CocoChorales', frame[frame.dataset.eq('coco')])]:
        if selected.empty:
            continue
        totals = selected.groupby('threshold')[list(COUNT_COLUMNS)].sum()
        result = pd.DataFrame({'Tracks': selected.groupby('threshold').size()})
        for metric, (numerator, denominator) in METRIC_COUNTS.items():
            result[metric] = 100 * totals[numerator] / totals[denominator].replace(0, float('nan'))
        result.insert(0, 'Group', label)
        parts.append(result.reset_index())
    return pd.concat(parts, ignore_index=True)


class _Status:
    def __init__(self):
        self.handle = None
        self.width = 0
        try:
            from IPython import get_ipython
            self.notebook = get_ipython() is not None and get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
        except ImportError:
            self.notebook = False

    def update(self, done, total, message):
        text = f'[{done}/{total}] {message}'
        if self.notebook:
            from IPython.display import display
            if self.handle is None:
                self.handle = display({'text/plain': text}, raw=True, display_id=True)
            else:
                self.handle.update({'text/plain': text}, raw=True)
        else:
            print('\r' + text.ljust(self.width), end='', flush=True)
            self.width = max(self.width, len(text))

    def finish(self):
        if not self.notebook:
            print()


def run_sweep(*, thresholds=DEFAULT_THRESHOLDS, workers=None, output=DEFAULT_OUTPUT,
              baseline=DEFAULT_BASELINE, max_tracks=None):
    """Run/resume independent track jobs; max_tracks is only for smoke checks.

    Existing experiment evidence is reused, never ordinary raw/smoothed caches.
    Keep a different output directory for smoke checks. Numerical failures abort
    the run instead of producing a leaderboard from partial coverage.
    """
    import pandas as pd
    from algorithms.Config import PYIN_TRACKING_UNV_THRESH, PYIN_POSTHOC_MIN_VOLUME
    from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker
    thresholds = tuple(sorted(set(float(t) for t in thresholds) | {PYIN_TRACKING_UNV_THRESH}))
    if any(not 0 < t <= PYIN_TRACKING_UNV_THRESH for t in thresholds):
        raise ValueError('Final thresholds must be in (0, fixed HMM threshold 0.990]')
    if workers is not None and (not isinstance(workers, int) or workers < 1):
        raise ValueError('workers must be a positive integer')
    if max_tracks is not None and (not isinstance(max_tracks, int) or max_tracks < 1):
        raise ValueError('max_tracks must be a positive integer')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    status = _Status()
    status.update(0, 449 if max_tracks is None else max_tracks, 'loading frozen cohort')
    try:
        with PitchBenchmarker.silence_runtime():
            jobs = _plan(baseline)
        if max_tracks is not None:
            jobs = jobs[:max_tracks]
        workers = min(workers or recommended_workers(), len(jobs))
        fingerprint = _fingerprint()
        manifest = dict(status='running', thresholds=thresholds, workers=workers,
                        hmm_threshold=PYIN_TRACKING_UNV_THRESH, volume_floor=PYIN_POSTHOC_MIN_VOLUME,
                        baseline=str(Path(baseline).resolve()), baseline_sha256=hashlib.sha256(Path(baseline).read_bytes()).hexdigest(), code=fingerprint,
                        tracks=[[t.dataset, t.track_id] for t, _ in jobs],
                        experiment='final gate only; fixed production HMM; exploratory tuning')
        manifest_path = output / 'run.json'
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if any(previous[k] != json.loads(json.dumps(manifest[k])) for k in ('code', 'tracks', 'baseline', 'baseline_sha256', 'hmm_threshold', 'volume_floor')):
                raise ValueError('Experiment inputs changed; choose a new output directory')
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        rows = []
        status.update(0, len(jobs), f'{workers} workers; fresh detection + fixed HMM or sweep evidence')
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_initialize_worker) as pool:
            futures = {pool.submit(_track, t, b, str(output), thresholds, fingerprint): t for t, b in jobs}
            try:
                for completed, future in enumerate(as_completed(futures), 1):
                    track = futures[future]
                    scored, reused = future.result()
                    rows.extend(scored)
                    status.update(completed, len(jobs), f'{track.dataset}/{track.track_id}: scored ({"cached evidence" if reused else "fresh"})')
            except BaseException:
                for future in futures:
                    future.cancel()
                manifest['status'] = 'failed'
                manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
                raise
        frame = pd.DataFrame(rows).sort_values(['dataset', 'track_id', 'threshold'])
        summary = summarize(frame)
        frame.to_csv(output / 'rows.csv', index=False)
        summary.to_csv(output / 'summary.csv', index=False)
        manifest['status'] = 'complete'
        controls = frame[frame.threshold.eq(PYIN_TRACKING_UNV_THRESH)]
        controls.to_csv(output / 'baseline_comparison.csv', index=False)
        manifest['fresh_controller_counts_match'] = True
        manifest['paper_baseline_mismatched_tracks'] = int((~controls.paper_baseline_counts_match).sum())
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        status.update(len(jobs), len(jobs), f'complete; fresh control verified; {manifest["paper_baseline_mismatched_tracks"]} paper-baseline differences; {output}')
        return summary
    finally:
        status.finish()
