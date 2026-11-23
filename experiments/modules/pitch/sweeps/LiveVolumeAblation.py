"""Replay causal volume gates on full capture windows from cached pYIN candidates."""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from algorithms.PitchDetector import PitchDetector
from experiments.modules.pitch.PitchBenchmarker import COUNT_COLUMNS, METRIC_COUNTS, PitchBenchmarker
from experiments.modules.pitch.PitchCache import PitchCache
from experiments.modules.pitch.PitchDetectorBase import PitchDetectorBase
from experiments.modules.pitch.competitors.Attune import Attune
from experiments.modules.pitch.sweeps.FinalVoicingSweep import DEFAULT_BASELINE, ROOT, _plan

OUTPUT = ROOT / 'experiments/results/caches/pitch/live_volume_reference_2026-11-05'


def live_controls():
    root = ROOT / 'experiments/results/caches/pitch/notebook_runs'
    controls = []
    for dataset, count in [('bach10-original', 'None'), ('urmp', 'None'), ('coco', '20')]:
        run = ('native_center_v9_pyin_joint_state_latency_samples__frame_counts_10ms_v1'
               '__methods_attune+pyin_framewise+praat+swiftf0+spice__instruments_all'
               f'__per_instrument_{count}__seed_0__max_infs__unv_0p985__dynamic_floor_0p03')
        frame = pd.read_csv(root / f'streaming_{dataset}' / run / 'rows.csv')
        controls.extend(frame[frame.model.eq('attune')].to_dict('records'))
    return {(r['dataset'], r['track_id']): r for r in controls}


def score_track(job):
    track, control = job
    dataset = PitchBenchmarker.dataset_for(track.dataset, PitchBenchmarker.Options())
    example = dataset.example(track)
    config = Attune().config_for(example.fmin, example.fmax)
    detector = PitchDetector(config=config)
    cache = PitchCache(example.stage_cache_path).read(PitchCache.RAW, config)
    if cache is None:
        raise ValueError(f'Missing raw candidates for {track.track_id}')
    # Centered cache index 16 is the first 4096-sample unpadded capture window.
    half_frame = config.w1 // 2
    n = int(control['latency_sample_count'])
    start, remainder = divmod(half_frame, config.h1)
    samples = round(control['audio_seconds'] * config.sr)
    if (remainder or config.sr != control['sample_rate_hz'] or
            config.w1 != control['frame_size_samples'] or config.h1 != control['hop_size_samples'] or
            n != 1 + (samples - config.w1) // config.h1):
        raise ValueError(f'Incompatible live capture grid for {track.track_id}')
    frames = cache[0].data[start:start+n]
    times = np.arange(n) * config.h1 / config.sr + half_frame / config.sr
    if len(frames) != n or not np.allclose([p.time for p in frames], times, rtol=0, atol=1e-10):
        raise ValueError(f'Cache times differ from the capture grid for {track.track_id}')
    rms = pd.Series([p.volume for p in frames], dtype=float)
    if not np.isfinite(rms).all():
        raise ValueError(f'Nonfinite RMS for {track.track_id}')
    # Rebuild pre-volume decisions, including candidates hidden by the saved gate.
    midi = np.array([detector._live_value(p.candidate_pitches, 1 - p.unvoiced_prob)[0] for p in frames])
    confidence = np.array([p.unvoiced_prob for p in frames]) < config.unv_thresh
    hz = np.where((midi != -1) & confidence, config.midi_to_freq(midi), 0)
    keep = example.ref_times <= control['audio_seconds'] + 1e-9
    rows = []
    for rule, reference in [('running_max', rms.cummax()), ('running_p95', rms.expanding().quantile(.95, interpolation='linear'))]:
        for alpha in (.03, .04):
            estimate = PitchDetectorBase.PitchEstimate.build(times, np.where(rms >= alpha * reference, hz, 0), 0)
            scores = PitchBenchmarker.score_frames(example.ref_times[keep], example.ref_freqs[keep], estimate.times, estimate.freqs)
            if scores['reference_hash'] != control['reference_hash']:
                raise ValueError(f'Reference changed for {track.track_id}')
            if rule == 'running_max' and alpha == .03:
                differences = {k: scores[k] - control[k] for k in COUNT_COLUMNS if scores[k] != control[k]}
                if differences:
                    raise ValueError(f'Cached replay differs from live control for {track.track_id}: {differences}')
            rows.append(dict(dataset=track.dataset, track_id=track.track_id, rule=rule, alpha=alpha, **scores))
    return rows


def run_live_ablation(workers=4):
    with PitchBenchmarker.silence_runtime():
        tracks = [track for track, _ in _plan(DEFAULT_BASELINE)]
    controls = live_controls()
    if {(t.dataset, t.track_id) for t in tracks} != set(controls):
        raise ValueError('Live and offline track selections differ')
    jobs = [(t, controls[t.dataset, t.track_id]) for t in tracks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = pd.DataFrame(row for group in pool.map(score_track, jobs) for row in group)
    tables = []
    for name, subset in [('All', rows), *rows.groupby('dataset')]:
        totals = subset.groupby(['rule', 'alpha'])[list(COUNT_COLUMNS)].sum()
        table = subset.groupby(['rule', 'alpha']).size().to_frame('Recordings')
        for metric, (num, den) in METRIC_COUNTS.items():
            table[metric] = 100 * totals[num] / totals[den]
        table['Median recording OA'] = 100 * subset.groupby(['rule', 'alpha'])['Overall Accuracy'].median()
        table['Pooled OA delta (pp)'] = table['Overall Accuracy'] - table.loc[('running_max', .03), 'Overall Accuracy']
        tables.append(table.assign(Dataset=name).reset_index())
    summary = pd.concat(tables, ignore_index=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows.to_csv(OUTPUT / 'paired_rows.csv', index=False)
    summary.to_csv(OUTPUT / 'summary.csv', index=False)
    return rows, summary
