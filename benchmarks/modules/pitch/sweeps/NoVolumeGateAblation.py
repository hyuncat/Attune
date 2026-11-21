"""Add one no-volume-gate pass to a completed volume sweep without rerunning it."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import pandas as pd

from benchmarks.modules.pitch.sweeps.FinalVoicingSweep import ROOT, DEFAULT_BASELINE, _plan, _initialize_worker
from benchmarks.modules.pitch.sweeps.VolumeReferenceAblation import summarize


def _job(args):
    track, baseline, output, code = args
    from algorithms.PitchSmoother import PitchSmoother, VoicingSmoother
    from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker
    from benchmarks.modules.pitch.PitchCache import PitchCache
    from benchmarks.modules.pitch.competitors.Attune import Attune
    with PitchBenchmarker.silence_runtime():
        example = PitchBenchmarker.dataset_for(track.dataset, PitchBenchmarker.Options()).example(track)
        config = Attune().config_for(example.fmin, example.fmax)
        path = Path(example.stage_cache_path)
        identity = dict(code=code, config=asdict(config), dataset=track.dataset, track_id=track.track_id,
                        raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        reference=hashlib.sha256(example.ref_times.tobytes() + example.ref_freqs.tobytes()).hexdigest())
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        result = Path(output) / 'tracks' / f'{key}.json'
        if result.exists():
            return json.loads(result.read_text()), True
        hit = PitchCache(path).read(PitchCache.RAW, config)
        if hit is None:
            raise ValueError(f'Missing/incompatible raw cache: {path}')
        raw, _ = hit
        # Read raw evidence with its original config, then disable only the offline
        # volume floor in BOTH HMM region selection and final voicing.
        no_volume = replace(config, posthoc_min_volume=0.0)
        tracked = PitchSmoother(config=no_volume).smooth(list(raw.data))
        gated = VoicingSmoother(config=no_volume).smooth(tracked)
        times, hz = Attune.melody(Attune._pitch_data(gated, no_volume, raw.t_origin), no_volume)
        hz = Attune.constrain_freqs_to_range(hz, example.fmin, example.fmax)
        score = PitchBenchmarker.score_frames(example.ref_times, example.ref_freqs, times, hz)
        if score['reference_hash'] != baseline['reference_hash']:
            raise ValueError(f'Reference changed: {track.track_id}')
        row = dict(dataset=track.dataset, track_id=track.track_id, instrument=baseline['instrument'],
                   variant='no_volume_gate', raw_cache=str(path), volume_rejected_fraction=0.0, **score)
        result.parent.mkdir(parents=True, exist_ok=True)
        temp = result.with_suffix('.tmp')
        temp.write_text(json.dumps(row))
        temp.replace(result)
        return row, False


def run_no_volume_ablation(*, comparison, workers=4):
    """Reuse completed five-rule scores; only compute the additional no-gate rule."""
    from benchmarks.modules.pitch.PitchBenchmarker import PitchBenchmarker
    comparison = Path(comparison)
    original = json.loads((comparison / 'manifest.json').read_text())
    if original['status'] != 'complete' or original['tracks'] != 449:
        raise ValueError('Requires the completed full-cohort volume sweep')
    for relative, digest in original['code'].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Comparison source changed: {relative}')
    code = dict(original['code'])
    code[str(Path(__file__).relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = comparison.with_name(comparison.name + '_with_no_gate')
    output.mkdir(parents=True, exist_ok=True)
    with PitchBenchmarker.silence_runtime():
        jobs = _plan(DEFAULT_BASELINE)
    manifest = dict(status='running', tracks=len(jobs), code=code, comparison=str(comparison),
                    cache_policy='raw candidates only; no detection or production writes')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    rows, reused = [], 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_initialize_worker) as pool:
        for i, (row, cached) in enumerate(pool.map(_job, [(t, b, output, code) for t, b in jobs]), 1):
            rows.append(row)
            reused += cached
            print(f'\rNo-volume gate: {i}/{len(jobs)} ({reused} cached)', end='', flush=True)
    print()
    previous = pd.read_csv(comparison / 'paired_rows.csv')
    new = pd.DataFrame(rows)
    control = previous[previous.variant.eq('current_p95')]
    paired = new.merge(control, on=['dataset', 'track_id'], suffixes=('_new', '_control'), validate='one_to_one')
    if len(paired) != 449 or not paired.reference_hash_new.eq(paired.reference_hash_control).all():
        raise ValueError('No-gate and control cohorts/references differ')
    combined = pd.concat([previous, new], ignore_index=True)
    summary = summarize(combined)
    combined.to_csv(output / 'paired_rows.csv', index=False)
    summary.to_csv(output / 'summary.csv', index=False)
    manifest.update(status='complete', result_cache_hits=reused, variants=list(combined.variant.unique()))
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return combined, summary
