"""Cache-only paired volume-rule ablation; never writes production caches."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from algorithms.PitchSmoother import VoicingSmoother
from experiments.modules.pitch.sweeps.FinalVoicingSweep import (
    ROOT, DEFAULT_BASELINE, _plan, _initialize_worker,
)

DEFAULT_OUTPUT = ROOT / 'experiments/results/caches/pitch/volume_reference_2026-11-04'
VARIANTS = ('current_p95', 'global_max_live_alpha', 'running_max_live_alpha',
            'global_max_post_alpha')
LOW_PERCENTILE_VARIANTS = ('current_p95', 'percentile_1', 'percentile_2',
                          'percentile_3', 'percentile_4')


def volume_thresholds(volumes, rule, config):
    """Threshold per frame; nonfinite samples never contribute to references."""
    values = np.asarray(volumes, dtype=float)
    clean = np.where(np.isfinite(values), np.maximum(values, 0), 0)
    finite = clean[np.isfinite(values)]
    if rule == 'current_p95':
        reference = np.percentile(finite, 95) if finite.size else 0.0
        return np.full(values.shape, config.posthoc_min_volume * reference)
    if rule in LOW_PERCENTILE_VARIANTS[1:]:
        percentile = int(rule.rsplit('_', 1)[1])
        floor = np.percentile(finite, percentile) if finite.size else 0.0
        return np.full(values.shape, floor)
    if rule == 'running_max_live_alpha':
        return config.min_volume * np.maximum.accumulate(clean)
    if rule not in ('global_max_live_alpha', 'global_max_post_alpha'):
        raise ValueError(rule)
    alpha = config.min_volume if rule == 'global_max_live_alpha' else config.posthoc_min_volume
    return np.full(values.shape, alpha * np.max(clean, initial=0))


def controller_for(rule):
    """Change volume eligibility only, retaining production confidence logic."""
    if rule == 'current_p95':
        return VoicingSmoother

    class VolumeController(VoicingSmoother):
        def volume_floor(self, volumes):
            return 0.0

        def decode(self, pitches, *, unvoiced_threshold=None):
            allowed = super().decode(pitches, unvoiced_threshold=unvoiced_threshold)
            present = [p for p in pitches if p is not None]
            volumes = np.asarray([max(0.0, float(p.volume)) for p in present])
            floors = volume_thresholds(volumes, rule, self.config)
            volume_allowed = np.zeros(len(pitches), dtype=bool)
            volume_allowed[[i for i, p in enumerate(pitches) if p is not None]] = volumes >= floors
            return allowed & volume_allowed
    return VolumeController


def _job(args):
    track, baseline, output, code, variants = args
    from experiments.modules.pitch.PitchBenchmarker import PitchBenchmarker
    from experiments.modules.pitch.PitchCache import PitchCache
    from experiments.modules.pitch.competitors.Attune import Attune
    module = importlib.import_module('algorithms.PitchSmoother')
    with PitchBenchmarker.silence_runtime():
        example = PitchBenchmarker.dataset_for(track.dataset, PitchBenchmarker.Options()).example(track)
        config = Attune().config_for(example.fmin, example.fmax)
        path = Path(example.stage_cache_path)
        if not path.exists():
            raise FileNotFoundError(f'Missing raw cache (no automatic detection): {path}')
        identity = dict(code=code, variants=variants, config=asdict(config), cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        reference=hashlib.sha256(example.ref_times.tobytes() + example.ref_freqs.tobytes()).hexdigest(),
                        dataset=track.dataset, track_id=track.track_id)
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        result_path = Path(output) / 'tracks' / f'{key}.json'
        if result_path.exists():
            return json.loads(result_path.read_text()), True
        hit = PitchCache(path).read(PitchCache.RAW, config)
        if hit is None:
            raise ValueError(f'Missing/incompatible raw cache: {path}')
        raw, _ = hit
        volumes = np.asarray([max(0.0, p.volume) for p in raw.data if p is not None])
        finite = volumes[np.isfinite(volumes)]
        peak = float(finite.max(initial=0))
        q95 = float(np.percentile(finite, 95)) if finite.size else 0.0
        current_floor = volume_thresholds(volumes, 'current_p95', config)
        rows = []
        control_hz = None
        for rule in variants:
            controller = controller_for(rule)
            # The production HMM instantiates this controller internally too.
            # Process-local patch is restored before leaving this variant.
            with patch.object(module, 'VoicingSmoother', controller):
                tracked = module.PitchSmoother(config=config).smooth(list(raw.data))
                gated = controller(config=config).smooth(tracked)
            times, hz = Attune.melody(Attune._pitch_data(gated, config, raw.t_origin), config)
            hz = Attune.constrain_freqs_to_range(hz, example.fmin, example.fmax)
            if control_hz is None:
                control_hz = hz.copy()
            score = PitchBenchmarker.score_frames(example.ref_times, example.ref_freqs, times, hz)
            if score['reference_hash'] != baseline['reference_hash']:
                raise ValueError(f'Reference changed: {track.track_id}')
            floor = volume_thresholds(volumes, rule, config)
            rows.append(dict(dataset=track.dataset, track_id=track.track_id, instrument=baseline['instrument'],
                             variant=rule, rms_max=peak, rms_q95=q95,
                             equivalent_max_alpha=config.posthoc_min_volume*q95/peak if peak else None,
                             volume_rejected_fraction=float(np.mean(volumes < floor)),
                             volume_mask_disagreement=float(np.mean((volumes >= floor) != (volumes >= current_floor))),
                             output_changed_fraction=float(np.mean(hz != control_hz)),
                             raw_cache=str(path), **score))
        result_path.parent.mkdir(parents=True, exist_ok=True)
        temp = result_path.with_suffix('.tmp')
        temp.write_text(json.dumps(rows))
        temp.replace(result_path)
        return rows, False


def summarize(rows):
    from experiments.modules.pitch.PitchBenchmarker import METRIC_COUNTS, COUNT_COLUMNS
    parts = []
    groups = [('All', rows), ('Coco + URMP', rows[rows.dataset.isin(['coco', 'urmp'])])]
    groups += list(rows.groupby('dataset'))
    for name, selected in groups:
        totals = selected.groupby('variant')[list(COUNT_COLUMNS)].sum()
        result = selected.groupby('variant').size().to_frame('Tracks')
        for metric, (num, den) in METRIC_COUNTS.items():
            result[metric] = 100 * totals[num] / totals[den].replace(0, np.nan)
        result['Median recording OA'] = selected.groupby('variant')['Overall Accuracy'].median() * 100
        control = selected[selected.variant.eq('current_p95')].set_index(['dataset', 'track_id'])['Overall Accuracy']
        deltas = selected['Overall Accuracy'].to_numpy() - control.reindex(pd.MultiIndex.from_frame(selected[['dataset', 'track_id']])).to_numpy()
        paired = selected.copy()
        paired['delta'] = deltas * 100
        result['Median paired OA delta (pp)'] = paired.groupby('variant').delta.median()
        result['Pooled OA delta (pp)'] = result['Overall Accuracy'] - result.loc['current_p95', 'Overall Accuracy']
        result.insert(0, 'Group', name)
        parts.append(result.reset_index())
    return pd.concat(parts, ignore_index=True)


def run_ablation(*, output=DEFAULT_OUTPUT, baseline=DEFAULT_BASELINE, workers=4, smoke=False,
                 variants=VARIANTS, per_instrument=None, seed=0):
    """Frozen 449 tracks; smoke selects one per dataset. Cache misses fail loudly."""
    from experiments.modules.pitch.PitchBenchmarker import PitchBenchmarker
    if not isinstance(workers, int) or workers < 1:
        raise ValueError('workers must be positive')
    variants = tuple(variants)
    if (not variants or variants[0] != 'current_p95' or len(set(variants)) != len(variants)
            or not set(variants) <= set(VARIANTS + LOW_PERCENTILE_VARIANTS)):
        raise ValueError('Use unique supported variants, with current_p95 first')
    if per_instrument is not None and (not isinstance(per_instrument, int) or per_instrument < 1):
        raise ValueError('per_instrument must be positive')
    if smoke and per_instrument is not None:
        raise ValueError('Choose smoke or per_instrument sampling, not both')
    output = Path(output)
    with PitchBenchmarker.silence_runtime():
        jobs = _plan(baseline)
    if per_instrument is not None:
        groups = {}
        for job in jobs:
            groups.setdefault((job[0].dataset, job[1]['instrument']), []).append(job)
        rng = np.random.default_rng(seed)
        selected = []
        for key in sorted(groups):
            group = sorted(groups[key], key=lambda job: job[0].track_id)
            indices = sorted(rng.choice(len(group), min(per_instrument, len(group)), replace=False))
            selected.extend(group[i] for i in indices)
        jobs = selected
    if smoke:
        jobs = [next(j for j in jobs if j[0].dataset == d) for d in sorted({t.dataset for t, _ in jobs})]
    paths = [Path(__file__), ROOT/'algorithms/Config.py', ROOT/'algorithms/PitchSmoother.py',
             ROOT/'algorithms/PitchDetector.py', ROOT/'experiments/modules/pitch/PitchCache.py',
             ROOT/'experiments/modules/pitch/PitchBenchmarker.py', ROOT/'experiments/modules/pitch/competitors/Attune.py']
    code = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    rows, reused = [], 0
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(status='running', code=code, smoke=smoke, tracks=len(jobs), variants=variants,
                    per_instrument=per_instrument, seed=seed,
                    selection=[dict(dataset=t.dataset, track_id=t.track_id, instrument=b['instrument']) for t, b in jobs],
                    baseline=str(baseline), cache_policy='current-version raw only; no detection or production writes')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    with ProcessPoolExecutor(max_workers=workers, initializer=_initialize_worker) as pool:
        for i, (result, cached) in enumerate(pool.map(_job, [(t, b, output, code, variants) for t, b in jobs]), 1):
            rows.extend(result)
            reused += cached
            print(f'\rVolume ablation: {i}/{len(jobs)} tracks ({reused} result-cache hits)', end='', flush=True)
    print()
    frame = pd.DataFrame(rows)
    summary = summarize(frame)
    frame.to_csv(output/'paired_rows.csv', index=False)
    summary.to_csv(output/'summary.csv', index=False)
    manifest.update(status='complete', result_cache_hits=reused, raw_cache_tracks=len(jobs))
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    return frame, summary
