"""Rebuild shared production pitch stages and verify the accepted 449-track sweep.

The compact sweep evidence lacks RMS and candidate lists required by downstream
note analysis. Recompute each track once, verify all accepted scoring counts,
and only then publish complete raw/smoothed PitchCache stages with real timings.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import shutil

from experiments.modules.pitch.sweeps.FinalVoicingSweep import (
    DEFAULT_BASELINE, DEFAULT_OUTPUT, ROOT, _initialize_worker, _plan, _Status,
)

OUTPUT = ROOT / 'experiments/results/caches/pitch/final_voicing_promotion_2026-10-08'


def _promote(track, expected, output):
    from experiments.modules.pitch.PitchBenchmarker import PitchBenchmarker
    with PitchBenchmarker.silence_runtime():
        return _promote_quiet(track, expected, Path(output))


def _promote_quiet(track, expected, output):
    from experiments.modules.pitch.PitchBenchmarker import PitchBenchmarker, COUNT_COLUMNS
    from experiments.modules.pitch.PitchCache import PitchCache
    from experiments.modules.pitch.competitors.Attune import Attune
    dataset = PitchBenchmarker.dataset_for(track.dataset, PitchBenchmarker.Options())
    example = dataset.example(track)
    adapter = Attune()
    config = adapter.config_for(example.fmin, example.fmax)
    assert config.pitch_tracking_unv_thresh == 0.990 and config.posthoc_unv_thresh == 0.9875
    cache = adapter.cache(example)
    stages = None
    if cache.has_current_timing(PitchCache.SMOOTHED, adapter.COMPUTE_CLOCK):
        pitch_data, timing = cache.read(PitchCache.SMOOTHED, config)
    else:
        recording = adapter.recording_for(config)
        recording.audio_data = adapter._audio_data(example, config)
        stages = adapter.detect_stages(recording)
        pitch_data, timing = stages.data[PitchCache.SMOOTHED], stages.timing[PitchCache.SMOOTHED]
    times, hz = adapter.melody(pitch_data, config)
    hz = adapter.constrain_freqs_to_range(hz, example.fmin, example.fmax)
    scores = PitchBenchmarker.score_frames(example.ref_times, example.ref_freqs, times, hz)
    differences = {k: [int(expected[k]), scores[k]] for k in COUNT_COLUMNS if int(expected[k]) != scores[k]}
    if differences or scores['reference_hash'] != expected['reference_hash']:
        raise RuntimeError(f'{track.dataset}/{track.track_id}: promotion differs from accepted sweep: {differences}')
    if stages is not None:
        if cache.path.exists():
            backup = output / 'previous_caches' / track.dataset / cache.path.name
            backup.parent.mkdir(parents=True, exist_ok=True)
            if not backup.exists():
                shutil.copy2(cache.path, backup)
        cache.write(stages)
    return dict(dataset=track.dataset, track_id=track.track_id, cache=str(cache.path),
                verified=True, reused=stages is None, **scores, **timing)


def run(*, workers=8, output=OUTPUT, max_tracks=None):
    import pandas as pd
    from experiments.modules.pitch.PitchBenchmarker import PitchBenchmarker
    from experiments.modules.pitch.PitchCache import PitchCache
    from experiments.modules.pitch.sweeps.FinalVoicingSweep import summarize
    expected = pd.read_csv(DEFAULT_OUTPUT / 'rows.csv')
    expected = expected[expected.threshold.eq(0.9875)]
    assert len(expected) == 449 and not expected.duplicated(['dataset', 'track_id']).any()
    expected = {(r['dataset'], r['track_id']): r for r in expected.to_dict('records')}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with PitchBenchmarker.silence_runtime():
        jobs = _plan(DEFAULT_BASELINE)
    if max_tracks is not None:
        jobs = jobs[:max_tracks]
    status = _Status()
    status.update(0, len(jobs), f'promoting final gate 0.9875; {workers} workers')
    manifest = dict(status='running', pitch_cache_version=PitchCache.VERSION,
                    tracking_threshold=0.990, final_threshold=0.9875, expected_tracks=len(jobs))
    manifest_path = output / 'run.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    rows = []
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_initialize_worker) as pool:
            futures = {pool.submit(_promote, t, expected[t.dataset, t.track_id], str(output)): t for t, _ in jobs}
            try:
                for i, future in enumerate(as_completed(futures), 1):
                    row = future.result()
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(output / 'rows.csv', index=False)
                    status.update(i, len(jobs), f'verified {row["dataset"]}/{row["track_id"]}')
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        frame = pd.DataFrame(rows).assign(threshold=0.9875)
        summarize(frame).to_csv(output / 'summary.csv', index=False)
        manifest['status'] = 'complete'
        manifest['verified_tracks'] = len(rows)
        status.update(len(rows), len(jobs), 'shared pitch caches promoted; all scoring counts verified')
    except BaseException:
        manifest['status'] = 'failed'
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        status.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max-tracks', type=int)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.workers < 1 or (args.max_tracks is not None and args.max_tracks < 1):
        parser.error('workers and max-tracks must be positive')
    run(workers=args.workers, max_tracks=args.max_tracks, output=args.output)
