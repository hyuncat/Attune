"""Presentation-only note sweep reports; does not alter inference fingerprints."""
from pathlib import Path
import numpy as np
import pandas as pd
from benchmarks.modules.sweeps import AppendixSweeps as base

SPEED_COLUMN = 'Audio (s) / Compute (s)'


def speed_table(rows, audio_seconds):
    """Pool durations/time, never average individual recordings' speed ratios.

    Sweeps time segmentation and production refinement with perf_counter inside each worker.
    Missing recovered timings remain unknown, rather than making cache hits
    look instantaneous or dropping slow/untimed recordings from the ratio.
    """
    keys = ['Method', *[label for label in base.PARAMETER_LABELS.values() if label in rows]]
    frame = rows.copy()
    frame['_audio_seconds'] = frame.track_id.map(audio_seconds)
    records = []
    for values, group in frame.groupby(keys, dropna=False, sort=False):
        values = values if isinstance(values, tuple) else (values,)
        # Legacy pre-refinement reports only contain segmentation_seconds.
        timing_column = 'compute_seconds' if 'compute_seconds' in group else 'segmentation_seconds'
        elapsed = pd.to_numeric(group[timing_column], errors='coerce')
        audio = group['_audio_seconds']
        valid = (np.isfinite(elapsed).all() and elapsed.ge(0).all()
                 and np.isfinite(audio).all() and audio.gt(0).all()
                 and elapsed.sum() > 0)
        records.append(dict(zip(keys, values)) | {
            SPEED_COLUMN: float(audio.sum() / elapsed.sum()) if valid else float('nan')})
    return pd.DataFrame(records, columns=[*keys, SPEED_COLUMN]), keys


def add_speed(results, manifest):
    """Append speed to completed result tables and CSVs, without running models."""
    import soundfile as sf
    from benchmarks.modules.pitch.datasets.CocoChorales import CocoChorales
    directory = Path(results['directory'])
    rows = pd.read_csv(directory / 'rows.csv')
    audio_seconds = {}
    selected = set(rows.track_id)
    for item in base.inputs(manifest):
        if item['track_id'] in selected:
            audio_seconds[item['track_id']] = sf.info(item['paths']['audio']).duration
    speeds, keys = speed_table(rows, audio_seconds)

    def append(table):
        return table.drop(columns=SPEED_COLUMN, errors='ignore').merge(
            speeds, on=keys, how='left', validate='one_to_one', sort=False)

    updated = dict(results, summary=append(results['summary']),
                   best_by_method=append(results['best_by_method']))
    for key, filename in [('summary', 'summary.csv'), ('best_by_method', 'best_by_method.csv')]:
        path = directory / filename
        temporary = path.with_suffix('.tmp.csv')
        updated[key].to_csv(temporary, index=False)
        temporary.replace(path)
    return updated
