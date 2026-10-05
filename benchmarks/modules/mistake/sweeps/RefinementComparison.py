"""Paired audio-injection comparison of regional note refiners.

Uses MistakeBenchmarker generation and onset scoring, with fresh production
note segmentation (never its historical note cache). No symbolic source IDs
are used to score newly segmented notes. Each checker gets independent state.
"""
from copy import deepcopy
from dataclasses import asdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter
import mir_eval
import numpy as np
import pandas as pd
from app_logic.user.ds.PitchData import PitchData
from notebooks.archive.MistakeChecker2 import MistakeChecker as Checker2
from notebooks.archive.MistakeChecker3 import MistakeChecker as Checker3
from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker
from benchmarks.modules.mistake.datasets.MistakeInjector import MistakeInjector
from algorithms.Config import Config
from benchmarks.modules.note.NoteBenchmarker import OneInstrumentScoreData
from benchmarks.modules.note.NoteDetectorBase import NoteDetectorBase
from benchmarks.paths import REPO_ROOT
ERROR_TYPES = ('substitution', 'insertion', 'deletion', 'short', 'long')

def fingerprint():
    from benchmarks.modules.mistake.MistakeCache import MistakeCache
    return MistakeCache.pipeline_fingerprint()

def prepare_case(midi, seed, rate, output):
    """Cache one injected/synthesized performance under a content-specific key."""
    midi = Path(midi).resolve()
    duration_error_min = max(0.3, Config().timing_tolerance + 0.05)
    spec = dict(source=str(midi), source_hash=hashlib.sha256(midi.read_bytes()).hexdigest(), seed=int(seed), rate=float(rate), weights=[0.2] * 5, duration_error_policy='relative_salient_v1', duration_factor_range=[0.5, 1.5], duration_error_min_sec=duration_error_min, timing_std_ms=0.0, duration_std=0.0, injector_hash=fingerprint()['benchmarks/modules/mistake/datasets/MistakeInjector.py'])
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
    bench = MistakeBenchmarker()
    bench.MISTAKE_DIR = Path(output) / 'cases' / key
    manifest = bench.MISTAKE_DIR / 'manifest.json'
    if manifest.exists():
        item = json.loads(manifest.read_text())
    else:
        injector = MistakeInjector(mistake_rate=rate, weights=(0.2,) * 5, timing_std_ms=0.0, duration_std=0.0, duration_error_min_sec=duration_error_min)
        item = bench.generate_mistake_db_track(midi, 'paired', seed, injector=injector)
        item.update(case_id=key, spec=spec)
        manifest.write_text(json.dumps(item, indent=2))
    return (bench, item)

def compare_case(midi, seed, rate, output):
    bench, item = prepare_case(midi, seed, rate, output)
    source = OneInstrumentScoreData(midi)
    performed = OneInstrumentScoreData(item['midi']).note_data
    config = bench.config_for_performance(performed)
    from app_logic.user.ds.AudioData import AudioData
    base = bench.attune.recording_for(config, score_data=source)
    base.audio_data = AudioData(audio_filepath=item['audio'], config=base.config)
    bench.load_mistake_pitches(base, item['pitch_data'], smooth=True)
    base.reset_analysis()
    base.detect_notes()
    initial = NoteDetectorBase.clone_note_data(base.note_data)
    truth = json.loads(Path(item['truth']).read_text())['truth']
    ref_intervals, ref_pitches = bench.notedata_to_intervals(performed, base.config)
    initial_intervals, _ = bench.notedata_to_intervals(initial, base.config)
    latency_shift = bench._latency_offset(ref_intervals[:, 0], initial_intervals[:, 0])
    rows = []
    for name, cls in [('Checker2', Checker2), ('Checker3', Checker3)]:
        rec = bench.attune.recording_for(replace(base.config), score_data=OneInstrumentScoreData(midi))
        rec.pitch_data = PitchData(rec.config)
        rec.pitch_data.data = deepcopy(base.pitch_data.data, {id(base.config): rec.config})
        rec.pitch_data.t_origin = base.pitch_data.t_origin
        rec.pitch_data.end_index = base.pitch_data.end_index
        rec.note_data = NoteDetectorBase.clone_note_data(initial)
        rec.repeat_splitter = cls(recording=rec)
        start = perf_counter()
        rec.resize_score(to_span='onset')
        rec.detect_mistakes()
        initial_cost = rec.mistake_detector.get_alignment_cost(rec.alignment)
        rec.stabilize_score_alignment()
        seconds = perf_counter() - start
        alignment = rec.alignment
        mistakes = [*alignment.pitch_mistakes, *alignment.timing_mistakes]
        counts = bench.score_onset(mistakes, truth, onset_tolerance=0.1, canonical=False)
        est_intervals, est_pitches = bench.notedata_to_intervals(rec.note_data, rec.config)
        matched = mir_eval.transcription.match_notes(ref_intervals, ref_pitches, est_intervals, est_pitches, onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=0.2)
        row = dict(case_id=item['case_id'], source=str(midi), seed=seed, rate=rate, checker=name, seconds=seconds, initial_notes=len(initial.times), final_notes=len(rec.note_data.times), initial_cost=initial_cost, final_cost=rec.mistake_detector.get_alignment_cost(alignment), note_tp=len(matched), note_fp=len(est_intervals) - len(matched), note_fn=len(ref_intervals) - len(matched), truth_events=len(truth), config=json.dumps(asdict(base.config), sort_keys=True))
        for kind in ERROR_TYPES:
            for suffix, value in zip(('tp', 'fp', 'fn'), counts.get(kind, (0, 0, 0))):
                row[f'{kind}_{suffix}'] = value
        for suffix in ('tp', 'fp', 'fn'):
            row[f'error_{suffix}'] = sum((row[f'{kind}_{suffix}'] for kind in ERROR_TYPES))
        row['note_f1'] = prf(row['note_tp'], row['note_fp'], row['note_fn'])[2]
        adjusted = mir_eval.transcription.match_notes(ref_intervals, ref_pitches, est_intervals + latency_shift, est_pitches, onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=0.2)
        row.update(latency_shift_sec=latency_shift, note_latency_tp=len(adjusted), note_latency_fp=len(est_intervals) - len(adjusted), note_latency_fn=len(ref_intervals) - len(adjusted))
        row['note_latency_f1'] = prf(row['note_latency_tp'], row['note_latency_fp'], row['note_latency_fn'])[2]
        row['error_f1'] = prf(row['error_tp'], row['error_fp'], row['error_fn'])[2]
        rows.append(row)
    return rows

def prf(tp, fp, fn):
    from benchmarks.modules.mistake.MistakeDetectorBase import MistakeDetectorBase
    return MistakeDetectorBase.prf(tp, fp, fn)

def summarize(rows):
    summaries = []
    for (rate, checker), group in rows.groupby(['rate', 'checker']):
        for metric in ('note', 'note_latency', 'error', *ERROR_TYPES):
            tp, fp, fn = [int(group[f'{metric}_{s}'].sum()) for s in ('tp', 'fp', 'fn')]
            precision, recall, f1 = prf(tp, fp, fn)
            summaries.append(dict(rate=rate, checker=checker, metric=metric, cases=len(group), tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1, mean_seconds=group.seconds.mean()))
    return pd.DataFrame(summaries)

def run_comparison(midis, output, seeds=(0,), rates=(0.0, 0.25)):
    """Sequential reproducible paired run; fail loudly, save each completed pair."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    midis = [str(Path(m).resolve()) for m in midis]
    metadata = dict(sources=midis, seeds=list(seeds), rates=list(rates), code=fingerprint(), scoring='onset-matched mistakes 100ms; notes 50ms/50c/20% offset', pipeline='shared audio/pitches/initial notes; production stabilization; synth edge trim')
    (output / 'run.json').write_text(json.dumps(metadata, indent=2))
    rows = []
    for midi in midis:
        for seed in seeds:
            for rate in rates:
                print(f'{Path(midi).stem}: seed={seed}, rate={rate}', flush=True)
                rows.extend(compare_case(midi, seed, rate, output))
                pd.DataFrame(rows).to_csv(output / 'rows.csv', index=False)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError('Select at least one MIDI, seed, and rate.')
    summarize(frame).to_csv(output / 'summary.csv', index=False)
    return frame
