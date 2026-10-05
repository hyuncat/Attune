"""Replay an exact saved mistake corpus, scoring every stage against performed MIDI.

Writes diagnostics only. No audio/model inference, no production configuration edits.
Frame truth is nominal MIDI occupancy, not acoustic attack/release annotation.
"""
from contextlib import redirect_stdout
from contextlib import redirect_stderr
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from algorithms.Config import Config
from app_logic.user.ds.PitchData import PitchData
from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker
from benchmarks.modules.mistake.MistakeDetectorBase import MistakeDetectorBase
from benchmarks.modules.mistake.provenance.InjectedCaseAudit import matched_indices
from benchmarks.modules.mistake.provenance.InjectedCaseAudit import support
from benchmarks.modules.note.NoteBenchmarker import OneInstrumentScoreData
from benchmarks.modules.note.NoteDetectorBase import NoteDetectorBase
from benchmarks.modules.pitch.PitchCache import PitchCache

def sequence(data):
    return [data.data[t] for t in data.times]

def note_matches(reference, estimated, tolerance=0.1, offsets=False, pitch=True):
    valid = np.zeros((len(estimated), len(reference)), dtype=bool)
    for i, est in enumerate(estimated):
        for j, ref in enumerate(reference):
            valid[i, j] = abs(est.start_time - ref.start_time) <= tolerance and (not pitch or abs(est.midi_num[0] - ref.midi_num[0]) <= 0.5) and (not offsets or abs(est.end_time - ref.end_time) <= max(0.05, 0.2 * ref.duration()))
    result = maximum_bipartite_matching(csr_matrix(valid), perm_type='column')
    return [(i, int(j)) for i, j in enumerate(result) if j >= 0]

def frame_reference(times, performed):
    values = np.full(len(times), -1.0)
    interior = np.zeros(len(times), bool)
    for n in performed:
        mask = (times >= n.start_time) & (times < n.end_time)
        values[mask] = n.midi_num[0]
        margin = min(0.05, n.duration() / 4)
        interior |= (times >= n.start_time + margin) & (times < n.end_time - margin)
    return (values, interior)

def frame_counts(pitches, performed):
    times = pitches.t_origin + np.arange(len(pitches.data)) * pitches.config.h1 / pitches.config.sr
    times = np.asarray([p.time if p is not None else t for p, t in zip(pitches.data, times)])
    truth, interior = frame_reference(times, performed)
    estimates = np.asarray([p.value if p is not None else -1 for p in pitches.data])
    ref, est = (truth >= 0, estimates >= 0)
    correct = ref & est & (np.abs(truth - estimates) <= 0.5)
    candidate = np.asarray([p is not None and any((abs(m - t) <= 0.5 and prob > 0 for m, prob in p.candidate_pitches)) for p, t in zip(pitches.data, truth)]) & ref
    return dict(frames=len(times), ref_frames=int(ref.sum()), est_frames=int(est.sum()), correct=int(correct.sum()), voiced_tp=int((ref & est).sum()), unvoiced_on_note=int((ref & ~est).sum()), wrong_pitch=int((ref & est & ~correct).sum()), octave_errors=int((ref & est & (np.abs(np.abs(truth - estimates) - 12) <= 0.5)).sum()), voiced_in_silence=int((~ref & est).sum()), candidate_correct=int(candidate.sum()), interior_frames=int(interior.sum()), interior_correct=int((interior & correct).sum()))

def snapshot(rec, reference, events):
    mistakes = MistakeDetectorBase.with_reference_score_ids(MistakeBenchmarker.label_pairs(rec.alignment.pairs, rec.config), reference, rec.score_data.note_data)
    predicted = MistakeDetectorBase.predicted_events(mistakes, reference)
    counts = MistakeDetectorBase.score_events(predicted, events)['audio_pitch']
    user, score = (sequence(rec.note_data), sequence(rec.score_data.note_data))
    ui, si = ({id(n): i for i, n in enumerate(user)}, {id(n): i for i, n in enumerate(score)})
    return dict(tp=counts[0], fp=counts[1], fn=counts[2], cost=rec.mistake_detector.get_alignment_cost(rec.alignment), notes=[dict(index=i, id=n.id, onset=n.start_time, end=n.end_time, pitch=n.midi_num[0]) for i, n in enumerate(user)], score=[dict(index=i, id=n.id, onset=n.start_time, end=n.end_time, pitch=n.midi_num[0]) for i, n in enumerate(score)], pairs=[(ui[id(u)] if u is not None else None, si[id(s)] if s is not None else None) for u, s in rec.alignment.pairs], predicted=predicted)

def audit(output):
    output = Path(output)
    run = json.loads((output / 'run.json').read_text())
    if run['status'] != 'complete':
        raise ValueError('Audit requires a completed run')
    saved = pd.read_csv(output / 'rows.csv')
    cases = saved.drop_duplicates('case_id')
    cases = pd.concat([cases[cases.rate > 0], cases[cases.rate == 0].drop_duplicates('source')])
    destination = output / 'pipeline_audit'
    destination.mkdir(exist_ok=True)
    tables = {k: [] for k in ['pitch', 'notes', 'mistakes', 'reproduction', 'performed_notes', 'events', 'trace']}
    bench = MistakeBenchmarker()
    for pos, row in enumerate(cases.itertuples(), 1):
        print(f'{pos}/{len(cases)} {row.instrument} seed={row.seed} rate={row.rate}', flush=True)
        case = output / 'cases' / row.case_id
        item = json.loads((case / 'manifest.json').read_text())
        with (destination / 'replay.log').open('a') as log, redirect_stdout(log), redirect_stderr(log):
            reference = OneInstrumentScoreData(item['source_midi']).note_data
            performed = OneInstrumentScoreData(item['midi']).note_data
            perf = sequence(performed)
            events = MistakeDetectorBase.truth_events(json.loads((case / 'net_truth.json').read_text())['net_truth'], reference, performed)
            cfg = Config(**{'min_note_length_cap': 0.0, **json.loads(row.config)})
            cache = PitchCache(item['pitch_data'])
            pitches = cache.read(PitchCache.SMOOTHED, cfg)[0]
            pitches.end_index = len(pitches.data)
            raw = cache.read(PitchCache.RAW, cfg)
            common = dict(case_id=row.case_id, source=row.source, instrument=row.instrument, seed=row.seed, rate=row.rate)
            for stage, data in [('smoothed', pitches)] + ([('raw', raw[0])] if raw else []):
                tables['pitch'].append(dict(**common, stage=stage, **frame_counts(data, perf)))

            def recording(initial=None, pitch_source=pitches):
                rec = bench.attune.recording_for(replace(cfg), score_data=OneInstrumentScoreData(item['source_midi']))
                rec.pitch_data = PitchData(rec.config)
                rec.pitch_data.data = deepcopy(pitch_source.data, {id(pitch_source.config): rec.config})
                rec.pitch_data.t_origin = pitch_source.t_origin
                rec.pitch_data.end_index = len(rec.pitch_data.data)
                if initial is not None:
                    rec.note_data = NoteDetectorBase.clone_note_data(initial)
                return rec
            base = recording()
            base.detect_notes()
            initial = NoteDetectorBase.clone_note_data(base.note_data)
            initial_match = {j for i, j in note_matches(perf, sequence(initial))}
            for j, n in enumerate(perf):
                extra = any((e['kind'] == 'extra' and abs(e['onset'] - n.start_time) < 1e-07 and (e['pitch'] == n.midi_num[0]) for e in events))
                freq = cfg.tuning * 2 ** ((n.midi_num[0] - 69) / 12)
                tables['performed_notes'].append(dict(**common, index=j, onset=n.start_time, end=n.end_time, pitch=n.midi_num[0], duration=n.duration(), extra=extra, matched_initial=j in initial_match, outside_range=freq < cfg.fmin - 1e-06 or freq > cfg.fmax + 1e-06, below_min_duration=n.duration() < base.config.note_detection_min_seconds(), repeated_previous=j > 0 and perf[j - 1].midi_num[0] == n.midi_num[0], **support(n, pitches)))
            snapshots = {}
            modes = ['Attune (no refinement)', 'Attune (Checker 3)', 'fit only', 'checker only', 'rounded pitch cost', 'onset cost 1', 'pitch only cost', 'no initial fit', 'Parangonar DualDTW']
            if 'Attune (repeat only)' in set(saved.method):
                modes.insert(1, 'Attune (repeat only)')
            for mode in modes:
                rec = recording(initial)
                if mode in ('Attune (Checker 3)', 'checker only'):
                    from notebooks.archive.MistakeChecker3 import MistakeChecker as LegacyRepeatSplitter
                    rec.repeat_splitter = LegacyRepeatSplitter(recording=rec)
                if mode not in ('no initial fit', 'Parangonar DualDTW') and rec.note_data.times:
                    rec.resize_score(to_span='onset')
                if mode == 'rounded pitch cost':
                    rec.mistake_detector.get_pitch_distance = lambda u, s: abs(float(np.rint(u.midi_num[0])) - s.midi_num[0])
                elif mode == 'onset cost 1':
                    rec.config.alignment_alpha_onset = 1.0
                    rec.config.alignment_alpha_duration = 0.0
                elif mode == 'pitch only cost':
                    rec.config.alignment_gamma_time = 0.0
                rec.detect_mistakes()
                if run.get('score_fit_protocol') == 'matched_onsets_once_v1' and mode not in ('Attune (Checker 3)', 'checker only', 'fit only', 'no initial fit', 'Parangonar DualDTW'):
                    rec.refit_score_alignment_once()
                if mode == 'Parangonar DualDTW':
                    rec.alignment.pairs = MistakeBenchmarker.external_pairs(mode, sequence(rec.score_data.note_data), sequence(rec.note_data), rec.score_data.bpm)
                elif mode == 'Attune (repeat only)':
                    rec.stabilize_score_alignment()
                elif mode == 'checker only':
                    rec.repeat_splitter.check_mistakes()
                elif mode == 'fit only':
                    rec.repeat_splitter = SimpleNamespace(check_mistakes=lambda **kw: None)
                    rec.stabilize_score_alignment()
                elif mode == 'Attune (Checker 3)':
                    checker = rec.repeat_splitter.check_mistakes
                    fit = rec.resize_score_to_aligned_onsets
                    steps = []

                    def traced_fit(*args, **kwargs):
                        before = snapshot(rec, reference, events)
                        result = fit(*args, **kwargs)
                        rec.detect_mistakes()
                        after = snapshot(rec, reference, events)
                        steps.append(dict(stage='fit', before=before, after=after))
                        return result

                    def traced_check(*args, **kwargs):
                        before = snapshot(rec, reference, events)
                        result = checker(*args, **kwargs)
                        after = snapshot(rec, reference, events)
                        steps.append(dict(stage='checker', before=before, after=after))
                        return result
                    rec.resize_score_to_aligned_onsets = traced_fit
                    rec.repeat_splitter.check_mistakes = traced_check
                    rec.stabilize_score_alignment()
                    for k, step in enumerate(steps):
                        tables['trace'].append(dict(**common, step=k, stage=step['stage'], **{prefix + '_' + key: value[key] for prefix, value in [('before', step['before']), ('after', step['after'])] for key in ['tp', 'fp', 'fn', 'cost']}))
                    snapshots['refinement_steps'] = steps
                snap = snapshot(rec, reference, events)
                snapshots[mode] = snap
                for metric, counts in MistakeDetectorBase.score_events(snap['predicted'], events).items():
                    tables['mistakes'].append(dict(**common, method=mode, metric=metric, tp=counts[0], fp=counts[1], fn=counts[2]))
                if mode in ['Attune (no refinement)', 'Attune (repeat only)', 'Attune (Checker 3)', 'Parangonar DualDTW']:
                    old = saved[(saved.case_id == row.case_id) & (saved.method == mode) & (saved.input == 'detected') & (saved.metric == 'audio_pitch') & np.isclose(saved.tolerance, 0.1)].iloc[0]
                    ok = tuple((snap[k] for k in ['tp', 'fp', 'fn'])) == tuple((old[k] for k in ['tp', 'fp', 'fn']))
                    tables['reproduction'].append(dict(**common, method=mode, reproduced=ok))
                    if not ok:
                        raise ValueError(f'Saved results did not reproduce: {row.case_id} {mode}')
                estimates = sequence(rec.note_data)
                if mode in ['Attune (no refinement)', 'Attune (repeat only)', 'Attune (Checker 3)']:
                    for tol in [0.05, 0.1, 0.2, 0.5]:
                        for offsets in [False, True]:
                            matches = note_matches(perf, estimates, tol, offsets)
                            tables['notes'].append(dict(**common, stage=mode, tolerance=tol, offsets=offsets, tp=len(matches), fp=len(estimates) - len(matches), fn=len(perf) - len(matches)))
                good_pred, good_truth = matched_indices(snap['predicted'], events)
                for i, e in enumerate(events):
                    tables['events'].append(dict(**common, method=mode, truth=True, correct=i in good_truth, **e))
                for i, e in enumerate(snap['predicted']):
                    tables['events'].append(dict(**common, method=mode, truth=False, correct=i in good_pred, **e))
            ideal = PitchData(cfg)
            ideal.t_origin = pitches.t_origin
            times = np.array([p.time if p else ideal.t_origin + i * cfg.h1 / cfg.sr for i, p in enumerate(pitches.data)])
            values, _ = frame_reference(times, perf)
            from app_logic.user.ds.PitchData import Pitch
            ideal.data = [Pitch(float(t), 0.1 if v >= 0 else 0.0, 0.0 if v >= 0 else 1.0, 0.0, cfg, candidates=[(float(v), 1.0)] if v >= 0 else [], value=float(v)) for t, v in zip(times, values)]
            rec = recording(pitch_source=ideal)
            rec.detect_notes()
            for tol in [0.05, 0.1, 0.2, 0.5]:
                for offsets in [False, True]:
                    matches = note_matches(perf, sequence(rec.note_data), tol, offsets)
                    tables['notes'].append(dict(**common, stage='oracle pitch -> segmenter', tolerance=tol, offsets=offsets, tp=len(matches), fp=len(rec.note_data.times) - len(matches), fn=len(perf) - len(matches)))
            snapshots['truth'] = events
            snapshots['performed'] = [dict(onset=n.start_time, end=n.end_time, pitch=n.midi_num[0]) for n in perf]
            (destination / (row.case_id + '.json')).write_text(json.dumps(snapshots, indent=2))
        for key, rows in tables.items():
            pd.DataFrame(rows).to_csv(destination / (key + '.csv'), index=False)
    (destination / 'metadata.json').write_text(json.dumps(dict(source_run_sha256=hashlib.sha256((output / 'run.json').read_bytes()).hexdigest(), audit_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), note='Nominal performed-MIDI frame truth. No latency fitting. 50-cent pitch gates; note offset gate max(50ms,20% reference duration). Clean seeds deduplicated. Ablations are exploratory on this test sample.'), indent=2))
    return {key: pd.DataFrame(value) for key, value in tables.items()}

def summary(output):
    """Read saved diagnostics, checking their mistake counts against this run."""
    output = Path(output)
    dest = output / 'pipeline_audit'
    saved = pd.read_csv(output / 'rows.csv')
    mistakes = pd.read_csv(dest / 'mistakes.csv')
    methods = ['Attune (no refinement)', 'Attune (repeat only)', 'Attune (Checker 3)', 'Parangonar DualDTW']
    methods = [method for method in methods if method in set(saved.method)]
    actual = mistakes[(mistakes.metric == 'audio_pitch') & mistakes.method.isin(methods)]
    expected = saved[(saved.metric == 'audio_pitch') & saved.method.isin(methods) & (saved.input == 'detected') & np.isclose(saved.tolerance, 0.1)]
    joined = actual.merge(expected, on=['case_id', 'method'], suffixes=('_audit', '_saved'), validate='one_to_one')
    injected = set(saved[saved.rate > 0].case_id)
    if len(joined) != len(actual) or set(actual[actual.rate > 0].case_id) != injected or any((not (joined[k + '_audit'] == joined[k + '_saved']).all() for k in ['tp', 'fp', 'fn'])):
        raise ValueError('Saved audit does not match the run. Rerun PipelineAudit and PipelineAuditDetails.')
    result = []

    def add(label, tp, fp, fn):
        result.append(dict(Stage=label, **{'Precision %': 100 * tp / (tp + fp), 'Recall %': 100 * tp / (tp + fn), 'F1 %': 200 * tp / (2 * tp + fp + fn)}))
    pitch = pd.read_csv(dest / 'pitch.csv')
    for label, g in pitch[pitch.rate > 0].groupby('stage'):
        n = g.sum(numeric_only=True)
        add('Pitch frames: ' + label, n.correct, n.est_frames - n.correct, n.ref_frames - n.correct)
    notes = pd.read_csv(dest / 'notes.csv')
    for stage in ['Attune (no refinement)', 'Attune (repeat only)', 'Attune (Checker 3)', 'oracle pitch -> segmenter']:
        if stage not in set(notes.stage):
            continue
        n = notes[(notes.rate > 0) & (notes.stage == stage) & np.isclose(notes.tolerance, 0.1) & ~notes.offsets][['tp', 'fp', 'fn']].sum()
        add('Notes @100ms: ' + stage, n.tp, n.fp, n.fn)
    for stage in methods:
        n = actual[(actual.rate > 0) & (actual.method == stage)][['tp', 'fp', 'fn']].sum()
        add('Mistakes: ' + stage, n.tp, n.fp, n.fn)
    return pd.DataFrame(result).set_index('Stage').round(1)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('output')
    args = parser.parse_args()
    audit(args.output)
