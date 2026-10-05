"""Explain saved pipeline-audit events and test pitch-preserving edit costs."""
from contextlib import redirect_stdout
from pathlib import Path
import json
import numpy as np
import pandas as pd
from algorithms.Config import Config
from app_logic.NoteData import Note
from app_logic.NoteData import NoteData
from benchmarks.modules.mistake.provenance.PipelineAudit import note_matches
from benchmarks.modules.mistake.provenance.PipelineAudit import sequence
from benchmarks.modules.mistake.provenance.PipelineAudit import snapshot
from benchmarks.modules.mistake.provenance.InjectedCaseAudit import matched_indices
from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker
from benchmarks.modules.note.NoteBenchmarker import OneInstrumentScoreData

def explain(output):
    output = Path(output)
    dest = output / 'pipeline_audit'
    robust_fit = json.loads((output / 'run.json').read_text()).get('score_fit_protocol') == 'matched_onsets_once_v1'
    rows = pd.read_csv(output / 'rows.csv')
    cases = rows[rows.rate > 0].drop_duplicates('case_id')
    support = pd.read_csv(dest / 'performed_notes.csv')
    classified = []
    ablations = []
    missed_notes = []
    pairs = []
    boundaries = []
    bench = MistakeBenchmarker()
    with (dest / 'details.log').open('w') as log, redirect_stdout(log):
        for r in cases.itertuples():
            d = json.loads((dest / (r.case_id + '.json')).read_text())
            manifest = json.loads((output / 'cases' / r.case_id / 'manifest.json').read_text())
            truth = json.loads((output / 'cases' / r.case_id / 'net_truth.json').read_text())
            reference = OneInstrumentScoreData(manifest['source_midi']).note_data
            performed = OneInstrumentScoreData(manifest['midi']).note_data
            perf = sequence(performed)
            by_id = performed.notes_by_id()
            per_index = {n.id: j for j, n in enumerate(perf)}
            common = dict(case_id=r.case_id, instrument=r.instrument, seed=r.seed)
            initial = d['Attune (no refinement)']['notes']
            initial_data = NoteData()
            for n in initial:
                initial_data.write_note(Note(n['id'], n['onset'], n['end'], [n['pitch']]))
            for i, j in note_matches(perf, sequence(initial_data), 0.2):
                n = sequence(initial_data)[i]
                boundaries.append(dict(**common, onset_error=n.start_time - perf[j].start_time, offset_error=n.end_time - perf[j].end_time))
            refs = sequence(reference)
            paired = {p['score_note_id']: p['performed_note_id'] for p in truth['pairs']}
            for mode in ['Attune (no refinement)', 'Attune (repeat only)', 'Attune (Checker 3)', 'Parangonar DualDTW']:
                if mode not in d:
                    continue
                s = d[mode]
                good, _ = matched_indices(s['predicted'], d['truth'])
                for i, event in enumerate(s['predicted']):
                    if i in good:
                        continue
                    if event['kind'] == 'extra':
                        same = [n for n in perf if abs(n.midi_num[0] - event['pitch']) <= 0.5]
                        if any((abs(n.start_time - event['onset']) <= 0.1 for n in same)):
                            reason = 'real performed note wrongly flagged'
                        elif any((n.start_time - 0.1 <= event['onset'] <= n.end_time + 0.1 for n in same)):
                            reason = 'same-pitch fragment or displaced onset'
                        else:
                            reason = 'pitch unsupported by nearby MIDI notes'
                    else:
                        ref = min(refs, key=lambda n: abs(n.start_time - event['onset']))
                        played = by_id.get(paired.get(ref.id))
                        if played is None:
                            reason = 'net-truth correspondence differs'
                        else:
                            found = any((abs(n['onset'] - played.start_time) <= 0.1 and abs(n['pitch'] - played.midi_num[0]) <= 0.5 for n in initial))
                            supp = support[(support.case_id == r.case_id) & (support['index'] == per_index[played.id])].iloc[0].pitch_support
                            if found:
                                reason = 'initial matching note exists: alignment/refinement'
                            elif supp >= 0.5:
                                reason = 'pitch present: segmentation/onset mismatch'
                            else:
                                reason = 'weak/absent true pitch in smoothed frames'
                    classified.append(dict(**common, method=mode, reason=reason, **event))
                for u, v in s['pairs']:
                    if u is not None and v is not None:
                        gap = abs(s['notes'][u]['pitch'] - s['score'][v]['pitch'])
                        pairs.append(dict(**common, method=mode, pitch_gap=gap))
            for tol in [0.1, 0.2]:
                matched = {j for i, j in note_matches(perf, sequence(initial_data), tol)}
                for j, n in enumerate(perf):
                    if j in matched:
                        continue
                    supp = support[(support.case_id == r.case_id) & (support['index'] == j)].iloc[0]
                    missed_notes.append(dict(**common, tolerance=tol, index=j, onset=n.start_time, pitch=n.midi_num[0], duration=n.duration(), extra=supp.extra, below_min_duration=supp.below_min_duration, support=supp.pitch_support, outside_range=supp.outside_range, repeated_previous=supp.repeated_previous))
            for mode in ['weighted edit replay', 'pitch-preserving rounded', 'pitch-preserving tolerance', 'pitch weight 4']:
                cfg = Config(**{'min_note_length_cap': 0.0, **json.loads(r.config)})
                rec = bench.attune.recording_for(cfg, score_data=OneInstrumentScoreData(manifest['source_midi']))
                from benchmarks.modules.note.NoteDetectorBase import NoteDetectorBase
                rec.note_data = NoteDetectorBase.clone_note_data(initial_data)
                rec.resize_score(to_span='onset')
                original = rec.mistake_detector.get_substitution_cost
                if mode.startswith('pitch-preserving'):

                    def cost(u, s):
                        different = round(u.midi_num[0]) != s.midi_num[0] if mode.endswith('rounded') else abs(u.midi_num[0] - s.midi_num[0]) >= cfg.pitch_tolerance
                        return rec.mistake_detector.get_insertion_cost(u) + rec.mistake_detector.get_deletion_cost(s) + 1 if different else original(u, s)
                    rec.mistake_detector.get_substitution_cost = cost
                if mode == 'pitch weight 4':
                    cfg.alignment_gamma_pitch = 4.0
                rec.detect_mistakes()
                if robust_fit:
                    rec.refit_score_alignment_once()
                snap = snapshot(rec, reference, d['truth'])
                if mode == 'weighted edit replay':
                    assert all((snap[k] == d['Attune (no refinement)'][k] for k in ['tp', 'fp', 'fn']))
                ablations.append(dict(**common, method=mode, **{k: snap[k] for k in ['tp', 'fp', 'fn']}))
    for name, data in [('false_positive_causes', classified), ('alignment_ablations', ablations), ('missed_notes', missed_notes), ('pair_pitch_gaps', pairs), ('boundary_errors', boundaries)]:
        pd.DataFrame(data).to_csv(dest / (name + '.csv'), index=False)

def oracle_segments(output):
    """Counterfactual: exact pitch/voicing, production segmentation and alignment."""
    from contextlib import redirect_stdout
    import json
    import numpy as np
    import pandas as pd
    from dataclasses import replace
    from algorithms.Config import Config
    from app_logic.user.ds.PitchData import Pitch
    from app_logic.user.ds.PitchData import PitchData
    from benchmarks.modules.pitch.PitchCache import PitchCache
    from benchmarks.modules.mistake.provenance.PipelineAudit import frame_reference
    from benchmarks.modules.mistake.provenance.PipelineAudit import sequence
    from benchmarks.modules.mistake.provenance.PipelineAudit import note_matches
    from benchmarks.modules.mistake.provenance.PipelineAudit import snapshot
    from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker
    from benchmarks.modules.note.NoteBenchmarker import OneInstrumentScoreData
    p = Path(output)
    dest = p / 'pipeline_audit'
    bench = MistakeBenchmarker()
    r = pd.read_csv(p / 'rows.csv').query('rate>0').drop_duplicates('case_id')
    out = []
    missing = []
    with (dest / 'oracle_segment.log').open('w') as log, redirect_stdout(log):
        for row in r.itertuples():
            item = json.load(open(p / 'cases' / row.case_id / 'manifest.json'))
            snap = json.load(open(dest / (row.case_id + '.json')))
            cfg = Config(**{'min_note_length_cap': 0.0, **json.loads(row.config)})
            real = PitchCache(item['pitch_data']).read(PitchCache.SMOOTHED, cfg)[0]
            perf = OneInstrumentScoreData(item['midi']).note_data
            reference = OneInstrumentScoreData(item['source_midi']).note_data
            times = np.array([f.time if f else real.t_origin + i * cfg.h1 / cfg.sr for i, f in enumerate(real.data)])
            values, _ = frame_reference(times, sequence(perf))
            for factor in [cfg.min_note_length_factor, 0.1]:
                conf = replace(cfg, min_note_length_factor=factor)
                rec = bench.attune.recording_for(conf, score_data=OneInstrumentScoreData(item['source_midi']))
                rec.pitch_data.data = [Pitch(float(t), 0.1 if v >= 0 else 0.0, 0.0 if v >= 0 else 1.0, 0.0, conf, candidates=[(float(v), 1.0)] if v >= 0 else [], value=float(v)) for t, v in zip(times, values)]
                rec.pitch_data.end_index = len(times)
                rec.pitch_data.t_origin = real.t_origin
                rec.detect_notes()
                matches = note_matches(sequence(perf), sequence(rec.note_data), 0.1)
                ok = {j for i, j in matches}
                seq = sequence(perf)
                for j, n in enumerate(seq):
                    if j not in ok:
                        missing.append(dict(case_id=row.case_id, instrument=row.instrument, factor=factor, index=j, duration=n.duration(), pitch=n.midi_num[0], repeated_previous=j > 0 and seq[j - 1].midi_num[0] == n.midi_num[0], min_seconds=conf.min_note_length * factor))
                rec.resize_score(to_span='onset')
                rec.detect_mistakes()
                if json.loads((p / 'run.json').read_text()).get('score_fit_protocol') == 'matched_onsets_once_v1':
                    rec.refit_score_alignment_once()
                state = snapshot(rec, reference, snap['truth'])
                out.append(dict(case_id=row.case_id, instrument=row.instrument, factor=factor, note_tp=len(matches), note_fp=len(rec.note_data.times) - len(matches), note_fn=len(seq) - len(matches), **{k: state[k] for k in ['tp', 'fp', 'fn']}))
    pd.DataFrame(out).to_csv(dest / 'oracle_segment_ablation.csv', index=False)
    pd.DataFrame(missing).to_csv(dest / 'oracle_missing_notes.csv', index=False)
    a = pd.DataFrame(out).groupby('factor')[['tp', 'fp', 'fn', 'note_tp', 'note_fp', 'note_fn']].sum()
    a['mistake_f1'] = 200 * a.tp / (2 * a.tp + a.fp + a.fn)
    a['note_f1'] = 200 * a.note_tp / (2 * a.note_tp + a.note_fp + a.note_fn)
    print(a.to_string())
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('output')
    args = p.parse_args()
    explain(args.output)
    oracle_segments(args.output)
