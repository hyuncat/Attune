"""Read saved injected cases; replay Attune without regenerating audio/pitches.

FP categories are evidence-based diagnostics, not causal proof. Matching uses
exactly the benchmark's one-to-one onset/pitch gates. Saved count checks expose
any drift between the saved run and current note/refinement code.
"""
import json
from pathlib import Path
from copy import deepcopy
from dataclasses import replace
import hashlib
import numpy as np
import pandas as pd
import pretty_midi
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from algorithms.Config import Config
from notebooks.archive.MistakeChecker3 import MistakeChecker as LegacyRepeatSplitter
from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker
from benchmarks.modules.mistake.MistakeDetectorBase import MistakeDetectorBase
from benchmarks.modules.note.NoteBenchmarker import OneInstrumentScoreData
from benchmarks.modules.note.NoteDetectorBase import NoteDetectorBase
from benchmarks.modules.pitch.PitchCache import PitchCache

def matched_indices(predicted, truth):
    valid = np.zeros((len(predicted), len(truth)), dtype=bool)
    for i, p in enumerate(predicted):
        for j, t in enumerate(truth):
            valid[i, j] = p['kind'] == t['kind'] and abs(p['pitch'] - t['pitch']) <= 0.5 and (abs(p['onset'] - t['onset']) <= 0.1)
    match = maximum_bipartite_matching(csr_matrix(valid), perm_type='column')
    return ({i for i, j in enumerate(match) if j >= 0}, {int(j) for j in match if j >= 0})

def near_note(event, notes, tolerance=0.1):
    return any((abs(n.start_time - event['onset']) <= tolerance and abs(n.midi_num[0] - event['pitch']) <= 0.5 for n in notes))

def support(note, pitches):
    margin = min(0.05, note.duration() / 4)
    frames = [p for p in pitches.data if p is not None and note.start_time + margin <= p.time < note.end_time - margin]
    return dict(frames=len(frames), voiced_fraction=np.mean([p.value != -1 for p in frames]) if frames else np.nan, pitch_support=np.mean([p.value != -1 and abs(p.value - note.midi_num[0]) <= 0.5 for p in frames]) if frames else np.nan, candidate_support=np.mean([any((abs(m - note.midi_num[0]) <= 0.5 and prob > 0 for m, prob in p.candidate_pitches)) for p in frames]) if frames else np.nan)

def audit(output):
    from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker
    output = Path(output)
    destination = output / 'injected_audit'
    destination.mkdir(exist_ok=True)
    saved = pd.read_csv(output / 'rows.csv')
    robust_fit = json.loads((output / 'run.json').read_text()).get('score_fit_protocol') == 'matched_onsets_once_v1'
    cases = saved.drop_duplicates('case_id')
    cases = pd.concat([cases[cases.rate > 0], cases[cases.rate == 0].drop_duplicates('source')])
    fp_rows, extra_rows, note_rows, checks, grid_rows = ([], [], [], [], [])
    for index, item_row in enumerate(cases.itertuples(), 1):
        case_dir = output / 'cases' / item_row.case_id
        item = json.loads((case_dir / 'manifest.json').read_text())
        print(f'{index}/{len(cases)} {item_row.instrument} rate={item_row.rate}', flush=True)
        reference = OneInstrumentScoreData(item['source_midi']).note_data
        performed = OneInstrumentScoreData(item['midi']).note_data
        perf = [performed.data[t] for t in performed.times]
        refs = list(reference.data.values())
        midi = pretty_midi.PrettyMIDI(item['midi'])
        grid_rows.append(dict(case_id=item_row.case_id, notes=sum((len(i.notes) for i in midi.instruments)), off_grid=sum((not float(n.pitch).is_integer() for i in midi.instruments for n in i.notes)), pitch_bends=sum((len(i.pitch_bends) for i in midi.instruments))))
        truth = json.loads((case_dir / 'net_truth.json').read_text())
        events = MistakeDetectorBase.truth_events(truth['net_truth'], reference, performed)
        cfg = Config(**{'min_note_length_cap': 0.0, **json.loads(item_row.config)})
        bench = MistakeBenchmarker()
        cached = PitchCache(item['pitch_data']).read(PitchCache.SMOOTHED, cfg)
        if cached is None:
            raise ValueError(f'Missing saved pitch stage: {item['pitch_data']}')
        pitches = cached[0]
        pitches.end_index = len(pitches.data)
        base = bench.attune.recording_for(cfg, score_data=OneInstrumentScoreData(item['source_midi']))
        base.pitch_data = pitches
        base.detect_notes()
        initial = NoteDetectorBase.clone_note_data(base.note_data)
        common = dict(case_id=item_row.case_id, instrument=item_row.instrument, rate=item_row.rate, seed=item_row.seed)
        for note in perf:
            overlaps = [n for n in perf if n is not note and min(n.end_time, note.end_time) > max(n.start_time, note.start_time)]
            note_rows.append(dict(**common, onset=note.start_time, pitch=note.midi_num[0], overlap=bool(overlaps), detected_onset_pitch=near_note(dict(onset=note.start_time, pitch=note.midi_num[0]), list(initial.data.values())), **support(note, pitches)))
        for method, refine in [('Attune (no refinement)', False), ('Attune (Checker 3)', True)]:
            rec = bench.attune.recording_for(replace(base.config), score_data=OneInstrumentScoreData(item['source_midi']))
            rec.note_data = NoteDetectorBase.clone_note_data(initial)
            from app_logic.user.ds.PitchData import PitchData
            data = PitchData(rec.config)
            data.data = deepcopy(pitches.data, {id(base.config): rec.config})
            data.end_index = len(data.data)
            data.t_origin = pitches.t_origin
            rec.pitch_data = data
            if rec.note_data.times:
                rec.resize_score(to_span='onset')
            rec.detect_mistakes()
            if robust_fit and method != 'Attune (Checker 3)':
                rec.refit_score_alignment_once()
            if refine:
                rec.repeat_splitter = LegacyRepeatSplitter(recording=rec)
                rec.stabilize_score_alignment()
            mistakes = MistakeBenchmarker.label_pairs(rec.alignment.pairs, rec.config)
            import benchmarks.modules.mistake.MistakeDetectorBase as _api_MistakeDetectorBase
            mistakes = with_reference_score_ids(mistakes, reference, rec.score_data.note_data)
            predictions = MistakeDetectorBase.predicted_events(mistakes, reference)
            good_pred, good_truth = matched_indices(predictions, events)
            current = MistakeDetectorBase.score_events(predictions, events)['audio_pitch']
            old = saved[(saved.case_id == item_row.case_id) & (saved.method == method) & (saved.input == 'detected') & (saved.metric == 'audio_pitch') & np.isclose(saved.tolerance, 0.1)].iloc[0]
            checks.append(dict(**common, method=method, reproduced=tuple(current) == (old.tp, old.fp, old.fn), tp=current[0], fp=current[1], fn=current[2], saved_tp=old.tp, saved_fp=old.fp, saved_fn=old.fn))
            for i, event in enumerate(predictions):
                if i in good_pred:
                    continue
                nearby_truth = [e for e in events if e['kind'] == event['kind'] and abs(e['pitch'] - event['pitch']) <= 0.5 and (abs(e['onset'] - event['onset']) <= 0.3)]
                if nearby_truth:
                    reason = 'near true error: timing gate or duplicate'
                elif event['kind'] == 'extra':
                    if near_note(event, perf):
                        reason = 'real performed note incorrectly flagged'
                    elif any((abs(n.midi_num[0] - event['pitch']) <= 0.5 and n.start_time - 0.1 <= event['onset'] <= n.end_time + 0.1 for n in perf)):
                        reason = 'same-pitch fragment / displaced onset'
                    else:
                        reason = 'pitch not supported by nearby performed notes'
                else:
                    score_note = min(refs, key=lambda n: abs(n.start_time - event['onset']))
                    pair = next((p for p in truth['pairs'] if p['score_note_id'] == score_note.id), None)
                    played = performed.notes_by_id().get(pair['performed_note_id']) if pair else None
                    if played is None:
                        reason = 'net-truth correspondence differs'
                    elif near_note(dict(onset=played.start_time, pitch=played.midi_num[0]), list(initial.data.values())):
                        reason = 'matching initial note exists: alignment/refinement'
                    elif support(played, pitches)['pitch_support'] >= 0.5:
                        reason = 'pitch mostly present: segmentation/onset mismatch'
                    else:
                        reason = 'target pitch weak/absent in smoothed track'
                fp_rows.append(dict(**common, method=method, reason=reason, **event))
            for i, event in enumerate(events):
                if event['kind'] != 'extra':
                    continue
                note = performed.data[event['onset']]
                nominal = (truth.get('score_onsets') or {}).get(str(note.id), note.start_time)
                active = [n for n in refs if n.start_time <= nominal < n.end_time]
                nearest = min(refs, key=lambda n: abs(n.start_time - nominal))
                gap = min((abs(note.midi_num[0] - n.midi_num[0]) for n in active or [nearest]))
                overlap = any((n is not note and min(n.end_time, note.end_time) > max(n.start_time, note.start_time) for n in perf))
                initial_found = near_note(event, list(initial.data.values()))
                extra_rows.append(dict(**common, method=method, **event, detected=i in good_truth, net_type=next((t['type'] for t in truth['net_truth'] if t['type'] in ('insertion', 'substitution') and t['time'] == event['onset'])), gap_semitones=gap, gap_group='0' if gap == 0 else '1' if gap == 1 else '2+', overlap=overlap, initial_note_found=initial_found, **support(note, pitches)))
    tables = dict(false_positives=fp_rows, extra_events=extra_rows, performed_notes=note_rows, reproduction=checks, pitch_grid=grid_rows)
    for name, items in tables.items():
        pd.DataFrame(items).to_csv(destination / f'{name}.csv', index=False)
    (destination / 'metadata.json').write_text(json.dumps(dict(note='Replay of current note/alignment code over saved smoothed pitches; no cache regeneration. Heuristic FP categories; pitch coverage uses nominal MIDI time with 50ms edge exclusion.', audit_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2))
    return {name: pd.DataFrame(items) for name, items in tables.items()}
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('output')
    args = parser.parse_args()
    result = audit(args.output)
    print(result['reproduction'].groupby('method').reproduced.agg(['sum', 'count']))
