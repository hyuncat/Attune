"""Paired repeat-only experiment on real URMP audio and official score MIDI.

Select distinct locally cached recordings with exact score/annotation pitch
sequences and a score repetition. Never use annotation timing as input.
"""
from pathlib import Path
from copy import deepcopy
from dataclasses import asdict
from dataclasses import replace
from contextlib import redirect_stdout
import hashlib
import json
import numpy as np
import pandas as pd
import pretty_midi
from app_logic.NoteData import Note
from app_logic.user.ds.AudioData import AudioData
from app_logic.user.ds.PitchData import PitchData
from benchmarks.modules.pitch.PitchCache import PitchCache
from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker
from benchmarks.modules.mistake.provenance.PipelineAudit import sequence
from benchmarks.modules.mistake.provenance.PipelineAudit import note_matches
from benchmarks.modules.mistake.sweeps.RepeatRefinementComparison import repeat_split
from benchmarks.modules.mistake.sweeps.RepeatRefinementComparison import groups
from benchmarks.modules.note.NoteBenchmarker import OneInstrumentScoreData
from benchmarks.modules.note.NoteDetectorBase import NoteDetectorBase
from benchmarks.paths import REPO_ROOT

def scan(root):
    records = []
    for wav in sorted(root.glob('*/AuSep_*.wav')):
        suffix = wav.stem.removeprefix('AuSep_')
        part = int(suffix.split('_')[0])
        annotations = wav.with_name('Notes_' + suffix + '.txt')
        scores = list(wav.parent.glob('Sco_*.mid'))
        if not annotations.exists() or not scores:
            continue
        pm = pretty_midi.PrettyMIDI(str(scores[0]))
        instruments = [i for i in pm.instruments if i.notes]
        if part > len(instruments):
            continue
        notes = sorted(instruments[part - 1].notes, key=lambda n: n.start)
        table = np.loadtxt(annotations, ndmin=2)
        pitches = np.rint(69 + 12 * np.log2(table[:, 1] / 440)).astype(int).tolist()
        cache = PitchCache.path_for(root, wav.parent.name + '/' + wav.stem)
        repeats = sum((a.pitch == b.pitch for a, b in zip(notes, notes[1:])))
        records.append(dict(wav=str(wav), score=str(scores[0]), annotation=str(annotations), part=part, instrument=suffix.split('_')[1], repeats=repeats, notes=len(notes), annotated=len(pitches), pitch_sequence_equal=pitches == [n.pitch for n in notes], cached=cache.exists(), cache=str(cache), track=wav.stem))
    return records

def run(root, output):
    root, output = (Path(root).resolve(), Path(output).resolve())
    output.mkdir(parents=True, exist_ok=True)
    inventory = scan(root)
    selected = [r for r in inventory if r['cached'] and r['pitch_sequence_equal'] and r['repeats']]
    unique, duplicates, seen_audio = ([], [], {})
    for item in selected:
        digest = hashlib.sha256(Path(item['wav']).read_bytes()).hexdigest()
        item['audio_sha256'] = digest
        if digest in seen_audio:
            duplicates.append(dict(track=item['track'], duplicate_of=seen_audio[digest]))
        else:
            seen_audio[digest] = item['track']
            unique.append(item)
    selected = unique
    (output / 'inventory.json').write_text(json.dumps(inventory, indent=2))
    code = ['benchmarks/modules/mistake/sweeps/RepeatCorpusComparison.py', 'benchmarks/modules/mistake/sweeps/RepeatRefinementComparison.py', 'algorithms/RepeatSplitter.py', 'algorithms/NoteDetector.py', 'notebooks/archive/MistakeChecker3.py', 'algorithms/MistakeDetector.py', 'app_logic/user/ds/Recording.py']
    (output / 'run.json').write_text(json.dumps(dict(selected=selected, excluded_duplicate_audio=duplicates, code={p: hashlib.sha256((REPO_ROOT / p).read_bytes()).hexdigest() for p in code}, protocol='real audio; official score timing; annotation timing used only for evaluation; no truth-based edge trimming', status='running'), indent=2))
    bench = MistakeBenchmarker()
    rows = []
    for i, item in enumerate(selected):
        print(f'{i + 1}/{len(selected)} {item['track']}', flush=True)
        with (output / (item['track'] + '.log')).open('w') as log, redirect_stdout(log):
            pm = pretty_midi.PrettyMIDI(item['score'])
            instrument = [x for x in pm.instruments if x.notes][item['part'] - 1]
            single = pretty_midi.PrettyMIDI(initial_tempo=120, resolution=960)
            single.instruments = [deepcopy(instrument)]
            midi = output / (item['track'] + '.mid')
            single.write(str(midi))
            source = OneInstrumentScoreData(midi)
            table = np.loadtxt(item['annotation'], ndmin=2)
            reference = [Note(j, float(t), float(t + d), [float(round(69 + 12 * np.log2(f / 440)))]) for j, (t, f, d) in enumerate(table)]
            score = sequence(source.note_data)
            assert [n.midi_num[0] for n in score] == [n.midi_num[0] for n in reference]
            repeat_indices = {j for j in range(1, len(score)) if score[j].midi_num == score[j - 1].midi_num}
            cfg = bench.attune.config_for(*bench.attune.range_from_midi(source.midi_numbers))
            base = bench.attune.recording_for(cfg, score_data=source)
            cached = PitchCache(item['cache']).read(PitchCache.SMOOTHED, cfg)
            cache_reused = cached is not None
            if cached is None:
                print('Refreshing incompatible pitch cache', flush=True)
                base.audio_data = AudioData(audio_filepath=item['wav'], config=cfg)
                if base.audio_data.sr != cfg.sr:
                    import librosa
                    audio = librosa.resample(base.audio_data.read_all(), orig_sr=base.audio_data.sr, target_sr=cfg.sr)
                    base.audio_data.data = audio
                    base.audio_data.sr = cfg.sr
                    base.audio_data.end_index = base.audio_data.capacity = len(audio)
                bench.attune.load_or_detect_pitches(base, item['cache'])
            else:
                base.pitch_data = cached[0]
            base.pitch_data.end_index = len(base.pitch_data.data)
            base.detect_notes()
            initial = NoteDetectorBase.clone_note_data(base.note_data)
            saved = dict(reference=[dict(onset=n.start_time, end=n.end_time, pitch=n.midi_num[0]) for n in reference], repeat_indices=sorted(repeat_indices), cache_reused=cache_reused, methods={})
            for method in ['No refinement', 'Repeat only', 'Checker 3']:
                rec = bench.attune.recording_for(replace(base.config), score_data=OneInstrumentScoreData(midi))
                rec.pitch_data = PitchData(rec.config)
                rec.pitch_data.data = deepcopy(base.pitch_data.data, {id(base.config): rec.config})
                rec.pitch_data.end_index = len(rec.pitch_data.data)
                rec.pitch_data.t_origin = base.pitch_data.t_origin
                rec.note_data = NoteDetectorBase.clone_note_data(initial)
                rec.resize_score(to_span='onset')
                rec.detect_mistakes()
                proposals = []
                if method == 'Repeat only':
                    rec.stabilize_score_alignment()
                    proposals = rec.repeat_splitter.proposals
                elif method == 'Checker 3':
                    from notebooks.archive.MistakeChecker3 import MistakeChecker as LegacyRepeatSplitter
                    rec.repeat_splitter = LegacyRepeatSplitter(recording=rec)
                    rec.stabilize_score_alignment()
                notes = sequence(rec.note_data)
                matches = note_matches(reference, notes, 0.1)
                matched_ref = {j for _, j in matches}
                alarms = MistakeBenchmarker.label_pairs(rec.alignment.pairs, rec.config)
                import benchmarks.modules.mistake.MistakeDetectorBase as _api_MistakeDetectorBase
                events = predicted_events(alarms, rec.score_data.note_data)
                rows.append(dict(track=item['track'], instrument=item['instrument'], method=method, note_tp=len(matches), note_fp=len(notes) - len(matches), note_fn=len(reference) - len(matches), repeat_tp=len(repeat_indices & matched_ref), repeat_total=len(repeat_indices), notes=len(notes), initial_notes=len(initial.times), pitch_alarms=len(events), cuts=sum((len(p['cuts']) for p in proposals if p['status'] == 'split')), config=json.dumps(asdict(rec.config), sort_keys=True), cache_reused=cache_reused))
                saved['methods'][method] = dict(notes=[dict(onset=n.start_time, end=n.end_time, pitch=n.midi_num[0]) for n in notes], matches=matches, proposals=proposals, events=events)
            (output / (item['track'] + '.json')).write_text(json.dumps(saved, indent=2))
        pd.DataFrame(rows).to_csv(output / 'rows.csv', index=False)
    table = pd.DataFrame(rows).groupby('method')[['note_tp', 'note_fp', 'note_fn', 'repeat_tp', 'repeat_total', 'pitch_alarms', 'cuts']].sum()
    table['note_f1'] = 200 * table.note_tp / (2 * table.note_tp + table.note_fp + table.note_fn)
    table['repeat_recall'] = 100 * table.repeat_tp / table.repeat_total
    table.to_csv(output / 'summary.csv')
    metadata = json.loads((output / 'run.json').read_text())
    metadata['status'] = 'complete'
    (output / 'run.json').write_text(json.dumps(metadata, indent=2))
    print(table.to_string())
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='benchmarks/datasets/urmp')
    p.add_argument('--output', default='benchmarks/results/repeat_refinement_urmp')
    args = p.parse_args()
    run(args.root, args.output)
