"""Small synthetic checks only; never runs the corpus sweeps."""
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / 'app.py').is_file())))
from pathlib import Path
import numpy as np
import pretty_midi, soundfile as sf
from experiments.modules.sweeps import AppendixSweeps as s
from app_logic.user.ds.PitchData import Pitch
from experiments.modules.pitch.competitors.Attune import Attune
from unittest.mock import patch

def smoke_notes():
 with tempfile.TemporaryDirectory() as d:
    root=Path(d)
    midi=pretty_midi.PrettyMIDI(); inst=pretty_midi.Instrument(40)
    for i,p in enumerate([60,62,64,65]):
        inst.notes.append(pretty_midi.Note(90,p,.1+i*.4,.45+i*.4))
    midi.instruments.append(inst); midi.write(str(root/'score.mid'))
    sf.write(root/'audio.wav',np.zeros(32000),16000)
    item=dict(track_id='synthetic',instrument='violin',paths=dict(midi=str(root/'score.mid'),audio=str(root/'audio.wav')))
    rec=s._recording(item,{})
    assert rec.config.sr==44100
    pitches=[]
    for t in np.arange(0,1.8,rec.config.h1/rec.config.sr):
        active=next((n.pitch for n in inst.notes if n.start<=t<n.end),None)
        pitches.append(Pitch(float(t),.2 if active else 0.,0. if active else 1.,0.,rec.config,
                             candidates=[] if active is None else [(active,1.)]))
    s.atomic(root/'pitch.pkl',pitches,True)
    variants=s.grid(dict(min_silence_duration_ms=(20.,40.),min_note_length_factor=(.5,),min_note_length_cap=(.2,),pitch_thresh=(.75,)))
    task=(item,str(root/'note'),variants,{},str(root/'pitch.pkl'))
    rows=s._note(task)
    assert len(rows)==2
    from app_logic.user.ds.Recording import Recording
    with patch.object(Recording,'detect_notes',side_effect=AssertionError('cache miss')):
        assert s._note(task)==rows
    # The E2 worker gets a prebuilt frontend, so this checks real production
    # alignment and independent weight jobs without rendering or pitch inference.
    for seed,rate in [(0,0.),(0,.25)]:
        folder=root/'mistake/frontend/synthetic'/f'{seed}_{rate}'
        s.atomic(folder/'notes.pkl',dict(config=s.asdict(rec.config),pitches=pitches,
            notes=s.blob(rows[0]['artifact']),truth=[],rate=rate,seed=seed),True)
    weights=s.grid(dict(alignment_gamma_pitch=(2.,4.),alignment_gamma_time=(1.,)))
    mt=(item,str(root/'mistake'),weights,{},(0,))
    results=s._mistake(mt)
    assert len(results)==2
    with patch.object(Recording,'align_score_and_refine',side_effect=AssertionError('cache miss')):
        assert s._mistake(mt)==results
    # Grid extension reuses saved frontend, evaluating only the new weights.
    s._mistake((item,str(root/'mistake'),s.grid(dict(alignment_gamma_pitch=(6.,),alignment_gamma_time=(1.,))),{},(0,)))
    print('Synthetic note + production alignment smoke passed; sample-rate and cache reuse verified')
    print('Note counts',[(r['tp'],r['fp'],r['fn']) for r in rows])


import sys,tempfile
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / 'app.py').is_file())))
from pathlib import Path
from unittest.mock import patch
import numpy as np, pretty_midi, soundfile as sf
from experiments.modules.sweeps import AppendixSweeps as s
from experiments.modules.pitch.datasets.CocoChorales import CocoChorales
from algorithms.PitchDetector import PitchDetector
from algorithms.PitchSmoother import PitchSmoother

def smoke_pitch():
 with tempfile.TemporaryDirectory() as d:
    d=Path(d); sr=16000
    t=np.arange(sr)/sr; sf.write(d/'audio.wav',.2*np.sin(2*np.pi*440*t),sr)
    pm=pretty_midi.PrettyMIDI(); ins=pretty_midi.Instrument(40)
    ins.notes=[pretty_midi.Note(90,69,0.,1.)]; pm.instruments=[ins]; pm.write(str(d/'score.mid'))
    record=dict(split='test',shard='',track='string_track999999',ensemble='string',stem='1_violin',stem_voice=1,f0_voice=0,instrument='violin',wav_member='',midi_member='',metadata_member='',mix_midi_member='',f0_path='')
    item=dict(track_id='synthetic',instrument='violin',record=record,paths=dict(audio=str(d/'audio.wav'),midi=str(d/'score.mid')))
    variants=s.grid(dict(posthoc_min_volume=(.04,),posthoc_unv_thresh=(.985,.9875)))
    task=(item,str(d/'pitch'),variants,str(d))
    ref_t=np.arange(100)/100; ref_f=np.ones(100)*440
    with patch.object(CocoChorales,'records_to_tracks',return_value=[object()]), patch.object(CocoChorales,'reference',return_value=(ref_t,ref_f)):
        rows=s._pitch(task)
        with patch.object(PitchDetector,'detect_pitches',side_effect=AssertionError('reran detector')),patch.object(PitchSmoother,'smooth',side_effect=AssertionError('reran HMM')):
            s._export_pitch((item,str(d/'pitch'),variants[0]))
            assert [r['score'] for r in s._pitch(task)] == [r['score'] for r in rows]
        cached = s.blob(rows[0]['artifact'])
        assert len(cached)>0
        assert np.isclose(cached[1].time-cached[0].time, 128/44100)
        assert .9 < cached[-1].time < 1.1
        assert not Path(rows[1]['artifact']).exists()
    print('Real one-second pitch inference, HMM reuse, winner export, and cache resume passed')
    print('Overall accuracy:',[r['score'] for r in rows])


def check_pitch_reporting_migration():
    from copy import deepcopy
    from experiments.modules.pitch.PitchBenchmarker import COUNT_COLUMNS, SCORING_VERSION
    helper = 'experiments/modules/sweeps/AppendixSweeps.py'
    predecessor = 'reviewed-reporting-fixture'
    variants = s.grid(dict(posthoc_min_volume=(.04,), posthoc_unv_thresh=(.985,)))
    current = dict(items=[dict(track_id='track', instrument='violin')], variants=variants,
                   identity=dict(code={helper: 'current', 'detector.py': 'same'}, packages={}))
    previous = deepcopy(current)
    previous['identity']['code'][helper] = predecessor
    row = dict(track_id='track', instrument='violin', variant=variants[0]['variant'],
               **variants[0]['params'], **{k: 1 for k in COUNT_COLUMNS}, scoring_version=SCORING_VERSION)
    with tempfile.TemporaryDirectory() as root, patch.object(s, '_PITCH_REPORTING_PREDECESSORS', {predecessor}):
        root = Path(root)
        s.atomic(root/'contract.json', previous)
        s.atomic(root/'track/rows.json', [row])
        s.atomic(root/'best.json', dict(old_report=True))
        s.pitch_contract(root, current)
        assert s.read(root/'contract.json') == current
        assert (root/'report_history'/s.digest(previous)[:16]/'best.json').is_file()
        assert s.read(root/'track/rows.json') == [row]
        # Idempotent, but an actual detector/input/grid change still fails.
        s.pitch_contract(root, current)
        for changed in ('detector', 'grid', 'input'):
            bad = deepcopy(current)
            if changed == 'detector': bad['identity']['code']['detector.py'] = 'changed'
            if changed == 'grid': bad['variants'] = []
            if changed == 'input': bad['items'][0]['instrument'] = 'flute'
            try:
                s.pitch_contract(root, bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f'Accepted incompatible {changed}')
    print('Reporting migration preserves caches/history and rejects incompatible inputs')


def check_benchmark_metrics():
    import pandas as pd
    from algorithms.Config import Config
    from app_logic.NoteData import Note, NoteData
    from experiments.modules.note.NoteBenchmarker import NoteEvaluation, NotebookConfig
    from experiments.modules.pitch.PitchBenchmarker import PitchBenchmarker

    ref_iv = np.array([[0., 1.], [2., 3.], [4., 5.]])
    ref_hz = np.full(3, 440.)
    detected = NoteData()
    for i, (start, end) in enumerate(((.075, 1.), (2., 2.5), (4., 5.))):
        detected.write_note(Note(i, start, end, [69]))
    metrics = s._note_metrics((ref_iv, ref_hz), detected, Config())
    # A 75 ms onset must fail; a correct onset with a bad offset must only
    # count toward Onset F1. This catches both previous sweep mismatches.
    assert metrics['tp'] == 2 and metrics['tp_offset'] == 1
    expected = NoteEvaluation.score_predictions(ref_iv, ref_hz,
        np.array([[.075, 1.], [2., 2.5], [4., 5.]]), ref_hz, NotebookConfig())
    assert all(metrics[k] == v for k, v in expected.items())
    perfect = dict(metrics, reference_notes=9, estimated_notes=9)
    for suffix in ('', '_offset'):
        perfect.update({k+suffix: v for k, v in
                        dict(tp=9, fp=0, fn=0, precision=1., recall=1., f1=1.).items()})
    frame = pd.DataFrame([metrics, perfect]).assign(variant='a')
    benchmark = frame.assign(method='a', dataset='coco', cpu_seconds=0., audio_seconds=0.)
    expected = NoteEvaluation.summarize(benchmark, by_dataset=False).iloc[0]
    actual = s.summarize_scores(frame, 'note').iloc[0]
    assert actual.score == expected.f1_offset and actual.f1 == expected.f1
    assert actual.score != frame.f1_offset.mean()
    shown = s.result_table(s.summarize_scores(frame, 'note'), 'note')
    assert shown.columns.is_unique and {'Onset F1', 'Transcription F1'} <= set(shown)

    pitch_rows = []
    for length, estimate in ((10, 440.), (100, 880.)):
        times = np.arange(length) * .01
        pitch_rows.append(PitchBenchmarker.score_frames(times, np.full(length, 440.),
                                                        times, np.full(length, estimate)))
    frame = pd.DataFrame(pitch_rows).assign(variant='a')
    actual = s.summarize_scores(frame, 'pitch').set_index('variant')
    expected = PitchBenchmarker.pool_scores(frame, by='variant')
    pd.testing.assert_frame_equal(actual[expected.columns], expected)
    assert actual.loc['a', 'score'] != frame['Overall Accuracy'].mean()
    print('Pitch and note benchmark scoring, tolerance, and pooling parity passed')


def check_presentation():
    import pandas as pd
    for stage, metric in s.METRIC_LABELS.items():
        table = s.result_table(pd.DataFrame([dict(variant='opaque', score=.9,
                    posthoc_min_volume=.04, posthoc_unv_thresh=.985)]), stage)
        assert 'variant' not in table and 'score' not in table
        assert metric in table and 'Min Volume' in table and 'Unv Thresh' in table

    from experiments.modules.mistake.MistakeBenchmarker import MistakeBenchmarker
    counts = pd.DataFrame([
        dict(variant='a', tp=9, fp=1, fn=0, clean_fp=99),
        dict(variant='a', tp=1, fp=0, fn=9, clean_fp=0),
    ])
    benchmark = counts.assign(input='detected', rate=.25, method='Attune',
                             tolerance=.1, metric='audio_pitch', seconds=0., score_notes=20)
    benchmark['f1'] = 2 * benchmark.tp / (2 * benchmark.tp + benchmark.fp + benchmark.fn)
    expected = MistakeBenchmarker.summarize(benchmark).iloc[0].f1
    actual = s.summarize_scores(counts, 'mistake').iloc[0]
    assert actual.score == expected and actual.clean_fp == 99
    assert actual.score != benchmark.f1.mean()

    class Capture:
        def __init__(self): self.messages = []
        def update(self, message): self.messages.append(message)

    with tempfile.TemporaryDirectory() as directory:
        variants = s.grid(dict(pitch_thresh=(.5, .75)))
        tasks = []
        for track in ('track_a', 'track_b'):
            s.atomic(Path(directory) / track / 'rows.json', [dict(variant=v['variant']) for v in variants])
            tasks.append((dict(track_id=track), directory, variants, {}, 'unused'))
        status = Capture()
        results = s.parallel(s._note, tasks, workers=2, status=status)
        updates = [m for m in status.messages if m.endswith('| Cached')]
        assert len(results) == 2 and len(updates) == 4
        assert updates[-1].startswith('4/4 | ')
        assert all('Min Pitch Distance (semitones)=' in m for m in updates)
        assert all(any(t in m for t in ('track_a', 'track_b')) for m in updates)
    # Notebook output must create exactly one display, then update it in place.
    from unittest.mock import MagicMock
    with patch('IPython.display.display') as display:
        line = s.StatusLine()
        line.notebook = True
        line.update('1/2 | track_a')
        line.update('2/2 | track_b')
        assert display.call_count == 1
        assert display.return_value.update.call_count == 1
    print('Readable tables, per-combination cache progress, and single display checks passed')


if __name__ == '__main__':
    check_pitch_reporting_migration()
    check_benchmark_metrics()
    check_presentation()
    smoke_notes()
    smoke_pitch()
    hashes = s.parallel(s.file_hash, [Path(__file__)] * 8, workers=8)
    assert len(set(hashes)) == 1
    print('Eight-worker spawn check passed')
