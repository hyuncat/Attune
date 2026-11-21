"""Tiny synthetic method-selection checks; no corpus sweep or pitch inference."""
import sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / 'app.py').is_file())))
import tempfile
from unittest.mock import patch
import numpy as np
import pretty_midi
import soundfile as sf
from app_logic.user.ds.PitchData import Pitch
from benchmarks.modules.sweeps import AppendixSweeps as base
from benchmarks.notebooks.sweeps.scripts import note_method_sweeps as runner


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(40)
        for i, pitch in enumerate((60, 62, 65)):
            instrument.notes.append(pretty_midi.Note(90, pitch, .1 + i*.5, .55 + i*.5))
        midi.instruments.append(instrument)
        midi.write(str(root / 'score.mid'))
        sf.write(root / 'audio.wav', np.zeros(28000), 16000)
        items = [dict(track_id=f'synthetic_{i}', instrument='violin', paths=dict(
            midi=str(root / 'score.mid'), audio=str(root / 'audio.wav'))) for i in range(2)]
        recording = base._recording(items[0], runner.BASELINE)
        pitches = []
        for t in np.arange(0, 1.7, recording.config.h1 / recording.config.sr):
            active = next((n.pitch for n in instrument.notes if n.start <= t < n.end), None)
            pitches.append(Pitch(float(t), .2 if active else 0., 0. if active else 1., 0.,
                                 recording.config, candidates=[] if active is None else [(active, 1.)]))
        base.atomic(root / 'pitches.pkl', pitches, True)
        common = runner.variants_for()
        assert len(common) == 6
        tasks = [(item, str(root / 'cache'), common, {}, str(root / 'pitches.pkl')) for item in items]
        from app_logic.user.ds.Recording import Recording
        production_refine = Recording.align_score_and_refine
        with patch.object(Recording, 'align_score_and_refine', autospec=True,
                          side_effect=production_refine) as refinement:
            batches = [runner._note(task) for task in tasks]
            assert refinement.call_count == 12  # All six methods, both tracks.
        assert all(len(batch) == 6 for batch in batches)
        for batch in batches:
            for row in batch:
                assert row['evaluation_stage'] == runner.EVALUATION_STAGE
                assert row['compute_seconds'] == row['segmentation_seconds'] + row['refinement_seconds']
                assert row['artifact'].endswith('.refined.pkl')
                assert Path(row['initial_artifact']).is_file()
        with patch.object(runner, 'detect_notes', side_effect=AssertionError('Repeated baseline inference')), \
             patch.object(Recording, 'align_score_and_refine', side_effect=AssertionError('Repeated refinement')):
            assert runner._note(tasks[0]) == batches[0]

        # A controlled split makes stage ordering observable, independent of
        # whether the production DP accepts a split for this simple fixture.
        from app_logic.NoteData import Note
        def split_first(rec):
            first = rec.note_data.read_note(i=0)
            end = first.end_time
            middle = (first.start_time + end) / 2
            first.end_time = middle
            rec.note_data.write_note(Note(1000, middle, end, list(first.midi_num)))
        with patch.object(Recording, 'align_score_and_refine', autospec=True, side_effect=split_first):
            split_row = runner._note((items[0], str(root / 'controlled'), common[:1], {},
                                      str(root / 'pitches.pkl')))[0]
        initial = base.blob(split_row['initial_artifact'])
        refined = base.blob(split_row['artifact'])
        assert len(refined.times) == len(initial.times) + 1
        assert split_row['estimated_notes'] == len(refined.times)
        from benchmarks.modules.note.NoteBenchmarker import NoteEvaluation
        reference = NoteEvaluation.reference_notes(dict(dataset='coco', reference=str(root / 'score.mid')))
        expected = base._note_metrics(reference, refined, recording.config)
        assert split_row['f1_offset'] == expected['f1_offset']

        # Rebuild a missing final artifact from the saved initial notes.
        resumed = batches[0][0]
        Path(resumed['artifact']).unlink()
        (root / 'cache' / items[0]['track_id'] / (common[0]['variant'] + '.json')).unlink()
        with patch.object(runner, 'detect_notes', side_effect=AssertionError('Initial cache ignored')), \
             patch.object(Recording, 'align_score_and_refine', autospec=True, side_effect=production_refine) as refinement:
            runner._note((items[0], str(root / 'cache'), common[:1], {}, str(root / 'pitches.pkl')))
            assert refinement.call_count == 1
        axes = {key: (value,) for key, value in runner.BASELINE.items()}
        axes['pitch_thresh'] = (.5, .75)
        axes['min_note_length_cap'] = (.1, .2)
        try:
            runner.winning_method_variants(root, axes=axes)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Parameter tuning must require completed method selection')
        comparison = root / 'methods' / base.digest(common)[:16]
        comparison.mkdir(parents=True)
        runner._finish(comparison, batches, common, items)
        variants, selection = runner.winning_method_variants(root, axes=axes)
        assert len(variants) == 4
        assert {v['method'] for v in variants} == {selection['method']}
        assert selection['method'] == base.read(comparison / 'best.json')['method']
        assert any(v['variant'] in {c['variant'] for c in common} for v in variants)
        tasks = [(item, str(root / 'cache'), variants, {}, str(root / 'pitches.pkl')) for item in items]
        batches = base.parallel(runner._note, tasks, workers=2)
        report = root / 'report'
        report.mkdir()
        summary, leaders = runner._finish(report, batches, variants, items)
        assert len(summary) == 4 and len(leaders) == 1
        assert leaders.Method.nunique() == 1
        assert 'variant' not in summary and 'score' not in summary
        assert {'Onset F1', 'Transcription F1'} <= set(summary)
        winners = base.read(report / 'best_by_method.json')
        assert set(winners) == {selection['method']}
        for winner in winners.values():
            assert set(winner['artifacts']) == {item['track_id'] for item in items}
            assert all(Path(a['path']).is_file() for a in winner['artifacts'].values())
        recording.config.min_note_length = 1.
        recording.config.min_note_length_factor = .5
        recording.config.min_note_length_cap = .1
        detector = runner.CappedRuptures(recording)
        assert detector._pelt_min_size_from_score() == int(np.ceil(.1 * recording.config.sr / recording.config.h1))
        recording.config.min_note_length_cap = .2
        assert detector._pelt_min_size_from_score() == int(np.ceil(.2 * recording.config.sr / recording.config.h1))
        conflicting = Pitch(0., .2, 0., 0., recording.config, candidates=[(48., 1.)], value=60.)
        assert detector._pelt_segment_pitch([conflicting]) == [60.]
        print('All six methods refine before scoring; final-note caches, resume, winner-only tuning and progress passed')


if __name__ == '__main__':
    main()
