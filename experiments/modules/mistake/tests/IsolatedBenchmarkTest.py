"""Exact-note input preservation and clustered significance contracts."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pretty_midi

from algorithms.Config import Config
from app_logic.NoteData import Note
from experiments.modules.mistake.MistakeBenchmarker import MistakeBenchmarker
from experiments.modules.mistake.datasets.NativeDatasets import digest, save_json


class IsolatedInputsTest(unittest.TestCase):
    def test_window_omissions_become_gaps_without_changing_matches(self):
        score, user = [object(), object()], [object(), object()]
        match = dict(label='match', score_id='0', performance_id='1')
        self.assertEqual(
            MistakeBenchmarker.validated_pairs([match], score, user, complete_unmatched=True),
            [(user[1], score[0]), (None, score[1]), (user[0], None)])
        with self.assertRaisesRegex(ValueError, 'omitted input notes'):
            MistakeBenchmarker.validated_pairs([match], score, user)
        for invalid in ([match, match], [dict(match, score_id='-1')]):
            with self.assertRaisesRegex(ValueError, 'Duplicate/invalid'):
                MistakeBenchmarker.validated_pairs(invalid, score, user, complete_unmatched=True)

    def test_author_and_coco_instrument_aliases_share_quotas(self):
        aliases = {
            'tenor_sax': 'saxophone',
            'Tenor Sax': 'saxophone',
            'tenor_saxophone': 'saxophone',
            'french_horn': 'horn',
            'contrabass': 'double_bass',
            'double bass': 'double_bass',
        }
        for name in MistakeBenchmarker.ISOLATED_INSTRUMENTS:
            aliases[name] = name
        for name, expected in aliases.items():
            with self.subTest(name=name):
                self.assertEqual(MistakeBenchmarker.isolated_instrument(name), expected)
        with self.assertRaisesRegex(ValueError, 'Unknown isolated instrument'):
            MistakeBenchmarker.isolated_instrument('unknown')

    def test_native_unlabelled_raw_note_is_audited_not_scored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = {}
            for key, values in {
                'score': [(60, 0, 1)], 'correct': [(60, 0, 1)],
                'extra': [], 'missed': [],
                'performed': [(60, 0, 1), (74, 0, 16)],
            }.items():
                midi = pretty_midi.PrettyMIDI()
                part = pretty_midi.Instrument(40)
                part.notes = [pretty_midi.Note(80, pitch, start, end)
                              for pitch, start, end in values]
                midi.instruments.append(part)
                path = root / (key + '.mid')
                midi.write(str(path))
                files[key] = str(path)
            task = dict(dataset='E', case=dict(files=files, track_id='piece/violin',
                        group='piece', instrument='violin'), directory=str(root / 'case'),
                        case_id='case', seed=0, rate=-1.0, tolerances=[0.1],
                        methods=['Attune (no refinement)'])
            with patch.object(MistakeBenchmarker, 'isolated_attune_pairs',
                              side_effect=lambda s, u, c: [(u[0], s[0])]) as align:
                rows = MistakeBenchmarker.isolated_case(task)
            self.assertEqual(len(align.call_args.args[1]), 1)
            self.assertEqual(align.call_args.args[1][0].velocity, 80)
            self.assertTrue(all(row['fp'] == 0 for row in rows))
            audit = json.loads((root / 'case/annotation_audit.json').read_text())
            self.assertEqual(audit['raw_performed_notes'], 2)
            self.assertEqual(audit['annotated_performed_notes'], 1)
            self.assertEqual(audit['unlabelled_raw_notes'][0]['pitch'], 74)
            # A later method can fail without discarding the completed method.
            task['methods'].append('Parangonar Automatic')
            with patch.object(MistakeBenchmarker, 'isolated_attune_pairs') as cached_align:
                with patch.object(MistakeBenchmarker, 'external_pairs',
                                  side_effect=RuntimeError('interrupted')):
                    with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                        MistakeBenchmarker.isolated_case(task)
                cached_align.assert_not_called()
            with patch.object(MistakeBenchmarker, 'isolated_attune_pairs') as cached_align:
                with patch.object(MistakeBenchmarker, 'external_pairs',
                                  side_effect=lambda m, s, u, b: [(u[0], s[0])]) as resume:
                    resumed = MistakeBenchmarker.isolated_case(task)
                cached_align.assert_not_called()
                resume.assert_called_once()
            self.assertEqual(len(resumed), 2 * len(rows))
            self.assertEqual(json.loads((root / 'case/rows.json').read_text()), resumed)
            with patch.object(MistakeBenchmarker, 'isolated_attune_pairs') as cached_align:
                with patch.object(MistakeBenchmarker, 'external_pairs') as cached_external:
                    self.assertEqual(MistakeBenchmarker.isolated_case(task), resumed)
                cached_align.assert_not_called()
                cached_external.assert_not_called()

    def test_annotated_input_retains_duplicate_and_simultaneous_events(self):
        event = dict(pitch=60, onset=0, end=1, velocity=80)
        notes = MistakeBenchmarker.isolated_annotated_notes(
            [event, event, dict(event, pitch=64, end=0.5)])
        self.assertEqual([n.midi_num[0] for n in notes], [64, 60, 60])
        self.assertEqual([n.id for n in notes], [0, 1, 2])

    def test_parallel_worker_budget_and_override(self):
        with patch('os.cpu_count', return_value=16), patch('psutil.virtual_memory') as memory:
            memory.return_value.available = 64 * 2**30
            self.assertEqual(MistakeBenchmarker.isolated_worker_count(), 15)
            memory.return_value.available = 5 * 2**30
            self.assertEqual(MistakeBenchmarker.isolated_worker_count(), 2)
            self.assertEqual(MistakeBenchmarker.isolated_worker_count(12), 12)

    def test_simultaneous_notes_and_overlap_are_not_collapsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'notes.mid'
            midi = pretty_midi.PrettyMIDI()
            part = pretty_midi.Instrument(40)
            part.notes = [pretty_midi.Note(80, 60, 0, 1),
                          pretty_midi.Note(80, 64, 0, 0.5),
                          pretty_midi.Note(80, 62, 0.25, 0.75)]
            midi.instruments.append(part)
            midi.write(str(path))
            notes = MistakeBenchmarker.isolated_midi_notes(path)
        self.assertEqual(len(notes), 3)
        self.assertEqual([n.midi_num[0] for n in notes], [64, 60, 62])
        self.assertEqual(notes[0].start_time, notes[1].start_time)
        self.assertGreater(notes[1].end_time, notes[2].start_time)
        with self.assertRaisesRegex(ValueError, 'simultaneous'):
            MistakeBenchmarker.isolated_note_data(notes)

    def test_alignment_keeps_every_event_without_acoustic_stages(self):
        score = [Note(0, 0, 1, [60]), Note(1, 1, 2, [62])]
        performed = [Note(0, 0, 0.5, [64]), Note(1, 0, 1, [60]),
                     Note(2, 1, 2, [62])]
        with patch.object(MistakeBenchmarker, 'synth_midi', side_effect=AssertionError('audio')):
            pairs = MistakeBenchmarker.isolated_attune_pairs(score, performed, Config())
        self.assertEqual(sorted(u.id for u, _ in pairs if u is not None), [0, 1, 2])
        self.assertEqual(sorted(s.id for _, s in pairs if s is not None), [0, 1])
        self.assertEqual([(n.start_time, n.end_time) for n in score], [(0, 1), (1, 2)])


class IsolatedProgressTest(unittest.TestCase):
    def test_global_counter_does_not_reset_between_methods_or_datasets(self):
        counts = {'Attune': 0, 'DualDTW': 0}
        line = MistakeBenchmarker.isolated_progress_line
        self.assertEqual(line('E1 [  8/260]: Attune: song', counts, 1040),
                         'E1 [  8/1040]: Attune: song')
        self.assertEqual(line('E1 [  3/260]: DualDTW: song', counts, 1040),
                         'E1 [ 11/1040]: DualDTW: song')
        self.assertEqual(line('E1 [  7/260]: Attune: song', counts, 1040),
                         'E1 [ 11/1040]: Attune: song')
        counts = {'Attune': 0, 'DualDTW': 0}
        self.assertEqual(line('E2 [  1/260]: Attune: song', counts, 1040, 520),
                         'E2 [521/1040]: Attune: song')

    def test_suite_total_uses_selected_methods_and_enabled_datasets(self):
        requests = [dict(dataset='E', methods=['Attune', 'baseline']),
                    dict(dataset='E2', methods=['Attune', 'baseline'])]
        with patch.object(MistakeBenchmarker, 'launch_isolated', return_value='handle') as launch:
            MistakeBenchmarker.launch_isolated_suite(requests)
        self.assertEqual(launch.call_args_list[0].kwargs['progress_total'], 1040)
        self.assertEqual(launch.call_args_list[1].kwargs['progress_offset'], 520)
        with patch.object(MistakeBenchmarker, 'launch_isolated') as launch:
            MistakeBenchmarker.launch_isolated_suite([dict(dataset='E')])
        self.assertEqual(launch.call_args.kwargs['progress_total'], 1300)


class IsolatedSignificanceTest(unittest.TestCase):
    def saved_outputs(self, root):
        outputs = {}
        methods = ['Attune (no refinement)', 'better', 'tied']
        for dataset in ('E', 'E2'):
            directory = root / dataset
            outputs[dataset] = directory
            cases, rows = [], []
            seeds = [0, 1] if dataset == 'E2' else [0]
            for group in range(5):
                for stem in range(2):
                    track = f'{group}/stem{stem}'
                    cases.append(dict(track_id=track, group=str(group)))
                    for seed in seeds:
                        for method in methods:
                            tp, fp, fn = (10, 0, 0) if method == 'better' else (8, 2, 2)
                            rows.append(dict(dataset=dataset, track_id=track, group=str(group),
                                             seed=seed, case_id=f'{track}/{seed}', method=method,
                                             input='oracle_notes', status='ok', metric='audio_pitch',
                                             tolerance=0.1, rate=0.25 if dataset == 'E2' else -1.0,
                                             tp=tp, fp=fp, fn=fn))
            save_json(directory / 'manifest.json', dict(cases=cases))
            pd.DataFrame(rows).to_csv(directory / 'rows.csv', index=False)
            save_json(directory / 'run.json', dict(
                status='complete', rows_sha256=digest(directory / 'rows.csv'),
                contract=dict(dataset=dataset, methods=methods, injection_seeds=seeds,
                              manifest_sha256=digest(directory / 'manifest.json'))))
        return outputs

    def rewrite_rows(self, directory, frame):
        frame.to_csv(directory / 'rows.csv', index=False)
        run = json.loads((directory / 'run.json').read_text())
        run['rows_sha256'] = digest(directory / 'rows.csv')
        save_json(directory / 'run.json', run)

    def test_exact_swaps_cluster_seeds_and_stems_and_correct_both_datasets(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = self.saved_outputs(Path(tmp))
            result, coverage = MistakeBenchmarker.isolated_significance(outputs)
        self.assertTrue(result.exact.all())
        self.assertTrue(result.permutations.eq(32).all())
        self.assertTrue(result.source_groups.eq(5).all())
        self.assertTrue(result.holm_family_size.eq(4).all())
        self.assertTrue(result.loc[result.method.eq('better'), 'p_value'].eq(2 / 32).all())
        self.assertTrue(result.loc[result.method.eq('better'), 'p_holm'].eq(0.25).all())
        for value in result.loc[result.method.eq('better'), 'difference_pp']:
            self.assertAlmostEqual(value, -20)
        self.assertTrue(result.loc[result.method.eq('tied'), 'p_holm'].eq(1).all())
        self.assertEqual(coverage.set_index('dataset').loc['E2', 'cases'], 20)

    def test_missing_pair_and_inconsistent_truth_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = self.saved_outputs(Path(tmp))
            directory = outputs['E2']
            frame = pd.read_csv(directory / 'rows.csv')
            self.rewrite_rows(directory, frame.iloc[1:])
            with self.assertRaisesRegex(ValueError, 'missing or duplicate'):
                MistakeBenchmarker.isolated_significance(outputs)
            frame.loc[0, 'fn'] += 1
            self.rewrite_rows(directory, frame)
            with self.assertRaisesRegex(ValueError, 'reference event counts'):
                MistakeBenchmarker.isolated_significance(outputs)

    def test_incomplete_run_and_changed_results_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = self.saved_outputs(Path(tmp))
            directory = outputs['E']
            run = json.loads((directory / 'run.json').read_text())
            run['status'] = 'failed'
            save_json(directory / 'run.json', run)
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                MistakeBenchmarker.isolated_significance(outputs)
            run['status'] = 'complete'
            save_json(directory / 'run.json', run)
            with (directory / 'rows.csv').open('a') as stream:
                stream.write('\n')
            with self.assertRaisesRegex(ValueError, 'results changed'):
                MistakeBenchmarker.isolated_significance(outputs)


class EndToEndSignificanceTest(unittest.TestCase):
    def test_exact_cluster_swaps_and_missing_pair_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = {d: Path(tmp) / d for d in ('E', 'E2')}
            for dataset, directory in outputs.items():
                directory.mkdir()
                sources = ['a', 'b', 'c']
                save_json(directory / 'run.json', dict(status='complete', sources=sources,
                          seeds=[0, 1], source_metadata={s: dict(group=s) for s in sources}))
                save_json(directory / 'manifest.json', dict(cases=[dict(case_id=s, piece=s) for s in sources]))
                rows = []
                for source in sources:
                    for seed in ([0] if dataset == 'E' else [0, 1]):
                        for method in ('Attune' if dataset == 'E' else 'Attune (repeat only)', 'PolyTune', 'LadderSym'):
                            attune = method.startswith('Attune')
                            rows.append(dict(case_id=source if dataset == 'E' else f'{source}-{seed}',
                                             source=source, group=source, piece=source, seed=seed,
                                             rate=0.25, input='detected' if attune else 'audio',
                                             method=method, protocol='common_pooled_events',
                                             metric='audio_pitch', tolerance=0.1,
                                             tp=10 if attune else 5, fp=0, fn=0 if attune else 5))
                pd.DataFrame(rows).to_csv(directory / 'rows.csv', index=False)
            result, coverage = MistakeBenchmarker.end_to_end_significance(outputs)
            self.assertTrue(result.exact.all())
            self.assertTrue(result.permutations.eq(8).all())
            self.assertTrue(result.p_value.eq(0.25).all())
            self.assertTrue(result.p_holm.eq(1).all())
            self.assertEqual(coverage.set_index('dataset').loc['E2', 'cases'], 6)
            path = outputs['E2'] / 'rows.csv'
            frame = pd.read_csv(path)
            # Reused model results may carry IDs from an older output folder.
            frame.loc[frame.method.ne('Attune (repeat only)'), 'case_id'] += '-cached'
            frame.to_csv(path, index=False)
            paired, paired_coverage = MistakeBenchmarker.end_to_end_significance(outputs)
            pd.testing.assert_frame_equal(result, paired)
            pd.testing.assert_frame_equal(coverage, paired_coverage)
            frame.iloc[1:].to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'missing or duplicate'):
                MistakeBenchmarker.end_to_end_significance(outputs)
            frame.loc[0, 'fn'] += 1
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'Reference counts differ'):
                MistakeBenchmarker.end_to_end_significance(outputs)


if __name__ == '__main__':
    unittest.main()
