"""Increasing n preserves completed evaluations and renders only new cases."""
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import pretty_midi
import soundfile as sf
from algorithms.Config import Config
from benchmarks.modules.mistake.MistakeBenchmarker import MistakeBenchmarker as B
from benchmarks.modules.mistake.datasets.NativeDatasets import digest, save_json
from benchmarks.modules.note.NoteBenchmarker import NoteBenchmarker


class IncrementalRunTests(unittest.TestCase):
    def test_e2_append_runs_only_new_sources_and_rejects_changed_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            sources = []
            for i in range(3):
                p = root / f'{i}.mid'
                p.write_bytes(str(i).encode())
                sources.append(str(p))
            calls = []

            def evaluate(midi, seed, rate, output, methods, tolerances, **kwargs):
                calls.append(midi)
                rows = [dict(source=midi, seed=seed, rate=rate, input='audio', method='LadderSym',
                             case_id=Path(midi).stem, tolerance=t, metric=m, tp=1, fp=0, fn=0,
                             f1=1.0, score_notes=2, seconds=1)
                        for t in tolerances for m in ('audio_pitch', 'audio_missed', 'audio_extra')]
                kwargs['on_result'](rows)

            options = dict(seeds=[0], rates=[.25], methods=['LadderSym'], tolerances=[.1],
                           parallel=False, input_kinds=('audio',), append=True)
            with patch.object(B, 'preflight', return_value={}), patch.object(B, 'evaluate_case', side_effect=evaluate), patch.object(B, 'summarize', side_effect=lambda f: f):
                B.run_comparison(sources[:2], root / 'run', **options)
                original = pd.read_csv(root / 'run/rows.csv')
                calls.clear()
                B.run_comparison(sources, root / 'run', **options)
                self.assertEqual(calls, sources[2:])
                rows = pd.read_csv(root / 'run/rows.csv')
                pd.testing.assert_frame_equal(original, rows.iloc[:len(original)].reset_index(drop=True))
                calls.clear()
                B.run_comparison(sources, root / 'run', **options)
                self.assertEqual(calls, [])
                Path(sources[0]).write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'missing or incompatible'):
                    B.run_comparison(sources, root / 'run', **options)
                self.assertEqual(calls, [])

    def test_request_rejects_shrink_and_settings_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources = [str(Path(tmp).resolve() / f'{i}.mid') for i in range(3)]
            prior = dict(sources=sources[:2], source_metadata={}, seeds=[0], rates=[.25], methods=['LadderSym'], tolerances=[.1])
            save_json(Path(tmp) / 'run.json', prior)
            request = dict(prior, sources=sources, output=tmp, append=True, expected_defaults=asdict(Config()), stage='audio', workers=1, polytune_device='cpu', neural_workers=1, polytune_last=True)
            with patch.object(B, 'run_comparison') as runner:
                B.run_request(request)
                self.assertTrue(runner.call_args.kwargs['append'])
                for changes in (dict(sources=sources[:1]), dict(seeds=[1])):
                    with self.assertRaisesRegex(ValueError, 'Cannot append'):
                        B.run_request(dict(request, **changes))

    def test_e1_renders_and_evaluates_only_appended_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            midi = pretty_midi.PrettyMIDI()
            instrument = pretty_midi.Instrument(56)
            instrument.notes.append(pretty_midi.Note(80, 72, .1, .3))
            midi.instruments.append(instrument)
            midi.write(str(root / 'score.mid'))
            pretty_midi.PrettyMIDI().write(str(root / 'empty.mid'))
            font = root / 'font.sf3'
            font.write_bytes(b'font')
            wav = root / 'render.wav'
            sf.write(wav, np.zeros(4410), 44100)
            cases = []
            for i in range(3):
                files = {k: str(root / ('empty.mid' if k in ('missed', 'extra') else 'score.mid')) for k in ('score', 'performed', 'correct', 'extra', 'missed')}
                cases.append(dict(track_id=f'{i}/0_trumpet', group=str(i), instrument='trumpet', files=files, hashes={k:digest(v) for k,v in files.items()}))
            cohort = root / 'cohort.json'
            save_json(cohort, dict(selection=dict(dataset='E'), provenance=dict(repo='fixture', revision='fixed', split_sha256='fixed'), cases=cases))
            base, output = root / 'base', root / 'output'
            save_json(base / 'run.json', dict(contract=dict(manifest=dict(dataset='coco'), device='cpu', midi_range=[21,108], tolerances=[.1], range_policy='performed_notes')))
            models = {m:SimpleNamespace(preflight=lambda: {}) for m in ('PolyTune','LadderSym')}
            batches = []

            def execute(tasks, **kwargs):
                batches.append([(t['case']['case_id'],t['method']) for t in tasks])
                for t in tasks:
                    rows = B.native_event_metrics([], [], t['tolerances'])
                    for row in rows:
                        row.update(dataset='coco',method=t['method'],case_id=t['case']['case_id'],piece=t['case']['piece'])
                    payload = dict(rows=rows, prediction={}, truth=[])
                    save_json(Path(t['directory'])/'result.json',payload)
                    kwargs['on_result'](payload)
                return {}

            def synth(source, *, out_dir, force):
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                return wav

            with patch.object(NoteBenchmarker,'SOUNDFONT_PATH',font), patch.object(NoteBenchmarker,'synth_midi',side_effect=synth) as render, patch('benchmarks.modules.mistake.MistakeBenchmarker.tree', return_value=[dict(path='0_trumpet.mid')]), patch('benchmarks.modules.mistake.MistakeBenchmarker.download',return_value=root/'score.mid'), patch.object(B,'native_models_for',return_value=models), patch.object(B,'native_code_identity',return_value={'fixture':'same'}), patch.object(B,'native_run_tasks',side_effect=execute):
                B.soundfont_prepare(base=base, output=output, cohort=cohort, tracks=2, distinct_pieces=True)
                B.soundfont_run(base=base,output=output)
                original=(output/'cases/0/0_trumpet/attune/result.json').read_bytes()
                render.reset_mock()
                B.soundfont_prepare(base=base, output=output, cohort=cohort, tracks=3, distinct_pieces=True)
                self.assertEqual(render.call_count,2)  # score + performance for one new track
                self.assertEqual(json.loads((output/'run.json').read_text())['status'],'preparing')
                B.soundfont_run(base=base,output=output)
                self.assertEqual(batches[-1], [('2/0_trumpet',m) for m in ('Attune','PolyTune','LadderSym')])
                self.assertEqual(original,(output/'cases/0/0_trumpet/attune/result.json').read_bytes())
                B.soundfont_run(base=base,output=output)
                self.assertEqual(batches[-1],[])
                with self.assertRaisesRegex(ValueError,'n decreased'):
                    B.soundfont_prepare(base=base,output=output,cohort=cohort,tracks=1,distinct_pieces=True)
                old=json.loads((output/'run.json').read_text())['contract']
                changed=deepcopy(old)
                changed['models']['LadderSym']={'checkpoint':'changed'}
                self.assertFalse(B.soundfont_append_contract(old,changed))


if __name__ == '__main__':unittest.main()
