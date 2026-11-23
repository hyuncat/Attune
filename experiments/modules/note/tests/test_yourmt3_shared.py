"""Shared-process lifecycle and checkpoint resume; no model downloads/inference."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import soundfile as sf
from experiments.modules.note.NoteBenchmarker import NoteEvaluation, NotebookConfig
from experiments.modules.note.NoteCache import NoteCache
from experiments.modules.note.experiments import YourMT3Shared as shared
from experiments.modules.note.tests import test_note_result_cache as fixtures


class YourMT3SharedTest(unittest.TestCase):
    def test_two_processes_two_requests_and_cleanup_on_close(self):
        processes = []
        class Process:
            def __init__(self, command, **kwargs):
                self.records = json.loads(Path(command[-1]).read_text())
                self.returncode = None
                self.terminated = False
                processes.append(self)
                for record in self.records:
                    Path(record['output']).write_text(json.dumps({'intervals': [], 'frequencies_hz': []}))
            def poll(self): return self.returncode
            def terminate(self): self.terminated = True; self.returncode = -15
            def wait(self, timeout=None): return self.returncode
        with tempfile.TemporaryDirectory() as output, patch.object(shared.subprocess,'Popen',Process):
            jobs=[({'job_id': str(i),'audio':'unused.wav'},None,output) for i in range(4)]
            iterator=shared.predictions(jobs)
            next(iterator)
            iterator.close()
        self.assertEqual(len(processes),2)
        self.assertEqual([len(p.records) for p in processes],[2,2])
        self.assertTrue(all(p.terminated for p in processes))
        self.assertEqual(shared.RECORDINGS_PER_MODEL,2)

    def test_dead_process_returns_error_for_every_unfinished_job(self):
        class Process:
            returncode=1
            def __init__(self,*args,**kwargs): pass
            def poll(self): return 1
            def wait(self,timeout=None): return 1
        with tempfile.TemporaryDirectory() as output, patch.object(shared.subprocess,'Popen',Process):
            jobs=[({'job_id':str(i),'audio':'unused.wav'},None,output) for i in range(3)]
            results=list(shared.predictions(jobs))
        self.assertEqual(len(results),3)
        self.assertTrue(all('error' in result for _,result in results))

    def test_checkpoint_each_completion_and_resume(self):
        tracks, provenance=fixtures.NoteResultCacheTest().fixture()
        for path in ('NoteDetectorBase','NoteCache','competitors/YourMT3','experiments/YourMT3Direct'):
            provenance['sources'][f'experiments/modules/note/{path}.py']='same'
        provenance['yourmt3_runtime']={'checkpoint':'same'}
        calls=[]
        interrupted=True
        def predictions(jobs):
            for payload in jobs:
                calls.append(payload[0]['track_id'])
                yield payload,dict(intervals=[[0,0.5]],frequencies_hz=[440],cpu_seconds=2.,
                    wall_seconds=3.,compute_clock='thread+amortized_model_startup',
                    model_pid=123,startup_cpu_share=0.5)
                if interrupted: raise KeyboardInterrupt
        with tempfile.TemporaryDirectory() as folder:
            audio=Path(folder)/'audio.wav';sf.write(audio,np.zeros(16000),16000)
            for track in tracks:
                track['audio']=str(audio)
            provenance['inputs'][str(audio)]='same'
            bench=NoteEvaluation(NotebookConfig(methods=('yourmt3',)))
            bench.runs_root=Path(folder)/'runs'
            with patch.object(bench,'select_tracks',return_value=tracks), patch.object(
                    bench,'provenance',side_effect=lambda t:copy.deepcopy(provenance)), patch.object(
                    bench,'summarize',return_value=pd.DataFrame()), patch.object(
                    bench,'paired_comparisons',return_value=pd.DataFrame()), patch.object(
                    NoteEvaluation,'reference_notes',return_value=(np.array([[0.,0.5]]),np.array([440.]))), patch.object(
                    shared,'predictions',side_effect=predictions), patch.object(
                    NoteEvaluation,'_predict',side_effect=AssertionError('Must not run another model')):
                with self.assertRaises(KeyboardInterrupt): bench.run_preliminary('coco')
                self.assertEqual(len(NoteCache.checkpoint_index(bench.runs_root)),1)
                interrupted=False
                rows=bench.run_preliminary('coco')
                self.assertEqual(calls,['0','1'])
                self.assertEqual(set(rows.status),{'ok'})
                self.assertTrue((rows.cpu_seconds==2.).all())
                bench.run_preliminary('coco')
                self.assertEqual(calls,['0','1'])
