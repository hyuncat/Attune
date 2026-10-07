"""Small arithmetic/annotation checks; never runs audio or pitch detection."""
import unittest
from types import SimpleNamespace
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import pandas as pd
from benchmarks.modules.vibrato.VibratoBenchmarker import YangMetrics as ym
from benchmarks.modules.vibrato.VibratoBenchmarker import VibratoBenchmarker
from benchmarks.modules.vibrato.VibratoDetectorBase import VibratoExample, VibratoEstimate
from benchmarks.modules.vibrato.datasets.YangFullDataset import YangFullDataset

class SoftYangTests(unittest.TestCase):
    def test_fractional_credit_misses_and_extras(self):
        counts = ym.soft_counts([1,1,0],[1,0,1],[1,1,0],[.9,0,1])
        self.assertEqual(counts,(.9,2.,2.))
        self.assertEqual(ym.soft_prf(*counts),(.45,.45,.45))
        self.assertAlmostEqual(ym.soft_prf(50,50,100)[2],2/3)

    def test_unknown_is_neither_missed_nor_false_positive(self):
        counts = ym.soft_counts([1,1,0],[1,1,1],[1,np.nan,0],[1,3,2])
        self.assertEqual(counts,(1.,2.,1.))

    def test_boundary_matching_and_parameter_coverage(self):
        t=np.arange(300)/100
        detected=(t>=.5)&(t<1.5)
        refs=[dict(start=.5,end=1.5,rate_hz=5,extent_semitones=.5),dict(start=2,end=2.5,rate_hz=5,extent_semitones=.5)]
        out=ym.evaluate(t,detected,np.full(300,4.5),np.full(300,.5),refs)
        self.assertEqual(out['yang_note_tp'],1)
        self.assertEqual(out['yang_note_fn'],1)
        self.assertEqual(out['yang_parameter_truth'],2)
        self.assertEqual(out['yang_parameter_matched'],1)
        self.assertAlmostEqual(out['yang_rate_sum'],.9)

    def test_short_runs_and_split_merge(self):
        t=np.arange(300)/100
        refs=[dict(start=.2,end=1.8)]
        out=ym.evaluate(t,((t>=.2)&(t<.8))|((t>=1)&(t<1.8))|((t>=2)&(t<2.2)),np.ones(300),np.ones(300),refs)
        self.assertEqual(out['yang_predicted_notes'],2)
        self.assertEqual(out['yang_split'],1)
        self.assertEqual(out['yang_parameter_truth'],0)
        out=ym.evaluate(t,(t>=.2)&(t<1.8),np.ones(300),np.ones(300),[dict(start=.2,end=.8),dict(start=1,end=1.8)])
        self.assertEqual(out['yang_merge'],1)
        self.assertLessEqual(out['yang_note_tp'],1)

    def test_scorer_pooling_and_detection_only(self):
        t=np.arange(200)/100
        truth=(t>=.5)&(t<1.5)
        refs=[dict(start=.5,end=1.5,rate_hz=5,extent_semitones=.5)]
        example=VibratoExample('fixture','fixture','test',t,np.full(200,60.),np.full(200,60.),np.where(truth,5.,0.),np.where(truth,100.,0.),truth,metadata={'yang_references':refs,'parameter_annotations':True})
        estimate=VibratoEstimate(np.where(truth,4.5,0.),np.where(truth,100.,0.),truth)
        bench=VibratoBenchmarker();method=SimpleNamespace(name='fixture',description='fixture',scores_center=False)
        row=bench._score(method,example,estimate,1.)
        self.assertAlmostEqual(row['rate_soft_f1'],.9)
        summary=bench.summarize(pd.DataFrame([row]))
        self.assertAlmostEqual(summary.iloc[0].rate_soft_f1,.9)
        self.assertEqual(summary.iloc[0].yang_matched_notes,1)
        self.assertIn('Extent Soft F1',bench.display_summary(summary))
        example.metadata['parameter_annotations']=False
        example.metadata['yang_references']=[dict(start=.5,end=1.5)]
        # Use NaN for unavailable parameter ground truth.
        unknown=VibratoExample('unknown','fixture','test',t,np.full(200,60.),np.full(200,60.),np.where(truth,np.nan,0.),np.where(truth,np.nan,0.),truth,metadata=example.metadata)
        other=bench._score(method,unknown,estimate,1.)
        combined=bench.summarize(pd.DataFrame([row,other])).iloc[0]
        self.assertAlmostEqual(combined.rate_soft_f1,.9)
        self.assertEqual(combined.yang_recordings,2)
        self.assertEqual(combined.yang_parameter_recordings,1)

    def test_coler_decimal_commas_and_negative_recordings(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'area.txt'
            p.write_text('0,0\t0,0\tvib_off\n0,5\t0,5\tvib_on\n1,5\t1,5\tvib_off\n')
            self.assertEqual(YangFullDataset.areas(SimpleNamespace(area_path=p)),[(.5,1.5)])
            p.write_text('0,0\t0,0\tvib_off\n')
            self.assertEqual(YangFullDataset.areas(SimpleNamespace(area_path=p)),[])

if __name__=='__main__':unittest.main()
