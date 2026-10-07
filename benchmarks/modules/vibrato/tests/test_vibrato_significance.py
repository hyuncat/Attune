import unittest
import numpy as np
import pandas as pd
from benchmarks.modules.vibrato.VibratoBenchmarker import VibratoBenchmarker


class PairedTests(unittest.TestCase):
    @staticmethod
    def rows(equal=False):
        rows=[]
        for i in range(6):
            for method in ('attune','peer'):
                r=dict(method=method,case_id=str(i),cluster=str(i),stratum='all')
                for name in VibratoBenchmarker.COUNT_NAMES:
                    for k,v in zip(('tp','fp','fn'),(10 if method=='attune' or equal else 5, 0, 0 if method=='attune' or equal else 5)):
                        r[f'{name}_{k}']=v
                rows.append(r)
        return pd.DataFrame(rows)

    def test_exact_swaps_and_identity(self):
        result=VibratoBenchmarker.paired_tests(self.rows(),('peer',),metrics=('aggregate_soft_f1',),draws=999)
        self.assertEqual(result.iloc[0].permutations,64)
        self.assertEqual(result.iloc[0].p_value,2/64)
        identical=VibratoBenchmarker.paired_tests(self.rows(True),('peer',),metrics=('aggregate_soft_f1',),draws=999)
        self.assertEqual(identical.iloc[0].p_value,1)
        self.assertEqual(identical.iloc[0].difference_pp,0)

    def test_pool_counts_not_case_scores(self):
        counts=np.zeros((2,5,3)); counts[0,:,:]=[1,0,0]; counts[1,:,:]=[0,0,9]
        self.assertAlmostEqual(float(VibratoBenchmarker.score(counts.sum(axis=0),'aggregate_f1')),2/11)
        self.assertAlmostEqual(float(VibratoBenchmarker.score(counts,'aggregate_f1').mean()),.5)

    def test_replication_keeps_clusters_and_results(self):
        rows=self.rows()
        copies=rows.copy(); copies['case_id']+='copy'
        a=VibratoBenchmarker.paired_tests(rows,('peer',),draws=999)
        b=VibratoBenchmarker.paired_tests(pd.concat([rows,copies]),('peer',),draws=999)
        np.testing.assert_allclose(a.difference_pp,b.difference_pp)
        np.testing.assert_allclose(a.p_holm,b.p_holm)
        self.assertTrue(b.clusters.eq(6).all())
        self.assertTrue((a.p_holm>=a.p_value).all())

    def test_single_cluster_rejected(self):
        rows=self.rows(); rows['cluster']='one'
        with self.assertRaises(ValueError): VibratoBenchmarker.paired_tests(rows,('peer',))

if __name__=='__main__': unittest.main()
