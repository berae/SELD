import unittest
import numpy as np
from analyze import matched_rows,polar,summarize,sed_counts


class AnalysisTests(unittest.TestCase):
    def test_same_class_gt_identity_matching(self):
        gt=[dict(frame=0,class_id=1,source_id=5,xyz=polar(0,0)),dict(frame=0,class_id=1,source_id=9,xyz=polar(90,0))]
        err,slot=matched_rows(gt,{0:[[1,0,90,0],[1,1,0,0]]})
        np.testing.assert_allclose(err,[0,0]);np.testing.assert_array_equal(slot,[1,0])
        err,slot=matched_rows(gt,{0:[[1,0,0,0]]})
        s=summarize(err);self.assertEqual(s['gt_denominator'],2);self.assertEqual(s['matched'],1);self.assertEqual(s['recall20'],.5)

    def test_no_angle_gate_in_diagnostic_matching(self):
        gt=[dict(frame=0,class_id=0,xyz=polar(0,0))]
        err,_=matched_rows(gt,{0:[[0,0,180,0]]})
        self.assertEqual(summarize(err)['matched'],1);self.assertEqual(summarize(err)['within20'],0)

    def test_sed_union_full_duration(self):
        gt=[dict(frame=0,class_id=0),dict(frame=0,class_id=0),dict(frame=599,class_id=1)]
        counts,p=sed_counts(gt,{0:[[0,0,0,0],[0,1,90,0]]})
        self.assertEqual(p.shape,(600,14));self.assertEqual(counts[0].sum(),1);self.assertEqual(counts[2].sum(),1)

    def test_common_partitions(self):
        x=np.array([1.,2.,np.nan,np.nan]);y=np.array([3.,np.nan,4.,np.nan])
        a=np.isfinite(x);b=np.isfinite(y)
        masks=[a&b,a&~b,~a&b,~a&~b]
        self.assertEqual([int(m.sum()) for m in masks],[1,1,1,1]);np.testing.assert_array_equal(np.sum(masks,axis=0),np.ones(4))


if __name__=='__main__':unittest.main()
