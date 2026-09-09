import inspect
import unittest
import numpy as np
from decoder import decode, labels, associate


class DecoderTests(unittest.TestCase):
    def fixture(self, n=5):
        prob = np.zeros((n, 2, 14)); prob[:, :, 0] = .9
        p = np.tile([[1., 0, 0], [0, 1., 0]], (n, 1, 1))
        return prob, p, np.zeros_like(p), np.arange(n), np.zeros(n, int)

    def test_slot_swap(self):
        a = list(self.fixture()); a[1][1] = a[1][1, ::-1]
        q, m, _ = decode(*a, 'smoothing')
        np.testing.assert_array_equal(m[1], [1, 0]); np.testing.assert_allclose(q, a[1])

    def test_wrap(self):
        a = list(self.fixture(2)); a[0][:, 1] = 0
        rad = np.deg2rad([179, -179]); a[1][:, 0, :2] = np.c_[np.cos(rad), np.sin(rad)]
        q, _, mask = decode(*a, 'smoothing')
        self.assertTrue(mask[1, 0]); self.assertLess(q[1, 0, 0], -.99)

    def test_onset_gap_class_chunk(self):
        for change in ('inactive', 'class', 'chunk', 'gap'):
            a = list(self.fixture())
            if change == 'inactive': a[0][1] = 0
            if change == 'class': a[0][2, :, 0] = 0; a[0][2, :, 1] = .9
            if change == 'chunk': a[4][2:] = 1
            if change == 'gap': a[3][2:] += 1
            q, _, mask = decode(*a, 'fusion')
            self.assertFalse(mask[0].any()); self.assertFalse(mask[2].any())

    def test_zero_nonfinite_shared_mask(self):
        a = list(self.fixture()); a[1][1, 0] = 0; a[2][3, 0] = np.nan
        s, _, sm = decode(*a, 'smoothing'); f, _, fm = decode(*a, 'fusion')
        np.testing.assert_array_equal(sm, fm); self.assertFalse(sm[1, 0]); self.assertFalse(sm[3, 0])

    def test_constant_dt_nonrecursive(self):
        a = list(self.fixture()); a[0][:, 1] = 0
        a[1][:, 0, 1] = np.arange(5)*.1; a[2][:, 0, 1] = 1.
        q, _, _ = decode(*a, 'fusion'); np.testing.assert_allclose(q, a[1], atol=1e-7)
        q, _, _ = decode(*a, 'smoothing')
        self.assertAlmostEqual(q[4, 0, 1], .35)

    def test_no_future_no_gt_sed_raw(self):
        a = list(self.fixture()); saved = a[0].copy()
        for mode in ('raw', 'smoothing', 'fusion'):
            q, _, _ = decode(*a, mode)
            shorter, _, _ = decode(*(x[:3] for x in a), mode)
            np.testing.assert_array_equal(q[:3], shorter)
            np.testing.assert_array_equal(a[0], saved)
            self.assertEqual([(k, [v[:2] for v in vs]) for k, vs in labels(a[0], q).items()],
                             [(k, [v[:2] for v in vs]) for k, vs in labels(a[0], a[1]).items()])
        self.assertNotIn('gt', inspect.signature(decode).parameters)
        np.testing.assert_array_equal(decode(*a, 'raw')[0], a[1])

    def test_ambiguity_and_gate(self):
        p = np.array([[1.,0,0], [1.,0,0]])
        m = associate(p,p,np.zeros(2),np.zeros(2),np.ones(2,bool),np.ones(2,bool))
        self.assertTrue((m == -1).all())
        m = associate(p,-p,np.zeros(2),np.zeros(2),np.ones(2,bool),np.ones(2,bool))
        self.assertTrue((m == -1).all())


if __name__ == '__main__':
    unittest.main()
