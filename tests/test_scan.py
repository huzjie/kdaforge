# -*- coding: utf-8 -*-
import unittest

from kdaforge.core.delta import delta_affine
from kdaforge.core.scan import scan_sequential, scan_chunked
from kdaforge.core.linalg import matmul, add, identity, zeros
from kdaforge.utils.random import stable_vector


def _combine(e1, e2):
    A1, b1 = e1
    A2, b2 = e2
    return matmul(A1, A2), add(matmul(b1, A2), b2)


class TestScan(unittest.TestCase):
    def test_chunked_equals_sequential(self):
        d = 4
        items = []
        for t in range(20):
            k = stable_vector("k:%d" % t, d)
            v = stable_vector("v:%d" % t, d)
            items.append(delta_affine(k, v, 0.6))
        ident = (identity(d), zeros(d, d))
        seq = scan_sequential(items, _combine, ident)
        chk = scan_chunked(items, _combine, ident, chunk=5)
        self.assertEqual(len(seq), len(chk))
        for (A1, b1), (A2, b2) in zip(seq, chk):
            for i in range(d):
                for j in range(d):
                    self.assertAlmostEqual(A1[i][j], A2[i][j], places=6)
                    self.assertAlmostEqual(b1[i][j], b2[i][j], places=6)


if __name__ == "__main__":
    unittest.main()
