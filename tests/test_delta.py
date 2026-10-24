# -*- coding: utf-8 -*-
import unittest

from kdaforge.core.delta import delta_update, delta_affine
from kdaforge.core.linalg import matmul, add, zeros
from kdaforge.utils.random import stable_vector


class TestDelta(unittest.TestCase):
    def test_single_write(self):
        k = [1.0, 0.0]
        v = [0.5, -0.5]
        S = delta_update(zeros(2, 2), k, v, beta=1.0)
        # S = outer(v, k)
        self.assertAlmostEqual(S[0][0], 0.5)
        self.assertAlmostEqual(S[1][0], -0.5)
        self.assertAlmostEqual(S[0][1], 0.0)
        self.assertAlmostEqual(S[1][1], 0.0)

    def test_affine_matches_sequential(self):
        k = stable_vector("k", 4)
        v = stable_vector("v", 4)
        beta = 0.7
        S0 = [stable_vector("s:%d" % i, 4) for i in range(4)]
        S_seq = delta_update(S0, k, v, beta)
        A, b = delta_affine(k, v, beta)
        S_aff = add(matmul(S0, A), b)
        for i in range(4):
            for j in range(4):
                self.assertAlmostEqual(S_seq[i][j], S_aff[i][j], places=9)

    def test_remove_then_add(self):
        # writing the same key twice with beta=1 replaces the old value
        k = [1.0, 0.0, 0.0]
        S = delta_update(zeros(3, 3), k, [1.0, 0.0, 0.0], 1.0)
        S = delta_update(S, k, [0.0, 0.0, 9.0], 1.0)
        self.assertAlmostEqual(S[0][0], 0.0)
        self.assertAlmostEqual(S[2][0], 9.0)


if __name__ == "__main__":
    unittest.main()
