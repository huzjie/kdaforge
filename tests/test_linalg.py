# -*- coding: utf-8 -*-
import math
import unittest

from kdaforge.core.linalg import (matvec, matmul, transpose, outer, softmax,
                                   sigmoid, layernorm, rmsnorm, identity, zeros)


class TestLinalg(unittest.TestCase):
    def test_matmul(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        B = [[5.0, 6.0], [7.0, 8.0]]
        C = matmul(A, B)
        self.assertEqual(C[0][0], 19.0)
        self.assertEqual(C[0][1], 22.0)
        self.assertEqual(C[1][0], 43.0)
        self.assertEqual(C[1][1], 50.0)

    def test_matvec(self):
        A = [[1.0, 2.0, 3.0]]
        self.assertEqual(matvec(A, [4.0, 5.0, 6.0]), [32.0])

    def test_transpose(self):
        A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        self.assertEqual(transpose(A), [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])

    def test_outer(self):
        self.assertEqual(outer([1.0, 2.0], [3.0, 4.0]),
                         [[3.0, 4.0], [6.0, 8.0]])

    def test_softmax_sums_to_one(self):
        p = softmax([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(p), 1.0, places=6)

    def test_sigmoid(self):
        self.assertAlmostEqual(sigmoid(0.0), 0.5, places=6)

    def test_identity(self):
        self.assertEqual(identity(3), [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_layernorm(self):
        x = layernorm([1.0, 3.0, 2.0, 6.0])
        self.assertAlmostEqual(sum(x), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
