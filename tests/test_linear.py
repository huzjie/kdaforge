# -*- coding: utf-8 -*-
import unittest

from kdaforge.core.linear import linear_recurrent, linear_chunked
from kdaforge.utils.random import stable_vector


class TestLinear(unittest.TestCase):
    def test_chunked_equals_recurrent(self):
        T, d = 30, 8
        keys = [stable_vector("k:%d" % t, d) for t in range(T)]
        values = [stable_vector("v:%d" % t, d) for t in range(T)]
        queries = [stable_vector("q:%d" % t, d) for t in range(T)]
        rec = linear_recurrent(keys, values, queries)
        chk = linear_chunked(keys, values, queries, chunk=6)
        for a, b in zip(rec, chk):
            for x, y in zip(a, b):
                self.assertAlmostEqual(x, y, places=5)


if __name__ == "__main__":
    unittest.main()
