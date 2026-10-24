# -*- coding: utf-8 -*-
"""Show the associative scan produces the same cumulative states as sequential."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kdaforge.core.delta import delta_affine
from kdaforge.core.scan import scan_sequential, scan_chunked
from kdaforge.core.linalg import matmul, add, identity, zeros
from kdaforge.utils.random import stable_vector


def combine(e1, e2):
    A1, b1 = e1
    A2, b2 = e2
    return matmul(A1, A2), add(matmul(b1, A2), b2)


d = 4
items = [delta_affine(stable_vector("k:%d" % t, d), stable_vector("v:%d" % t, d), 0.5)
         for t in range(16)]
seq = scan_sequential(items, combine, (identity(d), zeros(d, d)))
chk = scan_chunked(items, combine, (identity(d), zeros(d, d)), chunk=5)
maxdiff = max(abs(a - b) for (A1, b1), (A2, b2) in zip(seq, chk)
              for a, b in zip(sum(A1, []) + sum(b1, []), sum(A2, []) + sum(b2, [])))
print("max |sequential - chunked| = %.2e" % maxdiff)
