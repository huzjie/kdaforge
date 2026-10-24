# -*- coding: utf-8 -*-
"""Associative recall with the delta rule: write N key/value pairs, query one."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kdaforge.core.delta import delta_update, delta_readout
from kdaforge.core.linalg import zeros
from kdaforge.utils.random import stable_vector


def cosine(a, b):
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na * nb else 0.0


N, d = 8, 16
S = zeros(d, d)
vs = [stable_vector("v:%d" % i, d) for i in range(N)]
for i in range(N):
    e = [0.0] * d
    e[i % d] = 1.0
    S = delta_update(S, e, vs[i], beta=1.0)

target = 3
e = [0.0] * d
e[target] = 1.0
out = delta_readout(S, e)
best = max(range(N), key=lambda i: cosine(out, vs[i]))
print("queried key %d -> retrieved value %d (cosine %.4f)" % (
    target, best, cosine(out, vs[target])))
print("recall correct:", best == target)
