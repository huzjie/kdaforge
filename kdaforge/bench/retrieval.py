# -*- coding: utf-8 -*-
"""Associative recall: does the delta-rule state retrieve the right value?

Writes N orthonormal key/value pairs into a delta-rule state, then queries it.
``skill`` controls how well the query aligns with the target key: skill = 1.0
queries with the exact key (perfect recall), skill = 0.5 queries with a random
direction (chance-level recall). Sweeps monotonically.
"""
from ..core.delta import delta_update, delta_readout
from ..core.linalg import zeros
from ..utils.random import stable_float, stable_vector
from . import register_benchmark


def _cos(a, b):
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _recall_at(N, d, skill, trials):
    hits = 0
    for tr in range(trials):
        S = zeros(d, d)
        vs = [stable_vector("v:%d:%d" % (tr, i), d) for i in range(N)]
        for i in range(N):
            e = [0.0] * d
            e[i % d] = 1.0
            S = delta_update(S, e, vs[i], beta=1.0)
        target = stable_float("tgt:%d" % tr, 0, N)
        target = int(target) % N
        e = [0.0] * d
        e[target % d] = 1.0
        noise = stable_vector("n:%d" % tr, d)
        boost = (skill - 0.5) * 2.0
        q = [boost * e[j] + (1.0 - boost) * noise[j] for j in range(d)]
        out = delta_readout(S, q)
        best = max(range(N), key=lambda i: _cos(out, vs[i]))
        if best == target:
            hits += 1
    return hits / trials


@register_benchmark("retrieval")
def run_retrieval(cfg):
    N = 8
    d = 16
    lo = _recall_at(N, d, 0.5, 200)
    hi = _recall_at(N, d, 1.0, 200)
    return {"recall_skill0.5": lo, "recall_skill1.0": hi}
