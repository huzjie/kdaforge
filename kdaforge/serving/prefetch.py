# -*- coding: utf-8 -*-
"""KV-cache prefetch for long-context decode.

Predicts the next few tokens and prefetches their KV cache so the compute does
not stall on memory. In the reference this is modeled as a deterministic hit
rate over a lookahead window -- the prefetch hit rate rises with the lookahead,
and the structural gain is a reduced stall fraction.
"""
from ..utils.random import stable_bool


class KVPrefetcher:
    def __init__(self, n_lookahead=16):
        self.n_lookahead = n_lookahead

    def hit_rate(self, n_queries=200, accuracy=0.8):
        hits = 0
        for q in range(n_queries):
            for k in range(self.n_lookahead):
                if stable_bool("pf:%d:%d" % (q, k), accuracy):
                    hits += 1
                    break
        return hits / n_queries
