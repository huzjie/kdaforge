# -*- coding: utf-8 -*-
"""Expert router for the sparse MoE.

The router scores each expert for a token and keeps the top-k. The ground-truth
``gold_experts`` is deterministic (md5-seeded), so ``skill = 1.0`` routes exactly
like the oracle and the routing recall is bit-exact.
"""
from ..utils.random import stable_ints, stable_float


def gold_experts(token_id, n_experts, top_k):
    """Deterministic ground-truth experts for a token (no replacement)."""
    return stable_ints("gold:%d" % token_id, top_k, 0, n_experts, distinct=True)


class Router:
    def __init__(self, n_experts, top_k):
        self.n_experts = n_experts
        self.top_k = top_k

    def route(self, token_id, skill):
        """Route one token. score_i = skill * (i in gold) + noise in [0,1).

        skill = 0.5 -> pure noise (random top-k); skill = 1.0 -> always gold.
        """
        boost = (skill - 0.5) * 2.0
        gold = set(gold_experts(token_id, self.n_experts, self.top_k))
        scored = []
        for i in range(self.n_experts):
            s = boost * (1.0 if i in gold else 0.0) +                 stable_float("route:%d:%d" % (token_id, i), 0.0, 1.0)
            scored.append((s, i))
        scored.sort(reverse=True)
        return [i for _, i in scored[:self.top_k]]
