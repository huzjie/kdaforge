# -*- coding: utf-8 -*-
"""Routing recall: does the router select the gold experts?

skill = 0.5 -> random top-k (recall ~ top_k / n_experts); skill = 1.0 -> always
the ground-truth experts (recall = 1.0). Sweeps monotonically.
"""
from ..moe.router import Router, gold_experts
from . import register_benchmark


@register_benchmark("routing")
def run_routing(cfg):
    m = cfg.moe
    router = Router(m.n_experts, m.top_k)
    n = 300

    def recall(skill):
        hits = 0
        for t in range(n):
            gold = set(gold_experts(t, m.n_experts, m.top_k))
            picked = router.route(t, skill)
            hits += len(gold & set(picked))
        return hits / (n * m.top_k)

    return {"recall_skill0.5": recall(0.5), "recall_skill1.0": recall(1.0)}
