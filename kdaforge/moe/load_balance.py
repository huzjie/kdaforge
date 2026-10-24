# -*- coding: utf-8 -*-
"""Load balancing: Switch-style auxiliary loss + per-expert bias correction.

The "hot expert" problem: a learned gate naturally specializes, so a handful of
experts absorb most tokens and become the serving bottleneck. Two mechanisms
fix it -- an auxiliary loss that penalizes concentrated routing, and a
per-expert additive bias that is nudged down when an expert is over-used.
"""
from ..utils.metrics import mean


def aux_loss(counts, probs, n_experts):
    """Switch Transformer load-balance loss L = n_experts * sum_i f_i * P_i.

    f_i = fraction of tokens routed to expert i; P_i = mean gate probability.
    """
    total = sum(counts) or 1
    f = [c / total for c in counts]
    return n_experts * sum(f[i] * probs[i] for i in range(n_experts))


def imbalance_ratio(counts):
    """max load / mean load -- 1.0 means perfectly balanced."""
    m = mean(counts)
    if m == 0:
        return 0.0
    return max(counts) / m


class BiasCorrector:
    """Stable LatentMoE bias correction: keep a per-expert additive bias, and
    nudge it down when the expert is over-used relative to the mean."""

    def __init__(self, n_experts, lr=0.1):
        self.n_experts = n_experts
        self.lr = lr
        self.bias = [0.0] * n_experts

    def update(self, counts):
        m = mean(counts)
        for i in range(self.n_experts):
            overload = (counts[i] - m) / (m if m > 0 else 1)
            self.bias[i] -= self.lr * overload
