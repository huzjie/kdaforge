# -*- coding: utf-8 -*-
"""Speculative decoding -- the Highspeed (2-3x) serving trick.

A small draft model proposes ``K`` tokens; the big target model verifies them in
parallel and accepts the longest prefix that agrees. The closed-form expected
acceptance is Leviathan et al. 2023::

    expected = (1 - a^(K+1)) / (1 - a)
    speedup  = expected / (1 + K * c)

where ``a`` is the per-token acceptance rate and ``c`` the draft/target cost
ratio. ``skill`` drives ``a`` so the benchmark sweeps monotonically.
"""
from ..utils.random import stable_bool


def expected_acceptance(a, K):
    if abs(a - 1.0) < 1e-9:
        return float(K + 1)
    return (1.0 - a ** (K + 1)) / (1.0 - a)


def speedup(a, K, c):
    return expected_acceptance(a, K) / (1.0 + K * c)


class SpeculativeDecoder:
    def __init__(self, draft_len=6, cost_ratio=0.15):
        self.K = draft_len
        self.c = cost_ratio

    def acceptance_rate(self, skill):
        """Acceptance rate a: skill=0.5 -> 0.5, skill=1.0 -> 1.0."""
        return (skill - 0.5) * 2.0 * 0.5 + 0.5

    def simulate_round(self, skill, round_id):
        a = self.acceptance_rate(skill)
        accepted = 0
        for i in range(self.K):
            if stable_bool("spec:%d:%d" % (round_id, i), a):
                accepted += 1
            else:
                break
        return accepted

    def measure(self, skill, n_rounds=200):
        a = self.acceptance_rate(skill)
        accepted = sum(self.simulate_round(skill, r) for r in range(n_rounds))
        measured = accepted / n_rounds
        return measured, speedup(a, self.K, self.c)
