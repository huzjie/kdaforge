# -*- coding: utf-8 -*-
"""Expert FFN (deterministic single-hidden-layer) and the expert bank.

One md5 seed per expert generates a small weight matrix; the same expert id
always produces the same transform, keeping the whole bank deterministic.
"""
import math
import random
from ..utils.random import _seed


def _rng(key):
    return random.Random(_seed(key))


class Expert:
    def __init__(self, expert_id, d_model, expert_dim):
        self.expert_id = expert_id
        self.d_model = d_model
        self.expert_dim = expert_dim
        r = _rng("expert:%d" % expert_id)
        self.w1 = [[r.uniform(-0.1, 0.1) for _ in range(d_model)]
                   for _ in range(expert_dim)]
        self.b1 = [r.uniform(-0.1, 0.1) for _ in range(expert_dim)]
        self.w2 = [[r.uniform(-0.1, 0.1) for _ in range(expert_dim)]
                   for _ in range(d_model)]
        self.b2 = [r.uniform(-0.1, 0.1) for _ in range(d_model)]

    def forward(self, x):
        h = [math.tanh(sum(self.w1[j][i] * x[i] for i in range(self.d_model)) + self.b1[j])
             for j in range(self.expert_dim)]
        return [sum(self.w2[i][j] * h[j] for j in range(self.expert_dim)) + self.b2[i]
                for i in range(self.d_model)]


class ExpertBank:
    def __init__(self, n_experts, d_model, expert_dim):
        self.experts = [Expert(i, d_model, expert_dim) for i in range(n_experts)]
