# -*- coding: utf-8 -*-
"""KL divergence for distillation (student || teacher)."""
import math


def kl_divergence(p, q, eps=1e-9):
    s = 0.0
    for pi, qi in zip(p, q):
        s += pi * math.log((pi + eps) / (qi + eps))
    return s
