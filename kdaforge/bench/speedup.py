# -*- coding: utf-8 -*-
"""Speculative-decoding speedup: acceptance rate -> decode speedup.

skill = 0.5 -> 50% acceptance -> modest speedup; skill = 1.0 -> 100% acceptance
-> maximum speedup (K+1)/(1+K*c). Sweeps monotonically.
"""
from ..serving.speculate import SpeculativeDecoder
from . import register_benchmark


@register_benchmark("speedup")
def run_speedup(cfg):
    dec = SpeculativeDecoder(cfg.serving.draft_len)
    _, lo_sp = dec.measure(0.5)
    _, hi_sp = dec.measure(1.0)
    return {"speedup_skill0.5": lo_sp, "speedup_skill1.0": hi_sp,
            "acceptance_skill0.5": dec.acceptance_rate(0.5),
            "acceptance_skill1.0": dec.acceptance_rate(1.0)}
