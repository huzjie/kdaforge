# -*- coding: utf-8 -*-
"""MoE load-balance benchmark: how imbalanced is expert routing?

``imbalance = max_load / mean_load``. skill = 0.5 leaves the natural hot-expert
bias uncorrected (high imbalance); skill = 1.0 applies the Stable LatentMoE
bias correction (imbalance -> 1.0). Sweeps monotonically.
"""
from ..moe.latent_moe import StableLatentMoE
from . import register_benchmark


@register_benchmark("load_balance")
def run_load_balance(cfg):
    moe = StableLatentMoE(cfg.moe)
    n = 400
    lo, _ = moe.load_imbalance(n, skill=0.5)
    hi, _ = moe.load_imbalance(n, skill=1.0)
    return {"imbalance_skill0.5": lo, "imbalance_skill1.0": hi}
