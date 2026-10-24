# -*- coding: utf-8 -*-
"""Run directly: python examples/<this>.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kdaforge.config import load_config
from kdaforge.moe.latent_moe import StableLatentMoE

cfg = load_config()
moe = StableLatentMoE(cfg.moe)
for skill in (0.5, 0.75, 1.0):
    ratio, counts = moe.load_imbalance(400, skill=skill)
    print("skill=%.2f  max_load/mean_load = %.2f   top counts = %s" % (
        skill, ratio, sorted(counts, reverse=True)[:5]))
print("skill=1.0 applies the Stable LatentMoE bias correction -> balanced routing.")
