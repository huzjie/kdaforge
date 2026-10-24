# -*- coding: utf-8 -*-
"""Run directly: python examples/<this>.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kdaforge.config import load_config
from kdaforge.model import KDAForgeModel

cfg = load_config()
model = KDAForgeModel(cfg)
x = model.embed([0, 1, 2, 3])
out = model.forward(x)
print("declared scale: %.1fT / %.0fB active / %d experts" % (
    cfg.declared.total_params_b / 1000, cfg.declared.active_params_b,
    cfg.declared.n_experts))
print("hybrid stack: %d layers (%d KDA + %d MLA)" % (
    model.attn.n_layers, cfg.attention.kda_layers, cfg.attention.mla_layers))
print("forward output dim:", len(out[0]))
