# -*- coding: utf-8 -*-
"""Run directly: python examples/<this>.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kdaforge.config import load_config
import os

here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for name in ("default", "long_context", "balanced", "highspeed"):
    cfg = load_config(os.path.join(here, "kdaforge", "configs", name + ".yaml"))
    print("%-13s kda_layers=%d mla_layers=%d aux=%s bias_corr=%s draft=%d" % (
        name, cfg.attention.kda_layers, cfg.attention.mla_layers,
        cfg.moe.aux_loss_weight, cfg.moe.bias_correction, cfg.serving.draft_len))
