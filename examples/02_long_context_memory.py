# -*- coding: utf-8 -*-
"""Run directly: python examples/<this>.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kdaforge.config import load_config
from kdaforge.model import KDAForgeModel

cfg = load_config()
model = KDAForgeModel(cfg)
print("context length | full-attention KV | KDA hybrid KV | ratio")
print("-" * 60)
for L in (1024, 65536, 1000000):
    full, actual = model.attn.kv_memory(L)
    print("%-14d | %-18.3e | %-13.3e | %.1fx" % (L, full, actual, full / actual))
print()
print("Linear layers keep O(d_k*d_v) state; only the %d MLA layers need a KV cache."
      % cfg.attention.mla_layers)
