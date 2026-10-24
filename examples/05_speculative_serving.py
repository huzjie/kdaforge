# -*- coding: utf-8 -*-
"""Run directly: python examples/<this>.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kdaforge.config import load_config
from kdaforge.serving.speculate import SpeculativeDecoder

cfg = load_config()
dec = SpeculativeDecoder(cfg.serving.draft_len)
print("acceptance rate -> expected decode speedup (draft cost ratio 0.15)")
for skill in (0.5, 0.6, 0.8, 1.0):
    acc, sp = dec.measure(skill)
    print("  skill=%.2f  accept=%.2f  speedup=%.2fx" % (skill, acc, sp))
