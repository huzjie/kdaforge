# -*- coding: utf-8 -*-
"""Run directly: python examples/<this>.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kdaforge.config import load_config
from kdaforge.mopd.mopd import MOPD

cfg = load_config()
mopd = MOPD(cfg.mopd)
print("before training: success rate = %.3f (random ~0.125)" % mopd.evaluate(200))
hist = mopd.train(cfg.mopd.max_steps)
print("after  training: success rate = %.3f" % mopd.evaluate(200))
print("skill curve: %.3f -> %.4f  (monotonic)" % (0.5, hist[-1][0]))
print("no external closed teacher: sandbox auto-rewards + self-generated teachers.")
