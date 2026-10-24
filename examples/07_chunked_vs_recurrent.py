# -*- coding: utf-8 -*-
"""Run directly: python examples/<this>.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kdaforge.attention.delta_net import GatedDeltaNet
from kdaforge.utils.random import stable_vector

net = GatedDeltaNet(d_model=64, n_heads=4, head_dim=16)
x = [stable_vector("x:%d" % t, 64) for t in range(24)]
rec = net.forward(x, chunked=False)
chk = net.forward(x, chunked=True, chunk=8)
maxdiff = max(abs(a - b) for ra, rb in zip(rec, chk) for a, b in zip(ra, rb))
print("max |recurrent - chunked| = %.2e" % maxdiff)
print("chunked associative scan reproduces the sequential delta rule exactly.")
