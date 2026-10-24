# -*- coding: utf-8 -*-
"""MOPD post-training entry point: trains the student policy online against the
sandbox and the self-generated teacher ensemble, then reports the success-rate
curve. Demonstrates the "base model fixed, post-training does the work" story."""
from ..mopd.mopd import MOPD


def train_mopd(cfg):
    mopd = MOPD(cfg.mopd)
    hist = mopd.train(cfg.mopd.max_steps)
    acc = mopd.evaluate(200)
    s0 = 0.5
    s1 = hist[-1][0]
    print("[mopd] skill %.3f -> %.4f over %d steps" % (s0, s1, len(hist)))
    print("[mopd] final success rate = %.4f (random baseline ~0.125)" % acc)
    print("[mopd] final KL(student||teacher) = %.5f" % hist[-1][2])
    return acc
