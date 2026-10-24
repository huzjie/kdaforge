#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick end-to-end demo of the whole stack in one pass."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kdaforge.config import load_config
from kdaforge.model import KDAForgeModel
from kdaforge.moe.latent_moe import StableLatentMoE
from kdaforge.mopd.mopd import MOPD
from kdaforge.serving.speculate import SpeculativeDecoder


def main():
    cfg = load_config()
    print("== forward ==")
    model = KDAForgeModel(cfg)
    out = model.forward(model.embed([0, 1, 2, 3]))
    print("out dim:", len(out[0]))
    print("== memory ==")
    full, actual = model.attn.kv_memory(cfg.declared.context_len)
    print("KV @1M ctx: full=%.3e hybrid=%.3e ratio=%.2fx" % (full, actual, full / actual))
    print("== load balance ==")
    moe = StableLatentMoE(cfg.moe)
    lo, _ = moe.load_imbalance(400, skill=0.5)
    hi, _ = moe.load_imbalance(400, skill=1.0)
    print("imbalance skill0.5=%.2f skill1.0=%.2f" % (lo, hi))
    print("== mopd ==")
    m = MOPD(cfg.mopd)
    m.train(20)
    print("success rate: %.3f" % m.evaluate(100))
    print("== speculative ==")
    dec = SpeculativeDecoder(cfg.serving.draft_len)
    _, sp = dec.measure(1.0)
    print("speedup @skill1.0: %.2fx" % sp)


if __name__ == "__main__":
    main()
