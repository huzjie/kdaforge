# -*- coding: utf-8 -*-
"""KV memory benchmark: full attention vs the KDA hybrid stack.

Structural (skill-independent): it measures how many floats of KV state a
1M-token context would need if every layer were full attention, versus the
hybrid (linear layers keep a constant d_k*d_v state). The ratio is what makes
the 1M context affordable.
"""
from ..model import KDAForgeModel
from . import register_benchmark


@register_benchmark("memory")
def run_memory(cfg):
    model = KDAForgeModel(cfg)
    L = cfg.declared.context_len
    full, actual = model.attn.kv_memory(L)
    d = cfg.declared
    hd = cfg.attention.head_dim
    full_d = d.n_layers * L * cfg.attention.d_model * 2
    actual_d = d.mla_layers * L * cfg.attention.d_model * 2 + d.kda_layers * hd * hd
    return {"ratio_ref": full / actual,
            "ratio_declared_69_24": full_d / actual_d}
