# -*- coding: utf-8 -*-
"""KDAForgeModel: the full hybrid transformer (KDA + Gated MLA + Stable LatentMoE).

This is the reference model that ties the pieces together. It instantiates tiny
structures (declared scale lives in ``Config.declared``) so a forward pass runs
in milliseconds of pure Python.
"""
from .attention.hybrid import HybridAttention
from .moe.latent_moe import StableLatentMoE
from .core.linalg import add, rmsnorm, matvec, zeros
from .utils.random import stable_vector


class KDAForgeModel:
    def __init__(self, cfg):
        self.cfg = cfg
        a = cfg.attention
        self.attn = HybridAttention(a)
        self.moe = StableLatentMoE(cfg.moe)

    def forward(self, x):
        """x: list of token vectors [seq, d_model]. Returns [seq, d_model]."""
        h = self.attn.forward(x)
        out = []
        for t, ht in enumerate(h):
            ff = self.moe.forward([ht])[0]
            out.append(rmsnorm(add(ht, ff)))
        return out

    def n_params(self):
        """Declared total parameter count (informational)."""
        return self.cfg.declared.total_params_b

    def embed(self, token_ids):
        """Deterministic pseudo-embedding for a sequence of token ids."""
        d = self.cfg.attention.d_model
        return [stable_vector("emb:%d" % tid, d) for tid in token_ids]
