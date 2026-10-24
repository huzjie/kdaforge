# -*- coding: utf-8 -*-
"""HybridAttention: a stack mixing linear (KDA) and full (Gated MLA) layers.

Kimi K3.1 uses 69 KDA layers + 24 Gated MLA layers out of 93. The hybrid stack
here reproduces that *ratio* with a tiny layer count: it fills the first
``kda_layers`` slots with Gated DeltaNet and the remaining ``mla_layers`` with
Gated MLA. The point is that most of the depth never materializes a KV cache --
only the full-attention layers do.
"""
from .kda import KDALayer


class HybridAttention:
    def __init__(self, cfg):
        self.cfg = cfg
        self.layers = []
        for i in range(cfg.kda_layers):
            self.layers.append(KDALayer(
                cfg.d_model, cfg.n_heads, cfg.head_dim, KDALayer.LINEAR,
                beta_init=cfg.beta_init, chunked=True, chunk=cfg.chunk_size))
        for i in range(cfg.mla_layers):
            self.layers.append(KDALayer(
                cfg.d_model, cfg.n_heads, cfg.head_dim, KDALayer.FULL))

    @property
    def n_layers(self):
        return len(self.layers)

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = layer.forward(h)
        return h

    def kv_memory(self, seq_len):
        """KV-cache memory if every layer were full attention vs actual hybrid."""
        d = self.cfg.d_model
        full = self.n_layers * seq_len * d * 2
        # linear layers: constant state; full layers: KV cache
        actual = self.cfg.mla_layers * seq_len * d * 2 +             self.cfg.kda_layers * self.cfg.head_dim * self.cfg.head_dim
        return full, actual
