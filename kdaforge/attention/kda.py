# -*- coding: utf-8 -*-
"""KDALayer: one transformer block that is either a Gated DeltaNet (linear) or
a Gated MLA (full attention) layer, followed by an RMSNorm. The hybrid stack
instantiates 69 of the former and 24 of the latter (configurable here)."""
from ..core.linalg import rmsnorm, add
from .delta_net import GatedDeltaNet
from .gated_mla import GatedMLA


class KDALayer:
    LINEAR = "linear"
    FULL = "full"

    def __init__(self, d_model, n_heads, head_dim, kind=LINEAR, beta_init=0.5,
                 chunked=True, chunk=16):
        self.d_model = d_model
        self.kind = kind
        if kind == self.LINEAR:
            self.attn = GatedDeltaNet(d_model, n_heads, head_dim, beta_init)
        else:
            self.attn = GatedMLA(d_model, n_heads, head_dim)
        self.chunked = chunked
        self.chunk = chunk

    def forward(self, x):
        if self.kind == self.LINEAR:
            a = self.attn.forward(x, chunked=self.chunked, chunk=self.chunk)
        else:
            a = self.attn.forward(x)
        return [rmsnorm(add(xt, at)) for xt, at in zip(x, a)]
