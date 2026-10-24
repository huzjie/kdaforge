# -*- coding: utf-8 -*-
"""Gated Multi-head Latent Attention (the full-attention layers of KDA).

Only 24 of 93 K3.1 layers use full attention. To keep their cost down the keys
and values are first compressed into a low-rank *latent* space (down-projection
then up-projection), softmax attention runs in that space, and a SwiGLU-style
output gate modulates the result. This layer is quadratic in the sequence, so
it is only used where the model needs fine-grained token-to-token interaction.
"""
import math
from ..core.linalg import matvec, softmax, sigmoid, layernorm
from ..utils.random import stable_float


class GatedMLA:
    def __init__(self, d_model, n_heads, head_dim, latent_dim=None, gate=True):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.latent_dim = latent_dim or (head_dim // 2)
        self.gate_enabled = gate
        self._wcache = {}

    def _weights(self, name, rows, cols):
        if name in self._wcache:
            return self._wcache[name]
        W = [[stable_float("%s:%d:%d" % (name, j, i), -0.1, 0.1)
              for i in range(cols)] for j in range(rows)]
        self._wcache[name] = W
        return W

    def forward(self, x):
        d, hd, ld = self.d_model, self.head_dim, self.latent_dim
        Wq = self._weights("q", d, d)
        Wk_down = self._weights("kd", ld, d)
        Wv_down = self._weights("vd", ld, d)
        Wk_up = self._weights("ku", hd, ld)
        Wv_up = self._weights("vu", hd, ld)
        Wo = self._weights("o", d, d)
        T = len(x)
        q = [matvec(Wq, xt) for xt in x]
        outs = []
        for t in range(T):
            hvec = []
            for h in range(self.n_heads):
                # latent KV (compressed), then up-projected per head
                kt = matvec(Wk_up, matvec(Wk_down, x[t]))
                vt = matvec(Wv_up, matvec(Wv_down, x[t]))
                qt = q[t][h * hd:(h + 1) * hd]
                scores = []
                for s in range(t + 1):
                    ks = matvec(Wk_up, matvec(Wk_down, x[s]))
                    scores.append(sum(qt[j] * ks[j] for j in range(hd)) / math.sqrt(hd))
                attn = softmax(scores)
                ctx = [0.0] * hd
                for s in range(t + 1):
                    vs = matvec(Wv_up, matvec(Wv_down, x[s]))
                    w = attn[s]
                    for j in range(hd):
                        ctx[j] += w * vs[j]
                hvec.extend(ctx)
            out = matvec(Wo, hvec)
            if self.gate_enabled:
                g = sigmoid(stable_float("mla_gate:%d" % t, -3.0, 3.0))
                out = [o * g for o in out]
            outs.append(out)
        return outs
