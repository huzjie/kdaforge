# -*- coding: utf-8 -*-
"""Gated DeltaNet: the linear-attention head of Kimi Delta Attention.

Each head keeps a recurrent state ``S`` (d_v x d_k) and applies the delta rule
per token. The head exposes two equivalent implementations -- a sequential
``forward_recurrent`` and a chunked ``forward_chunked`` (associative scan) --
and ``forward`` dispatches by flag. The state size is constant in the sequence
length, which is the entire reason a 1M-token context becomes affordable.
"""
from ..core.linalg import matvec, outer, zeros, sigmoid
from ..core.delta import delta_update, delta_affine, delta_readout
from ..core.scan import scan_chunked
from ..utils.random import stable_float


class GatedDeltaNet:
    def __init__(self, d_model, n_heads, head_dim, beta_init=0.5, gate=True):
        assert d_model == n_heads * head_dim, "d_model must equal n_heads * head_dim"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.beta = [sigmoid(beta_init + 0.1 * stable_float("beta:%d" % h, -1, 1))
                     for h in range(n_heads)]
        self.gate_enabled = gate
        self._wcache = {}

    def _weights(self, kind, h):
        key = "w:%s:%d" % (kind, h)
        if key in self._wcache:
            return self._wcache[key]
        d, hd = self.d_model, self.head_dim
        W = [[stable_float(key + ":%d:%d" % (j, i), -0.1, 0.1)
              for i in range(d)] for j in range(hd)]
        self._wcache[key] = W
        return W

    def _project(self, x, kind, h):
        W = self._weights(kind, h)
        return [matvec(W, xt) for xt in x]

    def _gate(self, h, t):
        return sigmoid(stable_float("gate:%d:%d" % (h, t), -3.0, 3.0))

    def _head_recurrent(self, q, k, v, h):
        beta = self.beta[h]
        S = zeros(self.head_dim, self.head_dim)
        outs = []
        for t, qt in enumerate(q):
            outs.append(delta_readout(S, qt))
            vt = v[t]
            if self.gate_enabled:
                g = self._gate(h, t)
                vt = [vi * g for vi in vt]
            S = delta_update(S, k[t], vt, beta)
        return outs

    def _head_chunked(self, q, k, v, h, chunk):
        beta = self.beta[h]
        d = self.head_dim
        items = []
        for t, kt in enumerate(k):
            vt = v[t]
            if self.gate_enabled:
                g = self._gate(h, t)
                vt = [vi * g for vi in vt]
            items.append(delta_affine(kt, vt, beta))
        from ..core.linalg import identity
        ident = (identity(d), zeros(d, d))

        def combine(e1, e2):
            from ..core.linalg import matmul, add
            A1, b1 = e1
            A2, b2 = e2
            return matmul(A1, A2), add(matmul(b1, A2), b2)

        cum = scan_chunked(items, combine, ident, chunk)
        outs = []
        for t, qt in enumerate(q):
            if t == 0:
                S = zeros(d, d)
            else:
                S = cum[t - 1][1]
            outs.append(delta_readout(S, qt))
        return outs

    def forward(self, x, chunked=True, chunk=16):
        outs_by_head = []
        for h in range(self.n_heads):
            q = self._project(x, "q", h)
            k = self._project(x, "k", h)
            v = self._project(x, "v", h)
            if chunked:
                oh = self._head_chunked(q, k, v, h, chunk)
            else:
                oh = self._head_recurrent(q, k, v, h)
            outs_by_head.append(oh)
        # concatenate heads -> d_model per token
        T = len(x)
        return [[outs_by_head[h][t][j]
                 for h in range(self.n_heads) for j in range(self.head_dim)]
                for t in range(T)]

    @staticmethod
    def state_memory(head_dim):
        """Numbers of floats kept per head regardless of sequence length."""
        return head_dim * head_dim
