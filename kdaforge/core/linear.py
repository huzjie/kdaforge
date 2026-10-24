# -*- coding: utf-8 -*-
"""Standard linear attention (KV accumulation + normalizer) as a baseline.

This is the simpler cousin of the delta rule: the state accumulates
``S += outer(v, k)`` and the normalizer ``z += k``, and the output is
``S q / (z . q + eps)``. It has the same O(d_k * d_v) memory profile and the
same associative-scan chunking, so it serves as a clean sanity check that the
chunked scan reproduces the recurrent form exactly.
"""
from .linalg import matvec, outer, zeros


def _step(S, z, k, v):
    d_k = len(k)
    S2 = [[S[i][j] + v[i] * k[j] for j in range(d_k)] for i in range(len(v))]
    z2 = [z[j] + k[j] for j in range(d_k)]
    return S2, z2


def linear_recurrent(keys, values, queries, eps=1e-6):
    """Sequential linear attention. keys/values/queries are lists of vectors."""
    d_k = len(keys[0])
    d_v = len(values[0])
    S = zeros(d_v, d_k)
    z = [0.0] * d_k
    outs = []
    for t, q in enumerate(queries):
        # causal: query only sees keys up to t (exclusive)
        denom = sum(z[j] * q[j] for j in range(d_k)) + eps
        num = matvec(S, q)
        outs.append([n / denom for n in num])
        S, z = _step(S, z, keys[t], values[t])
    return outs


def linear_chunked(keys, values, queries, chunk=16, eps=1e-6):
    """Chunked linear attention via associative scan -- same result, O(L*d)."""
    from .scan import scan_chunked
    d_k = len(keys[0])
    d_v = len(values[0])
    ident = _identity(d_k, d_v)
    # each token is a monoid element (A, b) with A = I, b = outer(v, k)
    items = []
    for k, v in zip(keys, values):
        items.append((ident[0], outer(v, k)))
    cum = scan_chunked(items, _combine, _identity(d_k, d_v), chunk)
    outs = []
    for t, q in enumerate(queries):
        if t == 0:
            S, z = zeros(d_v, d_k), [0.0] * d_k
        else:
            _, b = cum[t - 1]
            S = b
            # z = sum of keys up to t-1
            z = [sum(keys[t2][j] for t2 in range(t)) for j in range(d_k)]
        denom = sum(z[j] * q[j] for j in range(d_k)) + eps
        num = matvec(S, q)
        outs.append([n / denom for n in num])
    return outs


def _identity(d_k, d_v):
    from .linalg import identity, zeros
    return identity(d_k), zeros(d_v, d_k)


def _combine(e1, e2):
    from .linalg import matmul, add
    A1, b1 = e1
    A2, b2 = e2
    return matmul(A1, A2), add(matmul(b1, A2), b2)
