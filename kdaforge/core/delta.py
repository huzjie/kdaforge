# -*- coding: utf-8 -*-
"""The delta rule (Gated DeltaNet) -- the heart of Kimi Delta Attention.

Convention: the recurrent state ``S`` is a ``d_v x d_k`` matrix (value-dim
rows, key-dim columns). The delta rule removes the old value stored at key
``k`` and writes the new value ``v``, gated by a per-token write intensity
``beta``::

    S_t = S_{t-1} - beta * (S_{t-1} k_t - v_t) k_t^T

This is an *affine* update ``S_t = S_{t-1} A_t + b_t`` with

    A_t = I - beta * k_t k_t^T        (d_k x d_k)
    b_t = beta * v_t k_t^T            (d_v x d_k)

which is why it composes over a sequence as a monoid (see ``core/scan.py``):
two transitions combine as ``(A1 A2, b1 A2 + b2)``. That single fact is what
lets us compute the whole layer with a parallel associative scan in O(L * d_k^2)
instead of O(L * d_k * d_v) sequentially -- and, more importantly, keep only
``d_k * d_v`` numbers of state no matter how long the context is.
"""
from .linalg import matvec, outer, zeros


def delta_update(S, k, v, beta):
    """One sequential delta-rule step. Returns the updated state.

    S: d_v x d_k ; k: [d_k] ; v: [d_v] ; beta in (0, 1).
    """
    d_k = len(k)
    Sk = matvec(S, k)                       # d_v
    err = [Sk[i] - v[i] for i in range(len(v))]
    return [[S[i][j] - beta * err[i] * k[j] for j in range(d_k)]
            for i in range(len(S))]


def delta_affine(k, v, beta):
    """Return the (A, b) pair of the affine update S <- S A + b."""
    d_k = len(k)
    A = [[-beta * k[i] * k[j] for j in range(d_k)] for i in range(d_k)]
    for i in range(d_k):
        A[i][i] += 1.0
    b = [[beta * v[i] * k[j] for j in range(d_k)] for i in range(len(v))]
    return A, b


def gated_delta(k, v, beta, gate):
    """Gated DeltaNet: scale the written value by a per-head gate in (0, 1)."""
    vg = [vi * gate for vi in v]
    return delta_affine(k, vg, beta)


def delta_readout(S, q):
    """Read the current state with query q -> value vector [d_v]."""
    return matvec(S, q)


def delta_state_zeros(d_k, d_v):
    return zeros(d_v, d_k)
