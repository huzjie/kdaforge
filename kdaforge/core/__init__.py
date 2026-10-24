# -*- coding: utf-8 -*-
"""Core numerical kernels: linear algebra, the delta rule, linear attention and
the associative scan. All operate on plain Python lists of floats -- no numpy,
no torch, no external dependency."""
from .linalg import (zeros, matvec, matvec_t, matmul, transpose, outer,
                     add, sub, scale, softmax, sigmoid, layernorm, rmsnorm,
                     identity, exp, logsumexp)
from .delta import delta_update, delta_affine, gated_delta, delta_readout
from .linear import linear_recurrent, linear_chunked
from .scan import scan_sequential, scan_chunked

__all__ = [
    "zeros", "matvec", "matvec_t", "matmul", "transpose", "outer",
    "add", "sub", "scale", "softmax", "sigmoid", "layernorm", "rmsnorm",
    "identity", "exp", "logsumexp",
    "delta_update", "delta_affine", "gated_delta", "delta_readout",
    "linear_recurrent", "linear_chunked", "scan_sequential", "scan_chunked",
]
