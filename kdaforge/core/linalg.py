# -*- coding: utf-8 -*-
"""Minimal linear algebra over Python lists of lists.

Matrices are ``[[float]]`` with ``A[i][j]`` meaning row i, column j. Vectors are
``[float]``. Everything is pure Python so the whole project runs anywhere.
"""
import math


def exp(x):
    return math.exp(x)


def zeros(r, c=None):
    if c is None:
        c = r
    return [[0.0] * c for _ in range(r)]


def identity(n):
    m = zeros(n, n)
    for i in range(n):
        m[i][i] = 1.0
    return m


def matvec(A, x):
    """A (r x k) times x (k) -> r."""
    return [sum(a * b for a, b in zip(row, x)) for row in A]


def matvec_t(A, x):
    """A^T (k x r) times x (r) -> k."""
    k = len(A[0])
    return [sum(A[i][j] * x[i] for i in range(len(A))) for j in range(k)]


def transpose(A):
    if not A:
        return []
    return [list(col) for col in zip(*A)]


def matmul(A, B):
    """A (r x k) times B (k x c) -> r x c."""
    Bt = transpose(B)
    return [[sum(a * b for a, b in zip(row, col)) for col in Bt] for row in A]


def outer(x, y):
    """x (r), y (c) -> r x c."""
    return [[xi * yj for yj in y] for xi in x]


def add(A, B):
    if A and isinstance(A[0], (list, tuple)):
        return [[a + b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]
    return [a + b for a, b in zip(A, B)]


def sub(A, B):
    if A and isinstance(A[0], (list, tuple)):
        return [[a - b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]
    return [a - b for a, b in zip(A, B)]


def scale(A, s):
    if A and isinstance(A[0], (list, tuple)):
        return [[a * s for a in row] for row in A]
    return [a * s for a in A]


def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]


def logsumexp(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def layernorm(x, eps=1e-5):
    m = sum(x) / len(x)
    v = sum((xi - m) ** 2 for xi in x) / len(x)
    inv = 1.0 / math.sqrt(v + eps)
    return [(xi - m) * inv for xi in x]


def rmsnorm(x, eps=1e-5):
    ms = sum(xi * xi for xi in x) / len(x)
    inv = 1.0 / math.sqrt(ms + eps)
    return [xi * inv for xi in x]
