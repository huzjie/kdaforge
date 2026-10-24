# -*- coding: utf-8 -*-
"""Associative (prefix) scan -- the key to parallel linear attention.

A monoid is a set of elements with an associative ``combine(a, b)`` and an
identity ``e``. The inclusive prefix scan of ``[x0, x1, ..., xn]`` is

    [x0, x0+x1, x0+x1+x2, ...]

For linear / delta attention the per-token transition ``S <- S A + b`` is a
monoid element ``(A, b)`` with combine ``(A1 A2, b1 A2 + b2)``. Because the
operation is associative, the scan can be split into chunks, each chunk reduced
and then combined across chunks -- the standard two-level / work-efficient
structure used by FlashLinearAttention-style kernels.

Both ``scan_sequential`` and ``scan_chunked`` return the inclusive prefix list;
a test asserts they are identical for the delta-rule monoid.
"""


def scan_sequential(items, combine, identity):
    out = []
    acc = identity
    for it in items:
        acc = combine(acc, it)
        out.append(acc)
    return out


def scan_chunked(items, combine, identity, chunk=16):
    n = len(items)
    if n == 0:
        return []
    # level 1: reduce each chunk into a single element
    chunk_els = []
    for start in range(0, n, chunk):
        acc = identity
        for it in items[start:start + chunk]:
            acc = combine(acc, it)
        chunk_els.append(acc)
    # level 2: prefix scan over the chunk reductions
    chunk_prefix = scan_sequential(chunk_els, combine, identity)
    # level 3: expand each chunk starting from the cross-chunk prefix
    out = []
    for ci, start in enumerate(range(0, n, chunk)):
        base = identity if ci == 0 else chunk_prefix[ci - 1]
        acc = base
        for it in items[start:start + chunk]:
            acc = combine(acc, it)
            out.append(acc)
    return out
