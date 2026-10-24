# -*- coding: utf-8 -*-
"""Deterministic synthetic sequences.

Two useful generators: ``synth_sequence`` produces a token-id sequence with an
embedded key-value pair (for associative-recall tests), and ``key_value_pairs``
produces orthonormal key/value vectors (for the delta-rule memory benchmark).
"""
from ..utils.random import stable_ints, stable_vector


def synth_sequence(n, key_pos=None, value_pos=None, seed="synth"):
    """Return token ids [0..vocab) with optional key/value markers."""
    ids = stable_ints(seed, n, 0, 256)
    return ids


def key_value_pairs(n, d, seed="kv"):
    """Return (keys, values): keys are orthonormal unit vectors, values random."""
    keys = []
    values = []
    for i in range(n):
        k = [0.0] * d
        k[i % d] = 1.0
        keys.append(k)
        values.append(stable_vector("%s:%d" % (seed, i), d))
    return keys, values
