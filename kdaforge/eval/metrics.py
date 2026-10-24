# -*- coding: utf-8 -*-
"""Evaluation metrics used by benchmarks."""


def cosine(a, b):
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def accuracy(preds, targets):
    if not preds:
        return 0.0
    return sum(1 for p, t in zip(preds, targets) if p == t) / len(preds)


def recall_at_k(ranked, target, k):
    return 1.0 if target in ranked[:k] else 0.0
