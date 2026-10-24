# -*- coding: utf-8 -*-
"""Mock backend: a trainable, deterministic policy over a token vocabulary.

``skill`` in [0.5, 1.0] controls answer correctness: 0.5 is random, 1.0 is
perfect. It shares the md5 oracle with the benchmarks, so ``skill = 1.0`` yields
exactly the ground truth.
"""
from ..utils.random import stable_bool
from . import register_backend


@register_backend("mock")
class MockBackend:
    def __init__(self, cfg):
        self.cfg = cfg
        self.skill = 0.5

    def generate(self, prompt, max_tokens=16, temperature=1.0):
        out = []
        for i in range(max_tokens):
            gold = stable_bool("gold:%s:%d" % (prompt, i), 0.5)
            boost = (self.skill - 0.5) * 2.0
            correct = stable_bool("ans:%s:%d" % (prompt, i), 0.5 + 0.5 * boost)
            out.append(1 if correct else 0)
        return out

    def train_step(self):
        self.skill += self.cfg.train.learning_rate * (1.0 - self.skill)

    @property
    def name(self):
        return "mock"
