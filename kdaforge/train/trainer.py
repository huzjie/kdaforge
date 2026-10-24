# -*- coding: utf-8 -*-
"""SkillTrainer: the monotonic mock-skill trainer used for sanity checks.

The update is ``skill += lr * (1 - skill)`` -- a geometric climb that is always
positive, so a low starting skill is never driven to zero by a random walk.
"""
from ..backends import get_backend


class SkillTrainer:
    def __init__(self, cfg, backend="mock"):
        self.backend = get_backend(backend, cfg)
        self.lr = cfg.train.learning_rate

    def train(self, steps=None):
        steps = steps or 40
        hist = [self.backend.skill]
        for _ in range(steps):
            self.backend.train_step()
            hist.append(self.backend.skill)
        return hist
