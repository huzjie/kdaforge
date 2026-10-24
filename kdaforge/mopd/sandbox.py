# -*- coding: utf-8 -*-
"""Deterministic sandbox that auto-grades actions -- the reward source.

Real post-training RL needs a cheap, automatic "grader" that says whether an
action was right. Here each task has a deterministic set of correct actions
(md5-seeded), so the reward is reproducible and the environment needs no
human label or closed model.
"""
from ..utils.random import stable_ints


class SandboxTask:
    def __init__(self, task_id, n_actions=8, n_correct=1):
        self.id = task_id
        self.n_actions = n_actions
        self.correct = set(stable_ints("corr:%s" % task_id, n_correct, 0, n_actions,
                                       distinct=True))

    def reward(self, action):
        return 1.0 if action in self.correct else 0.0


class Sandbox:
    def __init__(self, n_actions=8, n_correct=1):
        self.n_actions = n_actions
        self.n_correct = n_correct

    def make_task(self, task_id):
        return SandboxTask(task_id, self.n_actions, self.n_correct)
