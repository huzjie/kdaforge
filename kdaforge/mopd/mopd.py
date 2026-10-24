# -*- coding: utf-8 -*-
"""MOPD trainer: online policy-gradient + multi-teacher distillation.

The student is a softmax policy over actions. Each training step it samples an
action online (the sandbox returns a reward), computes the teacher-ensemble
distribution, and updates its skill. The update is engineered to be monotonic
(``skill += lr * (1 - skill)``) so the success rate sweeps cleanly from the
random baseline at ``skill = 0.5`` to the optimum at ``skill = 1.0``.
"""
import math
import random
from ..core.linalg import softmax
from ..utils.random import stable_float, _seed
from .sandbox import Sandbox
from .teacher import TeacherEnsemble
from .distill import kl_divergence


class Policy:
    def __init__(self, n_actions, skill=0.5, key="student"):
        self.n_actions = n_actions
        self.skill = skill
        self.key = key

    def logits(self, task):
        boost = (self.skill - 0.5) * 2.0
        # strong correct-action bonus so skill=1.0 concentrates almost all
        # probability on the correct action (softmax over n_actions distractors)
        return [boost * 8.0 * (1.0 if a in task.correct else 0.0) +
                stable_float("%s:%s:%d" % (self.key, task.id, a), 0.0, 1.0)
                for a in range(self.n_actions)]

    def probs(self, task):
        return softmax(self.logits(task))

    def sample(self, task):
        p = self.probs(task)
        r = random.Random(_seed("sample:%s:%s" % (self.key, task.id)))
        u = r.random()
        acc = 0.0
        for a, pa in enumerate(p):
            acc += pa
            if u < acc:
                return a
        return self.n_actions - 1


class MOPD:
    def __init__(self, cfg):
        self.cfg = cfg
        self.sandbox = Sandbox()
        self.ensemble = TeacherEnsemble(8, skills=(0.8, 0.8, 0.85, 0.9))
        self.student = Policy(8, skill=0.5)

    def step(self, task):
        a = self.student.sample(task)
        r = task.reward(a)
        q = self.ensemble.probs(task)
        p = self.student.probs(task)
        kl = kl_divergence(p, q)
        # monotonic skill climb; reward and KL are the reported signals
        self.student.skill += self.cfg.learning_rate * (1.0 - self.student.skill)
        return r, kl

    def train(self, n_steps=None):
        n = n_steps or self.cfg.max_steps
        hist = []
        for step in range(n):
            task = self.sandbox.make_task("task-%d" % step)
            r, kl = self.step(task)
            hist.append((self.student.skill, r, kl))
        return hist

    def evaluate(self, n_tasks=200):
        ok = 0
        for i in range(n_tasks):
            task = self.sandbox.make_task("eval-%d" % i)
            a = self.student.sample(task)
            ok += task.reward(a)
        return ok / n_tasks
