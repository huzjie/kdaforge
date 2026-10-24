# -*- coding: utf-8 -*-
"""Self-generated teachers -- no external closed model required.

A teacher is just a policy with a (higher) skill and a distinct noise key, i.e.
a *different rollout of the model itself*. An ensemble of a few such teachers
defines the distillation target. This is the "multi-teacher, online" part of
MOPD: the student learns from its own better checkpoints and the sandbox
reward, never from a black-box API teacher.
"""
from ..core.linalg import softmax
from ..utils.random import stable_float


class Teacher:
    def __init__(self, n_actions, skill, key):
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


class TeacherEnsemble:
    def __init__(self, n_actions, skills=(0.8, 0.8, 0.85, 0.9)):
        self.teachers = [Teacher(n_actions, s, "teacher%d" % i)
                         for i, s in enumerate(skills)]

    def probs(self, task):
        n = len(self.teachers)
        ps = [t.probs(task) for t in self.teachers]
        return [sum(p[a] for p in ps) / n for a in range(len(ps[0]))]
