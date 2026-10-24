# -*- coding: utf-8 -*-
"""MOPD -- Multi-teacher Online Policy Distillation (post-training).

Kimi K3.1's post-training runs RL over hundreds of domains without any external
closed-source teacher: the sandbox grades automatically, and the model's own
rollouts act as teachers. This package reproduces that loop with a clean
policy-gradient + distillation implementation over deterministic sandbox tasks.
"""
from .sandbox import SandboxTask, Sandbox
from .reward import advantage, baseline
from .teacher import Teacher, TeacherEnsemble
from .distill import kl_divergence
from .mopd import MOPD, Policy

__all__ = ["SandboxTask", "Sandbox", "advantage", "baseline",
           "Teacher", "TeacherEnsemble", "kl_divergence", "MOPD", "Policy"]
