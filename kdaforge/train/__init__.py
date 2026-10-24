# -*- coding: utf-8 -*-
"""Training entry points: generic skill trainer and MOPD post-training."""
from .trainer import SkillTrainer
from .mopd_trainer import train_mopd

__all__ = ["SkillTrainer", "train_mopd"]
