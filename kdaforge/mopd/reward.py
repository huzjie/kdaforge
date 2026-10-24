# -*- coding: utf-8 -*-
"""Reward helpers: advantage (reward - baseline) and a running mean baseline."""


def baseline(rewards):
    if not rewards:
        return 0.0
    return sum(rewards) / len(rewards)


def advantage(reward, base):
    return reward - base
