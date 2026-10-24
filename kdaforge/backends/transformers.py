# -*- coding: utf-8 -*-
"""HuggingFace transformers adapter (optional). Falls back to the local CPU
engine when the package is not installed."""
from . import register_backend
from .cpu import CpuBackend


@register_backend("transformers")
class TransformersBackend:
    def __init__(self, cfg):
        self.cfg = cfg
        self._fallback = CpuBackend(cfg)

    def generate(self, prompt, max_tokens=16, temperature=1.0):
        return self._fallback.generate(prompt, max_tokens, temperature)

    @property
    def name(self):
        return "transformers"
