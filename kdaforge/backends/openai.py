# -*- coding: utf-8 -*-
"""OpenAI-compatible adapter (optional). Falls back to the local CPU engine
when the remote endpoint is not configured."""
from . import register_backend
from .cpu import CpuBackend


@register_backend("openai")
class OpenAIBackend:
    def __init__(self, cfg):
        self.cfg = cfg
        self._fallback = CpuBackend(cfg)

    def generate(self, prompt, max_tokens=16, temperature=1.0):
        # In a real deployment this would POST to /v1/chat/completions.
        return self._fallback.generate(prompt, max_tokens, temperature)

    @property
    def name(self):
        return "openai"
