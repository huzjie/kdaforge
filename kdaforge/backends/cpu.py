# -*- coding: utf-8 -*-
"""CPU backend: runs the real KDAForgeModel forward in pure Python."""
from ..model import KDAForgeModel
from . import register_backend


@register_backend("cpu")
class CpuBackend:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = KDAForgeModel(cfg)

    def forward(self, token_ids):
        x = self.model.embed(token_ids)
        return self.model.forward(x)

    def generate(self, prompt, max_tokens=16, temperature=1.0):
        token_ids = [ord(c) % 100 for c in prompt[:32]] or [0]
        out = self.forward(token_ids)
        return [round(o[0], 4) for o in out[:max_tokens]]

    @property
    def name(self):
        return "cpu"
