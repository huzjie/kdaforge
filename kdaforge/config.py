# -*- coding: utf-8 -*-
"""Configuration for kdaforge.

A plain dataclass plus a minimal YAML loader (falling back to a tiny hand-rolled
parser ``utils/yamlish.py`` because the managed Python runtime has no pyyaml).
Declared scale numbers (2.8T / 104B active / 896 experts / 1M context) are
informational only; the reference implementation instantiates tiny structures
so a full forward pass runs in milliseconds of pure Python.
"""
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

try:  # pragma: no cover - optional
    import yaml as _pyyaml  # type: ignore
except Exception:  # pragma: no cover
    _pyyaml = None


@dataclass
class AttentionConfig:
    n_layers: int = 6            # declared 93
    kda_layers: int = 4          # declared 69 (linear-attention layers)
    mla_layers: int = 2          # declared 24 (full-attention layers)
    d_model: int = 64
    n_heads: int = 4
    head_dim: int = 16           # d_k = d_v = head_dim
    chunk_size: int = 16         # chunked scan block size
    beta_init: float = 0.5       # delta-rule write intensity init
    gate: bool = True            # Gated DeltaNet


@dataclass
class MoEConfig:
    n_experts: int = 16          # declared 896
    top_k: int = 4               # declared 16
    d_model: int = 64
    expert_dim: int = 128
    aux_loss_weight: float = 0.01
    bias_correction: bool = True  # Stable LatentMoE per-expert bias


@dataclass
class MopdConfig:
    n_teachers: int = 4
    n_rollouts: int = 32
    temperature: float = 0.8
    kl_weight: float = 0.5
    advantage_clip: float = 1.0
    learning_rate: float = 0.2
    max_steps: int = 40


@dataclass
class ServingConfig:
    draft_len: int = 6           # speculative draft tokens
    n_lookahead: int = 16        # KV prefetch lookahead
    batch_size: int = 8
    host: str = "127.0.0.1"
    port: int = 8321


@dataclass
class TrainConfig:
    learning_rate: float = 0.2
    max_steps: int = 40
    seed: int = 20261024


@dataclass
class BenchConfig:
    n_tasks: int = 200
    seq_len: int = 64
    seed: int = 20261024


@dataclass
class DeclaredConfig:
    total_params_b: float = 2800.0      # 2.8T
    active_params_b: float = 104.0      # ~104B active
    n_experts: int = 896
    top_k: int = 16
    context_len: int = 1000000          # 1M tokens
    n_layers: int = 93
    kda_layers: int = 69
    mla_layers: int = 24
    terminal_bench: float = 65.7        # Terminal-Bench 4.0
    speedup_low: float = 2.0
    speedup_high: float = 3.0

    def to_dict(self):
        return asdict(self)


@dataclass
class Config:
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    mopd: MopdConfig = field(default_factory=MopdConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    bench: BenchConfig = field(default_factory=BenchConfig)
    declared: DeclaredConfig = field(default_factory=DeclaredConfig)

    @classmethod
    def from_dict(cls, d):
        d = d or {}
        factories = {
            "attention": AttentionConfig, "moe": MoEConfig, "mopd": MopdConfig,
            "serving": ServingConfig, "train": TrainConfig,
            "bench": BenchConfig, "declared": DeclaredConfig,
        }
        kw = {}
        for key, factory in factories.items():
            kw[key] = factory(**{k: v for k, v in (d.get(key) or {}).items()})
        return cls(**kw)

    def to_dict(self):
        return asdict(self)


def load_config(path=None):
    if path is None:
        here = Path(__file__).parent
        path = here / "configs" / "default.yaml"
        if not path.exists():
            path = here.parent / "configs" / "default.yaml"
    p = Path(path)
    if not p.exists():
        return Config()
    text = p.read_text(encoding="utf-8")
    data = None
    if _pyyaml is not None:
        try:
            data = _pyyaml.safe_load(text)
        except Exception:
            data = None
    if data is None:
        from .utils.yamlish import parse
        data = parse(text)
    return Config.from_dict(data)
