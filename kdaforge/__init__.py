# -*- coding: utf-8 -*-
"""kdaforge - Kimi Delta Attention (KDA) hybrid linear attention + sparse MoE engine.

A pure-Python, zero-dependency reference implementation of the architectural
ideas behind Moonshot AI's Kimi K3.1 Pro (2026-10-24): a 2.8T-parameter sparse
MoE (104B active, 896 experts, top-16) that reaches a 1M-token context window by
replacing most of its quadratic attention with **hybrid linear attention**
(Kimi Delta Attention, KDA): 69 of 93 layers run a Gated DeltaNet that keeps a
constant-size recurrent state instead of a KV cache, the other 24 run gated
multi-head latent attention (Gated MLA).

The four composable pieces:

  * attention  - KDA hybrid stack: Gated DeltaNet (delta-rule recurrent state,
                 chunked via an associative scan) + Gated MLA, giving O(d_k*d_v)
                 memory per layer instead of O(L*d) KV cache.
  * moe        - Stable LatentMoE: 896-expert top-16 routing with a Switch-style
                 auxiliary load-balance loss and per-expert bias correction so
                 hot experts do not throttle serving throughput.
  * mopd       - Multi-teacher Online Policy Distillation: a sandbox gives
                 automatic rewards, the model's own rollouts act as teachers,
                 no external closed-source teacher is ever required.
  * serving    - Highspeed speculative decoding: draft-then-verify, routing-
                 aware batch scheduling and KV-cache prefetch (2-3x decode).

Everything is deterministic (md5-seeded) and shares a single oracle. ``skill =
0.5`` is a random router / untrained policy, ``skill = 1.0`` is the perfect one,
so every benchmark sweeps monotonically from a random baseline to the optimum.
"""

__version__ = "1.0.0"

from .config import Config, load_config
from .model import KDAForgeModel
from .attention.kda import KDALayer
from .moe.latent_moe import StableLatentMoE
from .serving.speculate import SpeculativeDecoder

__all__ = [
    "Config", "load_config", "KDAForgeModel", "KDALayer",
    "StableLatentMoE", "SpeculativeDecoder", "__version__",
]
