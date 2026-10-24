# -*- coding: utf-8 -*-
"""StableLatentMoE: the sparse feed-forward block of Kimi K3.1.

Combines a gate with top-k routing, the expert bank, an auxiliary load-balance
loss and a per-expert bias correction. ``skill`` controls how faithfully the
router tracks the balanced oracle -- at ``skill = 1.0`` routing is perfectly
balanced, at ``skill = 0.5`` the natural hot-expert bias is left uncorrected.
"""
from ..core.linalg import softmax
from ..utils.random import stable_float
from .router import gold_experts
from .experts import ExpertBank
from .load_balance import aux_loss, imbalance_ratio, BiasCorrector


class StableLatentMoE:
    def __init__(self, cfg):
        self.cfg = cfg
        self.n_experts = cfg.n_experts
        self.top_k = cfg.top_k
        self.d_model = cfg.d_model
        self.n_hot = max(2, cfg.n_experts // 8)
        self.hot_bias = 1.5
        self.bank = ExpertBank(cfg.n_experts, cfg.d_model, cfg.expert_dim)
        self.corrector = BiasCorrector(cfg.n_experts)

    def _raw_logits(self, token_id):
        """Natural (biased) gate logits: the first ``n_hot`` experts are hot."""
        logits = []
        for i in range(self.n_experts):
            base = self.hot_bias if i < self.n_hot else 0.0
            logits.append(base + stable_float("logit:%d:%d" % (token_id, i), 0.0, 1.0))
        return logits

    def route(self, token_id, skill, balance=True):
        """Pick top-k experts. skill controls how much the hot bias is corrected.

        skill = 0.5 -> boost 0 -> hot experts win every token (imbalanced).
        skill = 1.0 -> boost 1 -> hot bias fully subtracted (balanced routing).
        """
        boost = (skill - 0.5) * 2.0
        logits = self._raw_logits(token_id)
        if balance:
            for i in range(self.n_hot):
                logits[i] -= boost * self.hot_bias
        scored = sorted(enumerate(logits), key=lambda kv: kv[1], reverse=True)
        return [i for i, _ in scored[:self.top_k]]

    def forward(self, x, skill=1.0, balance=True):
        outs = []
        counts = [0] * self.n_experts
        for t, xt in enumerate(x):
            experts = self.route(t, skill, balance)
            acc = [0.0] * self.d_model
            for eid in experts:
                counts[eid] += 1
                eo = self.bank.experts[eid].forward(xt)
                for j in range(self.d_model):
                    acc[j] += eo[j]
            outs.append([a / self.top_k for a in acc])
        return outs

    def load_imbalance(self, n_tokens, skill=0.5):
        counts = [0] * self.n_experts
        for t in range(n_tokens):
            for eid in self.route(t, skill, balance=True):
                counts[eid] += 1
        return imbalance_ratio(counts), counts
