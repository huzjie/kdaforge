# -*- coding: utf-8 -*-
"""Routing-aware continuous batching.

Groups a stream of tokens into batches so that tokens sharing the same top-k
experts can reuse loaded expert weights, cutting per-token memory traffic. The
gain is structural (it depends on batch composition, not on ``skill``): a
perfectly clustered batch reuses experts maximally.
"""


class BatchScheduler:
    def __init__(self, batch_size=8):
        self.batch_size = batch_size

    def schedule(self, expert_ids, max_batch=None):
        """expert_ids: list of per-token expert tuples. Returns list of batches."""
        bs = max_batch or self.batch_size
        batches = [expert_ids[i:i + bs] for i in range(0, len(expert_ids), bs)]
        return batches

    def expert_reuse(self, expert_ids, max_batch=None):
        """Fraction of expert-loads avoided by grouping into batches."""
        batches = self.schedule(expert_ids, max_batch)
        total_loads = 0
        unique_loads = 0
        for batch in batches:
            union = set()
            for toks in batch:
                for e in toks:
                    total_loads += 1
                    union.add(e)
            unique_loads += len(union)
        if total_loads == 0:
            return 0.0
        return 1.0 - unique_loads / total_loads
