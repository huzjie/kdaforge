# -*- coding: utf-8 -*-
import unittest

from kdaforge.attention.delta_net import GatedDeltaNet
from kdaforge.attention.hybrid import HybridAttention
from kdaforge.config import Config
from kdaforge.utils.random import stable_vector


class TestKDA(unittest.TestCase):
    def test_delta_net_chunked_equals_recurrent(self):
        net = GatedDeltaNet(d_model=64, n_heads=4, head_dim=16)
        x = [stable_vector("x:%d" % t, 64) for t in range(20)]
        rec = net.forward(x, chunked=False)
        chk = net.forward(x, chunked=True, chunk=5)
        for a, b in zip(rec, chk):
            for u, v in zip(a, b):
                self.assertAlmostEqual(u, v, places=4)

    def test_state_memory_constant(self):
        self.assertEqual(GatedDeltaNet.state_memory(16), 256)

    def test_hybrid_stack(self):
        cfg = Config().attention
        h = HybridAttention(cfg)
        self.assertEqual(h.n_layers, cfg.kda_layers + cfg.mla_layers)
        x = [stable_vector("x:%d" % t, cfg.d_model) for t in range(8)]
        out = h.forward(x)
        self.assertEqual(len(out), 8)
        self.assertEqual(len(out[0]), cfg.d_model)

    def test_kv_memory_ratio(self):
        cfg = Config().attention
        h = HybridAttention(cfg)
        full, actual = h.kv_memory(1000000)
        self.assertGreater(full, actual)
        self.assertGreater(full / actual, 1.0)


if __name__ == "__main__":
    unittest.main()
