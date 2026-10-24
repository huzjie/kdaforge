# -*- coding: utf-8 -*-
import unittest

from kdaforge.moe.router import Router, gold_experts
from kdaforge.moe.load_balance import aux_loss, imbalance_ratio
from kdaforge.moe.latent_moe import StableLatentMoE
from kdaforge.config import Config


class TestMoE(unittest.TestCase):
    def test_gold_experts_distinct(self):
        g = gold_experts(7, 16, 4)
        self.assertEqual(len(set(g)), 4)
        self.assertTrue(all(0 <= e < 16 for e in g))

    def test_router_skill_one_matches_gold(self):
        r = Router(16, 4)
        for t in range(20):
            gold = set(gold_experts(t, 16, 4))
            self.assertEqual(set(r.route(t, 1.0)), gold)

    def test_aux_loss(self):
        counts = [10, 10, 10, 10]
        probs = [0.25, 0.25, 0.25, 0.25]
        self.assertAlmostEqual(aux_loss(counts, probs, 4), 1.0, places=6)

    def test_imbalance_balanced(self):
        self.assertAlmostEqual(imbalance_ratio([5, 5, 5, 5]), 1.0, places=6)

    def test_load_balance_improves(self):
        moe = StableLatentMoE(Config().moe)
        lo, _ = moe.load_imbalance(400, skill=0.5)
        hi, _ = moe.load_imbalance(400, skill=1.0)
        self.assertLess(hi, lo)


if __name__ == "__main__":
    unittest.main()
