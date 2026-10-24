# -*- coding: utf-8 -*-
import unittest

from kdaforge.mopd.sandbox import Sandbox
from kdaforge.mopd.distill import kl_divergence
from kdaforge.mopd.mopd import MOPD
from kdaforge.config import Config


class TestMOPD(unittest.TestCase):
    def test_reward(self):
        task = Sandbox().make_task("a")
        self.assertEqual(task.reward(list(task.correct)[0]), 1.0)
        wrong = next(a for a in range(task.n_actions) if a not in task.correct)
        self.assertEqual(task.reward(wrong), 0.0)

    def test_kl_zero_for_same(self):
        self.assertAlmostEqual(kl_divergence([0.5, 0.5], [0.5, 0.5]), 0.0, places=6)

    def test_skill_monotonic(self):
        m = MOPD(Config().mopd)
        hist = m.train(15)
        skills = [s for s, _, _ in hist]
        self.assertEqual(skills, sorted(skills))

    def test_success_improves(self):
        m = MOPD(Config().mopd)
        before = m.evaluate(100)
        m.train(20)
        after = m.evaluate(100)
        self.assertGreater(after, before)


if __name__ == "__main__":
    unittest.main()
