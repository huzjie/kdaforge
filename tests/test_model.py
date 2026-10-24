# -*- coding: utf-8 -*-
import unittest

from kdaforge.config import Config
from kdaforge.model import KDAForgeModel


class TestModel(unittest.TestCase):
    def test_forward(self):
        cfg = Config()
        model = KDAForgeModel(cfg)
        x = model.embed([0, 1, 2, 3])
        out = model.forward(x)
        self.assertEqual(len(out), 4)
        self.assertEqual(len(out[0]), cfg.attention.d_model)


if __name__ == "__main__":
    unittest.main()
