# -*- coding: utf-8 -*-
import unittest

from kdaforge.config import Config, load_config
from kdaforge.utils.yamlish import parse


class TestConfig(unittest.TestCase):
    def test_default(self):
        cfg = Config()
        self.assertEqual(cfg.declared.total_params_b, 2800.0)
        self.assertEqual(cfg.declared.n_experts, 896)
        self.assertEqual(cfg.declared.context_len, 1000000)

    def test_load_default_yaml(self):
        cfg = load_config()
        self.assertEqual(cfg.moe.n_experts, 16)
        self.assertEqual(cfg.attention.d_model, 64)

    def test_yamlish_block_list(self):
        d = parse("items:\n  - a\n  - b\n  - name: c\n")
        self.assertEqual(d["items"], ["a", "b", {"name": "c"}])


if __name__ == "__main__":
    unittest.main()
