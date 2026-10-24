# -*- coding: utf-8 -*-
import unittest

from kdaforge.tokenizer import Tokenizer


class TestTokenizer(unittest.TestCase):
    def test_round_trip(self):
        tok = Tokenizer()
        text = "hello, 混合线性注意力 KDA 1M context"
        ids = tok.encode(text)
        self.assertEqual(tok.decode(ids), text)

    def test_special_tokens_skipped(self):
        tok = Tokenizer()
        text = "abc"
        ids = tok.encode(text, add_bos=True, add_eos=True)
        self.assertEqual(tok.decode(ids), text)


if __name__ == "__main__":
    unittest.main()
