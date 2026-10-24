# -*- coding: utf-8 -*-
import unittest

from kdaforge.serving.speculate import SpeculativeDecoder, expected_acceptance, speedup


class TestSpeculate(unittest.TestCase):
    def test_expected_acceptance(self):
        self.assertAlmostEqual(expected_acceptance(1.0, 6), 7.0, places=6)
        self.assertAlmostEqual(expected_acceptance(0.5, 0), 1.0, places=6)

    def test_speedup_formula(self):
        self.assertAlmostEqual(speedup(1.0, 6, 0.15), 7.0 / 1.9, places=6)

    def test_acceptance_rate(self):
        dec = SpeculativeDecoder()
        self.assertAlmostEqual(dec.acceptance_rate(0.5), 0.5, places=6)
        self.assertAlmostEqual(dec.acceptance_rate(1.0), 1.0, places=6)

    def test_speedup_monotonic(self):
        dec = SpeculativeDecoder()
        _, lo = dec.measure(0.5)
        _, hi = dec.measure(1.0)
        self.assertGreater(hi, lo)


if __name__ == "__main__":
    unittest.main()
