"""Tests for `experiment.sweep`.

Copyright (c) 2023 The INVRS-IO authors.
"""

import unittest

from invrs_utils.experiment import sweep


class SweepTest(unittest.TestCase):
    def test_sweep_matches_expected(self):
        sa = sweep.sweep("a", [1, 2, 3])
        self.assertSequenceEqual(sa, [{"a": 1}, {"a": 2}, {"a": 3}])

    def test_zip(self):
        sa = sweep.sweep("a", [1, 2, 3])
        sb = sweep.sweep("b", [7, 8, 9])
        zipped = sweep.zip(sa, sb)
        self.assertSequenceEqual(
            zipped, [{"a": 1, "b": 7}, {"a": 2, "b": 8}, {"a": 3, "b": 9}]
        )

    def test_product(self):
        sa = sweep.sweep("a", [1, 2, 3])
        sb = sweep.sweep("b", [7, 8, 9])
        product = sweep.product(sa, sb)
        self.assertSequenceEqual(
            product,
            [
                {"a": 1, "b": 7},
                {"a": 1, "b": 8},
                {"a": 1, "b": 9},
                {"a": 2, "b": 7},
                {"a": 2, "b": 8},
                {"a": 2, "b": 9},
                {"a": 3, "b": 7},
                {"a": 3, "b": 8},
                {"a": 3, "b": 9},
            ],
        )
