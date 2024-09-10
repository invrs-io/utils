"""Tests for `experiment.experiment`.

Copyright (c) 2023 The INVRS-IO authors.
"""

import glob
import tempfile
import unittest

from parameterized import parameterized

from invrs_utils.experiment import experiment


@experiment.work_unit_fn
def work_unit_fn(wid_path, a, b, c, d):
    pass


class ExperimentTest(unittest.TestCase):
    @parameterized.expand(
        [
            [1, True, True],
            [1, False, True],
            [1, False, False],
            [3, False, False],
        ]
    )
    def test_experiment(self, num_workers, dry_run, randomize):
        sweeps = [
            {"a": 0, "b": 1, "c": 2, "d": 3},
            {"a": 4, "b": 5, "c": 6, "d": 7},
            {"a": 8, "b": 9, "c": 10, "d": 11},
            {"a": 12, "b": 13, "c": 14, "d": 15},
            {"a": 16, "b": 17, "c": 18, "d": 19},
        ]

        with tempfile.TemporaryDirectory() as experiment_path:
            experiment.run_experiment(
                experiment_path=experiment_path,
                sweeps=sweeps,
                work_unit_fn=work_unit_fn,
                randomize=True,
                dry_run=False,
            )
            self.assertSequenceEqual(
                set(glob.glob(f"{experiment_path}/*/*")),
                {
                    f"{experiment_path}/wid_0000/setup.json",
                    f"{experiment_path}/wid_0001/setup.json",
                    f"{experiment_path}/wid_0002/setup.json",
                    f"{experiment_path}/wid_0003/setup.json",
                    f"{experiment_path}/wid_0004/setup.json",
                },
            )
