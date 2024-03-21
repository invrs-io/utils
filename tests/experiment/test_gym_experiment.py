"""Test that runs an actual optimization experiemnt of an invrs-gym challenge.

Copyright (c) 2023 The INVRS-IO authors.
"""

import glob
import tempfile
import unittest

import jax
from parameterized import parameterized

from invrs_utils.experiment import data, experiment, sweep


class GymExperimentTest(unittest.TestCase):
    @parameterized.expand([[1], [2]])
    def test_gym_work_unit(self, num_replicas):
        with tempfile.TemporaryDirectory() as experiment_path:
            sweeps = sweep.sweep("beta", [2])

            @experiment.work_unit_fn
            def work_unit_fn(wid_path, beta):
                import invrs_opt
                from invrs_gym import challenges

                from invrs_utils.experiment import work_unit

                work_unit.run_work_unit(
                    key=jax.random.PRNGKey(0),
                    wid_path=wid_path,
                    challenge=challenges.metagrating(),
                    optimizer=invrs_opt.density_lbfgsb(beta=beta),
                    steps=3,
                    stop_on_zero_distance=False,
                    stop_requires_binary=True,
                    num_replicas=num_replicas,
                )

            experiment.run_experiment(
                experiment_path=experiment_path,
                sweeps=sweeps,
                work_unit_fn=work_unit_fn,
                workers=1,
                dry_run=False,
                randomize=False,
            )

            data.summarize_experiment(experiment_path, summarize_intervals=[(0, 10)])

            wid_paths = glob.glob(f"{experiment_path}/wid_*")
            for wid_path in wid_paths:
                if data.checkpoint_exists(wid_path):
                    data.load_work_unit_scalars(wid_path)
