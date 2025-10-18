"""Test that runs an actual optimization experiemnt of an invrs-gym challenge.

Copyright (c) 2025 invrs.io LLC
"""

import glob
import os
import tempfile
import unittest

import jax
from parameterized import parameterized

from invrs_utils.experiment import data, experiment, sweep


class GymExperimentTest(unittest.TestCase):
    @parameterized.expand([[1], [2]])
    def test_gym_work_unit(self, num_replicas):
        with tempfile.TemporaryDirectory() as experiment_path:

            @experiment.work_unit_fn
            def work_unit_fn(wid_path, beta, steps):
                import invrs_opt
                from invrs_gym import challenges

                from invrs_utils.experiment import work_unit

                work_unit.run_work_unit(
                    key=jax.random.PRNGKey(0),
                    wid_path=wid_path,
                    challenge=challenges.metagrating(),
                    optimizer=invrs_opt.density_lbfgsb(beta=beta),
                    steps=steps,
                    num_replicas=num_replicas,
                )

            sweeps = sweep.product(
                sweep.sweep("beta", [2]),
                sweep.sweep("steps", [3]),
            )
            experiment.run_experiment(
                experiment_path=experiment_path,
                sweeps=sweeps,
                work_unit_fn=work_unit_fn,
                dry_run=False,
                randomize=False,
            )

            # Delete the `completed.txt` files and run for some aditional steps.
            for fname in glob.glob(f"{experiment_path}/wid_*/completed.txt"):
                os.remove(fname)

            sweeps = sweep.product(
                sweep.sweep("beta", [2]),
                sweep.sweep("steps", [6]),
            )
            experiment.run_experiment(
                experiment_path=experiment_path,
                sweeps=sweeps,
                work_unit_fn=work_unit_fn,
                dry_run=False,
                randomize=False,
            )

            data.summarize_experiment(experiment_path, summarize_intervals=[(0, 10)])

            wid_paths = glob.glob(f"{experiment_path}/wid_*")
            for wid_path in wid_paths:
                _, df = data.load_work_unit_scalars(wid_path)
                self.assertEqual(len(df), 6 * num_replicas)
