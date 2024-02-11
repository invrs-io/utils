"""Tests a mock experiment including analysis while the experiment runs.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
import glob
import json
import multiprocessing as mp
import os
import tempfile
import time
import unittest

# Matches the experiment defined in `run_experiment`.
NUM_WORK_UNITS = 24


def run_analysis(experiment_path, timeout):
    """Builds a summary dataframe, sleeps, and repeats until timeout."""
    from invrs_utils.experiment import data

    df = None
    done = False
    max_time = time.time() + timeout
    while not done and time.time() < max_time:
        # Summary should have a single row per work unit.
        try:
            df = data.summarize_experiment(
                experiment_path, summarize_intervals=[(0, 500)]
            )
            done = len(df) == NUM_WORK_UNITS and df["wid.completed"].all()
        except ValueError:
            # If no work unit data is ready, a ValueError is raised.
            pass
        time.sleep(0.5)

    return df


def run_experiment(experiment_path, workers, steps):
    """Runs an experiment with multiple conditions and work units."""
    from invrs_utils.experiment import sweep

    sweeps = sweep.product(
        sweep.sweep("steps", [steps]),
        sweep.sweep("seed", [0, 1, 2]),
        sweep.product(
            sweep.zip(
                sweep.sweep("a", [5, 6]),
                sweep.sweep("b", [7, 8]),
            ),
            sweep.sweep("c", [9, 10, 11, 12]),
        ),
    )
    assert len(sweeps) == NUM_WORK_UNITS

    wid_paths = [experiment_path + f"/wid_{i:04}" for i in range(len(sweeps))]
    path_and_kwargs = list(zip(wid_paths, sweeps))

    with mp.Pool(processes=workers) as pool:
        return list(pool.imap_unordered(_run_work_unit, path_and_kwargs))


def _run_work_unit(path_and_kwargs):
    wid_path, kwargs = path_and_kwargs
    run_work_unit(wid_path, **kwargs)


def run_work_unit(wid_path, seed, steps, a, b, c):
    if os.path.isfile(f"{wid_path}/completed.txt"):
        return
    if not os.path.exists(wid_path):
        os.makedirs(wid_path)

    work_unit_config = locals()
    with open(wid_path + "/setup.json", "w") as f:
        json.dump(work_unit_config, f, indent=4)

    import invrs_opt
    import jax.numpy as jnp
    from invrs_gym.challenges import base as challenge_base
    from jax import random, tree_util
    from totypes import types

    from invrs_utils.experiment import work_unit

    class DummyComponent(challenge_base.Component):
        def init(self, key):
            k1, k2 = random.split(key)
            return {
                "bounded_array": types.BoundedArray(
                    array=random.normal(k1, (20, 20)),
                    lower_bound=-2,
                    upper_bound=2,
                ),
                "density2d": types.Density2DArray(
                    array=random.normal(k2, shape=(30, 40)),
                    lower_bound=0,
                    upper_bound=2,
                ),
            }

        def response(self, params):
            leaves = tree_util.tree_leaves(params)
            leaf_sums = tree_util.tree_map(
                lambda leaf: jnp.sum(jnp.abs(leaf) ** 2), leaves
            )
            return jnp.asarray(leaf_sums), {}

    @dataclasses.dataclass
    class DummyChallenge(challenge_base.Challenge):
        component: challenge_base.Component

        def loss(self, response):
            return jnp.sum(response)

        def distance_to_target(self, response):
            return jnp.sum(response) - 0.01

        def metrics(self, response, params, aux):
            return super().metrics(response, params, aux)

    work_unit.run_work_unit(
        key=random.PRNGKey(seed),
        wid_path=wid_path,
        challenge=DummyChallenge(component=DummyComponent()),
        optimizer=invrs_opt.lbfgsb(),
        steps=steps,
        stop_on_zero_distance=False,
        stop_requires_binary=True,
        save_interval_steps=10,
        max_to_keep=1,
        print_interval=None,
    )


class MockExperimentTest(unittest.TestCase):
    def test_mock_experiment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Launch a separate process that runs the experiment. This will itself
            # spawn multiple workers that carry out the experiment.
            p = mp.Process(
                target=run_experiment,
                kwargs={"experiment_path": tmpdir, "workers": 5, "steps": 50},
            )
            p.start()

            # Run the analysis. This will repeatedly summarize the experiment, and
            # return once all work units have been finished.
            df = run_analysis(experiment_path=tmpdir, timeout=100)
            p.join()
            self.assertIsNotNone(df)
            self.assertEqual(len(df), NUM_WORK_UNITS)
            self.assertTrue((df["wid.latest_step"] == 49).all())

            # Delete all the `completed.txt`, and run some additional steps.
            for path in glob.glob(f"{tmpdir}/wid_*/completed.txt"):
                os.remove(path)

            # Relaunch the experiment, which runs all work units for a few more steps.
            p = mp.Process(
                target=run_experiment,
                kwargs={"experiment_path": tmpdir, "workers": 5, "steps": 100},
            )
            p.start()
            p.join()

            df = run_analysis(experiment_path=tmpdir, timeout=100)
            self.assertIsNotNone(df)
            self.assertEqual(len(df), NUM_WORK_UNITS)
            self.assertTrue((df["wid.latest_step"] == 99).all())
