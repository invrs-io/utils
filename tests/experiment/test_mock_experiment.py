"""Tests a mock experiment including analysis while the experiment runs.

Copyright (c) 2023 The INVRS-IO authors.
"""

import json
import multiprocessing as mp
import os
import tempfile
import time
import unittest


NUM_WORK_UNITS = 40


def run_analysis(experiment_path, timeout):
    from invrs_utils.experiment import data

    done = False
    max_time = time.time() + timeout
    while not done and time.time() < max_time:
        # Summary should have a single row per work unit.
        df = data.summarize_experiment(experiment_path, summarize_intervals=[(0, 500)])
        time.sleep(0.5)
        done = len(df) == NUM_WORK_UNITS and df["wid.completed"].all()

    return df


def run_experiment(experiment_path, workers):
    from invrs_utils.experiment import sweep

    sweeps = sweep.product(
        sweep.sweep("seed", [0, 1, 2, 3, 4]),
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


def run_work_unit(wid_path, seed, a, b, c):
    print(f"Launching {wid_path}")
    if not os.path.exists(wid_path):
        os.makedirs(wid_path)

    work_unit_config = locals()
    with open(wid_path + "/setup.json", "w") as f:
        json.dump(work_unit_config, f, indent=4)

    import jax.numpy as jnp
    from jax import random
    from invrs_utils.experiment import checkpoint

    mngr = checkpoint.CheckpointManager(
        path=wid_path,
        save_interval_steps=10,
        max_to_keep=1,
    )

    dummy_loss = []
    dummy_distance = []

    key = random.PRNGKey(seed)
    for i in range(100):
        dummy_loss.append(jnp.exp(-(a + b + c) * i / 10))
        dummy_distance.append(jnp.exp(-(a + b + c) * i / 10) - 0.1)
        dummy_params = random.uniform(random.fold_in(key, i), shape=(50, 50))
        mngr.save(
            step=i,
            pytree={
                "scalars": {
                    "loss": jnp.asarray(dummy_loss),
                    "distance": jnp.asarray(dummy_distance),
                },
                "params": dummy_params,
            },
            force_save=False,
        )

    mngr.save(
        step=i,
        pytree={
            "scalars": {"loss": dummy_loss, "distance": dummy_distance},
            "params": dummy_params,
        },
        force_save=True,
    )
    with open(wid_path + "/completed.txt", "w") as f:
        os.utime(wid_path, None)
    print(f"Completed {wid_path}")


class MockExperimentTest(unittest.TestCase):
    def test_mock_experiment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Launch a separate process that runs the experiment. This will itself
            # spawn multiple workers that carry out the experiment.
            p = mp.Process(
                target=run_experiment,
                kwargs={"experiment_path": tmpdir, "workers": 5},
            )
            p.start()
            # Wait until some work units have started saving checkpoints.
            time.sleep(2)
            # Run the analysis. This will repeatedly summarize the experiment, and
            # return once all work units have been finished.
            df = run_analysis(experiment_path=tmpdir, timeout=100)
            p.join()

        self.assertEqual(len(df), NUM_WORK_UNITS)
