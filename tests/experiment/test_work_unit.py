"""Tests for `experiment.work_unit`.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
import glob
import json
import os
import tempfile
import unittest

import invrs_opt
import jax
import jax.numpy as jnp
from invrs_gym.challenges import base as challenge_base
from parameterized import parameterized
from totypes import types

from invrs_utils.experiment import checkpoint, work_unit


def dummy_challenge():
    class DummyComponent(challenge_base.Component):
        def init(self, key):
            del key
            return {"array": jnp.arange(100).reshape(10, 10).astype(float)}

        def response(self, params):
            return jnp.sum(params["array"], axis=0), {}

    @dataclasses.dataclass
    class DummyChallenge(challenge_base.Challenge):
        component: challenge_base.Component

        def loss(self, response):
            return jnp.sum(response**2)

        def distance_to_target(self, response):
            return jnp.sum(jnp.abs(response)) - 0.1

        def metrics(self, response, params, aux):
            return super().metrics(response, params, aux)

    return DummyChallenge(component=DummyComponent())


def dummy_challenge_with_density():
    class DummyComponent(challenge_base.Component):
        def init(self, key):
            del key
            return types.Density2DArray(
                array=jnp.arange(100).reshape(10, 10).astype(float)
            )

        def response(self, params):
            return jnp.sum(1 - params.array, axis=0), {}

    @dataclasses.dataclass
    class DummyChallenge(challenge_base.Challenge):
        component: challenge_base.Component

        def loss(self, response):
            return jnp.sum(response**2)

        def distance_to_target(self, response):
            return jnp.sum(jnp.abs(response)) - 0.1

        def metrics(self, response, params, aux):
            return super().metrics(response, params, aux)

    return DummyChallenge(component=DummyComponent())


class WorkUnitTest(unittest.TestCase):
    @parameterized.expand(
        [
            [
                dummy_challenge,
                {
                    "loss",
                    "distance",
                    "simulation_time",
                    "update_time",
                },
            ],
            [
                dummy_challenge_with_density,
                {
                    "loss",
                    "distance",
                    "simulation_time",
                    "update_time",
                    "binarization_degree",
                },
            ],
        ]
    )
    def test_optimize(self, challenge_fn, expected_scalars):
        with tempfile.TemporaryDirectory() as wid_path:

            def _run_work_unit():
                if not os.path.exists(wid_path):
                    os.makedirs(wid_path)

                work_unit_config = locals()
                del work_unit_config["challenge_fn"]
                with open(wid_path + "/setup.json", "w") as f:
                    json.dump(work_unit_config, f, indent=4)

                work_unit.run_work_unit(
                    key=jax.random.PRNGKey(0),
                    wid_path=wid_path,
                    challenge=challenge_fn(),
                    optimizer=invrs_opt.lbfgsb(maxcor=20),
                    steps=100,
                    stop_on_zero_distance=False,
                    stop_requires_binary=True,
                    save_interval_steps=10,
                    max_to_keep=3,
                )

            _run_work_unit()

            self.assertSequenceEqual(
                set(glob.glob(f"{wid_path}/*")),
                {
                    f"{wid_path}/setup.json",
                    f"{wid_path}/completed.txt",
                    f"{wid_path}/checkpoint_0079.json",
                    f"{wid_path}/checkpoint_0089.json",
                    f"{wid_path}/checkpoint_0099.json",
                },
            )
            ckpt = checkpoint.load(wid_path, step=99)
            self.assertSequenceEqual(
                set(ckpt.keys()), {"state", "scalars", "champion_result"}
            )
            self.assertSequenceEqual(set(ckpt["scalars"].keys()), expected_scalars)
            self.assertEqual(
                jnp.amin(ckpt["scalars"]["loss"]), ckpt["champion_result"]["loss"]
            )

    @parameterized.expand([[dummy_challenge], [dummy_challenge_with_density]])
    def test_optimize_with_early_stopping(self, challenge_fn):
        with tempfile.TemporaryDirectory() as wid_path:

            def _run_work_unit():
                if not os.path.exists(wid_path):
                    os.makedirs(wid_path)

                work_unit_config = locals()
                del work_unit_config["challenge_fn"]
                with open(wid_path + "/setup.json", "w") as f:
                    json.dump(work_unit_config, f, indent=4)

                work_unit.run_work_unit(
                    key=jax.random.PRNGKey(0),
                    wid_path=wid_path,
                    challenge=challenge_fn(),
                    optimizer=invrs_opt.lbfgsb(maxcor=20),
                    steps=100,
                    stop_on_zero_distance=True,
                    stop_requires_binary=True,
                    save_interval_steps=10,
                    max_to_keep=3,
                )

            _run_work_unit()

            latest_step = checkpoint.latest_step(wid_path)
            self.assertLess(latest_step, 99)
            self.assertSequenceEqual(
                set(glob.glob(f"{wid_path}/*")),
                {
                    f"{wid_path}/setup.json",
                    f"{wid_path}/completed.txt",
                    f"{wid_path}/checkpoint_{latest_step:04}.json",
                },
            )
            ckpt = checkpoint.load(wid_path, step=latest_step)
            self.assertEqual(
                jnp.amin(ckpt["scalars"]["loss"]), ckpt["champion_result"]["loss"]
            )
