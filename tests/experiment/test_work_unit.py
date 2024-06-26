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
import numpy as onp
from invrs_gym import challenges
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

        def _distance_to_target(self, response):
            return jnp.sum(jnp.abs(response)) - 0.1

        def metrics(self, response, params, aux):
            metrics = super().metrics(response, params, aux)
            metrics.update({"distance_to_target": self._distance_to_target(response)})
            return metrics

    return DummyChallenge(component=DummyComponent())


def dummy_challenge_with_no_distance_metric():
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

        def _distance_to_target(self, response):
            return jnp.sum(jnp.abs(response)) - 0.1

        def metrics(self, response, params, aux):
            metrics = super().metrics(response, params, aux)
            metrics.update({"distance_to_target": self._distance_to_target(response)})
            return metrics

    return DummyChallenge(component=DummyComponent())


class WorkUnitTest(unittest.TestCase):
    @parameterized.expand(
        [
            [
                dummy_challenge,
                {
                    "loss",
                    "distance_to_target",
                    "step_time",
                },
                1,
            ],
            [
                dummy_challenge_with_no_distance_metric,
                {
                    "loss",
                    "step_time",
                },
                1,
            ],
            [
                dummy_challenge_with_density,
                {
                    "loss",
                    "distance_to_target",
                    "step_time",
                    "binarization_degree",
                },
                1,
            ],
            [
                dummy_challenge_with_density,
                {
                    "loss",
                    "distance_to_target",
                    "step_time",
                    "binarization_degree",
                },
                10,
            ],
        ]
    )
    def test_optimize(self, challenge_fn, expected_scalars, num_replicas):
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
                    num_replicas=num_replicas,
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
            onp.testing.assert_array_equal(
                jnp.amin(ckpt["scalars"]["loss"], axis=0),
                ckpt["champion_result"]["loss"],
            )

    @parameterized.expand(
        [
            [dummy_challenge, 1, True],
            [dummy_challenge_with_density, 1, True],
            [dummy_challenge_with_density, 10, True],
            [dummy_challenge_with_no_distance_metric, 1, False],
        ]
    )
    def test_optimize_with_early_stopping(
        self, challenge_fn, num_replicas, should_stop_early
    ):
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
                    max_to_keep=1,
                    num_replicas=num_replicas,
                )

            _run_work_unit()

            latest_step = checkpoint.latest_step(wid_path)
            if should_stop_early:
                self.assertLess(latest_step, 99)
            print(glob.glob(f"{wid_path}/*"))
            self.assertSequenceEqual(
                set(glob.glob(f"{wid_path}/*")),
                {
                    f"{wid_path}/setup.json",
                    f"{wid_path}/completed.txt",
                    f"{wid_path}/checkpoint_{latest_step:04}.json",
                },
            )
            ckpt = checkpoint.load(wid_path, step=latest_step)
            onp.testing.assert_array_equal(
                jnp.amin(ckpt["scalars"]["loss"], axis=0),
                ckpt["champion_result"]["loss"],
            )

    @parameterized.expand(
        [
            [challenges.metagrating, 1],
            [challenges.ceviche_lightweight_mode_converter, 1],
            [challenges.metagrating, 2],
            [challenges.ceviche_lightweight_mode_converter, 2],
        ]
    )
    def test_actual_challenge(self, challenge_fn, num_replicas):
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
                    steps=5,
                    stop_on_zero_distance=True,
                    stop_requires_binary=True,
                    champion_requires_binary=False,
                    save_interval_steps=10,
                    max_to_keep=3,
                    num_replicas=num_replicas,
                )

            _run_work_unit()

            latest_step = checkpoint.latest_step(wid_path)
            self.assertEqual(latest_step, 4)
            self.assertSequenceEqual(
                set(glob.glob(f"{wid_path}/*")),
                {
                    f"{wid_path}/setup.json",
                    f"{wid_path}/completed.txt",
                    f"{wid_path}/checkpoint_{latest_step:04}.json",
                },
            )
            ckpt = checkpoint.load(wid_path, step=latest_step)
            onp.testing.assert_array_equal(
                jnp.amin(ckpt["scalars"]["loss"], axis=0),
                ckpt["champion_result"]["loss"],
            )
