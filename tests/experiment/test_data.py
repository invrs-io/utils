"""Tests for `experiment.data`.

Copyright (c) 2023 The INVRS-IO authors.
"""

import json
import os
import tempfile
import unittest

import numpy as onp

from invrs_utils.experiment import checkpoint, data

NUM_WORK_UNITS = 4
STEPS = 30
DISTANCE_ZERO_STEP = 13


def _loss(i):
    # A dummy loss function.
    return onp.sqrt(i)


def _distance(i):
    # A dummy distance function which is zero at i == 13.
    return onp.where(i == DISTANCE_ZERO_STEP, 0, i**2 + 0.1)


class DataTest(unittest.TestCase):
    def _dummy_run_work_unit(self, wid_path, config_dict):
        if not os.path.exists(wid_path):
            os.makedirs(wid_path)
        with open(f"{wid_path}/setup.json", "w") as f:
            json.dump(config_dict, f)
        mgr = checkpoint.CheckpointManager(
            path=wid_path,
            save_interval_steps=1,
            max_to_keep=1,
        )
        loss = []
        distance = []
        for i in range(STEPS):
            loss.append(_loss(i))
            distance.append(_distance(i))
            mgr.save(
                step=i,
                pytree={
                    "scalars": {
                        "loss": onp.asarray(loss),
                        "distance": onp.asarray(distance),
                    }
                },
            )
        with open(f"{wid_path}/completed.txt", "w"):
            os.utime(wid_path, None)

    def _dummy_experiment_data(self, experiment_path):
        for i in range(NUM_WORK_UNITS):
            wid_path = f"{experiment_path}/wid_{i:04}"
            self._dummy_run_work_unit(
                wid_path,
                {"a": i, "b": i**2, "c": i**3},
            )

    def test_load_work_unit_scalars(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._dummy_run_work_unit(tmpdir, {"a": "a_data", "b": "b_data", "c": 3})
            wid_config, df = data.load_work_unit_scalars(tmpdir)
        self.assertSequenceEqual(
            set(wid_config.keys()),
            {"a", "b", "c", "completed", "wid", "latest_step", "latest_time_utc"},
        )
        self.assertSequenceEqual(set(df.columns), {"loss", "distance"})
        self.assertEqual(len(df), STEPS)

    def test_summarize_experiment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._dummy_experiment_data(tmpdir)
            df = data.summarize_experiment(
                tmpdir, summarize_intervals=[(0, 10), (10, 20)]
            )
        expected_cols = [
            "wid.a",
            "wid.b",
            "wid.c",
            "wid.completed",
            "wid.wid",
            "wid.latest_step",
            "wid.latest_time_utc",
            data.SUMMARY_INTERVAL,
            data.LOSS_MIN,
            data.LOSS_MEAN,
            data.LOSS_PERCENTILE_10,
            data.DISTANCE_MIN,
            data.DISTANCE_MEAN,
            data.DISTANCE_PERCENTILE_10,
            data.DISTANCE_ZERO_STEP,
            data.DISTANCE_ZERO_COUNT,
        ]
        self.assertSequenceEqual(set(df.columns), set(expected_cols))

        onp.testing.assert_allclose(
            df[df["summary_interval"] == "000-010"][data.LOSS_MIN],
            onp.amin(_loss(onp.arange(0, 10))),
        )
        onp.testing.assert_allclose(
            df[df["summary_interval"] == "000-010"][data.LOSS_MEAN],
            onp.mean(_loss(onp.arange(0, 10))),
        )
        onp.testing.assert_allclose(
            df[df["summary_interval"] == "000-010"][data.LOSS_PERCENTILE_10],
            onp.percentile(_loss(onp.arange(0, 10)), 10),
        )

        onp.testing.assert_allclose(
            df[df["summary_interval"] == "000-010"][data.DISTANCE_MIN],
            onp.amin(_distance(onp.arange(0, 10))),
        )
        onp.testing.assert_allclose(
            df[df["summary_interval"] == "000-010"][data.DISTANCE_MEAN],
            onp.mean(_distance(onp.arange(0, 10))),
        )
        onp.testing.assert_allclose(
            df[df["summary_interval"] == "000-010"][data.DISTANCE_PERCENTILE_10],
            onp.percentile(_distance(onp.arange(0, 10)), 10),
        )

        onp.testing.assert_array_equal(
            df[df["summary_interval"] == "000-010"][data.DISTANCE_ZERO_COUNT],
            onp.zeros(NUM_WORK_UNITS),
        )
        onp.testing.assert_array_equal(
            df[df["summary_interval"] == "010-020"][data.DISTANCE_ZERO_COUNT],
            onp.ones(NUM_WORK_UNITS),
        )
        onp.testing.assert_array_equal(
            df[df["summary_interval"] == "000-010"][data.DISTANCE_ZERO_STEP],
            onp.full((NUM_WORK_UNITS,), onp.nan),
        )
        onp.testing.assert_array_equal(
            df[df["summary_interval"] == "010-020"][data.DISTANCE_ZERO_STEP],
            onp.full((NUM_WORK_UNITS,), DISTANCE_ZERO_STEP),
        )
