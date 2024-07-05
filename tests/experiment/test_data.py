"""Tests for `experiment.data`.

Copyright (c) 2023 The INVRS-IO authors.
"""

import json
import os
import tempfile
import unittest

import numpy as onp
from parameterized import parameterized

from invrs_utils.experiment import checkpoint, data

NUM_WORK_UNITS = 4
STEPS = 30


def _loss(i):
    # A dummy loss function which monotonically decreases with `i`.
    return onp.sqrt(i)


def _eval_metric(i):
    # A dummy eval metric which monotonically increases with `i`.
    return 1 - _loss(i)


class DataTest(unittest.TestCase):
    def _dummy_run_work_unit(self, wid_path, config_dict, num_replicas):
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
        eval_metric = []
        for i in range(STEPS):
            loss.append([_loss(i)] * num_replicas)
            eval_metric.append([_eval_metric(i)] * num_replicas)
            mgr.save(
                step=i,
                pytree={
                    "scalars": {
                        "loss": onp.asarray(loss),
                        "eval_metric": onp.asarray(eval_metric),
                    }
                },
            )
        with open(f"{wid_path}/completed.txt", "w"):
            os.utime(wid_path, None)

    def _dummy_experiment_data(self, experiment_path, num_replicas):
        for i in range(NUM_WORK_UNITS):
            wid_path = f"{experiment_path}/wid_{i:04}"
            self._dummy_run_work_unit(
                wid_path,
                {"a": i, "b": i**2, "c": i**3},
                num_replicas=num_replicas,
            )

    @parameterized.expand([[1], [10]])
    def test_load_work_unit_scalars(self, num_replicas):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._dummy_run_work_unit(
                tmpdir, {"a": "a_data", "b": "b_data", "c": 3}, num_replicas
            )
            wid_config, df = data.load_work_unit_scalars(tmpdir)
        self.assertSequenceEqual(
            set(wid_config.keys()),
            {"a", "b", "c", "completed", "wid", "latest_step", "latest_time_utc"},
        )
        self.assertSequenceEqual(
            set(df.columns), {"replica", "step", "loss", "eval_metric"}
        )
        self.assertEqual(len(df), STEPS * num_replicas)

    @parameterized.expand([[1], [10]])
    def test_summarize_experiment(self, num_replicas):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._dummy_experiment_data(tmpdir, num_replicas)
            df = data.summarize_experiment(
                tmpdir,
                summarize_intervals=[(0, 10), (10, 20)],
            )
        expected_cols = [
            "wid.a",
            "wid.b",
            "wid.c",
            "wid.completed",
            "wid.wid",
            "wid.latest_step",
            "wid.latest_time_utc",
            "wid.replica",
            data.SUMMARY_INTERVAL,
            data.LOSS_MIN,
            data.LOSS_MEAN,
            data.LOSS_P05,
            data.LOSS_P10,
            data.LOSS_P25,
            data.LOSS_P50,
            data.EVAL_METRIC_MAX,
            data.EVAL_METRIC_MEAN,
            data.EVAL_METRIC_P95,
            data.EVAL_METRIC_P90,
            data.EVAL_METRIC_P75,
            data.EVAL_METRIC_P50,
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
            df[df["summary_interval"] == "000-010"][data.LOSS_P05],
            onp.percentile(_loss(onp.arange(0, 10)), 5),
        )
        onp.testing.assert_allclose(
            df[df["summary_interval"] == "000-010"][data.LOSS_P10],
            onp.percentile(_loss(onp.arange(0, 10)), 10),
        )
        onp.testing.assert_allclose(
            df[df["summary_interval"] == "000-010"][data.LOSS_P25],
            onp.percentile(_loss(onp.arange(0, 10)), 25),
        )
        onp.testing.assert_allclose(
            df[df["summary_interval"] == "000-010"][data.LOSS_P50],
            onp.percentile(_loss(onp.arange(0, 10)), 50),
        )

        onp.testing.assert_allclose(
            df[df["summary_interval"] == "000-010"][data.EVAL_METRIC_MAX],
            onp.amax(_eval_metric(onp.arange(0, 10))),
        )
        onp.testing.assert_allclose(
            df[df["summary_interval"] == "000-010"][data.EVAL_METRIC_MEAN],
            onp.mean(_eval_metric(onp.arange(0, 10))),
        )
        onp.testing.assert_allclose(
            df[df["summary_interval"] == "000-010"][data.EVAL_METRIC_P95],
            onp.percentile(_eval_metric(onp.arange(0, 10)), 95),
        )
        onp.testing.assert_allclose(
            df[df["summary_interval"] == "000-010"][data.EVAL_METRIC_P90],
            onp.percentile(_eval_metric(onp.arange(0, 10)), 90),
        )
        onp.testing.assert_allclose(
            df[df["summary_interval"] == "000-010"][data.EVAL_METRIC_P75],
            onp.percentile(_eval_metric(onp.arange(0, 10)), 75),
        )
        onp.testing.assert_allclose(
            df[df["summary_interval"] == "000-010"][data.EVAL_METRIC_P50],
            onp.percentile(_eval_metric(onp.arange(0, 10)), 50),
        )
