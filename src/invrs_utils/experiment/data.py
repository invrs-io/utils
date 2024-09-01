"""Functions for loading and analyzing experiments.

Copyright (c) 2023 The INVRS-IO authors.
"""

import glob
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Sequence, Tuple

import jax
import numpy as onp
import pandas as pd

from invrs_utils.experiment import checkpoint, experiment

WID = "wid"
REPLICA = "replica"
STEP = "step"
LATEST_STEP = "latest_step"
LATEST_TIME_UTC = "latest_time_utc"
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

LOSS = "loss"
LOSS_MIN = "loss_min"
LOSS_MEAN = "loss_mean"
LOSS_P05 = "loss_p05"
LOSS_P10 = "loss_p10"
LOSS_P25 = "loss_p25"
LOSS_P50 = "loss_p50"

EVAL_METRIC = "eval_metric"
EVAL_METRIC_MAX = "eval_metric_max"
EVAL_METRIC_MEAN = "eval_metric_mean"
EVAL_METRIC_P95 = "eval_metric_p95"
EVAL_METRIC_P90 = "eval_metric_p90"
EVAL_METRIC_P75 = "eval_metric_p75"
EVAL_METRIC_P50 = "eval_metric_p50"

SUMMARY_INTERVAL = "summary_interval"

SCALARS = "scalars"
COMPLETED = "completed"

PREFIX_WID_COL = "wid."

PREFIX_WID_PATH = "wid_"
PREFIX_CHECKPOINT = "checkpoint_"


def summarize_experiment(
    experiment_path: str,
    summarize_intervals: Sequence[Tuple[int, int]],
) -> pd.DataFrame:
    """Generate a summary dataframe containing all work units in an experiment.

    See `summarize_work_unit` for details on the summary information.

    Args:
        experiment_path: The path containing saved experiment data.
        summarize_intervals: Sequence of `(lo, hi)` defining the intervals for which
            summary data is to be calculated.

    Returns:
        The dataframe containing the summary data.
    """
    dfs = []
    wid_paths = glob.glob(f"{experiment_path}/{PREFIX_WID_PATH}*")
    for wid_path in wid_paths:
        if checkpoint_exists(wid_path):
            wid_config, wid_df = load_work_unit_scalars(wid_path)
            for replica_id, wid_replica_df in wid_df.groupby(REPLICA):
                replica_summary_df = summarize_work_unit(
                    wid_config,
                    wid_replica_df,
                    summarize_intervals=summarize_intervals,
                )
                replica_summary_df[f"{PREFIX_WID_COL}{REPLICA}"] = replica_id
                dfs.append(replica_summary_df)
    return pd.concat(dfs)


def summarize_work_unit(
    wid_config: Dict,
    wid_df: pd.DataFrame,
    summarize_intervals: Sequence[Tuple[int, int]],
) -> pd.DataFrame:
    """Generates a dataframe summarizing an experiment work unit.

    For each summary interval, the following quantities are computed:
      - The minimum loss over the interval
      - The mean loss over the interval
      - The 5th, 10th, 25th, and 50th percentile of loss over the interal
      - The maximum eval metric over the interval (if `eval_metric` is a column in
        the work unit dataframe).
      - The mean eval metric over the interval
      - The 95th, 90th, 75th, and 50th percentile of eval metric over the interval

    Args:
        wid_config: Dictionary containing the work unit configuration. Config data
            is included in the columns of the output dataframe.
        wid_df: Dataframe containing the work unit scalars.
        summarize_intervals: Sequence of `(lo, hi)` defining the intervals for which
            summary data is to be calculated.

    Returns:
        The dataframe containing the summary data.
    """
    data: Dict[str, List[Any]] = {}

    loss = onp.asarray(wid_df[LOSS])
    for key in (
        SUMMARY_INTERVAL,
        LOSS_MIN,
        LOSS_MEAN,
        LOSS_P05,
        LOSS_P10,
        LOSS_P25,
        LOSS_P50,
    ):
        data[key] = []

    include_eval_metric = EVAL_METRIC in wid_df.columns
    if include_eval_metric:
        eval_metric = onp.asarray(wid_df[EVAL_METRIC])
        for key in (
            EVAL_METRIC_MAX,
            EVAL_METRIC_MEAN,
            EVAL_METRIC_P95,
            EVAL_METRIC_P90,
            EVAL_METRIC_P75,
            EVAL_METRIC_P50,
        ):
            data[key] = []

    for lo, hi in summarize_intervals:
        interval_loss = loss[lo : min(hi, len(loss))]
        data[SUMMARY_INTERVAL].append(f"{lo:03}-{hi:03}")

        data[LOSS_MIN].append(onp.amin(interval_loss))
        data[LOSS_MEAN].append(onp.mean(interval_loss))
        data[LOSS_P05].append(onp.percentile(interval_loss, 5))
        data[LOSS_P10].append(onp.percentile(interval_loss, 10))
        data[LOSS_P25].append(onp.percentile(interval_loss, 25))
        data[LOSS_P50].append(onp.percentile(interval_loss, 50))

        if include_eval_metric:
            interval_eval_metric = eval_metric[lo : min(hi, len(loss))]
            data[EVAL_METRIC_MAX].append(onp.amax(interval_eval_metric))
            data[EVAL_METRIC_MEAN].append(onp.mean(interval_eval_metric))
            data[EVAL_METRIC_P95].append(onp.percentile(interval_eval_metric, 95))
            data[EVAL_METRIC_P90].append(onp.percentile(interval_eval_metric, 90))
            data[EVAL_METRIC_P75].append(onp.percentile(interval_eval_metric, 75))
            data[EVAL_METRIC_P50].append(onp.percentile(interval_eval_metric, 50))

    df = pd.DataFrame.from_dict(data)
    for key, value in wid_config.items():
        df[f"{PREFIX_WID_COL}{key}"] = value
    return df


def checkpoint_exists(wid_path: str) -> bool:
    """Return `True` if a checkpoint file exists."""
    if not os.path.isfile(f"{wid_path}/{experiment.FNAME_WID_CONFIG}"):
        return False
    if not glob.glob(f"{wid_path}/{PREFIX_CHECKPOINT}*.json"):
        return False
    return True


def load_work_unit_scalars(wid_path: str) -> Tuple[Dict, pd.DataFrame]:
    """Loads the scalars for a work unit from an experiment.

    Args:
        wid_path: The path for the work unit.

    Returns:
        A dictionary containing the work unit configuration, and a dataframe
        containing the logged scalars.
    """
    assert checkpoint_exists(wid_path)
    latest_step: int = checkpoint.latest_step(wid_path)  # type: ignore[assignment]

    with open(f"{wid_path}/{experiment.FNAME_WID_CONFIG}") as f:
        wid_config = json.load(f)
    wid_config[WID] = wid_path.split("/")[-1]
    wid_config[COMPLETED] = os.path.isfile(f"{wid_path}/{experiment.FNAME_COMPLETED}")
    wid_config = flatten_nested(wid_config)
    wid_config[LATEST_STEP] = latest_step

    timestamp = os.path.getmtime(checkpoint.fname_for_step(wid_path, latest_step))
    wid_config[LATEST_TIME_UTC] = datetime.utcfromtimestamp(timestamp).strftime(
        TIME_FORMAT
    )

    scalars_path = f"{wid_path}/scalars_{latest_step:04}.json"
    if os.path.exists(scalars_path):
        scalars = checkpoint.load_scalars(scalars_path)
    else:
        latest_checkpoint = checkpoint.load(wid_path, latest_step)
        scalars = latest_checkpoint[SCALARS]

    scalars_shape = list(scalars.values())[0].shape
    num_steps = scalars_shape[0]
    num_replicas = 1 if len(scalars_shape) == 1 else scalars_shape[1]

    replica_scalars = {}
    for key, value in scalars.items():
        assert value.shape == scalars_shape
        replica_scalars[key] = value.flatten()

    replica_scalars[REPLICA] = onp.tile(
        onp.arange(num_replicas)[onp.newaxis, :], (num_steps, 1)
    ).flatten()

    replica_scalars[STEP] = onp.tile(
        onp.arange(num_steps)[:, onp.newaxis], (1, num_replicas)
    ).flatten()

    df = pd.DataFrame.from_dict(replica_scalars)
    return wid_config, df


def flatten_nested(nested_dict):
    """Flatten a nested dictionary."""
    path_with_leaves = jax.tree_util.tree_leaves_with_path(nested_dict)
    flat = {}
    for path, leaf in path_with_leaves:
        key = ".".join(
            [
                p.key if isinstance(p, jax.tree_util.DictKey) else str(p.idx)
                for p in path
            ]
        )
        flat[key] = leaf
    return flat
