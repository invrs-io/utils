"""Functions for loading and analyzing experiments.

Copyright (c) 2023 The INVRS-IO authors.
"""

import glob
import json
import os
from typing import Any, Dict, List, Sequence, Tuple

import jax
import numpy as onp
import pandas as pd
from totypes import json_utils

WID = "wid"

LOSS = "loss"
LOSS_MIN = "loss_min"
LOSS_MEAN = "loss_mean"
LOSS_PERCENTILE_10 = "loss_percentile_10"

DISTANCE = "distance"
DISTANCE_MIN = "distance_min"
DISTANCE_MEAN = "distance_mean"
DISTANCE_ZERO_STEP = "distance_zero_step"
DISTANCE_ZERO_COUNT = "distance_zero_count"
DISTANCE_PERCENTILE_10 = "distance_percentile_10"

SUMMARY_INTERVAL = "summary_interval"

SCALARS = "scalars"
COMPLETED = "completed"

PREFIX_WID_COL = "wid."

PREFIX_WID_PATH = "wid_"
PREFIX_CHECKPOINT = "checkpoint_"
FNAME_WID_CONFIG = "setup.json"
FNAME_COMPLETED = "completed.txt"


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
            dfs.append(
                summarize_work_unit(
                    wid_config, wid_df, summarize_intervals=summarize_intervals
                )
            )
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
      - The 10th percentile of loss over the interal
      - The minimum distance over the interval
      - The mean distance over the interval
      - The 10th percentile of distance over the interval
      - The number of steps where the distance was zero or negative
      - The first step (within each interval) where the distance is zero or negative

    Args:
        wid_config: Dictionary containing the work unit configuration. Config data
            is included in the columns of the output dataframe.
        wid_df: Dataframe containing the work unit scalars.
        summarize_intervals: Sequence of `(lo, hi)` defining the intervals for which
            summary data is to be calculated.

    Returns:
        The dataframe containing the summary data.
    """
    loss = onp.asarray(wid_df[LOSS])
    distance = onp.asarray(wid_df[DISTANCE])

    data: Dict[str, List[Any]] = {}
    for key in (
        SUMMARY_INTERVAL,
        LOSS_MIN,
        LOSS_MEAN,
        LOSS_PERCENTILE_10,
        DISTANCE_MIN,
        DISTANCE_MEAN,
        DISTANCE_PERCENTILE_10,
        DISTANCE_ZERO_COUNT,
        DISTANCE_ZERO_STEP,
    ):
        data[key] = []

    for lo, hi in summarize_intervals:
        interval_loss = loss[lo : min(hi, len(loss))]
        interval_distance = distance[lo : min(hi, len(loss))]

        data[SUMMARY_INTERVAL].append(f"{lo:03}-{hi:03}")

        data[LOSS_MIN].append(onp.amin(interval_loss))
        data[LOSS_MEAN].append(onp.mean(interval_loss))
        data[LOSS_PERCENTILE_10].append(onp.percentile(interval_loss, 10))
        data[DISTANCE_MIN].append(onp.amin(interval_distance))
        data[DISTANCE_MEAN].append(onp.mean(interval_distance))
        data[DISTANCE_PERCENTILE_10].append(onp.percentile(interval_distance, 10))
        data[DISTANCE_ZERO_COUNT].append(onp.sum(interval_distance <= 0))

        (zero_distance_steps,) = onp.where(interval_distance <= 0)
        if zero_distance_steps.size == 0:
            zero_distance_step = onp.nan
        else:
            zero_distance_step = lo + zero_distance_steps[0]
        data[DISTANCE_ZERO_STEP].append(zero_distance_step)

    df = pd.DataFrame.from_dict(data)
    for key, value in wid_config.items():
        df[f"{PREFIX_WID_COL}{key}"] = value
    return df


def checkpoint_exists(wid_path: str) -> bool:
    """Return `True` if a checkpoint file exists."""
    if not os.path.isfile(f"{wid_path}/{FNAME_WID_CONFIG}"):
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

    with open(f"{wid_path}/{FNAME_WID_CONFIG}") as f:
        wid_config = json.load(f)
    wid_config[WID] = wid_path.split("/")[-1]
    wid_config[COMPLETED] = os.path.isfile(f"{wid_path}/{FNAME_COMPLETED}")
    wid_config = flatten_nested(wid_config)

    checkpoint_fname = glob.glob(f"{wid_path}/{PREFIX_CHECKPOINT}*.json")
    checkpoint_fname.sort()
    with open(checkpoint_fname[-1]) as f:
        checkpoint = json_utils.pytree_from_json(f.read())
    df = pd.DataFrame.from_dict(checkpoint[SCALARS])
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
