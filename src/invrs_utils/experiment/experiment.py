"""Functions that enable running an optimization experiment.

Copyright (c) 2023 The INVRS-IO authors.
"""

import json
import multiprocessing as mp
import os
import random
import time
import traceback
from typing import Any, Callable, Dict, Sequence, Tuple

FNAME_WID_CONFIG = "setup.json"
FNAME_COMPLETED = "completed.txt"


def run_experiment(
    experiment_path: str,
    sweeps: Sequence[Dict[str, Any]],
    work_unit_fn: Callable[[Any], Any],
    workers: int,
    dry_run: bool,
    randomize: bool,
) -> None:
    """Runs an experiment."""
    # Set up checkpointing directory.
    wid_paths = [f"{experiment_path}/wid_{i:04}" for i in range(len(sweeps))]

    # Print some information about the experiment.
    print(
        f"Experiment:\n"
        f"  worker count = {max(1, workers)}\n"
        f"  work unit count = {len(sweeps)}\n"
        f"  experiment path = {experiment_path}\n"
        f"Work units:"
    )
    for wid_path, kwargs in zip(wid_paths, sweeps):
        print(f"  {wid_path}: {kwargs}")

    path_and_kwargs = list(zip(wid_paths, sweeps))
    if randomize:
        random.shuffle(path_and_kwargs)

    if dry_run:
        return

    if workers == 1:
        _ = list(map(work_unit_fn, path_and_kwargs))
    else:
        with mp.Pool(processes=workers) as pool:
            _ = list(pool.imap_unordered(work_unit_fn, path_and_kwargs))


def work_unit_fn(fn: Any) -> Callable[[Tuple[str, Dict[str, Any]]], Any]:
    """Wraps `fn` for use with `run_experiment`.

    The first argument to `fn` should be named `wid_path` and be the path to which work
    unit data will be saved. Remaining arguments are work unit hyperparameters.

    The wrapped function takes a single tuple as input, with the first element being
    the work unit path, and the second element being a dictionary giving keyword
    arguments to `fn`.

    Args:
        fn: The function to be wrapped.

    Returns:
        The wrapped function.
    """

    def wrapped_fn(path_and_kwargs: Tuple[str, Dict[str, Any]]) -> None:
        """Wraps `run_work_unit` so that it can be called by `map`."""
        wid_path, kwargs = path_and_kwargs

        if os.path.isfile(f"{wid_path}/{FNAME_COMPLETED}"):
            return
        if not os.path.exists(wid_path):
            os.makedirs(wid_path)
        with open(f"{wid_path}/{FNAME_WID_CONFIG}", "w") as f:
            json.dump(kwargs, f, indent=4)

        try:
            return fn(wid_path=wid_path, **kwargs)
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print(f"Exception: {wid_path}")
            tb = "".join(traceback.format_tb(e.__traceback__))
            message = f"{e}\n{tb}"
            with open(f"{wid_path}/exception_{str(int(time.time()))}.log", "w") as f:
                f.write(message)

    return wrapped_fn
