"""Functions that enable running an optimization experiment.

Copyright (c) 2023 The INVRS-IO authors.
"""

import json
import os
import random
import time
import traceback
from typing import Any, Callable, Dict, Sequence, Tuple

FNAME_WID_CONFIG = "setup.json"
FNAME_COMPLETED = "completed.txt"
LINE_LENGTH = 200


def run_experiment(
    experiment_path: str,
    sweeps: Sequence[Dict[str, Any]],
    work_unit_fn: Callable[[Any], Any],
    dry_run: bool,
    randomize: bool,
) -> None:
    """Runs an experiment."""
    # Set up checkpointing directory.
    wid_paths = [f"{experiment_path}/wid_{i:04}" for i in range(len(sweeps))]

    # Print some information about the experiment.
    print(
        f"Experiment {experiment_path.split('/')[-1]} (path={experiment_path}, "
        f"work_units={len(sweeps)})"
    )
    for wid_path, kwargs in zip(wid_paths, sweeps):
        kwarg_strs = [f"{keyword}={value}" for keyword, value in kwargs.items()]
        lines = [f"  {wid_path.split('/')[-1]}: "]
        for i, ks in enumerate(kwarg_strs):
            if len(lines[-1]) < LINE_LENGTH:
                lines[-1] += f"{ks}"
            else:
                lines.append(f"            {ks}")
            if i < len(kwarg_strs) - 1:
                lines[-1] += ", "
        for line in lines:
            print(line)

    path_and_kwargs = list(zip(wid_paths, sweeps))
    if randomize:
        random.shuffle(path_and_kwargs)

    if dry_run:
        return

    list(map(work_unit_fn, path_and_kwargs))


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
