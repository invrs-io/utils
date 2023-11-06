"""Defines a simple checkpoint manager.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
import glob
import os
import time
from typing import Any, Callable, List, Optional, Union

from totypes import json_utils

SERIALIZE_FN = json_utils.json_from_pytree
DESERIALIZE_FN = json_utils.pytree_from_json


@dataclasses.dataclass
class CheckpointManager:
    """A simple checkpoint manager with an orbax-like API.

    Example usage is as follows:

        mngr = checkpoint.CheckpointManager(
            path="experiment/wid_0000",
            save_interval_steps=10,
            max_to_keep=1,
        )

        # Initialize from a checkpoint if one exists.
        if mngr.latest_step() is not None:
            latest_step = mngr.latest_step()
            (params, state, scalars) = mngr.restore(latest_step)
        else:
            latest_step = -1
            params = ...  # initial parameters
            state = optimizer.init(params)
            scalars = {}

        for i in range(latest_step + 1, steps):
            # Update parameters, state, and scalars.
            mngr.save((params, state, scalars))

        mngr.save((params, state, scalars), force_save=True)

    Attributes:
        path: The path where checkpoints are to be saved.
        save_interval_steps: The save interval, which defines the steps at which
            checkpoints are saved. At other steps, calls to `save` do nothing, unless
            `force_save` is `True`.
        max_to_keep: The maximum number of checkpoints to keep.
        serialize_fn: Function which serializes the pytree.
        deserialize_fn: Function which deserializes the pytree.
    """

    path: str
    save_interval_steps: int
    max_to_keep: int
    serialize_fn: Callable[[Any], str] = SERIALIZE_FN
    deserialize_fn: Callable[[str], Any] = DESERIALIZE_FN

    def __post_init__(self):
        """Validates of `CheckpointManager` attributes."""
        if not os.path.exists(self.path):
            raise ValueError(f"`path` does not exist, got {self.path}.")

    def latest_step(self) -> Optional[int]:
        """Return the latest checkpointed step, or `None` if no checkpoints exist."""
        return latest_step(self.path)

    def save(self, step: int, pytree: Any, force_save: bool = False) -> None:
        """Save a pytree checkpoint."""
        if (step + 1) % self.save_interval_steps != 0 and not force_save:
            return
        serialized = self.serialize_fn(pytree)
        temp_fname = f"{self.path}/temp_{str(int(time.time()))}.json"
        with open(temp_fname, "w") as f:
            f.write(serialized)
        os.rename(temp_fname, fname_for_step(self.path, step))
        steps = checkpoint_steps(self.path)
        steps.sort()
        steps_to_delete = steps[: -self.max_to_keep]
        for step in steps_to_delete:
            os.remove(fname_for_step(self.path, step))

    def restore(self, step: int) -> Any:
        """Restore a pytree checkpoint."""
        return load(self.path, step, deserialize_fn=self.deserialize_fn)


def latest_step(wid_path: str) -> Optional[int]:
    """Return the latest checkpointed step, or `None` if no checkpoints exist."""
    steps = checkpoint_steps(wid_path)
    steps.sort()
    return None if len(steps) == 0 else steps[-1]


def checkpoint_steps(wid_path: str) -> List[int]:
    """Return the chackpoint filename for the given step."""
    fnames = glob.glob(fname_for_step(wid_path, step="*"))
    return [step_for_fname(f) for f in fnames]


def load(
    wid_path: str,
    step: int,
    deserialize_fn: Callable[[str], Any] = DESERIALIZE_FN,
) -> Any:
    """Load the checkpoitn for the given step from the `wid_path`."""
    with open(fname_for_step(wid_path, step)) as f:
        data = f.read()
    return deserialize_fn(data)


def fname_for_step(wid_path: str, step: Union[int, str]) -> str:
    """Return the filename for the given step."""
    step_str = f"{step:04}" if isinstance(step, int) else str(step)
    return f"{wid_path}/checkpoint_{step_str}.json"


def step_for_fname(checkpoint_fname: str) -> int:
    """Return the step for the given checkpoint filename."""
    return int(checkpoint_fname.split("_")[-1][:-5])
