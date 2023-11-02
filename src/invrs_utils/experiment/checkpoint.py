"""Defines a simple checkpoint manager.

Copyright (c) 2023 The INVRS-IO authors.
"""

import dataclasses
import glob
import os
from typing import Any, Callable, List, Optional

from totypes import json_utils


@dataclasses.dataclass
class CheckpointManager:
    """A simple checkpoint manager with an orbax-like API.

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
    serialize_fn: Callable[[Any], str] = json_utils.json_from_pytree
    deserialize_fn: Callable[[str], Any] = json_utils.pytree_from_json

    def latest_step(self) -> Optional[int]:
        """Return the latest checkpointed step, or `None` if no checkpoints exist."""
        steps = self._checkpoint_steps()
        steps.sort()
        return None if len(steps) == 0 else steps[-1]

    def save(self, step: int, pytree: Any, force_save: bool = False) -> None:
        """Save a pytree checkpoint."""
        if (step + 1) % self.save_interval_steps != 0 and not force_save:
            return
        with open(self._checkpoint_fname(step), "w") as f:
            f.write(self.serialize_fn(pytree))
        steps = self._checkpoint_steps()
        steps.sort()
        steps_to_delete = steps[: -self.max_to_keep]
        for step in steps_to_delete:
            os.remove(self._checkpoint_fname(step))

    def restore(self, step: int) -> Any:
        """Restore a pytree checkpoint."""
        with open(self._checkpoint_fname(step)) as f:
            return self.deserialize_fn(f.read())

    def _checkpoint_steps(self) -> List[int]:
        """Return the steps for which checkpoint files exist."""
        fnames = glob.glob(self.path + "/checkpoint_*.json")
        return [int(f.split("_")[-1][:-5]) for f in fnames]

    def _checkpoint_fname(self, step: int) -> str:
        """Return the chackpoint filename for the given step."""
        return self.path + f"/checkpoint_{step:04}.json"
