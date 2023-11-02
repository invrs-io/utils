"""Tests for `experiment.checkpoint`.

Copyright (c) 2023 The INVRS-IO authors.
"""

import glob
import tempfile
import unittest

from parameterized import parameterized

from invrs_utils.experiment import checkpoint


class CheckpointManagerTest(unittest.TestCase):
    @parameterized.expand(
        [
            [1, 1],
            [1, 5],
            [5, 1],
            [5, 5],
        ]
    )
    def test_checkpoints(self, save_interval_steps, max_to_keep):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = checkpoint.CheckpointManager(
                path=tmpdir,
                save_interval_steps=save_interval_steps,
                max_to_keep=max_to_keep,
            )
            max_steps = 30
            for i in range(max_steps):
                mgr.save(
                    step=i,
                    pytree={"a": i, "b": 2 * i},
                    force_save=False,
                )
            checkpoints = glob.glob(tmpdir + "/checkpoint_*.json")
            idxs = range(-1, max_steps, save_interval_steps)
            idxs = idxs[-max_to_keep:]
            expected = [tmpdir + f"/checkpoint_{idx:04}.json" for idx in idxs]
            self.assertSequenceEqual(set(checkpoints), set(expected))

            self.assertEqual(mgr.latest_step(), max_steps - 1)

            restored = mgr.restore(mgr.latest_step())
            self.assertSequenceEqual(
                restored, {"a": max_steps - 1, "b": 2 * (max_steps - 1)}
            )

    def test_force_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = checkpoint.CheckpointManager(
                path=tmpdir,
                save_interval_steps=100,
                max_to_keep=1,
            )
            max_steps = 30
            for i in range(max_steps):
                mgr.save(
                    step=i,
                    pytree={"a": i, "b": 2 * i},
                    force_save=False,
                )
            checkpoints = glob.glob(tmpdir + "/checkpoint_*.json")
            self.assertFalse(checkpoints)

            mgr.save(
                step=max_steps - 1,
                pytree={"a": i, "b": 2 * i},
                force_save=True,
            )
            checkpoints = glob.glob(tmpdir + "/checkpoint_*.json")
            self.assertSequenceEqual(
                checkpoints, [tmpdir + f"/checkpoint_{max_steps - 1:04}.json"]
            )
