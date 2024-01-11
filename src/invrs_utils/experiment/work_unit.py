"""Functions related to a single optimization work unit.

Copyright (c) 2023 The INVRS-IO authors.
"""

import os
import time
from typing import Any, Dict, Optional, Protocol, Tuple

import jax
import jax.numpy as jnp

from invrs_utils.experiment import checkpoint

PyTree = Any


def run_work_unit(
    key: jax.Array,
    wid_path: str,
    challenge: "Challenge",
    optimizer: "Optimizer",
    steps: int,
    stop_on_zero_distance: bool,
    stop_requires_binary: bool,
    save_interval_steps: int = 10,
    max_to_keep: int = 1,
    print_interval: Optional[int] = 300,
) -> None:
    """Runs a work unit.

    The work unit uses the `optimizer` to optimize the `challenge`, completing the
    specified number of `steps` but potentially stopping early if target performance
    has been achieved.

    Checkpoints are saved to the `wid_path`. These include the optimizer state, scalars
    (such as loss, distance to target, and any scalar metrics), and quantities
    associated with the best solution so far encountered (the "champion"); the best
    solution is the one which has the highest binarization degree and lowest loss.

    Args:
        key: Random number generator key used to initialize parameters.
        wid_path: The path where work unit checkpoints are to be stored.
        challenge: The challenge to be optimized.
        optimizer: The optimizer for the work unit.
        steps: The number of optimization steps.
        stop_on_zero_distance: Determines if the optimization run should be stopped
            early if zero distance to target is achieved, along with other optional
            criteria.
        stop_requires_binary: Determines if density arrays in the design must be binary
            for early stopping on zero distance.
        save_interval_steps: The interval at which checkpoints are saved to `wid_path`.
        max_to_keep: The maximum number of checkpoints to keep.
        print_interval: Optional, the seconds elapsed between updates.
    """
    if os.path.isfile(f"{wid_path}/completed.txt"):
        return
    if not os.path.exists(wid_path):
        raise ValueError(f"{wid_path} does not exist.")

    print(f"{wid_path} starting")

    # Create a basic checkpoint manager that can serialize custom types.
    mngr = checkpoint.CheckpointManager(
        path=wid_path, save_interval_steps=save_interval_steps, max_to_keep=max_to_keep
    )

    def loss_fn(
        params: Any,
    ) -> Tuple[jnp.ndarray, Tuple[Any, jnp.ndarray, Dict[str, Any], Dict[str, Any]]]:
        response, aux = challenge.component.response(params)
        loss = challenge.loss(response)
        distance = challenge.distance_to_target(response)
        metrics = challenge.metrics(response, params, aux)
        return loss, (response, distance, metrics, aux)

    # Use a jit-compiled value-and-grad function, if the challenge supports it.
    value_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    try:
        dummy_params = challenge.component.init(jax.random.PRNGKey(0))
        value_and_grad_fn = jax.jit(value_and_grad_fn).lower(dummy_params).compile()
    except jax.errors.UnexpectedTracerError:
        pass

    if mngr.latest_step() is not None:
        latest_step: int = mngr.latest_step()  # type: ignore[assignment]
        latest_checkpoint = mngr.restore(latest_step)
        state = latest_checkpoint["state"]
        scalars = latest_checkpoint["scalars"]
        champion_result = latest_checkpoint["champion_result"]
    else:
        latest_step = -1  # Next step is `0`.
        latent_params = challenge.component.init(key)
        state = optimizer.init(latent_params)
        scalars = {}
        champion_result = {}

    def _log_scalar(name: str, value: float) -> None:
        if name not in scalars:
            scalars[name] = jnp.zeros((0,))
        scalars[name] = jnp.concatenate([scalars[name], jnp.asarray([float(value)])])

    last_print_time = time.time()
    last_print_step = latest_step

    for i in range(latest_step + 1, steps):
        params = optimizer.params(state)
        t0 = time.time()
        (loss_value, (response, distance, metrics, aux)), grad = value_and_grad_fn(
            params
        )
        t1 = time.time()

        if print_interval is not None and (
            time.time() > last_print_time + print_interval
        ):
            print(
                f"{wid_path} is now at step {i} "
                f"({(time.time() - last_print_time) / (i - last_print_step):.1f}"
                f"s / step)"
            )
            last_print_time = time.time()
            last_print_step = i

        _log_scalar("loss", loss_value)
        _log_scalar("distance", distance)
        _log_scalar("simulation_time", t1 - t0)
        _log_scalar("update_time", time.time() - t0)
        for name, metric_value in metrics.items():
            if _is_scalar(metric_value):
                _log_scalar(name, metric_value)
        binarization = metrics["binarization_degree"]
        if (
            i == 0
            or (binarization is None and loss_value < champion_result["loss"])
            or (
                binarization is not None
                and (
                    binarization > champion_result["binarization_degree"]
                    or (
                        binarization == champion_result["binarization_degree"]
                        and loss_value < champion_result["loss"]
                    )
                )
            )
        ):
            champion_result = {
                "step": i,
                "loss": loss_value,
                "binarization_degree": binarization,
                "params": params,
                "response": response,
                "distance": distance,
                "metrics": metrics,
                "aux": aux,
            }
        ckpt_dict = {
            "state": state,
            "scalars": scalars,
            "champion_result": champion_result,
        }
        state = optimizer.update(
            value=loss_value, params=params, grad=grad, state=state
        )
        mngr.save(i, ckpt_dict)
        if (
            stop_on_zero_distance
            and distance <= 0
            and (binarization in (1, None) or not stop_requires_binary)
        ):
            break

    mngr.save(i, ckpt_dict, force_save=True)
    with open(f"{wid_path}/completed.txt", "w"):
        os.utime(wid_path, None)

    print(f"{wid_path} finished")


def _is_scalar(x: Any) -> bool:
    """Returns `True` if `x` is a scalar, i.e. it can be cast as a float."""
    try:
        float(x)
        return True
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Protocols defining the `Challenge` and `Optimizer` objects.
# -----------------------------------------------------------------------------


class Component(Protocol):
    def init(self, key: jax.Array) -> PyTree:
        ...

    def response(self, params: PyTree) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        ...


class Challenge(Protocol):
    component: Component

    def loss(self, response: PyTree) -> jnp.ndarray:
        ...

    def distance_to_target(self, response: PyTree) -> jnp.ndarray:
        ...

    def metrics(
        self, response: PyTree, Params: PyTree, aux: Dict[str, Any]
    ) -> Dict[str, Any]:
        ...


class InitFn(Protocol):
    def __call__(self, params: PyTree) -> PyTree:
        ...


class ParamsFn(Protocol):
    def __call__(self, state: PyTree) -> PyTree:
        ...


class UpdateFn(Protocol):
    def __call__(
        self, *, grad: PyTree, value: float, params: PyTree, state: PyTree
    ) -> PyTree:
        ...


class Optimizer(Protocol):
    init: InitFn
    update: UpdateFn
    params: ParamsFn
