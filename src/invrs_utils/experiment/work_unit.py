"""Functions related to a single optimization work unit.

Copyright (c) 2023 The INVRS-IO authors.
"""

import os
import time
from typing import Any, Callable, Dict, Optional, Protocol, Tuple

import jax
import jax.numpy as jnp
from jax import tree_util

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
    champion_requires_binary: bool = True,
    response_kwargs_fn: Callable[[int], Dict[str, Any]] = lambda _: {},
    save_interval_steps: int = 10,
    max_to_keep: int = 1,
    print_interval: Optional[int] = 300,
    use_jit: bool = True,
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
        champion_requires_binary: Determines if a new champion must have a greater
            degree of binarization than the previous champion. If `True`, champion
            results tend to become more binary, even at the cost of performance (loss).
        response_kwargs_fn: Function which computes keyword arguments to be supplied to
            the `challenge.component.response` method, given the step number. This
            enables e.g. evaluation with random wavelengths at each step.
        save_interval_steps: The interval at which checkpoints are saved to `wid_path`.
        max_to_keep: The maximum number of checkpoints to keep.
        print_interval: Optional, the seconds elapsed between updates.
        use_jit: If `True`, the entire optimization step will be jit-compiled.
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

    if mngr.latest_step() is not None:
        latest_step: int = mngr.latest_step()  # type: ignore[assignment]
        latest_checkpoint = mngr.restore(latest_step)
        # When saving/loading checkpoints, tuples are generally converted to lists. To
        # ensure that the restored state treedef matches the original treedef, create
        # a dummy state and use its treedef with the leaves from the restored state.
        dummy_params = challenge.component.init(key)
        treedef = tree_util.tree_structure(optimizer.init(dummy_params))
        state = tree_util.tree_unflatten(
            treedef, tree_util.tree_leaves(latest_checkpoint["state"])
        )
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

    def _step_fn(step: int, state: Any) -> Any:
        # Compute kwargs that override the default response calculation for this step.
        response_kwargs = response_kwargs_fn(step)

        def loss_fn(
            params: Any,
        ) -> Tuple[jnp.ndarray, Any]:
            response, aux = challenge.component.response(params, **response_kwargs)
            loss = challenge.loss(response)
            distance = challenge.distance_to_target(response)
            metrics = challenge.metrics(response, params, aux)
            return loss, (response, distance, metrics, aux)

        params = optimizer.params(state)
        (value, (response, distance, metrics, aux)), grad = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)

        # It can occur that even a properly-configured challenge yields a `nan` loss;
        # in these cases, do not update the state.
        state = jax.lax.cond(
            jnp.isnan(value),
            lambda grad, value, params, state: state,  # No update
            lambda grad, value, params, state: optimizer.update(
                grad=grad, value=value, params=params, state=state
            ),
            grad,
            value,
            params,
            state,
        )
        return state, (params, value, response, distance, metrics, aux)

    if use_jit:
        _step_fn = jax.jit(_step_fn)

    last_print_time = time.time()
    last_print_step = latest_step

    for i in range(latest_step + 1, steps):
        t0 = time.time()
        state, (params, loss_value, response, distance, metrics, aux) = _step_fn(
            step=i, state=state
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
        _log_scalar("step_time", t1 - t0)
        for name, metric_value in metrics.items():
            if _is_scalar(metric_value):
                _log_scalar(name, metric_value)

        is_new_champion = _is_new_champion(
            step=i,
            loss_value=loss_value,
            binarization=metrics["binarization_degree"],
            champion_result=champion_result,
            requires_binary=champion_requires_binary,
        )
        if is_new_champion:
            champion_result = {
                "step": i,
                "loss": loss_value,
                "binarization_degree": metrics["binarization_degree"],
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
        mngr.save(i, ckpt_dict)
        if (
            stop_on_zero_distance
            and distance <= 0
            and (
                metrics["binarization_degree"] in (1, None) or not stop_requires_binary
            )
        ):
            break

    mngr.save(i, ckpt_dict, force_save=True)
    with open(f"{wid_path}/completed.txt", "w"):
        os.utime(wid_path, None)

    print(f"{wid_path} finished")


def _is_new_champion(
    step: int,
    loss_value: float,
    binarization: Optional[float],
    champion_result: Dict[str, Any],
    requires_binary: bool,
) -> bool:
    """Determine whether a new champion is to be crowned."""
    if step == 0:
        return True
    if binarization is None or not requires_binary:
        return loss_value < champion_result["loss"]
    if binarization > champion_result["binarization_degree"]:
        return True
    return loss_value < champion_result["loss"]


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
