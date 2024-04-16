"""Functions related to a single optimization work unit.

Copyright (c) 2023 The INVRS-IO authors.
"""

import functools
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
    num_replicas: int = 1,
    pipeline: bool = False,
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
        num_replicas: The number of replicas for the work unit. Each replica is
            identical except for the random seed used to generate initial parameters.
        pipeline: If `True`, the optimization is pipelined.
    """
    if os.path.isfile(f"{wid_path}/completed.txt"):
        return
    if not os.path.exists(wid_path):
        raise ValueError(f"{wid_path} does not exist.")

    if pipeline and num_replicas == 1:
        pipeline = False

    print(f"{wid_path} starting")

    # Create a basic checkpoint manager that can serialize custom types.
    mngr = checkpoint.CheckpointManager(
        path=wid_path, save_interval_steps=save_interval_steps, max_to_keep=max_to_keep
    )

    def eval_fn(step: int, state: Any) -> Tuple[Any, jnp.ndarray, Any, Any]:
        """Performs the evaluation required for a single optimization step."""

        def loss_fn(params: Any) -> Tuple[jnp.ndarray, Any]:
            response_kwargs = response_kwargs_fn(step)
            response, aux = challenge.component.response(params, **response_kwargs)
            loss = challenge.loss(response)
            distance = challenge.distance_to_target(response)
            metrics = challenge.metrics(response, params, aux)
            return loss, (response, distance, metrics, aux)

        params = optimizer.params(state)
        (value, aux), grad = jax.value_and_grad(loss_fn, has_aux=True)(params)
        return params, value, aux, grad

    # -------------------------------------------------------------------------
    # Load state from checkpoint, or initialize new state.
    # -------------------------------------------------------------------------

    if mngr.latest_step() is not None:
        latest_step: int = mngr.latest_step()  # type: ignore[assignment]
        latest_checkpoint = mngr.restore(latest_step)
        state = latest_checkpoint["state"]
        scalars = latest_checkpoint["scalars"]
        champion_result = latest_checkpoint["champion_result"]
    else:
        latest_step = -1  # Next step is `0`.
        if not pipeline:
            keys = jax.random.split(key, num=num_replicas)
            del key
            latent_params = jax.vmap(challenge.component.init)(keys)
            state = jax.vmap(optimizer.init)(latent_params)
        else:
            # When the optimization is pipelined, two optimizer states are maintained.
            # One is updated while the value-and-gradient evaluation for the second is
            # performed. Then, the evaluation is performed for the first optimizer,
            # while the second optimizer state is updated.
            keys = jax.random.split(key, num=num_replicas)
            keys_a = keys[: num_replicas // 2]
            keys_b = keys[num_replicas // 2 :]
            del key, keys
            latent_params_a = jax.vmap(challenge.component.init)(keys_a)
            latent_params_b = jax.vmap(challenge.component.init)(keys_b)
            state_a = jax.vmap(optimizer.init)(latent_params_a)
            state_b = jax.vmap(optimizer.init)(latent_params_b)
            state = (state_a, state_b, jax.vmap(eval_fn, in_axes=(None, 0))(0, state_b))
        scalars = {}
        champion_result = {}

    # -------------------------------------------------------------------------
    # Define core calculations performed at each step.
    # -------------------------------------------------------------------------

    if not pipeline:

        @jax.jit
        @functools.partial(jax.vmap, in_axes=(None, 0))
        def _step_fn(step: int, state: Any) -> Any:
            params, value, (response, distance, metrics, aux), grad = eval_fn(
                step, state
            )
            updated_state = optimizer.update(
                grad=grad, value=value, params=params, state=state
            )
            # If `value` is `nan`, do not update the state.
            updated_state = _select_if_not_nan(value, updated_state, state)
            return updated_state, (params, value, response, distance, metrics, aux)

    else:

        @jax.jit
        def _step_fn(step: int, state: Any) -> Any:
            state_a, state_b, results_b = state

            # State a: evaluate, and then update. No update if value is `nan`.
            params_a, value_a, aux_a, grad_a = jax.vmap(eval_fn, in_axes=(None, 0))(
                step, state_a
            )
            updated_state_a = jax.vmap(optimizer.update)(
                grad=grad_a, value=value_a, params=params_a, state=state_a
            )
            updated_state_a = jax.vmap(_select_if_not_nan)(
                value_a, updated_state_a, state_a
            )

            # State b: update, and then evaluate. No update if value is `nan`.
            params_b, value_b, aux_b, grad_b = results_b
            updated_state_b = jax.vmap(optimizer.update)(
                grad=grad_b, value=value_b, params=params_b, state=state_b
            )
            updated_state_b = jax.vmap(_select_if_not_nan)(
                value_b, updated_state_b, state_b
            )
            next_results_b = jax.vmap(eval_fn, in_axes=(None, 0))(
                step + 1, updated_state_b
            )

            # Merge the a results and b results.
            params, value, response, distance, metrics, aux = tree_util.tree_map(
                lambda a, b: jnp.concatenate([a, b]),
                (params_a, value_a) + aux_a,
                (params_b, value_b) + aux_b,
            )
            updated_state = (updated_state_a, updated_state_b, next_results_b)
            return updated_state, (params, value, response, distance, metrics, aux)

    # -------------------------------------------------------------------------
    # Run the optimization loop.
    # -------------------------------------------------------------------------

    last_print_time = time.time()
    last_print_step = latest_step

    def _log_scalar(name: str, value: jnp.ndarray) -> None:
        if name not in scalars:
            assert value.ndim == 1
            scalars[name] = jnp.zeros((0, value.size))
        scalars[name] = jnp.concatenate([scalars[name], value[jnp.newaxis, :]])

    for i in range(latest_step + 1, steps):
        t0 = time.time()
        (
            state,
            (params, loss_value, response, distance, metrics, aux),
        ) = _step_fn(i, state)
        t1 = time.time()

        if print_interval is not None and (
            time.time() > last_print_time + print_interval
        ):
            print(
                f"{wid_path} is now at step {i} "
                f"({(t1 - last_print_time) / (i - last_print_step):.1f}s / step)"
            )
            last_print_time = time.time()
            last_print_step = i

        _log_scalar("loss", loss_value)
        _log_scalar("distance", distance)
        _log_scalar("step_time", jnp.full((num_replicas,), t1 - t0))
        for name, metric_value in metrics.items():
            if _is_scalar(metric_value, num_replicas):
                _log_scalar(name, metric_value)

        champion_result = _update_champion_result(
            candidate={
                "step": jnp.full((num_replicas,), i),
                "loss": loss_value,
                "binarization_degree": metrics["binarization_degree"],
                "params": params,
                "response": response,
                "distance": distance,
                "metrics": metrics,
                "aux": aux,
            },
            champion=champion_result,
            requires_binary=champion_requires_binary,
        )
        ckpt_dict = {
            "state": state,
            "scalars": scalars,
            "champion_result": champion_result,
        }
        mngr.save(i, ckpt_dict)
        if (
            stop_on_zero_distance
            and jnp.all(champion_result["distance"] <= 0)
            and (
                not stop_requires_binary
                or champion_result["metrics"]["binarization_degree"] is None
                or jnp.all(champion_result["metrics"]["binarization_degree"] == 1)
            )
        ):
            break

    mngr.save(i, ckpt_dict, force_save=True)
    with open(f"{wid_path}/completed.txt", "w"):
        os.utime(wid_path, None)

    print(f"{wid_path} finished")


def _select_if_not_nan(
    maybe_nan: jnp.ndarray,
    true_value: Any,
    false_value: Any,
) -> Any:
    """Return `true_value` if `maybe_nan` is not `nan`, otherwise `false_value`."""
    false_value = tree_util.tree_unflatten(
        treedef=tree_util.tree_structure(true_value),
        leaves=tree_util.tree_leaves(false_value),
    )
    return jax.lax.cond(jnp.isnan(maybe_nan), lambda: false_value, lambda: true_value)


def _update_champion_result(
    candidate: Dict[str, Any],
    champion: Dict[str, Any],
    requires_binary: bool,
) -> Dict[str, Any]:
    """Updates champion results."""
    if candidate["step"][0] == 0:
        return candidate

    assert candidate["loss"].ndim == 1
    num_replicas = candidate["loss"].size

    is_new_champion = []
    for i in range(num_replicas):
        # If binarization is not required or relevant, new champ if loss is lower.
        if candidate["binarization_degree"] is None or not requires_binary:
            is_new_champion.append(candidate["loss"][i] < champion["loss"][i])
        # If binarization is required, new champ if binarization is greater.
        elif candidate["binarization_degree"][i] > champion["binarization_degree"][i]:
            is_new_champion.append(True)
        # If binarization is required, no new champ if binarization is lesser.
        elif candidate["binarization_degree"][i] < champion["binarization_degree"][i]:
            is_new_champion.append(False)
        # If binarization is exactly equal, new champ if loss is lower.
        else:
            is_new_champion.append(candidate["loss"][i] < champion["loss"][i])

    old_champion_leaves = tree_util.tree_leaves(champion)
    candidate_leaves = tree_util.tree_leaves(candidate)
    new_champion_leaves = [
        jnp.where(
            jnp.reshape(
                jnp.asarray(is_new_champion), (num_replicas,) + (1,) * (new.ndim - 1)
            ),
            new,
            old,
        )
        for new, old in zip(candidate_leaves, old_champion_leaves)
    ]

    return tree_util.tree_unflatten(
        tree_util.tree_structure(candidate), new_champion_leaves
    )


def _is_scalar(x: Any, num_replicas: int) -> bool:
    """Returns `True` if `x` is a scalar, i.e. a singleton for each replica."""
    return isinstance(x, jnp.ndarray) and x.shape == (num_replicas,)


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
        self, *, grad: PyTree, value: jnp.ndarray, params: PyTree, state: PyTree
    ) -> PyTree:
        ...


class Optimizer(Protocol):
    init: InitFn
    update: UpdateFn
    params: ParamsFn
