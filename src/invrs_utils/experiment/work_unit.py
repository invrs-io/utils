"""Functions related to a single optimization work unit.

Copyright (c) 2023 The INVRS-IO authors.
"""

import functools
import os
import time
import tqdm
import warnings
from typing import Any, Callable, Dict, Optional, Protocol, Tuple

import jax
import jax.numpy as jnp
from jax import tree_util
from totypes import json_utils

from invrs_utils.experiment import checkpoint

PyTree = Any

SAVE_ALL = "all"
SAVE_BINARY = "binary"


def run_work_unit(
    key: jax.Array,
    wid_path: str,
    challenge: "Challenge",
    optimizer: "Optimizer",
    steps: int,
    champion_requires_binary: bool = True,
    response_kwargs_fn: Callable[[int], Dict[str, Any]] = lambda _: {},
    save_params_strategy: Optional[str] = None,
    save_interval_steps: int = 10,
    max_to_keep: int = 1,
    num_replicas: int = 1,
) -> None:
    """Runs a work unit.

    The work unit uses the `optimizer` to optimize the `challenge`, completing the
    specified number of `steps` but potentially stopping early if target performance
    has been achieved.

    Checkpoints are saved to the `wid_path`. These include the optimizer state, scalars
    (such as loss, eval metric, and any scalar metrics), and quantities associated with
    the best solution so far encountered (the "champion"); the best solution is the one
    which has the highest binarization degree and highest eval metric.

    Args:
        key: Random number generator key used to initialize parameters.
        wid_path: The path where work unit checkpoints are to be stored.
        challenge: The challenge to be optimized.
        optimizer: The optimizer for the work unit.
        steps: The number of optimization steps.
        champion_requires_binary: Determines if a new champion must have a greater
            degree of binarization than the previous champion. If `True`, champion
            results tend to become more binary, even at the cost of performance (loss).
        response_kwargs_fn: Function which computes keyword arguments to be supplied to
            the `challenge.component.response` method, given the step number. This
            enables e.g. evaluation with random wavelengths at each step.
        save_params_strategy: If `True`, all binary designs will be saved to the
            work unit directory. If `False`, only the champion design stored along with
            the checkpoint will be retained.
        save_interval_steps: The interval at which checkpoints are saved to `wid_path`.
        max_to_keep: The maximum number of checkpoints to keep.
        num_replicas: The number of replicas for the work unit. Each replica is
            identical except for the random seed used to generate initial parameters.
    """
    if save_params_strategy not in (SAVE_ALL, SAVE_BINARY, None):
        raise ValueError(
            f"Unrecognized `save_params_strategy`, got {save_params_strategy}"
        )
    if os.path.isfile(f"{wid_path}/completed.txt"):
        return
    if not os.path.exists(wid_path):
        raise ValueError(f"{wid_path} does not exist.")

    # Create a basic checkpoint manager that can serialize custom types.
    mngr = checkpoint.CheckpointManager(
        path=wid_path, save_interval_steps=save_interval_steps, max_to_keep=max_to_keep
    )

    keys = jax.random.split(key, num_replicas)
    del key

    maybe_latest_step = mngr.latest_step()
    if maybe_latest_step is not None and maybe_latest_step + 1 >= steps:
        return

    if maybe_latest_step is not None:
        latest_step: int = mngr.latest_step()  # type: ignore[assignment]
        latest_checkpoint = mngr.restore(latest_step)
        state = latest_checkpoint["state"]
        scalars = latest_checkpoint["scalars"]
        champion_result = latest_checkpoint["champion_result"]
    else:
        latest_step = -1  # Next step is `0`.
        latent_params = jax.vmap(challenge.component.init)(keys)
        state = jax.vmap(optimizer.init)(latent_params)
        scalars = {}
        champion_result = {}

    @jax.jit
    @functools.partial(jax.vmap, in_axes=(None, 0))
    def _step_fn(step: int, state: Any) -> Any:
        def loss_fn(params: Any) -> Tuple[jnp.ndarray, Any]:
            response_kwargs = response_kwargs_fn(step)
            response, aux = challenge.component.response(params, **response_kwargs)
            loss = challenge.loss(response)
            eval_metric = challenge.eval_metric(response)
            metrics = challenge.metrics(response, params, aux)
            return loss, (response, eval_metric, metrics, aux)

        params = optimizer.params(state)
        (value, (response, eval_metric, metrics, aux)), grad = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)
        updated_state = optimizer.update(
            grad=grad, value=value, params=params, state=state
        )

        # Ensure the previous state has the proper tree structure. If the update is
        # corrupted due to presence of `nan` in the loss value or gradient, we may
        # return the previous state with no update.
        previous_state = tree_util.tree_unflatten(
            treedef=tree_util.tree_structure(updated_state),
            leaves=tree_util.tree_leaves(state),
        )

        grad_contains_nan = jnp.any(
            jnp.asarray(
                [jnp.any(jnp.isnan(leaf)) for leaf in tree_util.tree_leaves(grad)]
            )
        )
        skip_update = jnp.isnan(value) | grad_contains_nan
        state = jax.lax.cond(skip_update, lambda: previous_state, lambda: updated_state)
        return state, skip_update, (params, value, response, eval_metric, metrics, aux)

    def _log_scalar(name: str, value: jnp.ndarray) -> None:
        if name not in scalars:
            assert value.ndim == 1
            scalars[name] = jnp.zeros((0, value.size))
        scalars[name] = jnp.concatenate([scalars[name], value[jnp.newaxis, :]])

    for i in tqdm.trange(latest_step + 1, steps, desc=wid_path.split("/")[-1]):
        t0 = time.time()
        (
            state,
            skip_update,
            (params, loss_value, response, eval_metric, metrics, aux),
        ) = _step_fn(i, state)
        t1 = time.time()

        for replica, skipped in enumerate(skip_update):
            if skipped:
                warnings.warn(
                    f"Skipped update for replica {replica} due to `nan` values at "
                    f"step {i}."
                )

        _log_scalar("loss", loss_value)
        _log_scalar("eval_metric", eval_metric)
        _log_scalar("step_time", jnp.full((num_replicas,), t1 - t0))
        for name, metric_value in metrics.items():
            if _is_scalar(metric_value, num_replicas):
                _log_scalar(name, metric_value)

        champion_result = _update_champion_result(
            candidate={
                "step": jnp.full((num_replicas,), i),
                "loss": loss_value,
                "eval_metric": eval_metric,
                "binarization_degree": metrics["binarization_degree"],
                "params": params,
                "response": response,
                "metrics": metrics,
                "aux": aux,
            },
            champion=champion_result,
            requires_binary=champion_requires_binary,
        )
        ckpt_dict = dict(state=state, scalars=scalars, champion_result=champion_result)
        mngr.save(i, ckpt_dict)
        checkpoint.save_scalars(i, scalars, wid_path=wid_path)

        if save_params_strategy == SAVE_ALL:
            should_save_params = True
        elif save_params_strategy == SAVE_BINARY:
            binarization = metrics["binarization_degree"]
            should_save_params = (binarization is None) or (1 in binarization)
        else:
            should_save_params = False
        if should_save_params:
            serialized_params = json_utils.json_from_pytree(params)
            params_path = f"{wid_path}/params"
            if not os.path.exists(params_path):
                os.mkdir(params_path)
            with open(f"{params_path}/params_{i:04}.json", "w") as f:
                f.write(serialized_params)

    mngr.save(i, ckpt_dict, force_save=True)
    with open(f"{wid_path}/completed.txt", "w"):
        os.utime(wid_path, None)


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

    new_champ = []
    for i in range(num_replicas):
        # If the old champion has a `nan` eval metric.
        if jnp.isnan(champion["eval_metric"][i]):
            new_champ.append(True)
        # If binarization is not required/relevant, new champ if eval metric is higher.
        elif candidate["binarization_degree"] is None or not requires_binary:
            new_champ.append(candidate["eval_metric"][i] > champion["eval_metric"][i])
        # If binarization is required, new champ if binarization is greater.
        elif candidate["binarization_degree"][i] > champion["binarization_degree"][i]:
            new_champ.append(True)
        # If binarization is required, no new champ if binarization is lesser.
        elif candidate["binarization_degree"][i] < champion["binarization_degree"][i]:
            new_champ.append(False)
        # If binarization is exactly equal, new champ if loss is lower.
        else:
            new_champ.append(candidate["eval_metric"][i] > champion["eval_metric"][i])

    old_champion_leaves = tree_util.tree_leaves(champion)
    candidate_leaves = tree_util.tree_leaves(candidate)
    new_champion_leaves = [
        jnp.where(
            jnp.reshape(
                jnp.asarray(new_champ), (num_replicas,) + (1,) * (new.ndim - 1)
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

    def eval_metric(self, response: PyTree) -> jnp.ndarray:
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
