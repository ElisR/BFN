"""Module for training the model."""
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Array, Int, Key, PyTree

import bfn.continuous.loss_and_sample as las


@partial(jax.jit, static_argnums=(0, 2))
def make_step(
    model: nn.Module,
    x_batch: Int[Array, "batch *shape"],
    optim: optax.GradientTransformation,
    opt_state,
    params: PyTree,
    sigma_1: float,
    *,
    key: Key
):
    """Calculate loss & grad for a batch and update model according to optimiser.

    Args:
        model: The BFN continuous model to be trained.
        x_batch: The input data, a JAX array of integers of shape (batch, *shape).
        optim: Optax optimiser.
        opt_state: Optax optimiser state.
        params: The parameters of the model.
        sigma_1: The final value of beta at t = 1.
        key: The random key to be used during loss calculation.

    Returns:
        The loss, updated parameters, and updated optimiser state.
    """
    batch_size = x_batch.shape[0]

    def loss_for_batch(params, key):
        keys = jr.split(key, batch_size)
        loss = jnp.mean(
            jax.vmap(las.loss, in_axes=(None, None, 0, None))(
                params, model, x_batch, sigma_1, key=keys
            )
        )
        return loss

    loss, grads = jax.value_and_grad(loss_for_batch)(params, key)
    updates, opt_state = optim.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return loss, params, opt_state
