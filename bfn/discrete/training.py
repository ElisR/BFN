"""Module for training the model."""
import jax
import jax.random as jr
import jax.numpy as jnp
from functools import partial
import optax

import bfn.discrete.train_and_sample as tas


@partial(jax.jit, static_argnums=(0, 2))
def make_step(model, x, optim, opt_state, params, beta_1, *, key):
    """Calculate loss & grad and update model according to optimiser.

    Args:
        model: The BFN discrete model to be trained.
        x: The input data, a JAX array of integers of shape (..., D).
        optim: Optax optimiser.
        opt_state: Optax optimiser state.
        params: The parameters of the model.
        beta_1: The final value of beta at t = 1.
        key: The random key to be used during loss calculation.

    Returns:
        The loss, updated parameters, and updated optimiser state.
    """
    loss, grads = jax.value_and_grad(tas.loss)(params, model, x, beta_1, key=key)
    updates, opt_state = optim.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return loss, params, opt_state


@partial(jax.jit, static_argnums=(0, 2))
def make_step_batch(model, x_batch, optim, opt_state, params, beta_1, *, key):
    """Calculate loss & grad for a batch and update model according to optimiser.

    Args:
        model: The BFN discrete model to be trained.
        x_batch: The input data, a JAX array of integers of shape (N, D).
        optim: Optax optimiser.
        opt_state: Optax optimiser state.
        params: The parameters of the model.
        beta_1: The final value of beta at t = 1.
        key: The random key to be used during loss calculation.

    Returns:
        The loss, updated parameters, and updated optimiser state.
    """
    batch_size = x_batch.shape[0]

    def loss_for_batch(params, key):
        keys = jr.split(key, batch_size)
        loss = jnp.mean(jax.vmap(tas.loss, in_axes=(None, None, 0, None))(params, model, x_batch, beta_1, key=keys))
        return loss

    loss, grads = jax.value_and_grad(loss_for_batch)(params, key)
    updates, opt_state = optim.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return loss, params, opt_state