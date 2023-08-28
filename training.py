"""Module for training the model."""
import jax
import train_and_sample as tas
from functools import partial
import optax


# TODO Make things work with batches
@partial(jax.jit, static_argnums=(0, 2))
def make_step(model, x, optim, opt_state, params, *, key):
    """Calculate loss & grad and update model according to optimiser.

    Args:
        model: The BFN discrete model to be trained.
        x: The input data, a JAX array of integers of shape (..., D).
        optim: Optax optimiser.
        opt_state: Optax optimiser state.
        params: The parameters of the model.
        key: The random key to be used during loss calculation.

    Returns:
        The loss, updated parameters, and updated optimiser state.
    """
    beta = 1.0
    loss, grads = jax.value_and_grad(tas.loss)(params, model, x, beta, key=key)
    updates, opt_state = optim.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return loss, params, opt_state
