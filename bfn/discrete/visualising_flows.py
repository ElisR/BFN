"""Functions for visualising the Bayesian flow for discrete probability distributions."""
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Float, Int, Key, Array


@partial(jax.jit, static_argnums=(1, 4), static_argnames=("bins",))
def flow(x: Int[Array, "D"], num_cats: int, t: Float, beta_1: float, samples: int, *, key: Key, bins: int = 100) -> tuple[Float[Array, "s s s"], list[Float[Array, "bpo"]]]:
    r"""Calculate the flow distribution for a certain observation as a histogram.

    This is the distribution of thetas after time t has elapsed.
    $p_F(\cdot | \mathbf{x}; t)$ in the paper.

    Args:
        x: The observation.
        num_cats: The number of categories.
        t: The time elapsed.
        beta_1: The accuracy at t=1.
        samples: The number of samples for Monte-Carlo density approximation.
        key: The random key.
        bins: The number of bins for the histogram. (Too many will crash matplotlib.)

    Returns:
        A tuple of the (bins, bins, bins) density and (bins + 1,) edges for the histogram.
    """
    beta_t = beta_1 * (t**2)
    oh_x = jax.nn.one_hot(x, num_cats, axis=-2)
    normals = jr.normal(key, shape=(samples, *oh_x.shape))
    y = beta_t * (num_cats * oh_x - 1) + jnp.sqrt(beta_t * num_cats) * normals
    thetas_y = jax.nn.softmax(y, axis=-2)

    thetas_y = thetas_y[:, :, 0]  # TODO Make this a reshape
    density = jnp.histogramdd(thetas_y, bins=bins, range=[(0.0, 1.0)]*num_cats, density=True)
    return density


def make_ternary_vals(h: Float[Array, "s s s"], edges: list[Float[Array, "bpo"]]):
    """Make the values for a ternary plot from a histogram.

    mpltern pseudocolor plot wants a list of (t, l, r, v).
    Here t, l, r are the ternary coordinates and v are values.

    TODO Work out why axes are perhaps permuted.

    Args:
        h: The histogram from `jnp.histogramdd`.
        edges: The edges from `jnp.histogramdd`.

    Returns:
        A tuple of (l, t, r, v).
    """
    edges_trimmed = jax.tree_util.tree_map(lambda edge: 0.5 * (edge[1:] + edge[:-1]), edges)
    tlr = jax.tree_util.tree_map(lambda x: x.ravel(), jnp.meshgrid(*edges_trimmed))
    v = (h / h.max()).ravel()
    t, l, r = jax.tree_util.tree_map(lambda x: x[v > 0], tlr)
    v = v[v > 0]
    return l, t, r, v


@partial(jax.jit, static_argnums=(1, 3))
def sample_theta(
    x: Int[Array, "D"], num_cats: int, beta_1: float, steps: int, *, key: Key
) -> Float[Array, "steps cats D"]:
    r"""Produce a stochastic trajectory of categorical distribution parameters.

    $\boldsymbol{\theta}_i \leftarrow h(\boldsymbol{\theta}_{i-1}, \mathbf{y}, \alpha)$$

    Args:
        x: The observation.
        num_cats: The number of categories.
        beta_1: The final value of beta at t = 1.
        steps: The number of sampling steps.
        key: The random key.

    Returns:
        A trajectory of categorical distribution parameters.
    """
    d = x.shape[-1]
    theta_prior = jnp.ones((num_cats, d), dtype=jnp.float32) / num_cats
    oh_x = jax.nn.one_hot(x, num_cats, axis=-2)

    def time_step(theta_key: tuple[Float[Array, "cats D"], Key], i: Int):
        theta, key = theta_key
        alpha = beta_1 * (2 * i - 1) / (steps**2)
        key, y_key = jr.split(key, 2)

        # Sample y
        normals = jr.normal(y_key, shape=(num_cats, d))
        y = alpha * (num_cats * oh_x - 1) + jnp.sqrt(alpha * num_cats) * normals

        theta_prime = y + jnp.log(theta)
        theta = jax.nn.softmax(theta_prime, axis=-2)
        return (theta, key), theta

    i_s = jnp.arange(1, steps + 1)
    _, theta_timeline = jax.lax.scan(time_step, (theta_prior, key), i_s)
    return theta_timeline
