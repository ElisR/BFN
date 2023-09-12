"""Functions for visualising the Bayesian flow for continuous probability distributions."""
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Float, Int, Key, Array


@partial(jax.jit, static_argnums=(2,), static_argnames=("bins", "time_slices"))
def flow_old(x: Float[Array, "D"], sigma_1: float, samples: int, *, key: Key, mu_range: tuple[float], bins: int = 1000, time_slices: int = 100):
    r"""Calculate the flow distribution for a certain observation as a histogram.

    This is an inefficient way to do it because Gaussian probability is easily evaluated.
    TODO Directly evaluate analytic probability.

    Args:
        x: The observation to calculate the flow distribution for.
        sigma_1: The standard deviation of the t=1 Gaussian.
        samples: The number of samples to use to calculate the flow distribution.
        key: The random key to use for sampling.
        mu_range: The range of mu values to use for the histogram.
        bins: The number of bins to use for the histogram.
        time_slices: The number of time slices to use for the flow distribution.

    Returns:
        The flow distribution as histogram:
        Densities of shape (time_slices, bins) and edges of shape (time_slices, bins + 1)
    """
    ts = jnp.linspace(0, 1, time_slices)
    gamma_ts = 1 - jnp.power(sigma_1, 2 * ts)

    normals = jr.normal(key, shape=(1, samples, *x.shape))
    x = jnp.expand_dims(x, (0, 1))
    gamma_ts = jnp.expand_dims(gamma_ts, (1, 2))
    ys = gamma_ts * x + jnp.sqrt((1 - gamma_ts) * gamma_ts) * normals

    # Using lambda function because of kwargs
    return jax.vmap(lambda y_t: jnp.histogram(y_t, bins=bins, density=True, range=mu_range))(ys)


@partial(jax.jit, static_argnames=("bins", "time_slices"))
def flow(x: Float[Array, "D"], sigma_1: float, *, mu_range: tuple[float], bins: int = 1000, time_slices: int = 100):
    """Calculate the flow distribution for a certain observationi as a histogram.
    
    More efficient way than the old way.

    Args:
        x: The observation to calculate the flow distribution for.
        sigma_1: The standard deviation of the t=1 Gaussian.
        samples: The number of samples to use to calculate the flow distribution.
        mu_range: The range of mu values to use for the histogram.
        bins: The number of bins to use for the histogram.
        time_slices: The number of time slices to use for the flow distribution.

    Returns:
        The flow distribution as histogram:
        Densities of shape (time_slices, bins) and edges of shape (bins,)
    """
    ts = jnp.linspace(0, 1, time_slices)
    gamma_ts = 1 - jnp.power(sigma_1, 2 * ts)
    gamma_ts = jnp.expand_dims(gamma_ts, (1,))

    mus = jnp.linspace(*mu_range, num=bins)
    mus = jnp.expand_dims(mus, (0,))

    x = jnp.expand_dims(x, (0,))
    ps = jnp.exp(-0.5 * jnp.square((mus - gamma_ts * x) / jnp.sqrt(gamma_ts * (1 - gamma_ts)))) / jnp.sqrt(2 * jnp.pi * gamma_ts * (1 - gamma_ts))
    ps = jnp.nan_to_num(ps, nan=0.0)
    return ps, mus[0]


@partial(jax.jit, static_argnums=(2,))
def sample_mu(x: Float[Array, "D"], sigma_1: float, steps: int, *, key: Key):
    """Produce a stochastic trajectory for mu parameters, given observation.

    Args:
        x: The observation to calculate the flow distribution for.
        sigma_1: The standard deviation of the t=1 Gaussian.
        steps: The number of steps to use for the stochastic trajectory.
        key: The random key to use for sampling.

    Returns:
        The stochastic trajectory for mu parameters in shape (steps, D)
    """
    mu_prior = jnp.zeros_like(x)
    rho_0 = jnp.array([1.0])    

    def time_step(mu_rho_key: tuple[Float[Array, "D"], Float, Key], i: Int):
        mu, rho, key = mu_rho_key
        key, y_key = jr.split(key)
        alpha = jnp.power(sigma_1, - 2 * i / steps) * (1 - jnp.power(sigma_1, 2 / steps))
        normals = jr.normal(y_key, shape=x.shape)
        y = x + (1 / jnp.sqrt(alpha)) * normals

        mu = (rho * mu + alpha * y) / (rho + alpha)
        rho = rho + alpha
        return (mu, rho, key), mu

    i_s = jnp.arange(1, steps + 1)
    _, mu_timeline = jax.lax.scan(time_step, (mu_prior, rho_0, key), i_s)
    return mu_timeline
