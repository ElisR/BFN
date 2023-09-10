"""Module containing loss functions and sampling methods for continuous distributions with Bayesian Flow Networks"""
from functools import partial
import jax.numpy as jnp
import jax
import jax.random as jr
from jaxtyping import Array, Float, Int, Key, PyTree
import flax.linen as nn

T_MIN = 1e-10


def loss(
    dist_params: PyTree,
    output_dist: nn.Module,
    x: Float[Array, "D"],
    sigma_1: float,
    *,
    key: Key
) -> float:
    """Return continuous-time Bayesian Flow Networks continuous loss."""
    d = output_dist.D
    assert x.shape == (d,)

    y_key, t_key = jr.split(key)
    t = jr.uniform(t_key, minval=T_MIN)  # Use minimum time as hinted in paper
    gamma = 1 - jnp.power(sigma_1, 2 * t)

    normals = jr.normal(y_key, shape=x.shape)
    mu = gamma * x + jnp.sqrt((1 - gamma) * gamma) * normals
    x_hat = output_dist.apply({"params": dist_params}, mu, t, gamma)

    l_infty = - jnp.log(sigma_1) * jnp.power(sigma_1, -2 * t) * jnp.linalg.norm(x_hat - x, ord=2)
    return l_infty


@partial(jax.jit, static_argnums=(1, 3))
def sample(
    dist_params: PyTree, output_dist: nn.Module, sigma_1: Float, steps: int, *, key: Key
) -> Float[Array, "D"]:
    """Sample from the Bayesian Flow Network for continuous data.

    Args:
        dist_params: Parameters of the neural network.
        output_dist: Neural network that transforms parameters of categorical distribution.
        sigma_1: The standard deviation of the t=1 Gaussian.
        n: The number of sampling steps.
        key: The random key to be used for sampling.
    """
    d = output_dist.D
    mu_prior = jnp.zeros((d,), dtype=jnp.float32)
    rho_0 = jnp.array(1.0)

    def time_step(mu_rho_key: tuple[Float[Array, "D"], Float, Key], i: Int):
        mu, rho, key = mu_rho_key
        t = (i - 1) / steps

        gamma = 1 - jnp.power(sigma_1, 2 * t)
        x_hat = output_dist.apply({"params": dist_params}, mu, t, gamma)

        alpha = jnp.power(sigma_1, -2 * i / steps) * (1 - jnp.power(sigma_1, 2 / steps))
        key, y_key = jr.split(key)
        epsilon = jr.normal(y_key, shape=(d,))
        y = x_hat + jnp.sqrt(1 / alpha) * epsilon
        mu = (rho * mu + alpha * y) / (rho + alpha)
        rho = rho + alpha
        return (mu, rho, key), mu
    (mu, _, _), _ = jax.lax.scan(time_step, (mu_prior, rho_0, key), jnp.arange(1, steps + 1))

    x_hat = output_dist.apply({"params": dist_params}, mu, jnp.array(1.0), 1 - jnp.power(sigma_1, 2))
    return x_hat
