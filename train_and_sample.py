"""Module containing loss functions for the discrete BFN case."""

import jax.numpy as jnp
import jax
import jax.random as jr
from jaxtyping import Array, Float, Int, Key, PyTree
import flax.linen as nn


class InnerNetwork(nn.Module):
    """Neural network that acts on categorical distribution."""
    K: int  # Number of categories
    D: int  # Number of variables

    def setup(self):
        self.position_mixer = nn.Dense(features=self.D)
        self.category_mixer = nn.Dense(features=self.K)

    def __call__(self, thetas: Float[Array, "K D"], t: float) -> Float[Array, "K D"]:
        """Return the output distribution of the model, given an underyling NN architecture."""
        thetas = self.category_mixer(thetas.T).T
        thetas = jnp.tanh(thetas)
        thetas = self.position_mixer(thetas)
        return thetas


class DiscreteOutputDistribution(nn.Module):
    """Module that takes in a set of thetas and returns new categorical distribution."""
    K: int  # Number of categories
    D: int  # Number of variables

    @nn.compact
    def __call__(self, thetas: Float[Array, "K D"], t: float) -> Float[Array, "K D"]:
        """Return the output distribution of the model, given an underyling NN architecture."""
        # assert 0 <= t <= 1  # Get ConcretizationTypeError with this

        # Rescale thetas to be between -1 and 1 before sending to inner network
        thetas = 2 * thetas - 1

        thetas_out = InnerNetwork(K=self.K, D=self.D)(thetas, t)
        return jax.nn.softmax(thetas_out, axis=-2)


def loss(x: Int[Array, "D"], dist_params: PyTree, output_dist: DiscreteOutputDistribution, beta: float, *, key: Key) -> float:
    """Return the Bayesian Flow Networks discrete loss.

    Args:
        x: Input array of integers, where integers are in the range [0, K).
        dist_params: Parameters of the neural network.
        output_dist: Neural network that transforms parameters of categorical distribution.
        beta: The final value of beta at t = 1.
        key: The random key to be used for sampling.

    Returns:
        The continous time loss.
    """
    k, d = output_dist.K, output_dist.D
    assert x.shape == (d,)
    y_key, t_key = jr.split(key)
    t = jr.uniform(t_key)
    beta_t = beta * (t**2)

    oh_x = jax.nn.one_hot(x, k, axis=-2)
    normals = jr.normal(y_key, shape=(k, d))
    # TODO Check variance or std
    y = beta_t * (k * oh_x - 1) + jnp.sqrt(beta_t * k) * normals
    thetas = jax.nn.softmax(y, axis=-2)

    # Apply the neural network
    thetas_out = output_dist.apply(dist_params, thetas, t)

    l_infty = k * beta * t * jnp.linalg.norm(thetas_out - oh_x, ord=2)
    return l_infty


def sample(dist_params: PyTree, output_dist: DiscreteOutputDistribution, beta: float, n: int, *, key: Key) -> Float[Array, "cats"]:
    """Sample from the Bayesian Flow Network for discrete data.

    Args:
        dist_params: Parameters of the neural network.
        output_dist: Neural network that transforms parameters of categorical distribution.
        beta: The final value of beta at t = 1.
        n: The number of sampling steps.
        key: The random key to be used for sampling.

    Returns:
        The sampled data.
    """
    num_cats, d = output_dist.K, output_dist.D
    theta_prior = jnp.ones((num_cats, d)) / num_cats

    def time_step(theta_key: tuple[Float[Array, "cats D"]], i: int):
        theta, key = theta_key
        t = (i - 1) / n
        alpha = beta * (2 * i - 1) / (n**2)
        key, k_key, y_key = jr.split(key, 3)

        # Sample k
        theta_k = output_dist.apply(dist_params, theta, t)
        k = jr.categorical(k_key, theta_k, axis=-2)

        # Sample y
        # TODO Check variance or std
        oh_k = jax.nn.one_hot(k, num_cats, axis=-2)
        normals = jr.normal(y_key, shape=(num_cats, d))
        y = alpha * (num_cats * oh_k - 1) + jnp.sqrt(alpha * num_cats) * normals

        theta_prime = jnp.exp(y) * theta
        theta = theta_prime / jnp.sum(theta_prime, axis=-2, keepdims=True)
        return (theta, key), k

    i_s = jnp.arange(1, n + 1)
    (theta, key), _ = jax.lax.scan(time_step, (theta_prior, key), i_s)

    k = jr.categorical(key, theta, axis=-2)
    return k
