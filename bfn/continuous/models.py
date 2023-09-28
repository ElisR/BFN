"""Module holding wrapper around neural networks so that BFN is trainable."""
import jax.numpy as jnp
import jax
from jaxtyping import Array, Float
import flax.linen as nn


class ContinuousOutputDistribution(nn.Module):
    """Module that takes in mean vector and outputs estimated data."""
    shape: tuple[int, ...]  # Shape of the data, needed for sampling
    network: nn.Module  # Network that takes in mean vector and outputs epsilon

    x_min: float = -1.0  # Lower clipping threshold
    x_max: float = 1.0  # Upper clipping threshold
    t_min: float = 1e-5  # Threshold at which x is set to zero

    @nn.compact
    def __call__(self, mu: Float[Array, "*shape"], t: Float, gamma: Float) -> Float[Array, "*shape"]:
        """Return an esimate x_hat of the data given the mean vector mu and time t."""
        epsilon = self.network(mu, t)

        # Fix attempt
        # TODO Find a nicer way
        mu_factor = jax.lax.cond(t < self.t_min, lambda _: 0.0, lambda _: 1 / gamma, None)
        epsilon_factor = jax.lax.cond(t < self.t_min, lambda _: 0.0, lambda _: jnp.sqrt((1 - gamma) / gamma), None)
        x_hat = mu * mu_factor - epsilon * epsilon_factor

        x_hat = jnp.clip(x_hat, a_min=self.x_min, a_max=self.x_max)
        return x_hat
