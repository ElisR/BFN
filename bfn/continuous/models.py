"""Module holding some simple neural networks for transforming between continuous distributions."""
import jax.numpy as jnp
import jax
from jaxtyping import Array, Float
import flax.linen as nn


class InnerNetwork(nn.Module):
    """Neural network that acts on continuous distribution."""

    shape: tuple[int]  # Dimensionality of the data
    inner_dim: int  # Number of inner dimensions

    @nn.compact
    def __call__(self, mu: Float[Array, "*shape"], t: Float) -> Float[Array, "*shape"]:
        """Return the output distribution of the model, given an underyling NN architecture."""
        residual = mu

        # An arbitrary way to condition on time
        # Project t to radial basis functions in range [0, 1]
        #radial = jnp.exp(-(t - jnp.linspace(0, 1, num=self.D))**2)
        #mu = mu + nn.Dense(features=self.D)(radial)

        mu = nn.Dense(features=self.shape[-1], use_bias=True, name="position_mixer")(mu)
        mu = nn.tanh(mu)
        mu = nn.Dense(features=self.inner_dim, use_bias=True, name="embedding_mixer")(mu.T).T
        mu = nn.tanh(mu)
        return mu + residual, None


class ScannedInnerNetwork(nn.Module):
    shape: tuple[int]  # Dimensionality of the data
    inner_dim: int = 5  # Number of inner dimensions

    @nn.compact
    def __call__(self, mu, t):
        mu = jnp.expand_dims(mu, axis=0)
        mu = nn.Dense(features=self.inner_dim, use_bias=True, name="embedding")(mu.T).T
        ScanInner = nn.scan(
            InnerNetwork,
            variable_axes={"params": 0},
            variable_broadcast=False,
            in_axes=(nn.broadcast,),
            split_rngs={"params": True},
            length=10
        )
        mu, _ = ScanInner(self.shape, self.inner_dim)(mu, t)
        mu = mu[0, :]
        return mu


class ContinuousOutputDistribution(nn.Module):
    """Module that takes in mean vector and outputs estimated data."""

    shape: tuple[int]  # Dimensionality of the data
    x_min: float = -1.0  # Lower clipping threshold
    x_max: float = 1.0  # Upper clipping threshold
    t_min: float = 1e-5  # Threshold at which x is set to zero

    @nn.compact
    def __call__(self, mu: Float[Array, "D"], t: Float, gamma: Float):
        """Return an esimate x_hat of the data given the mean vector mu and time t."""
        epsilon = ScannedInnerNetwork(self.shape)(mu, t)

        # Dodgy step for small t
        #x_hat = mu / gamma - epsilon * jnp.sqrt((1 - gamma) / gamma)
        #x_hat = jnp.nan_to_num(x_hat)  # Hack that breaks gradients?

        # Fix attempt
        # TODO Find a nicer way
        mu_factor = jax.lax.cond(t < self.t_min, lambda _: 0.0, lambda _: 1 / gamma, None)
        epsilon_factor = jax.lax.cond(t < self.t_min, lambda _: 0.0, lambda _: jnp.sqrt((1 - gamma) / gamma), None)
        x_hat = mu * mu_factor - epsilon * epsilon_factor

        # Cond breaks gradients
        #x_hat = jax.lax.cond(t < T_MIN, lambda x: x, jnp.zeros_like, x_hat)

        x_hat = jnp.clip(x_hat, a_min=self.x_min, a_max=self.x_max)
        return x_hat
