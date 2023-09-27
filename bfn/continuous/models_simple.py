"""Module holding some simple neural network for demos."""
import jax.numpy as jnp
from jaxtyping import Array, Float
import flax.linen as nn


class InnerNetwork(nn.Module):
    """Neural network that acts on continuous distribution."""

    inner_dim: int  # Number of inner dimensions

    @nn.compact
    def __call__(self, mu: Float[Array, "*shape"], t: Float) -> Float[Array, "*shape"]:
        """Return the output distribution of the model, given an underyling NN architecture."""
        residual = mu

        # An arbitrary way to condition on time
        # Project t to radial basis functions in range [0, 1]
        #radial = jnp.exp(-(t - jnp.linspace(0, 1, num=self.D))**2)
        #mu = mu + nn.Dense(features=self.D)(radial)

        mu = nn.Dense(features=mu.shape[-1], use_bias=True, name="position_mixer")(mu)
        mu = nn.tanh(mu)
        mu = nn.Dense(features=self.inner_dim, use_bias=True, name="embedding_mixer")(mu.T).T
        mu = nn.tanh(mu)
        return mu + residual, None


class ScannedInnerNetwork(nn.Module):
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
        mu, _ = ScanInner(self.inner_dim)(mu, t)
        mu = mu[0, :]
        return mu