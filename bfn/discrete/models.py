"""Module containing model for transforming between categorical distributions."""
import jax.numpy as jnp
import jax
from jaxtyping import Array, Float
import flax.linen as nn
from einops import rearrange


class InnerNetwork(nn.Module):
    """Neural network that acts on categorical distribution."""
    K: int  # Number of categories

    # TODO Consider turning time argument into a JAX array with a single float
    @nn.compact
    def __call__(self, thetas: Float[Array, "K D"], t: Float) -> Float[Array, "K D"]:
        """Return the output distribution of the model, given an underyling NN architecture."""
        d = thetas.shape[-1]
        residual = thetas

        # An arbitrary way to condition on time
        # Project t to radial basis functions in range [0, 1]
        #radial = jnp.exp(-(t - jnp.linspace(0, 1, num=self.K))**2)
        #thetas = thetas + nn.Dense(features=self.K)(radial)[:, None]

        thetas = nn.Dense(features=self.K, name="category_mixer")(thetas.T).T  # TODO Be careful with bigger shape
        thetas = nn.gelu(thetas)
        thetas = nn.Dense(features=d, name="position_mixer")(thetas)
        thetas = nn.gelu(thetas)
        return thetas + residual, None


class MultipleMLP(nn.Module):
    K: int  # Number of categories
    scale: int = 2  # Scale of the inner network

    @nn.compact
    def __call__(self, thetas: Float[Array, "K *shape"], t: Float) -> Float[Array, "K *shape"]:
        """Return the output distribution of the model, given an underyling NN architecture."""
        # Rescale thetas to be between -1 and 1 before sending to inner network
        thetas = 2 * thetas - 1

        scanned_inner = nn.scan(
            InnerNetwork,
            variable_axes={"params": 0},
            variable_broadcast=False,
            in_axes=(nn.broadcast,),
            split_rngs={"params": True},
            length=10
        )
        thetas = nn.Dense(features=self.scale*self.K)(thetas.T).T
        thetas, _ = scanned_inner(self.scale*self.K)(thetas, t)
        thetas = nn.Dense(features=self.K)(thetas.T).T
        return thetas


# TODO Refactor this to take a simpler network
class DiscreteOutputDistribution(nn.Module):
    """Module that takes in a set of thetas and returns new categorical distribution."""
    K: int  # Number of categories
    shape: tuple[int]  # Dimensionality of data

    network: nn.Module  # Inner neural network

    @nn.compact
    def __call__(self, thetas: Float[Array, "K *shape"], t: Float) -> Float[Array, "K *shape"]:
        """Return the output distribution of the model, given an underyling NN architecture."""
        # Rescale thetas to be between -1 and 1 before sending to inner network
        thetas = 2 * thetas - 1
        thetas = self.network(thetas, t)
        return jax.nn.softmax(thetas, axis=0)
