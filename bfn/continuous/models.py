"""Module holding some simple neural networks for transforming between continuous distributions."""
from typing import Any
import jax.numpy as jnp
import jax
from jaxtyping import Array, Float
import flax.linen as nn

import einops


class InnerNetwork(nn.Module):
    """Neural network that acts on continuous distribution."""

    D: int  # Number of variables
    inner_dim: int  # Number of inner dimensions

    @nn.compact
    def __call__(self, mu: Float[Array, "D"], t: Float) -> Float[Array, "D"]:
        """Return the output distribution of the model, given an underyling NN architecture."""
        residual = mu

        # An arbitrary way to condition on time
        # Project t to radial basis functions in range [0, 1]
        #radial = jnp.exp(-(t - jnp.linspace(0, 1, num=self.D))**2)
        #mu = mu + nn.Dense(features=self.D)(radial)

        mu = nn.Dense(features=self.D, use_bias=True, name="position_mixer")(mu)
        mu = nn.tanh(mu)
        mu = nn.Dense(features=self.inner_dim, use_bias=True, name="embedding_mixer")(mu.T).T
        mu = nn.tanh(mu)
        return mu + residual, None


class ScannedInnerNetwork(nn.Module):
    D: int
    inner_dim: int = 5

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
        mu, _ = ScanInner(self.D, self.inner_dim)(mu, t)
        mu = mu[0, :]
        return mu


class ContinuousOutputDistribution(nn.Module):
    """Module that takes in mean vector and outputs estimated data."""

    D: int  # Dimensionality of the data
    x_min: float = -1.0  # Lower clipping threshold
    x_max: float = 1.0  # Upper clipping threshold
    t_min: float = 1e-5  # Threshold at which x is set to zero

    @nn.compact
    def __call__(self, mu: Float[Array, "D"], t: Float, gamma: Float):
        """Return an esimate x_hat of the data given the mean vector mu and time t."""
        epsilon = ScannedInnerNetwork(self.D)(mu, t)

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


class MLP(nn.Module):
    """Basic MLP architecture."""
    hidden_dim: int

    @nn.compact
    def __call__(self, y):
        in_dim = y.shape[-1]
        y = nn.Dense(features=self.hidden_dim, use_bias=True)(y)
        y = nn.relu(y)
        y = nn.Dense(features=in_dim, use_bias=True)(y)
        return y


class MixerBlock(nn.Module):
    """Basic block of the MLP-Mixer architecture."""
    mix_patch_size: int  # Size of the patch mixing MLP
    mix_hidden_size: int  # Size of channel mixing MLP

    @nn.compact
    def __call__(self, y: Float[Array, "c D"]) -> Float[Array, "c D"]:
        """Apply mixer block to input, with image as one size"""
        y = y + MLP(self.mix_patch_size)(nn.LayerNorm()(y))
        y = y.T
        y = y + MLP(self.mix_hidden_size)(nn.LayerNorm()(y))
        y = y.T
        return y, None


class Mixer2D(nn.Module):
    """Basic MLP-Mixer architecture."""
    D: int  # Dimensionality of the data
    size: int  # Size in each dimension
    num_blocks: int  # Number of mixer blocks
    patch_size: int  # Size of the patches
    hidden_size: int  # Size of the hidden layers during convolution
    mix_patch_size: int  # Size of the patch mixing MLP
    mix_hidden_size: int  # Size of channel mixing MLP

    @nn.compact
    def __call__(self, y: Float[Array, "D"], t: Float) -> Float[Array, "D"]:
        """Apply MLP-Mixer to input."""
        height, width = self.size, self.size
        assert self.D == height * width
        assert height % self.patch_size == 0
        assert width % self.patch_size == 0

        # Stack time as a channel
        y = einops.rearrange(y, "(h w) -> h w 1", h=height, w=width)
        t = einops.repeat(t, "-> h w 1", h=height, w=width)
        y = jnp.concatenate([y, t], axis=-1)

        # Apply convolutional layer
        y = nn.Conv(features=self.hidden_size, kernel_size=(self.patch_size, self.patch_size), strides=(self.patch_size, self.patch_size))(y)
        patch_height, patch_width, _ = y.shape
        y = einops.rearrange(y, "h w c -> (h w) c")

        # Apply mixer blocks sequentially
        y, _ = nn.scan(MixerBlock, variable_axes={"params": 0}, split_rngs={"params": True}, length=self.num_blocks)(self.mix_patch_size, self.mix_hidden_size)(y.T)

        # Rearrange and apply final convolutional layer
        y = nn.LayerNorm()(y)
        y = einops.rearrange(y, "c (h w) -> h w c", h=patch_height, w=patch_width)
        y = nn.ConvTranspose(features=1, kernel_size=(self.patch_size, self.patch_size), strides=(self.patch_size, self.patch_size))(y)
        y = einops.rearrange(y, "h w 1 -> (h w)")
        return y


class ContinuousOutputDistributionMixer(nn.Module):
    """Module that takes in mean vector and outputs estimated data."""

    D: int  # Dimensionality of the data
    size: int  # Size along each axis
    num_blocks: int  # Number of mixer blocks
    patch_size: int  # Size of the patches
    hidden_size: int  # Size of the hidden layers during convolution
    mix_patch_size: int  # Size of the patch mixing MLP
    mix_hidden_size: int  # Size of channel mixing MLP

    x_min: float = -1.0  # Lower clipping threshold
    x_max: float = 1.0  # Upper clipping threshold
    t_min: float = 1e-10  # Threshold at which x is set to zero

    @nn.compact
    def __call__(self, mu: Float[Array, "D"], t: Float, gamma: Float):
        """Return an esimate x_hat of the data given the mean vector mu and time t."""
        epsilon = Mixer2D(self.D, self.size, self.num_blocks, self.patch_size, self.hidden_size, self.mix_patch_size, self.mix_hidden_size)(mu, t)

        # Fix attempt
        # TODO Find a nicer way
        mu_factor = jax.lax.cond(t < self.t_min, lambda _: 0.0, lambda _: 1 / gamma, None)
        epsilon_factor = jax.lax.cond(t < self.t_min, lambda _: 0.0, lambda _: jnp.sqrt((1 - gamma) / gamma), None)
        x_hat = mu * mu_factor - epsilon * epsilon_factor

        x_hat = jnp.clip(x_hat, a_min=self.x_min, a_max=self.x_max)
        return x_hat