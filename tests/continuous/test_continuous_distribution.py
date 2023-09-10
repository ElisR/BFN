import jax.numpy as jnp
import jax.random as jr
import pytest

import bfn.continuous.loss_and_sample as las
import bfn.continuous.models as models
import bfn.continuous.models_mnist as models_mnist


@pytest.mark.parametrize(("d",), [(5,), (10,), (1,)])
def test_continuous_output_distribution(d: int):
    model = models.ContinuousOutputDistribution(d)

    key, subkey1, subkey2 = jr.split(jr.PRNGKey(0), 3)
    example_x = jr.uniform(subkey1, (d,))
    mu_prior = jnp.zeros_like(example_x)

    variables = model.init(subkey2, mu_prior, 0.5, 0.2)
    params = variables["params"]

    # Defining some arbitrary values for t and gamma
    t = 0.5
    sigma_1 = 0.02
    gamma = 1 - jnp.power(sigma_1, 2 * t)

    out = model.apply({"params": params}, mu_prior, t, gamma)
    assert out.shape == (d,)

    # Test loss
    key, subkey = jr.split(key)
    loss_eval = las.loss(params, model, example_x, sigma_1, key=subkey)

    assert loss_eval.shape == ()

    # Test sampling
    key, subkey = jr.split(key)
    steps = 10
    final_sample = las.sample(params, model, sigma_1, steps, key=subkey)
    assert final_sample.shape == (d,)


@pytest.mark.parametrize(("d", "mix_patch_size", "mix_hidden_size"), [(5, 2, 3), (10, 3, 4), (1, 1, 1)])
def test_mixer_block(d: int, mix_patch_size: int, mix_hidden_size: int):
    model = models_mnist.MixerBlock(mix_patch_size, mix_hidden_size)

    data_key, params_key = jr.split(jr.PRNGKey(0), 2)
    example_x = jr.uniform(data_key, (2, d))

    variables = model.init(params_key, example_x)
    out, _ = model.apply(variables, example_x)
    assert out.shape == example_x.shape


def test_continuous_mixer():
    d = 784
    size = int(jnp.sqrt(d))

    data_key, params_key = jr.split(jr.PRNGKey(0), 2)
    example_x = jr.uniform(data_key, (d,))

    model = models_mnist.ContinuousOutputDistributionMixer(D=d, size=size, num_blocks=4, patch_size=4, hidden_size=64, mix_patch_size=4, mix_hidden_size=4)

    t = jnp.array(0.5)
    sigma_1 = 0.1
    gamma = 1 - jnp.power(sigma_1, 2 * t)

    variables = model.init(params_key, example_x, t, gamma)
    out = model.apply(variables, example_x, t, gamma)
    assert out.shape == example_x.shape
