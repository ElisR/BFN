import jax.numpy as jnp
import jax.random as jr
import pytest

import train_and_sample as tas

# TODO Make fixture and split up unit tests


@pytest.mark.parametrize(("k", "d"), [(5, 10), (10, 5), (1, 1)])
def test_discrete_output_distribution(k: int, d: int):
    dod = tas.DiscreteOutputDistribution(k, d)

    # Test distribution
    key, subkey1, subkey2 = jr.split(jr.PRNGKey(0), 3)
    thetas_example = jr.uniform(subkey1, (k, d))
    variables = dod.init(subkey2, thetas_example, 0.5)
    params = variables["params"]
    out = dod.apply(variables, thetas_example, 0.5)

    assert isinstance(out, jnp.ndarray)
    assert out.shape == (k, d)

    # Test loss
    key, subkey1, subkey2 = jr.split(key, 3)
    x = jr.randint(subkey1, shape=(d,), minval=0, maxval=k)
    loss_eval = tas.loss(params, dod, x, 0.5, key=subkey2)
    assert loss_eval.shape == ()

    # Test sampling
    key, subkey = jr.split(key)
    final_sample = tas.sample(params, dod, 0.5, 100, key=subkey)
    assert final_sample.shape == (d,)
