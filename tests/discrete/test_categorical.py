import jax.numpy as jnp
import jax.random as jr
import pytest

import bfn.discrete.loss_and_sample as las
import bfn.discrete.models as models


@pytest.mark.parametrize(("k", "d"), [(5, 10), (10, 5), (1, 1)])
def test_discrete_output_distribution(k: int, d: int):
    dod = models.DiscreteOutputDistribution(k, (d,), models.MultipleMLP(k))

    # Test distribution
    key, subkey1, subkey2 = jr.split(jr.PRNGKey(0), 3)
    thetas_example = jr.uniform(subkey1, (k, d))
    t = jnp.array(0.5, dtype=jnp.float32)
    variables = dod.init(subkey2, thetas_example, t)
    params = variables["params"]
    out = dod.apply(variables, thetas_example, t)

    assert isinstance(out, jnp.ndarray)
    assert out.shape == (k, d)

    # Test loss
    key, subkey1, subkey2 = jr.split(key, 3)
    x = jr.randint(subkey1, shape=(d,), minval=0, maxval=k)
    loss_eval = las.loss(params, dod, x, 0.5, key=subkey2)
    assert loss_eval.shape == ()

    # Test sampling
    key, subkey = jr.split(key)
    steps = 10
    final_sample, thetas_output = las.sample(params, dod, 0.5, steps, key=subkey)
    assert final_sample.shape == (d,)
    assert thetas_output.shape[0] == steps
