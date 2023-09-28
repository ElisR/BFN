import pytest
import jax.numpy as jnp
import jax.random as jr
import optax

import bfn.discrete.example_data as example_data
import bfn.discrete.training as training
import bfn.discrete.loss_and_sample as las
import bfn.discrete.models as models

@pytest.fixture(name="data")
def fixture_tokenized_strings() -> example_data.StringDataset:
    reference_string = "examplestring"
    return example_data.StringDataset(reference_string, 100, 0.1)


@pytest.mark.parametrize(("num_epochs",), [(20,)])
def test_basic_training(data: example_data.StringDataset, num_epochs: int):
    model = models.DiscreteOutputDistribution(data.num_cats, (data.d,), models.MultipleMLP(data.num_cats))
    thetas_prior = jnp.ones((data.num_cats, data.d)) / data.num_cats

    variables = model.init(jr.PRNGKey(0), thetas_prior, 1.0)
    params = variables["params"]

    optim = optax.adam(1e-3)
    opt_state = optim.init(params)

    losses = []
    key = jr.PRNGKey(0)
    beta = 1.0
    for _ in range(num_epochs):
        epoch_loss = 0.0
        for x in data:
            key, subkey = jr.split(key)
            x = jnp.expand_dims(x, 0)
            loss, params, opt_state = training.make_step(model, x, optim, opt_state, params, beta, key=subkey)
            epoch_loss += loss
        losses.append(epoch_loss)

    assert len(losses) == num_epochs

    output, _ = las.sample(params, model, 1.0, 10, key=key)
    output_string = example_data.detokenize_string(output)
    assert len(output_string) == data.d
