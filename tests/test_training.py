import pytest
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Int, Array

import bfn.example_data as example_data
import bfn.training as training
import bfn.train_and_sample as tas

@pytest.fixture(name="tokenized_strings")
def fixture_tokenized_strings():
    reference_string = "example string"
    corrupted_strings = example_data.corrupt_string(reference_string, 100, 0.1)
    tokenized_strings = [example_data.tokenize_string(string) for string in corrupted_strings]
    return tokenized_strings


@pytest.mark.parametrize("num_epochs", [50])
def test_basic_training(tokenized_strings: list[Int[Array, "D"]], num_epochs: int):
    num_cats = 27
    d = len(tokenized_strings[0])

    model = tas.DiscreteOutputDistribution(num_cats, d)
    thetas_prior = jnp.ones((num_cats, d)) / num_cats

    variables = model.init(jr.PRNGKey(0), thetas_prior, 1.0)
    params = variables["params"]

    optim = optax.adam(1e-3)
    opt_state = optim.init(params)

    losses = []
    key = jr.PRNGKey(0)
    beta = 1.0
    for _ in range(num_epochs):
        epoch_loss = 0.0
        for x in tokenized_strings:
            key, subkey = jr.split(key)
            loss, params, opt_state = training.make_step(model, x, optim, opt_state, params, beta, key=subkey)
            epoch_loss += loss
        losses.append(epoch_loss)

    assert len(losses) == num_epochs

    output = tas.sample(params, model, 1.0, 10, key=key)
    output_string = example_data.detokenize_string(output)
    assert len(output_string) == d
