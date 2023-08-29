import jax.numpy as jnp
import pytest

from bfn.example_data import detokenize_string, tokenize_string


@pytest.mark.parametrize("input_string, expected_output", [
    ("hello world", jnp.array([8, 5, 12, 12, 15, 0, 23, 15, 18, 12, 4], dtype=jnp.int32)),
    ("", jnp.array([], dtype=jnp.int32)),
])
def test_detokenize_string(input_string, expected_output):
    tokenized_string = tokenize_string(input_string)
    assert jnp.all(tokenized_string == expected_output)

    output = detokenize_string(tokenized_string)
    assert jnp.all(output == input_string)
