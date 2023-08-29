"""Example data for testing BFNs ability to learn discrete data."""
import random
from jaxtyping import Int, Array
import jax.numpy as jnp
from torch.utils.data import Dataset


def corrupt_string(reference: str, length: int, probability: float) -> list[str]:
    """Corrupts a string by randomly replacing characters with random lowercase letters.

    Args:
        reference: The string to be corrupted.
        length: The number of corrupted strings to be generated.
        probability: The probability of a character being replaced.

    Returns:
        A list of corrupted strings.
    """
    ASCII_LOWER_START, ASCII_LOWER_END = 97, 122
    output_list = []

    for _ in range(length):
        modified_string = [
            chr(random.randint(ASCII_LOWER_START, ASCII_LOWER_END))
            if char.islower() and random.random() < probability
            else char
            for char in reference
        ]
        output_list.append("".join(modified_string))

    return output_list


class StringDataset(Dataset):
    """Dataset of strings."""

    def __init__(self, reference: str, length: int, probability: float):
        """Initialise the dataset.

        Args:
            reference: The reference string.
            length: The number of corrupted strings to be generated.
            probability: The probability of a character being replaced.
        """
        self.reference = reference
        self.corrupted_strings = corrupt_string(reference, length, probability)
        self.num_cats = 27
        self.d = len(self.corrupted_strings[0])

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.corrupted_strings)

    def __getitem__(self, idx: int) -> Int[Array, "D"]:
        """Return the corrupted string at the given index."""
        return tokenize_string(self.corrupted_strings[idx])

    def get_raw(self, idx: int) -> str:
        """Return the raw string at the given index."""
        return self.corrupted_strings[idx]


def tokenize_string(reference: str) -> Int[Array, "D"]:
    """Tokenizes a string by converting each character to its ASCII code.

    Resets all non-lowercase characters to 0 and subtracts 96 from lowercase characters.
    Thus a is 1, b is 2, ..., z is 26.

    Args:
        reference: The string to be tokenized.

    Returns:
        Array of integers representing the string.
    """
    chars = [ord(char) for char in reference]
    fixed_chars = [(char - 96 if char >= 97 and char < 122 else 0) for char in chars]
    return jnp.array(fixed_chars, dtype=jnp.int32)


def detokenize_string(tokenized_string: Int[Array, "D"]) -> str:
    """Detokenizes a string by converting each token to its ASCII character.

    Args:
        tokenized_string: The string to be detokenized.

    Returns:
        The detokenized string.
    """
    chars = [(chr(char + 96) if char != 0 else " ") for char in tokenized_string]
    return "".join(chars)
