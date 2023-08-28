"""Example data for testing BFNs ability to learn discrete data."""
import random


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
