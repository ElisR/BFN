"""Module containing some useful functions for generating example data for training."""
import jax.numpy as jnp
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset

from jaxtyping import Int, Array


class MNISTDataset(Dataset):
    """Dataset of MNIST images, with pixels normalised to [-1.0, 1.0]."""

    def __init__(self) -> None:
        def transform(pic):
            """Function for transforming PIL image to JAX array."""
            normalised = jnp.ravel(jnp.array(pic, dtype=jnp.float32)) / 255.0
            return 2 * normalised - 1

        self.mnist = MNIST(root='./data', train=False, download=True, transform=transform)
        self.d = len(self.mnist[0][0])

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.mnist)

    def __getitem__(self, idx: int) -> Int[Array, "D"]:
        """Return the corrupted string at the given index."""
        return self.mnist[idx][0]
