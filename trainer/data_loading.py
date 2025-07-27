# In a file named data_loading.py

import torch
import numpy as np
from torch import Tensor
from numpy.typing import NDArray


def get_batch(
    dataset: NDArray, batch_size: int, context_length: int, device: str
) -> tuple[Tensor, Tensor]:
    """
    Samples a batch of language modeling inputs and their corresponding targets
    from a tokenized dataset.

    Args:
        dataset: A 1D numpy array containing the sequence of token IDs.
        batch_size: The number of sequences in a single batch.
        context_length: The length of each input sequence.
        device: The PyTorch device ('cpu', 'cuda', etc.) to move the tensors to.

    Returns:
        A tuple containing two tensors:
        - The input sequences of shape (batch_size, context_length).
        - The target sequences of shape (batch_size, context_length).
    """
    # Your implementation for the data loading function goes here.
    # This should include:
    # 1. Randomly selecting `batch_size` starting indices from the dataset.
    # 2. Slicing the dataset to create the input sequences (x).
    # 3. Slicing the dataset to create the target sequences (y), which are
    #    offset by one token from the inputs.
    # 4. Converting both from numpy arrays to torch tensors.
    # 5. Moving the tensors to the specified `device`.
    total_number_of_tokens = dataset.size
    ix = torch.randint(
        total_number_of_tokens - context_length,
        (batch_size,)
    )
    x = torch.stack(
        [
            torch.from_numpy((dataset[i : i + context_length]).astype(np.int64))
            for i in ix
        ]
    )
    y = torch.stack(
        [
            torch.from_numpy((dataset[i + 1 : i + 1 + context_length]).astype(np.int64))
            for i in ix
        ]
    )

    x = x.to(device=device)
    y = y.to(device=device)

    return x, y
