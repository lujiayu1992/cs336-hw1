# In a file named softmax.py

import torch
from torch import Tensor


def softmax(x: Tensor, dim: int) -> Tensor:
    """
    Applies the softmax operation to a tensor along a given dimension.

    Your implementation should be numerically stable.

    Args:
        x: The input tensor of any shape.
        dim: The dimension along which to apply softmax.

    Returns:
        A tensor of the same shape as the input, with the specified
        dimension normalized into a probability distribution.
    """
    # Your implementation for the numerically stable softmax function goes here.
    # Do not use torch.nn.functional.softmax.
    max_values, _ = torch.max(x, dim=dim, keepdim=True)
    stable_x = x - max_values
    numerator = torch.exp(stable_x)
    denominator = torch.sum(numerator, dim=dim, keepdim=True)
    return numerator / denominator
