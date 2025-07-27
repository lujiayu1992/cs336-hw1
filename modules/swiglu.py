# In a file named swiglu.py

import torch
from torch import Tensor
import torch.nn as nn
from .silu import silu
from .linear import Linear


class SwiGLU(nn.Module):
    """
    A custom implementation of the SwiGLU feed-forward network,
    [cite_start]as specified in the assignment document [cite: 638-641].
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Constructs the SwiGLU module.

        Args:
            d_model: The dimensionality of the input and output.
            d_ff: The dimensionality of the inner feed-forward layer.
            device: The device to store parameters on.
            dtype: The data type of the parameters.
        """
        super().__init__()
        self.w1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies the SwiGLU transformation to the input tensor.
        The formula is: W2(SiLU(W1(x)) * W3(x))

        Args:
            x: The input tensor, typically of shape (..., d_model).

        Returns:
            The output tensor of the same shape as the input.
        """
        # Your implementation for the forward pass goes here.
        # [cite_start]You may use torch.sigmoid for the SiLU activation[cite: 651].
        return self.w2(silu(self.w1(x))*self.w3(x))
