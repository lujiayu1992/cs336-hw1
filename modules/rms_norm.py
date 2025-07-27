# In a file named rmsnorm.py

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F, init
import math
from einops import reduce, einsum


class RMSNorm(nn.Module):
    """
    A custom implementation of Root Mean Square Layer Normalization,
    [cite_start]as specified in the assignment document[cite: 583].
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Constructs the RMSNorm module.

        Args:
            d_model: The dimensionality of the input tensor.
            eps: A small value added to the denominator for numerical stability.
            device: The device to store parameters on.
            dtype: The data type of the parameters.
        """
        super().__init__()
        # Your implementation for initializing the learnable gain parameter (self.weight)
        # [cite_start]to all ones goes here[cite: 517].
        self.eps = eps
        self.d_model = d_model
        factory_kwargs = {"device": device, "dtype": dtype}
        self.gain = nn.Parameter(torch.ones(d_model, **factory_kwargs))

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies RMSNorm to the input tensor.

        Args:
            x: The input tensor, typically of shape (..., d_model).

        Returns:
            The normalized tensor of the same shape as the input.
        """
        # Your implementation for the forward pass goes here.
        # Remember to upcast the input to float32 for the calculation and then
        # [cite_start]downcast the result back to the original dtype [cite: 590-598].
        in_dtype = x.dtype
        x = x.to(torch.float32)
        x_sq_sum = reduce(x**2, " ... d_model -> ... 1", "mean")
        rms = torch.sqrt(x_sq_sum + self.eps)
        rms_norm = x * self.gain / rms
        return rms_norm.to(in_dtype)
