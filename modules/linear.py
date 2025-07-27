import torch
from torch import Tensor
import torch.nn as nn
from einops import einsum
from torch.nn import functional as F, init
import math


class Linear(nn.Module):
    """
    A custom implementation of a linear transformation layer without a bias term,
    [cite_start]as specified in the assignment document [cite: 521-523].
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Constructs the linear transformation module.

        Args:
            in_features: The number of input features.
            out_features: The number of output features.
            device: The device to store the parameters on.
            dtype: The data type of the parameters.
        """
        super().__init__()
        # Your implementation for initializing the weight parameter (self.W) goes here,
        # [cite_start]making sure to use nn.Parameter and the specified initialization scheme[cite: 515, 541, 543].
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.parameter.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        self.in_features = in_features
        self.out_features = out_features
        self._reset_parameters()
        
    def _reset_parameters(self) -> None:
        # Initialize the weight parameter using the Kaiming uniform distribution
        # init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # In ASI, we use truncated normal distribution
        trunc_normal_std = math.sqrt(2 / (self.in_features + self.out_features))
        
        init.trunc_normal_(
            self.weight, 
            mean = 0, 
            std = trunc_normal_std,
            a = -3 * trunc_normal_std,
            b = 3 * trunc_normal_std
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies the linear transformation to the input tensor.

        Args:
            x: The input tensor of shape (..., in_features).

        Returns:
            The output tensor of shape (..., out_features).
        """
        # Your implementation for the forward pass (y = xW^T) goes here.
        # [cite_start]Do not use nn.Linear or nn.functional.linear[cite: 542].
        return einsum( x, self.weight," ... in_d,  out_d in_d -> ... out_d")
