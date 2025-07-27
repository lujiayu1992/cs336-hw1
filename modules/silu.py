# In a file named silu.py

from torch import Tensor
import torch

def silu(x: Tensor) -> Tensor:
    """
    Applies the Sigmoid Linear Unit (SiLU) activation function to the input tensor.
    The formula is: x * sigmoid(x).
    
    This implementation should not use nn.functional.silu. You may, however,
    [cite_start]use torch.sigmoid for numerical stability[cite: 651].
    
    Args:
        x: The input tensor of any shape.
        
    Returns:
        The output tensor of the same shape as the input.
    """
    return x * torch.sigmoid(x)