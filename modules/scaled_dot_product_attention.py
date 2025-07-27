# In a file named attention.py

import torch
from torch import Tensor
from typing import Optional
from .softmax import softmax
from einops import einsum
import math


def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Computes the scaled dot-product attention.

    Your implementation should support an optional boolean mask. For positions
    where the mask is False, the attention probabilities should be zero.

    Args:
        Q: The query tensor of shape (..., num_queries, d_k).
        K: The key tensor of shape (..., num_keys, d_k).
        V: The value tensor of shape (..., num_keys, d_v).
        mask: An optional boolean mask of shape (..., num_queries, num_keys).

    Returns:
        The output tensor of shape (..., num_queries, d_v).
    """
    # Your implementation for scaled dot-product attention goes here.
    # This should include:
    # 1. The matrix multiplication of Q and K^T.
    # 2. Scaling by the square root of d_k.
    # 3. Applying the mask (if provided) by setting masked positions to -infinity
    #    before the softmax.
    # 4. Applying the softmax function (your custom one).
    # 5. The matrix multiplication with V.
    d_k = Q.shape[-1]
    product = einsum(
        Q, K, "...  num_queries  d_k, ... num_keys d_k-> ... num_queries num_keys"
    ) / math.sqrt(d_k)
    if mask is not None:
        product = product.masked_fill(~mask, -torch.inf)
    scaled_attention = softmax(product, -1)
    return einsum(
        scaled_attention,
        V,
        "... num_queries num_keys, ... num_keys d_v -> ... num_queries d_v",
    )
