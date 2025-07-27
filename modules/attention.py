# In a file named attention.py

import torch
from torch import Tensor
import torch.nn as nn
from .linear import Linear
from .rope import RotaryPositionalEmbedding
from typing import Optional
from einops import einsum, rearrange
from .scaled_dot_product_attention import scaled_dot_product_attention


class CausalMultiHeadSelfAttention(nn.Module):
    """
    A custom implementation of causal multi-head self-attention with RoPE,
    [cite_start]as specified in the assignment document [cite: 723-732].
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Constructs the CausalMultiHeadSelfAttention module.

        Args:
            d_model: The dimensionality of the model's embeddings.
            num_heads: The number of attention heads.
            max_seq_len: The maximum sequence length for RoPE pre-computation.
            theta: The theta parameter for RoPE.
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.device = device
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Your implementation for initializing the Q, K, V, and output projection
        # Linear layers goes here. It's recommended to combine Q, K, V into a single
        # large linear layer for efficiency.
        # self.qkv_proj = Linear(...)
        # self.o_proj = Linear(...)

        # Your implementation for initializing the RoPE module goes here.
        # self.rope = RotaryPositionalEmbedding(...)
        self.Q = Linear(d_model, d_model, **factory_kwargs)
        self.K = Linear(d_model, d_model, **factory_kwargs)
        self.V = Linear(d_model, d_model, **factory_kwargs)
        self.O = Linear(d_model, d_model, **factory_kwargs)
        self.d_k = d_model // num_heads

        self.rope = RotaryPositionalEmbedding(
            d_k=d_model / num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            **factory_kwargs,
        )
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len, **factory_kwargs)).bool(),
            persistent=False,
        )

    def forward(self, x: Tensor, token_positions: Optional[Tensor] = None) -> Tensor:
        """
        Performs the forward pass for causal multi-head self-attention.

        Args:
            x: The input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
            The output tensor of the same shape as the input.
        """
        # Your implementation for the forward pass goes here.
        # This should include:
        # 1. Projecting the input to get Q, K, and V for all heads.
        # 2. Reshaping Q, K, V to have a separate 'num_heads' dimension.
        # 3. Applying RoPE to the Q and K tensors.
        # 4. Calling your scaled_dot_product_attention function with a causal mask.
        # 5. Reshaping the output and applying the final output projection.
        query = self.Q(x)
        key = self.K(x)
        value = self.V(x)

        query = rearrange(
            query, " ... T (n_head d_k) ->... n_head T d_k", n_head=self.num_heads
        )
        key = rearrange(
            key, " ... T (n_head d_k) ->... n_head T d_k", n_head=self.num_heads
        )
        value = rearrange(
            value, " ... T (n_head d_k) ->... n_head T d_k", n_head=self.num_heads
        )

        if token_positions is not None:
            query = self.rope(query, token_positions)
            key = self.rope(key, token_positions)
        T = x.shape[-2]
        attention = scaled_dot_product_attention(
            query, key, value, self.causal_mask[:T, :T]
        )
        return self.O(rearrange(attention, "... n_head T d_v -> ... T (n_head d_v)"))
