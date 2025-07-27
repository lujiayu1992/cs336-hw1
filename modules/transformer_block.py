# In a file named transformer.py

import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional

# Assume these are your custom-built modules from previous steps
from .embedding import Embedding
from .attention import CausalMultiHeadSelfAttention
from .swiglu import SwiGLU
from .rms_norm import RMSNorm


class TransformerBlock(nn.Module):
    """
    A custom implementation of a pre-norm Transformer block,
    [cite_start]as specified in the assignment document [cite: 752-756].
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Constructs the TransformerBlock module.

        Args:
            d_model: The dimensionality of the model's embeddings.
            num_heads: The number of attention heads.
            d_ff: The dimensionality of the feed-forward inner layer.
            max_seq_len: The maximum sequence length for RoPE.
            theta: The theta parameter for RoPE.
        """
        super().__init__()
        # Your implementation for initializing the CausalMultiHeadSelfAttention,
        # SwiGLU, and two RMSNorm layers goes here.
        factory_kwargs = {"device": device, "dtype": dtype}
        self.multihead_self_attention = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            **factory_kwargs,
        )
        self.swiglu = SwiGLU(d_model=d_model, d_ff=d_ff, **factory_kwargs)
        self.rms_norm_1 = RMSNorm(d_model=d_model, **factory_kwargs)
        self.rms_norm_2 = RMSNorm(d_model=d_model, **factory_kwargs)

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs the forward pass for the Transformer block.

        Args:
            x: The input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
            The output tensor of the same shape as the input.
        """
        # Your implementation of the pre-norm block logic goes here:
        # 1. First residual connection: x = x + Attention(RMSNorm(x))
        # 2. Second residual connection: x = x + FFN(RMSNorm(x))
        seq_len = x.shape[-2]
        token_positions = torch.arange(seq_len, device=x.device)
        atten = x + self.multihead_self_attention(
            x=self.rms_norm_1(x), token_positions=token_positions
        )
        y = atten + self.swiglu(self.rms_norm_2(atten))
        return y
