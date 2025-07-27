# In a file named rope.py

import torch
from torch import Tensor
import torch.nn as nn
from einops import einsum, rearrange


class RotaryPositionalEmbedding(nn.Module):
    """
    A custom implementation of Rotary Position Embeddings (RoPE),
    [cite_start]as specified in the assignment document [cite: 656-659].
    """

    def __init__(
        self,
        d_k: int,
        max_seq_len: int,
        theta: float = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Constructs the RoPE module.

        Args:
            d_k: The dimensionality of the query and key vectors.
            max_seq_len: The maximum sequence length that the model will process.
            theta: The theta parameter for RoPE.
            device: The device to store the pre-computed buffers on.
        """
        super().__init__()
        # Your implementation for pre-computing the sine and cosine values goes here.
        # It's recommended to store these in non-learnable buffers using
        # self.register_buffer("sin_cached", ...)
        # self.register_buffer("cos_cached", ...)
        factory_kwargs = {'device': device, 'dtype': dtype}
        if d_k % 2 != 0:
            raise ValueError("d_k must be even.")
        dim = torch.arange(1, d_k/2+1, **factory_kwargs)
        sequence = torch.arange(max_seq_len, **factory_kwargs)
        theta_i_k = einsum(sequence, theta ** (-2 * (dim-1) / d_k), "i,k->i k")
        self.register_buffer(
            "rope_cache",
            torch.stack((torch.cos(theta_i_k), torch.sin(theta_i_k)), dim=-1),
            persistent=False,
        )

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        """
        Applies RoPE to the input tensor.

        Args:
            x: The input tensor (queries or keys) of shape (..., seq_len, d_k).
            token_positions: A tensor of shape (seq_len,) specifying the absolute
                             positions of the tokens in the sequence.

        Returns:
            The transformed tensor of the same shape as the input.
        """
        # Your implementation for the forward pass goes here.
        # This involves:
        # 1. Slicing the pre-computed sin/cos buffers using the token_positions.
        # 2. Applying the pairwise rotation to the input tensor x.
        x_complex = torch.view_as_complex(
            rearrange(x, "... (d_k two) -> ... d_k two", two=2)
        )
        r_complex = torch.view_as_complex(self.rope_cache[token_positions])
        product_complex = x_complex* r_complex
        product_real = torch.view_as_real(product_complex)
        product = rearrange(product_real,"... d_k two -> ...(d_k two)", two=2)
        return product
