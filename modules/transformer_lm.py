# In a file named transformer.py

import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional

# Assume these are your custom-built modules from previous steps
from .embedding import Embedding
from .transformer_block import (
    TransformerBlock,
)  # Note: Corrected the typo from your file list
from .rms_norm import RMSNorm
from .linear import Linear
from .softmax import softmax


class TransformerLM(nn.Module):
    """
    A custom implementation of a full Transformer Language Model,
    [cite_start]as specified in the assignment document [cite: 765-767].
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Constructs the TransformerLM module.

        Args:
            vocab_size: The size of the vocabulary.
            context_length: The maximum sequence length.
            d_model: The dimensionality of the model's embeddings.
            num_layers: The number of Transformer blocks.
            num_heads: The number of attention heads.
            d_ff: The dimensionality of the feed-forward inner layer.
            rope_theta: The theta parameter for RoPE.
            device: The device to store parameters on.
            dtype: The data type of the parameters.
        """
        super().__init__()
        # Your implementation for initializing the model components goes here:
        # 1. An Embedding layer for token embeddings.
        # 2. An nn.ModuleList containing `num_layers` of your TransformerBlock.
        # 3. A final RMSNorm layer to apply after the last block.
        # 4. A final Linear layer to act as the language model head.
        self.embedding = Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype
        )
        self.transformer_layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        self.rms_norm = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, device=device)

    def forward(self, token_ids: Tensor) -> Tensor:
        """
        Performs the forward pass for the full Transformer Language Model.

        Args:
            token_ids: The input tensor of token IDs, shape (batch_size, sequence_length).

        Returns:
            The output logits tensor of shape (batch_size, sequence_length, vocab_size).
        """
        # Your implementation of the full forward pass goes here:
        # 1. Get token embeddings.
        # 2. Pass the embeddings through all the TransformerBlocks in sequence.
        # 3. Apply the final RMSNorm.
        # 4. Apply the language model head to get the final logits.
        x = self.embedding(token_ids)
        for layer in self.transformer_layers:
            x = layer(x)
        normalized = self.rms_norm(x)
        logits = self.lm_head(normalized)
        return logits
