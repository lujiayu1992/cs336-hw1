# In a file named embedding.py

import torch
from torch import Tensor
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F, init


class Embedding(nn.Module):
    """
    A custom implementation of an embedding layer that maps token IDs to dense vectors,
    [cite_start]as specified in the assignment document [cite: 548-549].
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Constructs the embedding module.

        Args:
            num_embeddings: The size of the vocabulary (total number of possible tokens).
            embedding_dim: The dimensionality of the embedding vectors (d_model).
            device: The device to store the parameters on.
            dtype: The data type of the parameters.
        """
        super().__init__()
        # Your implementation for initializing the embedding matrix goes here.
        # [cite_start]Make sure to use nn.Parameter and the specified initialization scheme[cite: 567, 570].
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))
        init.trunc_normal_(
            self.weight, 
            mean = 0,
            std = 1,
            a = -3,
            b = 3
        )
        self.d_model = embedding_dim
        
        

    def forward(self, token_ids: Tensor) -> Tensor:
        """
        Looks up the embedding vectors for the given token IDs.

        Args:
            token_ids: A tensor of integer token IDs, typically of shape (batch_size, sequence_length).

        Returns:
            A tensor of the corresponding embedding vectors, of shape (..., embedding_dim).
        """
        return self.weight[token_ids]
        
