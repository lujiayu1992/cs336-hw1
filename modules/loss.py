# In a file named loss.py

import torch
from torch import Tensor
from .softmax import softmax
from einops import reduce


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """
    Computes the average cross-entropy loss.

    Your implementation should be numerically stable and handle inputs with
    arbitrary batch dimensions.

    Args:
        inputs: A tensor of shape (..., vocab_size) containing the raw,
                unnormalized logits for each example.
        targets: A tensor of shape (...) containing the integer class indices
                 for each example.

    Returns:
        A scalar tensor representing the average cross-entropy loss.
    """
    # Your implementation for the numerically stable cross-entropy function goes here.
    # This should include:
    # 1. Subtracting the maximum logit for numerical stability.
    # 2. Calculating the log-sum-exp of the logits.
    # 3. Selecting the target logits and calculating the final loss.
    # 4. Averaging the loss across all examples.
    # Do not use torch.nn.functional.cross_entropy.
    max_logits = reduce(inputs, "... vocab_size -> ... ()", "max")
    stable_logits = inputs - max_logits
    log_sum_exp = torch.log(torch.sum(torch.exp(stable_logits), dim=-1, keepdim=True))
    target_logits = torch.gather(stable_logits, dim=-1, index=targets.unsqueeze(-1))
    log_probs = target_logits - log_sum_exp
    return -log_probs.mean()
