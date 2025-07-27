# In a file named gradient_clipping.py

import torch
from torch import Tensor
from typing import Iterable


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float
) -> None:
    """
    Clips the gradients of a collection of parameters in-place.

    The function computes the total L2 norm of all gradients combined and,
    if it exceeds max_l2_norm, scales all gradients down to have a total
    norm of max_l2_norm.

    Args:
        parameters: An iterable of model parameters (e.g., from model.parameters()).
        max_l2_norm: The maximum allowed L2 norm for the combined gradients.
    """
    # Your implementation for gradient clipping goes here.
    # This should include:
    # 1. Calculating the total L2 norm of all gradients from the `parameters` iterable.
    # 2. Checking if the total norm exceeds `max_l2_norm`.
    # 3. If it does, calculating the scaling factor.
    # 4. Iterating through the parameters again and scaling their gradients in-place
    #    (e.g., using `p.grad.mul_(clip_coef)`).
    eps = 1e-6
    grads = [param.grad for param in parameters if param.grad is not None]
    if len(grads) == 0:
        return
    l2_norm = torch.sqrt(sum((grad.detach() ** 2).sum() for grad in grads))
    if l2_norm > max_l2_norm:
        scaling_factor = max_l2_norm/(l2_norm+eps)
        for grad in grads:
            grad.mul_(scaling_factor)