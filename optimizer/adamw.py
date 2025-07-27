# In a file named adamw.py

import torch
from torch.optim import Optimizer
from typing import Optional, Callable, Iterable
import math


class AdamW(Optimizer):
    """
    A custom implementation of the AdamW optimizer, as specified in the
    [cite_start]assignment document[cite: 907, 911].
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        """
        Constructs the AdamW optimizer.

        Args:
            params: An iterable of parameters to optimize.
            lr: The learning rate (alpha).
            betas: A tuple containing beta1 and beta2.
            eps: A small value for numerical stability.
            weight_decay: The weight decay rate (lambda).
        """
        # Your implementation for the constructor goes here.
        # This should include:
        # 1. Checks for valid hyperparameter values (e.g., betas in [0, 1]).
        # 2. Creating a `defaults` dictionary with all the hyperparameters.
        # 3. Calling the superclass constructor: super().__init__(params, defaults).
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        """
        Performs a single optimization step.

        Args:
            closure (optional): A closure that re-evaluates the model
                                and returns the loss.
        """
        # Your implementation for the optimization step goes here.
        # This should follow Algorithm 1 on page 32 of the PDF, including:
        # 1. Initializing state for the moments (m and v) and the step (t).
        # 2. Updating the first and second moment estimates.
        # 3. Computing the bias-corrected learning rate for the current step.
        # 4. Updating the parameters.
        # 5. Applying the decoupled weight decay.
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if "t" not in state:
                    state["t"] = 1
                t = state["t"]
                first_moment = state.get("first_moment", torch.zeros_like(p.data))
                second_moment = state.get("second_moment", torch.zeros_like(p.data))
                first_moment = beta1 * first_moment + (1 - beta1) * grad
                second_moment = beta2 * second_moment + (1 - beta2) * grad**2
                state["first_moment"] = first_moment
                state["second_moment"] = second_moment

                current_lr = math.sqrt(1 - beta2**t) / (1 - beta1**t) * lr
                p.data.addcdiv_(
                    first_moment, torch.sqrt(second_moment) + eps, value=-current_lr
                )
                p.data.add_(p.data, alpha=-lr * weight_decay)
                state["t"] += 1

        return loss
