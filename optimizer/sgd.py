import torch
import math
from torch.optim import Optimizer
from typing import Optional, Callable, Iterable

class SGD(Optimizer):
    """
    A custom implementation of the Stochastic Gradient Descent optimizer with a
    [cite_start]decaying learning rate, as specified in the assignment document [cite: 855-879].
    """
    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 1e-3):
        """
        Constructs the SGD optimizer.

        Args:
            params: An iterable of parameters to optimize.
            lr: The learning rate.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        """
        Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if "t" not in state:
                    state["t"] = 0
                
                t = state["t"]

                # Apply the SGD update rule with decaying learning rate
                p.data.add_(grad, alpha=-lr / math.sqrt(t + 1))

                # Increment step
                state["t"] += 1
        
        return loss

# --- Main execution block for the learning_rate_tuning experiment ---
if __name__ == "__main__":
    
    learning_rates_to_test = [1.0, 1e1, 1e2, 1e3]
    num_iterations = 10

    print("--- Starting Learning Rate Tuning Experiment ---")

    for lr in learning_rates_to_test:
        print(f"\n--- Testing Learning Rate: {lr} ---")
        
        # Initialize weights and optimizer for each run
        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
        opt = SGD([weights], lr=lr)
        
        for t in range(num_iterations):
            opt.zero_grad()
            loss = (weights**2).mean()
            
            # Print the loss at each step to observe its behavior
            print(f"Iteration {t+1:2d}: Loss = {loss.cpu().item():.4f}")
            
            # Check for divergence
            if torch.isinf(loss) or torch.isnan(loss):
                print("Divergence detected!")
                break
                
            loss.backward()
            opt.step()