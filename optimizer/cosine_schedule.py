# In a file named cosine_schedule.py
import math


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Calculates the learning rate at a given iteration based on a cosine
    schedule with a linear warmup phase.

    This function implements the three stages described in the assignment:
    1. Linear Warmup: Linearly increases the learning rate from 0 to max_learning_rate.
    2. Cosine Annealing: Decays the learning rate from max_learning_rate to min_learning_rate.
    3. Post-annealing: Keeps the learning rate constant at min_learning_rate.

    Args:
        it: The current iteration number.
        max_learning_rate: The maximum learning rate (alpha_max).
        min_learning_rate: The minimum learning rate (alpha_min).
        warmup_iters: The number of warmup iterations (T_w).
        cosine_cycle_iters: The number of cosine annealing iterations (T_c).

    Returns:
        The learning rate for the given iteration.
    """
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    elif it > cosine_cycle_iters:
        return min_learning_rate
    else:
        return min_learning_rate + 0.5 * (
            1
            + math.cos(
                (it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi
            )
        ) * (max_learning_rate - min_learning_rate)
