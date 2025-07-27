# In a file named checkpointing.py

import torch
import os
from typing import IO, BinaryIO


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Serializes a model, optimizer, and iteration number to a file or file-like object.

    This implementation should follow the requirements on page 36 of the assignment PDF,
    using `state_dict()` for the model and optimizer and `torch.save()` to write
    the data.

    Args:
        model: The model to serialize.
        optimizer: The optimizer to serialize.
        iteration: The current training iteration number.
        out: The path or file-like object to write the checkpoint to.
    """
    # Your implementation for saving the checkpoint goes here.
    # This should involve:
    # 1. Creating a dictionary containing the model's state_dict, the optimizer's
    #    state_dict, and the iteration number.
    # 2. Using torch.save() to write this dictionary to the `out` path/object.
    state = {
        "iteration": iteration,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Loads a checkpoint from a file or file-like object into a model and optimizer.

    This implementation should follow the requirements on page 36 of the assignment PDF,
    using `torch.load()` to read the data and `load_state_dict()` to restore the
    states.

    Args:
        src: The path or file-like object to read the checkpoint from.
        model: The model to restore the state into.
        optimizer: The optimizer to restore the state into.

    Returns:
        The iteration number that was saved in the checkpoint.
    """
    # Your implementation for loading the checkpoint goes here.
    # This should involve:
    # 1. Using torch.load() to load the dictionary from the `src` path/object.
    # 2. Calling model.load_state_dict() with the saved model state.
    # 3. Calling optimizer.load_state_dict() with the saved optimizer state.
    # 4. Returning the saved iteration number.
    state = torch.load(src)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    return state["iteration"]
