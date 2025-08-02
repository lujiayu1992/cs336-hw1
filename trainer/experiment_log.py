"""
Experiment tracking and management module.

This module provides tools for managing and recording deep learning experiments,
tracking validation and training losses, and recording experimental configurations
and results. Supports both local logging and Weights & Biases integration.
"""

import os
import time
import json
import logging
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import torch

import numpy as np
import wandb

# Configure logging
logger = logging.getLogger(__name__)


class ExperimentLogger:
    """
    Experiment tracker for recording and managing training experiments.

    Supports both local file and Weights & Biases recording modes.
    """

    def __init__(
        self,
        config: Dict[str, Any],
    ):
        """
        Initialize the experiment tracker.

        Args:
            experiment_name: Name of the experiment
            base_dir: Base directory for saving experiment data
            use_wandb: Whether to use Weights & Biases
            wandb_project: W&B project name
            wandb_entity: W&B entity (organization or username)
            config: Experiment configuration parameters
            tags: Experiment tags
            resume: Whether to resume a previous experiment
            wandb_id: W&B run ID to resume
        """
        self.start_time = time.time()
        if config["training"]["wandb"]:
            wandb.init(
                project=config["training"]["wandb_project"],
                name=config["training"]["wandb_run"],
                config=config,
            )

    def log_loss(self, step: int, loss: torch.Tensor, lr: float, is_train: bool):
        """
        Logs training or validation loss and associated metrics to W&B.

        Args:
            step (int): The current training step.
            loss (torch.Tensor): The loss value for the current step.
            lr (float): The current learning rate.
            is_train (bool): True if logging a training step, False for validation.
        """
        elapsed_time = time.time() - self.start_time
        if is_train:
            # For a training step, log the loss and the learning rate
            log_data = {
                "train/loss": loss.item(),
                "train/lr": lr,
                "time/wall_clock_time_sec": elapsed_time,
            }
        else:
            # For a validation step, only log the loss
            log_data = {
                "val/loss": loss.item(),
                "time/wall_clock_time_sec": elapsed_time,
            }

        # The 'step' argument ensures the x-axis of your plots is the training step
        wandb.log(log_data, step=step)
