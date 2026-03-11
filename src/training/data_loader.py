"""
Converts replay buffer samples into PyTorch tensors for training.
"""

import torch
import numpy as np

from .replay_buffer import TrainingExample


def examples_to_tensors(
    examples: list[TrainingExample],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a batch of training examples into GPU-ready tensors.

    Returns:
        states:  [batch, 1008]
        policies: [batch, 15876]
        values:   [batch]
    """
    states = np.array([ex.state for ex in examples], dtype=np.float32)
    policies = np.array([ex.policy for ex in examples], dtype=np.float32)
    values = np.array([ex.value for ex in examples], dtype=np.float32)

    return (
        torch.from_numpy(states).to(device),
        torch.from_numpy(policies).to(device),
        torch.from_numpy(values).to(device),
    )
