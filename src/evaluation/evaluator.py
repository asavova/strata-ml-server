"""
Position evaluator — wraps the neural network for inference.
"""

import logging
from typing import Optional

import torch
import numpy as np

from ..network import StrataNetwork

logger = logging.getLogger("strata.evaluation")


class Evaluator:
    """
    Thread-safe position evaluator.

    Handles:
    - Single position evaluation
    - Batch evaluation for MCTS
    - Automatic device management
    """

    def __init__(self, network: StrataNetwork, device: torch.device):
        self.network = network
        self.device = device
        self.eval_count = 0

    @torch.no_grad()
    def evaluate(
        self,
        state: list[float],
        legal_mask: Optional[list[float]] = None,
    ) -> tuple[float, list[float]]:
        """
        Evaluate a single position.

        Args:
            state:      1008-float encoded board state
            legal_mask: 15876-float binary mask (optional)

        Returns:
            (value, policy) tuple
        """
        self.network.eval()
        self.eval_count += 1

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mask_tensor = None
        if legal_mask is not None:
            mask_tensor = torch.FloatTensor(legal_mask).unsqueeze(0).to(self.device)

        policy, value = self.network(state_tensor, mask_tensor)

        return (
            value.item(),
            policy.squeeze(0).cpu().tolist(),
        )

    @torch.no_grad()
    def evaluate_batch(
        self,
        states: list[list[float]],
        legal_masks: Optional[list[list[float]]] = None,
    ) -> list[tuple[float, list[float]]]:
        """
        Evaluate multiple positions in a single forward pass.

        Significantly faster than calling evaluate() in a loop.
        """
        self.network.eval()
        self.eval_count += len(states)

        state_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        mask_tensor = None
        if legal_masks is not None:
            mask_tensor = torch.FloatTensor(np.array(legal_masks)).to(self.device)

        policies, values = self.network(state_tensor, mask_tensor)

        results = []
        for i in range(len(states)):
            results.append((
                values[i].item(),
                policies[i].cpu().tolist(),
            ))

        return results

    def stats(self) -> dict:
        return {"total_evaluations": self.eval_count}
