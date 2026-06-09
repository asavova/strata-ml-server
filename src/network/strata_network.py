"""
Main neural network for Strata.

Dual-headed architecture:
    Input (1008) -> Shared Trunk (residual blocks) -> Policy Head (15876)
                                                   -> Value Head  (1)
"""

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .residual_block import ResidualBlock
from .heads import PolicyHead, ValueHead

logger = logging.getLogger("strata.network")


class StrataNetwork(nn.Module):
    """
    Neural network for position evaluation and move policy prediction.

    Input:  flat float[1008] — 126 nodes * 8 channels
    Output: (policy[15876], value scalar)
    """

    def __init__(
        self,
        state_size: int = 1008,
        hidden_size: int = 512,
        policy_size: int = 15876,
        num_residual_blocks: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.policy_size = policy_size

        # Input projection
        self.input_fc = nn.Linear(state_size, hidden_size)
        self.input_bn = nn.BatchNorm1d(hidden_size)

        # Residual trunk
        self.trunk = nn.Sequential(
            *[ResidualBlock(hidden_size, dropout) for _ in range(num_residual_blocks)]
        )

        # Output heads
        self.policy_head = PolicyHead(hidden_size, policy_size)
        self.value_head = ValueHead(hidden_size)

        # Initialize weights
        self._init_weights()

        param_count = sum(p.numel() for p in self.parameters())
        logger.info(
            f"StrataNetwork created: {num_residual_blocks} residual blocks, "
            f"{param_count:,} parameters"
        )

    def _init_weights(self):
        """Xavier initialization for faster convergence."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        state: torch.Tensor,
        legal_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            state:      [batch, 1008] encoded board state
            legal_mask: [batch, 15876] binary mask (1=legal, 0=illegal)

        Returns:
            policy: [batch, 15876] move probability distribution
            value:  [batch] position evaluation in [-1, 1]
        """
        x = F.relu(self.input_bn(self.input_fc(state)))
        x = self.trunk(x)

        policy = self.policy_head(x, legal_mask)
        value = self.value_head(x)

        return policy, value

    def save(self, path: str, metadata: dict[str, Any] | None = None):
        """Save model weights and optional metadata to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if metadata is None:
            torch.save(self.state_dict(), path)
        else:
            torch.save(
                {
                    "state_dict": self.state_dict(),
                    "metadata": metadata,
                },
                path,
            )
        logger.info(f"Model saved to {path}")

    def load(self, path: str, device: torch.device):
        """Load model weights from disk."""
        if Path(path).exists():
            checkpoint = torch.load(path, map_location=device, weights_only=True)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                self.load_state_dict(checkpoint["state_dict"])
                logger.info(f"Model loaded from {path}")
                return checkpoint.get("metadata", {})
            self.load_state_dict(checkpoint)
            logger.info(f"Model loaded from {path}")
            return {}
        logger.warning(f"No model found at {path}")
        return None
