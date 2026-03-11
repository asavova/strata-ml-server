"""Residual block with batch normalization."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Pre-activation residual block.

    Architecture:
        x -> BN -> ReLU -> Linear -> BN -> ReLU -> Linear -> + x
    """

    def __init__(self, size: int, dropout: float = 0.1):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(size)
        self.fc1 = nn.Linear(size, size)
        self.bn2 = nn.BatchNorm1d(size)
        self.fc2 = nn.Linear(size, size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(x))
        out = self.fc1(out)
        out = F.relu(self.bn2(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return F.relu(out + residual)
