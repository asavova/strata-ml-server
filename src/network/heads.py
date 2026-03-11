"""Policy and value output heads."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyHead(nn.Module):
    """
    Outputs a probability distribution over all possible moves.

    Architecture:
        hidden -> Linear -> BN -> ReLU -> Linear -> (mask) -> softmax
    """

    def __init__(self, hidden_size: int, policy_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, policy_size)

    def forward(self, x: torch.Tensor, legal_mask: torch.Tensor = None) -> torch.Tensor:
        out = F.relu(self.bn(self.fc1(x)))
        logits = self.fc2(out)

        if legal_mask is not None:
            # Set illegal move logits to -inf so softmax gives them 0 probability
            illegal = (legal_mask == 0)
            logits = logits.masked_fill(illegal, float("-inf"))

        return F.softmax(logits, dim=-1)


class ValueHead(nn.Module):
    """
    Outputs a single scalar in [-1, 1] representing position evaluation.

    Architecture:
        hidden -> Linear -> BN -> ReLU -> Linear -> ReLU -> Linear -> tanh
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, 256)
        self.bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn(self.fc1(x)))
        out = F.relu(self.fc2(out))
        return torch.tanh(self.fc3(out)).squeeze(-1)
