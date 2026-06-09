"""
Training loop for the Strata neural network.
"""

import logging
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from ..config import TrainingConfig, PathsConfig
from ..network import StrataNetwork
from .replay_buffer import ReplayBuffer
from .data_loader import examples_to_tensors

logger = logging.getLogger("strata.training")


class Trainer:
    """
    Handles the training loop: sampling from replay buffer,
    computing losses, updating weights, and checkpointing.
    """

    def __init__(
        self,
        network: StrataNetwork,
        device: torch.device,
        config: TrainingConfig,
        paths: PathsConfig,
    ):
        self.network = network
        self.device = device
        self.config = config
        self.paths = paths

        self.optimizer = optim.Adam(
            network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = MultiStepLR(
            self.optimizer,
            milestones=config.lr_schedule.step_milestones,
            gamma=config.lr_schedule.gamma,
        )

        self.step_count = 0
        self.best_loss = float("inf")
        self.loss_history: list[float] = []
        self.latest_checkpoint: dict | None = None

        logger.info(
            f"Trainer initialized: lr={config.learning_rate}, "
            f"batch_size={config.batch_size}, "
            f"weight_decay={config.weight_decay}"
        )

    def train_step(
        self,
        replay_buffer: ReplayBuffer,
        batch_size: int | None = None,
        learning_rate: float | None = None,
    ) -> dict:
        """
        Run one training step.

        Returns dict with loss metrics.
        """
        if replay_buffer.size() < self.config.min_buffer_for_training:
            return {
                "success": False,
                "reason": f"Not enough data: {replay_buffer.size()} < {self.config.min_buffer_for_training}",
                "loss": 0.0,
            }

        self.network.train()

        # Sample mini-batch
        effective_batch_size = batch_size or self.config.batch_size
        batch = replay_buffer.sample(effective_batch_size)
        states, target_policies, target_values = examples_to_tensors(batch, self.device)

        # Forward pass
        pred_policy, pred_value = self.network(states)

        # Policy loss: cross-entropy with target distribution
        # Add epsilon to avoid log(0)
        policy_loss = -torch.sum(
            target_policies * torch.log(pred_policy + 1e-8)
        ) / len(batch)

        # Value loss: MSE
        value_loss = F.mse_loss(pred_value, target_values)

        # L2 regularization is handled by optimizer weight_decay
        total_loss = policy_loss + value_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)

        original_lrs = [group["lr"] for group in self.optimizer.param_groups]
        effective_lr = learning_rate or original_lrs[0]
        if learning_rate is not None:
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate

        try:
            self.optimizer.step()
        finally:
            if learning_rate is not None:
                for group, original_lr in zip(self.optimizer.param_groups, original_lrs):
                    group["lr"] = original_lr

        self.scheduler.step()

        self.step_count += 1
        loss_val = total_loss.item()
        self.loss_history.append(loss_val)

        # Checkpoint
        if self.step_count % self.config.checkpoint_interval == 0:
            self._save_checkpoint(loss_val)

        logger.info(
            f"Step {self.step_count}: "
            f"total_loss={loss_val:.4f}, "
            f"policy_loss={policy_loss.item():.4f}, "
            f"value_loss={value_loss.item():.4f}, "
            f"lr={self.scheduler.get_last_lr()[0]:.6f}"
        )

        return {
            "success": True,
            "loss": loss_val,
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "step": self.step_count,
            "lr": effective_lr,
            "scheduled_lr": self.scheduler.get_last_lr()[0],
            "batch_size": len(batch),
            "buffer_size": replay_buffer.size(),
        }

    def _save_checkpoint(self, current_loss: float):
        """Save model checkpoint. Track best model separately."""
        metadata = self._checkpoint_metadata(current_loss)
        self.network.save(self.paths.latest_model, metadata=metadata)
        self.latest_checkpoint = metadata

        if current_loss < self.best_loss:
            self.best_loss = current_loss
            best_metadata = self._checkpoint_metadata(current_loss)
            self.network.save(self.paths.best_model, metadata=best_metadata)
            self.latest_checkpoint = best_metadata
            logger.info(f"New best model saved (loss={current_loss:.4f})")

    def _checkpoint_metadata(self, current_loss: float) -> dict:
        best_loss = min(self.best_loss, current_loss)
        return {
            "step": self.step_count,
            "loss": current_loss,
            "best_loss": best_loss if math.isfinite(best_loss) else None,
            "lr": self.scheduler.get_last_lr()[0],
        }

    def get_status(self) -> dict:
        """Return training status summary."""
        recent_loss = (
            sum(self.loss_history[-100:]) / len(self.loss_history[-100:])
            if self.loss_history
            else 0.0
        )
        return {
            "step_count": self.step_count,
            "best_loss": self.best_loss if math.isfinite(self.best_loss) else None,
            "recent_avg_loss": recent_loss,
            "lr": self.scheduler.get_last_lr()[0] if self.step_count > 0 else self.config.learning_rate,
            "latest_checkpoint": self.latest_checkpoint,
        }
