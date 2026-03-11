"""
Fixed-size replay buffer for training examples from self-play.
"""

import logging
import random
from collections import deque
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("strata.training.buffer")


@dataclass
class TrainingExample:
    """One training example from self-play."""
    state: list[float]      # 1008 floats
    policy: list[float]     # 15876 floats (sparse)
    value: float            # -1 to 1


class ReplayBuffer:
    """
    Circular replay buffer.

    Stores the most recent N training examples from self-play.
    Supports random sampling for mini-batch training.
    """

    def __init__(self, max_size: int = 200_000):
        self.buffer: deque[TrainingExample] = deque(maxlen=max_size)
        self.max_size = max_size
        self.total_added = 0
        logger.info(f"ReplayBuffer created with max_size={max_size:,}")

    def add(self, example: TrainingExample):
        """Add a single training example."""
        self.buffer.append(example)
        self.total_added += 1

    def add_batch(self, examples: list[TrainingExample]):
        """Add multiple training examples."""
        for ex in examples:
            self.add(ex)
        logger.info(
            f"Added {len(examples)} examples. "
            f"Buffer size: {len(self.buffer):,} / {self.max_size:,}"
        )

    def sample(self, batch_size: int) -> list[TrainingExample]:
        """Sample a random mini-batch."""
        actual_size = min(batch_size, len(self.buffer))
        return random.sample(list(self.buffer), actual_size)

    def size(self) -> int:
        return len(self.buffer)

    def stats(self) -> dict:
        """Return buffer statistics."""
        values = [ex.value for ex in self.buffer] if self.buffer else [0]
        return {
            "size": len(self.buffer),
            "max_size": self.max_size,
            "total_added": self.total_added,
            "mean_value": float(np.mean(values)),
            "std_value": float(np.std(values)),
        }
