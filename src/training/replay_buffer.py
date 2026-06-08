"""
Fixed-size replay buffer for training examples from self-play.
"""

import logging
import json
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

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

    def save_snapshot(self, path: str) -> None:
        """Persist the current buffer contents to a JSON snapshot."""
        snapshot_path = Path(path)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "max_size": self.max_size,
            "total_added": self.total_added,
            "examples": [asdict(example) for example in self.buffer],
        }
        snapshot_path.write_text(json.dumps(payload), encoding="utf-8")
        logger.info(
            f"ReplayBuffer snapshot saved: {len(self.buffer):,} examples to {path}"
        )

    def load_snapshot(self, path: str) -> bool:
        """Load a JSON snapshot if it exists, preserving the current max size."""
        snapshot_path = Path(path)
        if not snapshot_path.exists():
            logger.info(f"No replay buffer snapshot found at {path}")
            return False

        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        examples = payload.get("examples", [])
        self.buffer.clear()
        for raw in examples[-self.max_size:]:
            self.buffer.append(
                TrainingExample(
                    state=raw["state"],
                    policy=raw["policy"],
                    value=raw["value"],
                )
            )
        self.total_added = max(int(payload.get("total_added", 0)), len(self.buffer))
        logger.info(
            f"ReplayBuffer snapshot loaded: {len(self.buffer):,} examples from {path}"
        )
        return True

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
