"""Tests for the training pipeline."""

import torch
from src.network import StrataNetwork
from src.training import Trainer, ReplayBuffer
from src.training.replay_buffer import TrainingExample
from src.config import TrainingConfig, PathsConfig


def _make_example() -> TrainingExample:
    """Create a random training example."""
    import random
    state = [random.random() for _ in range(1008)]
    policy = [0.0] * 15876
    policy[random.randint(0, 15875)] = 1.0
    value = random.uniform(-1, 1)
    return TrainingExample(state=state, policy=policy, value=value)


def test_replay_buffer():
    buf = ReplayBuffer(max_size=100)
    assert buf.size() == 0

    for _ in range(50):
        buf.add(_make_example())
    assert buf.size() == 50

    for _ in range(80):
        buf.add(_make_example())
    assert buf.size() == 100  # capped at max_size


def test_training_step(tmp_path):
    net = StrataNetwork(state_size=1008, hidden_size=64, num_residual_blocks=1)
    device = torch.device("cpu")

    train_config = TrainingConfig(
        batch_size=16,
        min_buffer_for_training=10,
        checkpoint_interval=999,
    )
    paths_config = PathsConfig(
        latest_model=str(tmp_path / "latest.pt"),
        best_model=str(tmp_path / "best.pt"),
    )

    trainer = Trainer(net, device, train_config, paths_config)
    buf = ReplayBuffer(max_size=1000)

    # Not enough data
    result = trainer.train_step(buf)
    assert result["success"] is False

    # Add enough data
    for _ in range(20):
        buf.add(_make_example())

    result = trainer.train_step(buf)
    assert result["success"] is True
    assert result["loss"] > 0


def test_training_step_uses_one_step_overrides(tmp_path):
    net = StrataNetwork(state_size=1008, hidden_size=64, num_residual_blocks=1)
    device = torch.device("cpu")

    train_config = TrainingConfig(
        learning_rate=0.001,
        batch_size=16,
        min_buffer_for_training=10,
        checkpoint_interval=999,
    )
    paths_config = PathsConfig(
        latest_model=str(tmp_path / "latest.pt"),
        best_model=str(tmp_path / "best.pt"),
    )

    trainer = Trainer(net, device, train_config, paths_config)
    buf = ReplayBuffer(max_size=1000)
    for _ in range(20):
        buf.add(_make_example())

    result = trainer.train_step(buf, batch_size=4, learning_rate=0.005)

    assert result["success"] is True
    assert result["batch_size"] == 4
    assert result["lr"] == 0.005
    assert trainer.optimizer.param_groups[0]["lr"] == train_config.learning_rate
