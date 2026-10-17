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
    state = buf.state_dict()
    restored = ReplayBuffer(max_size=100)
    restored.load_state_dict(state)
    assert restored.size() == 100
    assert restored.stats()["total_added"] == 130
    assert restored.buffer[-1] == buf.buffer[-1]


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


def test_training_checkpoint_restores_training_state_and_replay_buffer(tmp_path):
    device = torch.device("cpu")
    train_config = TrainingConfig(
        batch_size=4,
        min_buffer_for_training=4,
        checkpoint_interval=999,
    )
    paths_config = PathsConfig(
        latest_model=str(tmp_path / "latest.pt"),
        best_model=str(tmp_path / "best.pt"),
        training_checkpoint=str(tmp_path / "training.pt"),
    )
    network = StrataNetwork(state_size=1008, hidden_size=64, num_residual_blocks=1)
    trainer = Trainer(network, device, train_config, paths_config)
    replay_buffer = ReplayBuffer(max_size=10)
    for _ in range(6):
        replay_buffer.add(_make_example())

    assert trainer.train_step(replay_buffer)["success"] is True
    trainer.best_loss = 0.25
    trainer.save_checkpoint(replay_buffer)
    expected_weights = {
        key: value.detach().clone()
        for key, value in network.state_dict().items()
    }

    restored_network = StrataNetwork(
        state_size=1008, hidden_size=64, num_residual_blocks=1
    )
    restored_trainer = Trainer(
        restored_network, device, train_config, paths_config
    )
    restored_buffer = ReplayBuffer(max_size=10)
    assert restored_trainer.load_checkpoint(restored_buffer) is True

    assert restored_trainer.step_count == 1
    assert restored_trainer.best_loss == 0.25
    assert restored_trainer.optimizer.state
    assert restored_buffer.size() == 6
    assert restored_buffer.stats()["total_added"] == 6
    for key, value in restored_network.state_dict().items():
        assert torch.equal(value, expected_weights[key])
