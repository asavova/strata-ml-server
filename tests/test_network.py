"""Tests for the neural network."""

import torch
from src.network import StrataNetwork


def test_network_forward_pass():
    """Verify the network produces correct output shapes."""
    net = StrataNetwork(
        state_size=1008,
        hidden_size=128,
        policy_size=15876,
        num_residual_blocks=2,
        dropout=0.0,
    )
    net.eval()

    state = torch.randn(4, 1008)
    mask = torch.ones(4, 15876)

    policy, value = net(state, mask)

    assert policy.shape == (4, 15876), f"Expected (4, 15876), got {policy.shape}"
    assert value.shape == (4,), f"Expected (4,), got {value.shape}"
    assert torch.all(policy >= 0), "Policy should be non-negative"
    assert torch.allclose(policy.sum(dim=1), torch.ones(4), atol=1e-5), "Policy should sum to 1"
    assert torch.all(value >= -1) and torch.all(value <= 1), "Value should be in [-1, 1]"


def test_network_legal_masking():
    """Verify illegal moves get zero probability."""
    net = StrataNetwork(
        state_size=1008,
        hidden_size=64,
        policy_size=15876,
        num_residual_blocks=1,
    )
    net.eval()

    state = torch.randn(1, 1008)
    mask = torch.zeros(1, 15876)
    mask[0, 0] = 1.0
    mask[0, 100] = 1.0
    mask[0, 5000] = 1.0

    policy, _ = net(state, mask)

    for i in range(15876):
        if i not in (0, 100, 5000):
            assert policy[0, i].item() < 1e-6, f"Illegal move {i} has prob {policy[0, i].item()}"


def test_network_save_load(tmp_path):
    """Verify model can be saved and loaded."""
    net1 = StrataNetwork(state_size=1008, hidden_size=64, num_residual_blocks=1)
    path = str(tmp_path / "test_model.pt")
    net1.save(path)

    net2 = StrataNetwork(state_size=1008, hidden_size=64, num_residual_blocks=1)
    loaded = net2.load(path, torch.device("cpu"))
    assert loaded == {}

    state = torch.randn(1, 1008)
    net1.eval()
    net2.eval()
    p1, v1 = net1(state)
    p2, v2 = net2(state)
    assert torch.allclose(p1, p2, atol=1e-6)
    assert torch.allclose(v1, v2, atol=1e-6)


def test_network_save_load_with_metadata(tmp_path):
    """Verify metadata checkpoint format remains loadable."""
    net1 = StrataNetwork(state_size=1008, hidden_size=64, num_residual_blocks=1)
    path = str(tmp_path / "metadata_model.pt")
    metadata = {"step": 12, "loss": 0.5, "best_loss": 0.5, "lr": 0.001}
    net1.save(path, metadata=metadata)

    net2 = StrataNetwork(state_size=1008, hidden_size=64, num_residual_blocks=1)
    loaded = net2.load(path, torch.device("cpu"))

    assert loaded == metadata
