"""Tests for the evaluator."""

import pytest
import torch
from src.network import StrataNetwork
from src.evaluation import Evaluator


def test_single_evaluation():
    net = StrataNetwork(state_size=1008, hidden_size=64, num_residual_blocks=1)
    evaluator = Evaluator(net, torch.device("cpu"))

    state = [0.5] * 1008
    mask = [0.0] * 15876
    mask[0] = 1.0
    mask[100] = 1.0

    value, policy = evaluator.evaluate(state, mask)

    assert -1.0 <= value <= 1.0
    assert len(policy) == 15876
    assert evaluator.eval_count == 1


def test_batch_evaluation():
    net = StrataNetwork(state_size=1008, hidden_size=64, num_residual_blocks=1)
    evaluator = Evaluator(net, torch.device("cpu"))

    states = [[0.5] * 1008 for _ in range(4)]
    masks = [[1.0] * 15876 for _ in range(4)]

    results = evaluator.evaluate_batch(states, masks)

    assert len(results) == 4
    for value, policy in results:
        assert -1.0 <= value <= 1.0
        assert len(policy) == 15876


def test_evaluator_rejects_empty_legal_mask():
    net = StrataNetwork(state_size=1008, hidden_size=64, num_residual_blocks=1)
    evaluator = Evaluator(net, torch.device("cpu"))

    state = [0.5] * 1008
    mask = [0.0] * 15876

    with pytest.raises(ValueError, match="at least one legal move"):
        evaluator.evaluate(state, mask)


def test_batch_evaluation_rejects_mismatched_masks():
    net = StrataNetwork(state_size=1008, hidden_size=64, num_residual_blocks=1)
    evaluator = Evaluator(net, torch.device("cpu"))

    states = [[0.5] * 1008 for _ in range(2)]
    masks = [[1.0] * 15876]

    with pytest.raises(ValueError, match="same number of items"):
        evaluator.evaluate_batch(states, masks)
