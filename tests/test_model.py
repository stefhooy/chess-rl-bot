"""Tests for ChessNet — output shapes, value range, policy validity."""

import numpy as np
import pytest
import torch

from chess_rl.model.chess_net import ChessNet, build_model


def _small_model():
    """Tiny ChessNet for fast CPU tests."""
    return ChessNet(num_res_blocks=2, num_channels=32)


def test_forward_policy_shape():
    model = _small_model()
    model.eval()
    with torch.no_grad():
        policy, _ = model(torch.zeros(2, 119, 8, 8))
    assert policy.shape == (2, 4096)


def test_forward_value_shape():
    model = _small_model()
    model.eval()
    with torch.no_grad():
        _, value = model(torch.zeros(2, 119, 8, 8))
    assert value.shape == (2, 1)


def test_value_in_minus_one_to_one():
    model = _small_model()
    model.eval()
    with torch.no_grad():
        _, value = model(torch.randn(8, 119, 8, 8))
    assert (value >= -1.0).all()
    assert (value <= 1.0).all()


def test_policy_is_log_probabilities():
    model = _small_model()
    model.eval()
    with torch.no_grad():
        policy, _ = model(torch.randn(1, 119, 8, 8))
    # Log-softmax output: all values ≤ 0, exp sums to 1
    assert (policy <= 0).all()
    assert abs(policy.exp().sum().item() - 1.0) < 1e-4


def test_predict_single_board_shapes():
    model = _small_model()
    board_tensor = np.zeros((119, 8, 8), dtype=np.float32)
    policy_probs, value = model.predict(board_tensor)
    assert policy_probs.shape == (4096,)
    assert isinstance(value, float)


def test_predict_value_in_range():
    model = _small_model()
    board_tensor = np.zeros((119, 8, 8), dtype=np.float32)
    _, value = model.predict(board_tensor)
    assert -1.0 <= value <= 1.0


def test_predict_policy_sums_to_one():
    model = _small_model()
    board_tensor = np.zeros((119, 8, 8), dtype=np.float32)
    policy_probs, _ = model.predict(board_tensor)
    assert abs(policy_probs.sum() - 1.0) < 1e-4


def test_build_model_returns_chessnet():
    model = build_model(device="cpu")
    assert isinstance(model, ChessNet)


def test_batch_size_one_works():
    model = _small_model()
    model.eval()
    with torch.no_grad():
        policy, value = model(torch.zeros(1, 119, 8, 8))
    assert policy.shape == (1, 4096)
    assert value.shape == (1, 1)


def test_different_inputs_give_different_outputs():
    model = _small_model()
    model.eval()
    x1 = torch.zeros(1, 119, 8, 8)
    x2 = torch.ones(1, 119, 8, 8)
    with torch.no_grad():
        p1, v1 = model(x1)
        p2, v2 = model(x2)
    assert not torch.allclose(p1, p2)
