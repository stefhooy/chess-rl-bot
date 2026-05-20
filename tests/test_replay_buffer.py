"""Tests for ReplayBuffer — capacity, FIFO eviction, sampling."""

import numpy as np
import pytest
import torch

from chess_rl.training.replay_buffer import ReplayBuffer


def _make_sample(value: float = 0.0):
    state = np.zeros((119, 8, 8), dtype=np.float32)
    policy = np.ones(4096, dtype=np.float32) / 4096
    return state, policy, value


def test_empty_buffer_length():
    buf = ReplayBuffer(capacity=10)
    assert len(buf) == 0


def test_push_increases_length():
    buf = ReplayBuffer(capacity=10)
    buf.push(*_make_sample())
    assert len(buf) == 1


def test_push_multiple():
    buf = ReplayBuffer(capacity=10)
    for i in range(5):
        buf.push(*_make_sample(float(i)))
    assert len(buf) == 5


def test_push_many():
    buf = ReplayBuffer(capacity=10)
    samples = [_make_sample(float(i)) for i in range(4)]
    buf.push_many(samples)
    assert len(buf) == 4


def test_fifo_eviction_at_capacity():
    buf = ReplayBuffer(capacity=3)
    for i in range(5):
        buf.push(*_make_sample(float(i)))
    assert len(buf) == 3  # oldest entries evicted


def test_capacity_not_exceeded():
    buf = ReplayBuffer(capacity=10)
    for i in range(20):
        buf.push(*_make_sample())
    assert len(buf) == 10


def test_sample_state_shape():
    buf = ReplayBuffer(capacity=100)
    for i in range(10):
        buf.push(*_make_sample())
    states, _, _ = buf.sample(batch_size=4, device="cpu")
    assert states.shape == (4, 119, 8, 8)


def test_sample_policy_shape():
    buf = ReplayBuffer(capacity=100)
    for i in range(10):
        buf.push(*_make_sample())
    _, policies, _ = buf.sample(batch_size=4, device="cpu")
    assert policies.shape == (4, 4096)


def test_sample_value_shape():
    buf = ReplayBuffer(capacity=100)
    for i in range(10):
        buf.push(*_make_sample())
    _, _, values = buf.sample(batch_size=4, device="cpu")
    assert values.shape == (4,)


def test_sample_returns_float32_tensors():
    buf = ReplayBuffer(capacity=100)
    for i in range(10):
        buf.push(*_make_sample())
    states, policies, values = buf.sample(batch_size=4, device="cpu")
    assert states.dtype == torch.float32
    assert policies.dtype == torch.float32
    assert values.dtype == torch.float32


def test_sample_capped_at_buffer_size():
    buf = ReplayBuffer(capacity=100)
    buf.push(*_make_sample())
    states, _, _ = buf.sample(batch_size=50, device="cpu")
    assert states.shape[0] == 1  # only 1 sample in buffer


def test_is_ready_false_when_empty():
    buf = ReplayBuffer(capacity=100)
    assert not buf.is_ready(batch_size=1)


def test_is_ready_true_when_enough():
    buf = ReplayBuffer(capacity=100)
    for _ in range(5):
        buf.push(*_make_sample())
    assert buf.is_ready(batch_size=5)


def test_is_ready_false_when_not_enough():
    buf = ReplayBuffer(capacity=100)
    for _ in range(4):
        buf.push(*_make_sample())
    assert not buf.is_ready(batch_size=5)
