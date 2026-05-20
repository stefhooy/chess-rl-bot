"""Tests for ChessEnv — action encoding, legal moves, rewards."""

import chess
import numpy as np
import pytest

from chess_rl.environment.chess_env import ChessEnv


def test_reset_returns_correct_shape():
    env = ChessEnv()
    obs = env.reset()
    assert obs.shape == (119, 8, 8)
    assert obs.dtype == np.float32


def test_legal_moves_mask_shape_and_count():
    env = ChessEnv()
    env.reset()
    mask = env.legal_moves_mask()
    assert mask.shape == (4096,)
    assert mask.dtype == bool
    assert mask.sum() == 20  # 20 legal moves from starting position


def test_uci_action_roundtrip():
    assert ChessEnv.action_to_uci(ChessEnv.uci_to_action("e2e4")) == "e2e4"
    assert ChessEnv.action_to_uci(ChessEnv.uci_to_action("a1h8")) == "a1h8"
    assert ChessEnv.action_to_uci(ChessEnv.uci_to_action("g1f3")) == "g1f3"


def test_action_encoding_formula():
    # action = from_square * 64 + to_square
    action = ChessEnv.uci_to_action("e2e4")
    assert action == chess.E2 * 64 + chess.E4


def test_step_returns_correct_types():
    env = ChessEnv()
    env.reset()
    action = ChessEnv.uci_to_action("e2e4")
    obs, reward, done, info = env.step(action)
    assert obs.shape == (119, 8, 8)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "fen" in info
    assert "move" in info


def test_step_non_terminal_reward_is_zero():
    env = ChessEnv()
    env.reset()
    action = ChessEnv.uci_to_action("e2e4")
    _, reward, done, _ = env.step(action)
    assert reward == 0.0
    assert done is False


def test_step_records_correct_uci():
    env = ChessEnv()
    env.reset()
    action = ChessEnv.uci_to_action("g1f3")
    _, _, _, info = env.step(action)
    assert info["move"] == "g1f3"


def test_illegal_action_raises():
    env = ChessEnv()
    env.reset()
    bad_action = ChessEnv.uci_to_action("e2e5")  # illegal jump
    with pytest.raises(ValueError):
        env.step(bad_action)


def test_checkmate_reward():
    # Fool's mate: shortest possible checkmate
    env = ChessEnv()
    env.reset()
    for uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:
        action = ChessEnv.uci_to_action(uci)
        obs, reward, done, _ = env.step(action)
    assert done is True
    assert reward == 1.0  # Black (last to move) won


def test_is_game_over_false_at_start():
    env = ChessEnv()
    env.reset()
    assert not env.is_game_over()


def test_legal_moves_mask_only_legal_actions():
    env = ChessEnv()
    env.reset()
    mask = env.legal_moves_mask()
    legal_actions = {m.from_square * 64 + m.to_square for m in env.board.legal_moves}
    assert set(np.where(mask)[0]) == legal_actions


def test_multiple_steps_update_board():
    env = ChessEnv()
    env.reset()
    initial_fen = env.board.fen()
    env.step(ChessEnv.uci_to_action("e2e4"))
    assert env.board.fen() != initial_fen
