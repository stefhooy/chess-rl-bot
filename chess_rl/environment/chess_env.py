"""Chess environment wrapping python-chess.

Action space : 4096 = 64 × 64  (from_square × to_square).
               action = from_square * 64 + to_square
Promotions   : always default to queen.
Reward       : from the perspective of the player who just moved.
               +1.0 win | -1.0 loss | 0.0 draw or non-terminal.
"""

import chess
import numpy as np
from collections import deque
from typing import Any, Deque, Dict, Tuple

from chess_rl.environment.encoder import encode_board


class ChessEnv:
    """Gymnasium-style chess environment for AlphaZero-style self-play.

    Usage::

        env = ChessEnv()
        obs = env.reset()                    # (119, 8, 8) float32
        mask = env.legal_moves_mask()        # (4096,) bool
        obs, reward, done, info = env.step(action)
    """

    ACTION_SPACE: int = 4096

    def __init__(self) -> None:
        self.board: chess.Board = chess.Board()
        self.board_history: Deque[chess.Board] = deque(maxlen=8)
        self._reset_internals()

    # ── Public API ─────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """Reset to the starting position and return the initial observation."""
        self._reset_internals()
        return self.get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Apply *action* and return (observation, reward, done, info).

        Args:
            action: Integer in [0, 4095], encoded as from_sq * 64 + to_sq.

        Returns:
            obs:    (119, 8, 8) float32 tensor of the new board state.
            reward: Scalar from the moved player's perspective.
            done:   True if the game has ended.
            info:   Dict with 'fen' and 'move' (UCI string).
        """
        move = self._action_to_move(action)
        self.board.push(move)
        self.board_history.append(self.board.copy())

        obs = self.get_observation()
        reward, done = self._compute_reward()
        info: Dict[str, Any] = {"fen": self.board.fen(), "move": move.uci()}

        return obs, reward, done, info

    def legal_moves_mask(self) -> np.ndarray:
        """Return a boolean mask of shape (4096,) — True at every legal action."""
        mask = np.zeros(self.ACTION_SPACE, dtype=bool)
        for move in self.board.legal_moves:
            mask[move.from_square * 64 + move.to_square] = True
        return mask

    def get_observation(self) -> np.ndarray:
        """Return the current board encoded as a (119, 8, 8) float32 array."""
        return encode_board(self.board, self.board_history)

    def is_game_over(self) -> bool:
        """True if the game has reached a terminal state."""
        return self.board.is_game_over()

    def result(self) -> str:
        """PGN result string: '1-0', '0-1', or '1/2-1/2'."""
        return self.board.result()

    # ── Static helpers ─────────────────────────────────────────────────────

    @staticmethod
    def action_to_uci(action: int) -> str:
        """Convert a flat action index to a UCI string (e.g. 'e2e4')."""
        from_sq = action // 64
        to_sq = action % 64
        return chess.square_name(from_sq) + chess.square_name(to_sq)

    @staticmethod
    def uci_to_action(uci: str) -> int:
        """Convert a UCI string (e.g. 'e2e4') to a flat action index."""
        move = chess.Move.from_uci(uci)
        return move.from_square * 64 + move.to_square

    # ── Internal helpers ───────────────────────────────────────────────────

    def _reset_internals(self) -> None:
        self.board = chess.Board()
        self.board_history.clear()
        self.board_history.append(self.board.copy())

    def _action_to_move(self, action: int) -> chess.Move:
        """Resolve a flat action index to a legal chess.Move.

        Tries queen-promotion first so pawn advances to the back rank are
        handled without requiring the caller to specify a promotion piece.
        """
        from_sq = action // 64
        to_sq = action % 64

        for promotion in (chess.QUEEN, None):
            move = chess.Move(from_sq, to_sq, promotion=promotion)
            if move in self.board.legal_moves:
                return move

        raise ValueError(
            f"Action {action} (from={chess.square_name(from_sq)}, "
            f"to={chess.square_name(to_sq)}) is illegal.\n"
            f"FEN: {self.board.fen()}"
        )

    def _compute_reward(self) -> Tuple[float, bool]:
        """Reward from the perspective of the player who just moved.

        After board.push(), board.turn is the NEXT player, so the player who
        just moved = not board.turn.
        """
        outcome = self.board.outcome()
        if outcome is None:
            return 0.0, False

        winner = outcome.winner  # chess.WHITE (True) | chess.BLACK (False) | None
        if winner is None:
            return 0.0, True  # draw

        just_moved_white = not self.board.turn   # True if White just moved
        white_won = bool(winner)                 # True if White won

        return (1.0 if white_won == just_moved_white else -1.0), True
