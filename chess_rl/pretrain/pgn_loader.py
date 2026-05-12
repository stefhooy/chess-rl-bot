"""Lazy PGN streamer for supervised pretraining on grandmaster games.

Streams one game at a time — never loads the full file into RAM.
Filters by minimum Elo and valid result, then yields one
(board_tensor, move_index, result_value) triple per position.

result_value convention (matches environment reward):
  +1.0  if the player to move at this position eventually won
  -1.0  if they eventually lost
   0.0  if the game was drawn
"""

import chess
import chess.pgn
import numpy as np
from collections import deque
from typing import Generator, Iterator, Tuple

from torch.utils.data import IterableDataset

from chess_rl.environment.encoder import encode_board
from chess_rl.config import MIN_ELO, PRETRAIN_VAL_SPLIT

# Only keep decisive or drawn games; skip abandoned / forfeited ones
VALID_RESULTS = frozenset({"1-0", "0-1", "1/2-1/2"})


# ── Low-level generator ────────────────────────────────────────────────────

def stream_positions(
    pgn_path: str,
    min_elo: int = MIN_ELO,
    split: str = "train",
    val_split: float = PRETRAIN_VAL_SPLIT,
) -> Generator[Tuple[np.ndarray, int, float], None, None]:
    """Lazily yield (board_tensor, move_index, result_value) from a PGN file.

    Args:
        pgn_path:  Path to the .pgn file (plain text, not compressed).
        min_elo:   Both players must have at least this Elo rating.
        split:     'train' or 'val'. Games are split by index so no game
                   contributes positions to both sets.
        val_split: Fraction of games reserved for validation (default 0.10).

    Yields:
        board_tensor:  np.ndarray of shape (119, 8, 8), float32.
        move_index:    Flat action index = from_square * 64 + to_square.
        result_value:  +1 / -1 / 0 from the moving player's perspective.
    """
    val_every = max(1, round(1.0 / val_split))   # e.g. 10 for val_split=0.10
    game_index = 0

    with open(pgn_path, encoding="utf-8", errors="ignore") as fh:
        while True:
            try:
                game = chess.pgn.read_game(fh)
            except Exception:
                continue                         # skip malformed games
            if game is None:
                break                           # end of file

            # ── Result filter ──────────────────────────────────────────────
            result = game.headers.get("Result", "*")
            if result not in VALID_RESULTS:
                continue

            # ── Elo filter ─────────────────────────────────────────────────
            try:
                white_elo = int(game.headers.get("WhiteElo", "0") or "0")
                black_elo = int(game.headers.get("BlackElo", "0") or "0")
            except ValueError:
                continue
            if white_elo < min_elo or black_elo < min_elo:
                continue

            # ── Train / val split by game index ───────────────────────────
            is_val = (game_index % val_every == 0)
            game_index += 1
            if split == "val" and not is_val:
                continue
            if split == "train" and is_val:
                continue

            # ── Yield one position per move ────────────────────────────────
            board = game.board()
            history: deque = deque(maxlen=8)
            history.append(board.copy())

            for move in game.mainline_moves():
                # Encode the position BEFORE the move is played
                board_tensor = encode_board(board, history)

                # Flat action index
                move_index = move.from_square * 64 + move.to_square

                # result_value from the perspective of the player about to move
                if result == "1-0":
                    result_value = 1.0 if board.turn == chess.WHITE else -1.0
                elif result == "0-1":
                    result_value = -1.0 if board.turn == chess.WHITE else 1.0
                else:
                    result_value = 0.0

                yield board_tensor, move_index, result_value

                # Advance board state for the next position
                board.push(move)
                history.append(board.copy())


# ── Dataset ────────────────────────────────────────────────────────────────

class PGNDataset(IterableDataset):
    """torch IterableDataset that streams positions from a PGN file.

    Use with DataLoader (recommended: num_workers=0 for simplicity,
    or implement worker_init_fn for multi-worker sharding).

    Args:
        pgn_path:  Path to the .pgn file.
        min_elo:   Minimum Elo for both players.
        split:     'train' or 'val'.
        val_split: Fraction of games held out for validation.
    """

    def __init__(
        self,
        pgn_path: str,
        min_elo: int = MIN_ELO,
        split: str = "train",
        val_split: float = PRETRAIN_VAL_SPLIT,
    ) -> None:
        self.pgn_path = pgn_path
        self.min_elo = min_elo
        self.split = split
        self.val_split = val_split

    def __iter__(self) -> Iterator[Tuple[np.ndarray, int, float]]:
        return stream_positions(
            self.pgn_path,
            min_elo=self.min_elo,
            split=self.split,
            val_split=self.val_split,
        )


# ── Reporting helper ────────────────────────────────────────────────────────

def scan_games(
    pgn_path: str,
    min_elo: int = MIN_ELO,
) -> Tuple[int, int]:
    """Count qualifying games and total positions without building tensors.

    This reads move lists (to count positions) but skips tensor encoding,
    so it is much faster than a full pass.  Use for the pretrain.py banner.

    Returns:
        (num_games, num_positions)
    """
    n_games = 0
    n_positions = 0

    with open(pgn_path, encoding="utf-8", errors="ignore") as fh:
        while True:
            try:
                game = chess.pgn.read_game(fh)
            except Exception:
                continue
            if game is None:
                break

            result = game.headers.get("Result", "*")
            if result not in VALID_RESULTS:
                continue

            try:
                white_elo = int(game.headers.get("WhiteElo", "0") or "0")
                black_elo = int(game.headers.get("BlackElo", "0") or "0")
            except ValueError:
                continue

            if white_elo < min_elo or black_elo < min_elo:
                continue

            n_games += 1
            n_positions += sum(1 for _ in game.mainline_moves())

    return n_games, n_positions
