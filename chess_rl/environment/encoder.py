"""119-channel AlphaZero-style board encoder.

Layout (119 planes total):
  Planes   0–111  : 8 time-steps × 14 planes each
                     per step: 6 white piece types + 6 black piece types
                                + 2 repetition planes
  Planes 112–118  : auxiliary (color, castling ×4, en-passant, halfmove clock)
"""

import chess
import numpy as np
from collections import deque
from typing import Deque, Optional

PIECE_TYPES = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]

HISTORY_LEN: int = 8
PLANES_PER_STEP: int = 14   # 6 white + 6 black + 2 repetition
AUX_PLANES: int = 7
TOTAL_PLANES: int = HISTORY_LEN * PLANES_PER_STEP + AUX_PLANES  # 119


def encode_board(
    board: chess.Board,
    history: Deque[chess.Board],
) -> np.ndarray:
    """Encode board + history as a (119, 8, 8) float32 numpy array.

    Args:
        board:   The current board position (after the latest move).
        history: Deque of board states, most-recent at the RIGHT (index -1).
                 Should include the current board as the last element.

    Returns:
        planes: np.ndarray of shape (119, 8, 8), dtype float32.
    """
    planes = np.zeros((TOTAL_PLANES, 8, 8), dtype=np.float32)

    # Reverse so index 0 = most recent, index 7 = oldest
    hist_list: list[Optional[chess.Board]] = list(history)[::-1]
    while len(hist_list) < HISTORY_LEN:
        hist_list.append(None)  # pad older slots with zeros

    # ── Piece planes + repetition planes ──────────────────────────────────
    for t, hist_board in enumerate(hist_list):
        if hist_board is None:
            continue

        base = t * PLANES_PER_STEP

        for i, piece_type in enumerate(PIECE_TYPES):
            # White pieces → channels base+0 … base+5
            for sq in hist_board.pieces(piece_type, chess.WHITE):
                planes[base + i, sq // 8, sq % 8] = 1.0
            # Black pieces → channels base+6 … base+11
            for sq in hist_board.pieces(piece_type, chess.BLACK):
                planes[base + 6 + i, sq // 8, sq % 8] = 1.0

        # Repetition planes (base+12, base+13) — only meaningful for t=0
        if t == 0:
            if board.is_repetition(2):
                planes[base + 12, :, :] = 1.0
            if board.is_repetition(3):
                planes[base + 13, :, :] = 1.0

    # ── Auxiliary planes (112–118) ─────────────────────────────────────────
    aux = HISTORY_LEN * PLANES_PER_STEP  # 112

    # 112: side to move (1 = White, 0 = Black)
    planes[aux + 0, :, :] = float(board.turn == chess.WHITE)

    # 113–116: castling rights
    planes[aux + 1, :, :] = float(board.has_kingside_castling_rights(chess.WHITE))
    planes[aux + 2, :, :] = float(board.has_queenside_castling_rights(chess.WHITE))
    planes[aux + 3, :, :] = float(board.has_kingside_castling_rights(chess.BLACK))
    planes[aux + 4, :, :] = float(board.has_queenside_castling_rights(chess.BLACK))

    # 117: en-passant square (single hot in 8×8 plane)
    if board.ep_square is not None:
        planes[aux + 5, board.ep_square // 8, board.ep_square % 8] = 1.0

    # 118: halfmove clock normalised to [0, 1] (50-move rule resets at 100)
    planes[aux + 6, :, :] = board.halfmove_clock / 100.0

    return planes
