"""Tests for the 119-plane AlphaZero board encoder."""

import chess
import numpy as np
from collections import deque

from chess_rl.environment.encoder import encode_board, TOTAL_PLANES


def _start_planes():
    board = chess.Board()
    history = deque([board.copy()], maxlen=8)
    return encode_board(board, history), board


def test_output_shape():
    planes, _ = _start_planes()
    assert planes.shape == (119, 8, 8)
    assert planes.dtype == np.float32


def test_total_planes_constant():
    assert TOTAL_PLANES == 119


def test_piece_planes_are_binary():
    planes, _ = _start_planes()
    piece_planes = planes[:112]
    assert np.all((piece_planes == 0) | (piece_planes == 1))


def test_white_pawn_count_at_start():
    planes, _ = _start_planes()
    # White pawns: plane 0 (PAWN, WHITE, t=0)
    # Squares a2-h2 = 8,9,...,15 → row = sq//8 = 1, col = sq%8 = 0-7
    assert planes[0, 1, :].sum() == 8.0


def test_black_pawn_count_at_start():
    planes, _ = _start_planes()
    # Black pawns: plane 6 (PAWN, BLACK, t=0)
    # Squares a7-h7 = 48-55 → row = sq//8 = 6
    assert planes[6, 6, :].sum() == 8.0


def test_total_piece_count_at_start():
    planes, _ = _start_planes()
    # 16 white pieces on planes 0-5, 16 black pieces on planes 6-11
    assert planes[0:6].sum() == 16.0
    assert planes[6:12].sum() == 16.0


def test_side_to_move_white():
    planes, _ = _start_planes()
    # Plane 112: 1.0 when White to move
    assert planes[112, 0, 0] == 1.0


def test_side_to_move_black():
    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))
    history = deque([board.copy()], maxlen=8)
    planes = encode_board(board, history)
    # Now Black to move → plane 112 = 0.0
    assert planes[112, 0, 0] == 0.0


def test_castling_rights_at_start():
    planes, _ = _start_planes()
    assert planes[113, 0, 0] == 1.0  # White kingside
    assert planes[114, 0, 0] == 1.0  # White queenside
    assert planes[115, 0, 0] == 1.0  # Black kingside
    assert planes[116, 0, 0] == 1.0  # Black queenside


def test_no_en_passant_at_start():
    planes, _ = _start_planes()
    # Plane 117: en-passant square — all zeros at start
    assert planes[117].sum() == 0.0


def test_en_passant_after_double_push():
    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))
    history = deque([board.copy()], maxlen=8)
    planes = encode_board(board, history)
    # e4 double push sets en-passant on e3 (square 20 → row=2, col=4)
    assert planes[117, 2, 4] == 1.0


def test_halfmove_clock_zero_at_start():
    planes, _ = _start_planes()
    # Plane 118: halfmove clock / 100 = 0.0 at start
    assert planes[118, 0, 0] == 0.0


def test_history_padding_with_zeros():
    # Fresh board with only 1 history entry — older slots should be zero
    board = chess.Board()
    history = deque([board.copy()], maxlen=8)
    planes = encode_board(board, history)
    # t=1 planes (step 1, oldest): should all be zero since no history
    base = 1 * 14  # PLANES_PER_STEP = 14
    assert planes[base: base + 14].sum() == 0.0


def test_king_position_at_start():
    planes, _ = _start_planes()
    # White king (KING=5): plane 5 at t=0
    # White king starts on e1 = square 4 → row=0, col=4
    assert planes[5, 0, 4] == 1.0
    # Black king (plane 11): starts on e8 = square 60 → row=7, col=4
    assert planes[11, 7, 4] == 1.0
