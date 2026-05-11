"""Board encoder re-exported from environment.encoder.

Keeping model/ self-contained: chess_net.py imports from here
rather than reaching into the environment package.

All constants and encode_board() live in environment/encoder.py
(single source of truth). This module just re-exports them.
"""

from chess_rl.environment.encoder import (  # noqa: F401
    encode_board,
    PIECE_TYPES,
    HISTORY_LEN,
    PLANES_PER_STEP,
    AUX_PLANES,
    TOTAL_PLANES,
)
