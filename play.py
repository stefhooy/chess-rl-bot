"""Human vs Chess Bot CLI.

Loads a trained ChessNet checkpoint and lets you play against it
interactively in the terminal.

Usage
-----
    python play.py --model checkpoints/best.pt
    python play.py --model checkpoints/best.pt --simulations 200 --color black

In-game commands
----------------
    hint    Show what the bot would play in your position.
    eval    Print the numeric position evaluation.
    resign  Forfeit the game.
"""

import argparse
import sys
from pathlib import Path

# Force UTF-8 output on Windows so unicode chess pieces render correctly.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import chess
import torch

from chess_rl.config import BEST_MODEL_PATH, DEVICE
from chess_rl.environment.chess_env import ChessEnv
from chess_rl.mcts.mcts import MCTS
from chess_rl.model.chess_net import ChessNet, build_model

# ── Unicode piece table ────────────────────────────────────────────────────
_UNICODE: dict = {
    (chess.PAWN,   chess.WHITE): "♙",   # ♙
    (chess.KNIGHT, chess.WHITE): "♘",   # ♘
    (chess.BISHOP, chess.WHITE): "♗",   # ♗
    (chess.ROOK,   chess.WHITE): "♖",   # ♖
    (chess.QUEEN,  chess.WHITE): "♕",   # ♕
    (chess.KING,   chess.WHITE): "♔",   # ♔
    (chess.PAWN,   chess.BLACK): "♟",   # ♟
    (chess.KNIGHT, chess.BLACK): "♞",   # ♞
    (chess.BISHOP, chess.BLACK): "♝",   # ♝
    (chess.ROOK,   chess.BLACK): "♜",   # ♜
    (chess.QUEEN,  chess.BLACK): "♛",   # ♛
    (chess.KING,   chess.BLACK): "♚",   # ♚
}


# ── Display helpers ────────────────────────────────────────────────────────

def render_board(board: chess.Board, player_is_white: bool = True) -> str:
    """Return a unicode board string with file/rank coordinates.

    Args:
        board:           Current position.
        player_is_white: True  → rank 8 at top  (White's view).
                         False → rank 1 at top  (Black's view).
    """
    ranks = range(7, -1, -1) if player_is_white else range(0, 8)
    files = range(0, 8) if player_is_white else range(7, -1, -1)

    file_chars = " ".join("abcdefgh"[f] for f in files)
    border = "   +-" + "-" * 15 + "-+"

    lines = [f"    {file_chars}", border]

    for rank in ranks:
        row = []
        for file in files:
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            if piece is None:
                row.append("·")           # ·  middle dot for empty square
            else:
                row.append(_UNICODE.get((piece.piece_type, piece.color), "?"))
        lines.append(f" {rank + 1} | {' '.join(row)} | {rank + 1}")

    lines += [border, f"    {file_chars}"]
    return "\n".join(lines)


def eval_bar(value: float, width: int = 10) -> str:
    """Text evaluation bar from White's perspective.

    Args:
        value: Float in [-1, +1].  Positive = White winning.
        width: Number of block characters in the bar.

    Returns e.g.  '[████████░░] +0.72'
    """
    filled = max(0, min(width, round((value + 1) / 2 * width)))
    bar = "█" * filled + "░" * (width - filled)   # █ and ░
    sign = "+" if value >= 0 else ""
    return f"[{bar}] {sign}{value:.2f}"


def format_history(sans: list) -> str:
    """Format a list of SAN strings as '1.e4 e5  2.Nf3 Nc6'."""
    parts = []
    for i, san in enumerate(sans):
        if i % 2 == 0:
            parts.append(f"{i // 2 + 1}.{san}")
        else:
            parts.append(san)
    # Group into move pairs separated by two spaces
    tokens = []
    for i in range(0, len(parts), 2):
        tokens.append(" ".join(parts[i: i + 2]))
    return "  ".join(tokens)


# ── Move parsing ───────────────────────────────────────────────────────────

def parse_move(text: str, board: chess.Board):
    """Parse a player move in UCI or SAN format.

    Returns a legal chess.Move, or None if the input is invalid.
    """
    text = text.strip()

    # Try UCI (e.g. 'e2e4', 'e7e8q')
    try:
        move = chess.Move.from_uci(text)
        if move in board.legal_moves:
            return move
        # Append queen promotion for bare 4-char input reaching the back rank
        if len(text) == 4:
            move = chess.Move.from_uci(text + "q")
            if move in board.legal_moves:
                return move
    except ValueError:
        pass

    # Try SAN (e.g. 'e4', 'Nf3', 'O-O')
    try:
        move = board.parse_san(text)
        if move in board.legal_moves:
            return move
    except (ValueError, chess.IllegalMoveError, chess.AmbiguousMoveError):
        pass

    return None


def action_to_move(action: int, board: chess.Board) -> chess.Move:
    """Resolve a flat action index to a legal chess.Move (queen promo default)."""
    from_sq = action // 64
    to_sq = action % 64
    for promo in (chess.QUEEN, None):
        m = chess.Move(from_sq, to_sq, promotion=promo)
        if m in board.legal_moves:
            return m
    raise ValueError(f"Action {action} is illegal — FEN: {board.fen()}")


# ── Model loading ──────────────────────────────────────────────────────────

def load_model(path: str, device: str) -> ChessNet:
    """Load a ChessNet from a checkpoint file."""
    model = build_model(device=device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play chess against the trained bot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=str(BEST_MODEL_PATH),
        help="Path to a .pt checkpoint file.",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=200,
        help="MCTS rollouts per bot move (higher = stronger but slower).",
    )
    parser.add_argument(
        "--color",
        choices=["white", "black"],
        default="white",
        help="Your colour.",
    )
    parser.add_argument(
        "--device",
        default=DEVICE,
        help="Torch device: 'cpu' or 'cuda'.",
    )
    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Train first  : python train.py --iterations 100")
        print("Or pretrain  : python pretrain.py --pgn chess_rl/games/file.pgn")
        sys.exit(1)

    print(f"Loading {model_path}...")
    model = load_model(str(model_path), args.device)
    mcts = MCTS(model, num_simulations=args.simulations, device=args.device)

    env = ChessEnv()
    env.reset()

    player_color = chess.WHITE if args.color == "white" else chess.BLACK
    player_is_white = player_color == chess.WHITE
    move_history: list = []

    print()
    print("=" * 52)
    print("  Chess RL Bot")
    print("=" * 52)
    print(f"  You     : {args.color.capitalize()}")
    print(f"  Bot     : {'Black' if player_is_white else 'White'}")
    print(f"  Sims    : {args.simulations} per move")
    print()
    print("  Commands: hint | eval | resign")
    print("  Moves   : e2e4  or  e4  or  Nf3  or  O-O")
    print("=" * 52)

    # ── Game loop ──────────────────────────────────────────────────────────
    while not env.is_game_over():
        print()
        print(render_board(env.board, player_is_white))

        # Evaluation bar (always from White's perspective)
        obs = env.get_observation()
        _, raw_val = model.predict(obs)
        white_eval = raw_val if env.board.turn == chess.WHITE else -raw_val
        print(f"\n  Eval  : {eval_bar(white_eval)}")

        if move_history:
            print(f"  Moves : {format_history(move_history)}")

        is_player_turn = (env.board.turn == player_color)

        if is_player_turn:
            # ── Human turn ────────────────────────────────────────────
            print()
            while True:
                try:
                    text = input("  Your move : ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n  Game interrupted.")
                    return

                if not text:
                    continue

                cmd = text.lower()

                if cmd == "resign":
                    winner = "Black" if player_is_white else "White"
                    print(f"\n  You resigned.  {winner} wins.")
                    return

                if cmd == "hint":
                    print("  Thinking...", end="  ", flush=True)
                    action, _, top = mcts.get_best_move(env, add_noise=False)
                    hint_move = action_to_move(action, env.board)
                    hint_san = env.board.san(hint_move)
                    cands = "  ".join(
                        f"{m}({p:.0%})" for m, p in top
                    )
                    print(f"Hint: {hint_san}  [{cands}]")
                    continue

                if cmd == "eval":
                    print(
                        f"  Score : {white_eval:+.3f}  "
                        f"({'White' if white_eval >= 0 else 'Black'} is better)"
                    )
                    continue

                move = parse_move(text, env.board)
                if move is None:
                    print(f"  Unknown move '{text}' — try 'e2e4' or 'Nf3'.")
                    continue

                san = env.board.san(move)
                env.step(move.from_square * 64 + move.to_square)
                move_history.append(san)
                break

        else:
            # ── Bot turn ──────────────────────────────────────────────
            print()
            print("  Bot thinking...", end="  ", flush=True)
            action, _, top_moves = mcts.get_best_move(env, add_noise=False)

            cands = "  ".join(f"{m}({p:.0%})" for m, p in top_moves)
            print(cands)

            move = action_to_move(action, env.board)
            san = env.board.san(move)
            env.step(action)
            move_history.append(san)
            print(f"  Bot plays : {san}")

    # ── Game over ──────────────────────────────────────────────────────────
    print()
    print(render_board(env.board, player_is_white))
    print()

    outcome = env.board.outcome()
    if outcome is None:
        result_str = "Game ended (move limit)."
    elif outcome.winner is None:
        reason = outcome.termination.name.replace("_", " ").title()
        result_str = f"Draw by {reason}."
    elif outcome.winner == player_color:
        reason = outcome.termination.name.replace("_", " ").title()
        result_str = f"You win by {reason}!"
    else:
        reason = outcome.termination.name.replace("_", " ").title()
        result_str = f"Bot wins by {reason}."

    print(f"  Result  : {result_str}")
    print(f"  Moves   : {len(move_history)} half-moves")
    print(f"  History : {format_history(move_history)}")
    print()
    print(f"  Replay  : python play.py --model {args.model}")


if __name__ == "__main__":
    main()
