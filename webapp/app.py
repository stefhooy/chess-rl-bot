"""Web UI for Marvin - Chess RL Bot.

Usage
-----
    cd chess-rl-bot
    pip install flask          # one-time
    python webapp/app.py --model checkpoints/pretrained.pt --simulations 50

Then open http://localhost:5000 in your browser.
"""

from __future__ import annotations

import argparse
import chess
import csv
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from chess_rl.config import BEST_MODEL_PATH, DEVICE, INITIAL_ELO
from chess_rl.environment.chess_env import ChessEnv
from chess_rl.mcts.mcts import MCTS
from chess_rl.model.chess_net import build_model

app = Flask(__name__, template_folder="templates")

_model = None
_args  = None

# Human-game learning state
_pos_buffer: list = []   # (obs, policy, mover) tuples from current game
_online_opt        = None
_human_games       = 0   # total games Marvin has learned from

_game: dict = {
    "env":          None,
    "player_color": chess.WHITE,
    "move_history": [],
    "last_move":    None,
}


# ── Helpers ────────────────────────────────────────────────────────────────

def _mcts() -> MCTS:
    return MCTS(_model, num_simulations=_args.simulations, device=_args.device)


def _action_to_move(action: int, board: chess.Board):
    from_sq, to_sq = action // 64, action % 64
    for promo in (chess.QUEEN, None):
        m = chess.Move(from_sq, to_sq, promotion=promo)
        if m in board.legal_moves:
            return m
    return None


def _result_str(outcome: chess.Outcome) -> str:
    term = outcome.termination.name.replace("_", " ").title()
    pc = _game["player_color"]
    if outcome.winner is None:
        return f"Draw by {term}."
    return f"You win by {term}!" if outcome.winner == pc else f"Marvin wins by {term}."


def _build_state() -> dict:
    env: ChessEnv = _game["env"]
    if env is None:
        return {"status": "idle"}

    board = env.board
    obs = env.get_observation()
    _, raw = _model.predict(obs)
    white_eval = float(raw if board.turn == chess.WHITE else -raw)

    outcome = board.outcome()
    if outcome is not None:
        status, result = "over", _result_str(outcome)
    elif env.is_game_over():
        status, result = "over", "Game ended (move limit)."
    else:
        status, result = "playing", None

    return {
        "fen":         board.fen(),
        "turn":        "white" if board.turn == chess.WHITE else "black",
        "playerColor": "white" if _game["player_color"] == chess.WHITE else "black",
        "status":      status,
        "result":      result,
        "evaluation":  white_eval,
        "moveHistory": _game["move_history"],
        "lastMove":    _game["last_move"],
        "topMoves":    [],
        "isCheck":     board.is_check(),
    }


def _do_bot_move() -> dict:
    env: ChessEnv = _game["env"]
    board = env.board

    # Record position + MCTS policy before moving
    obs_before = env.get_observation().copy()
    mover      = board.turn

    action, policy, top = _mcts().get_best_move(env, add_noise=False)
    _pos_buffer.append((obs_before, policy, mover))

    move = _action_to_move(action, board)
    if move is None:
        return _build_state()

    san = board.san(move)
    _game["last_move"] = {
        "from": chess.square_name(move.from_square),
        "to":   chess.square_name(move.to_square),
    }
    env.step(action)
    _game["move_history"].append(san)

    state = _build_state()
    state["topMoves"] = [{"move": m, "prob": float(p)} for m, p in top]
    return state


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/new_game", methods=["POST"])
def new_game():
    global _pos_buffer
    data  = request.get_json(force=True)
    color = data.get("color", "white")

    _pos_buffer = []
    _game["env"]          = ChessEnv()
    _game["env"].reset()
    _game["player_color"] = chess.WHITE if color == "white" else chess.BLACK
    _game["move_history"] = []
    _game["last_move"]    = None

    state = _build_state()
    if _game["player_color"] == chess.BLACK:
        state = _do_bot_move()
    return jsonify(state)


@app.route("/api/move", methods=["POST"])
def make_move():
    env: ChessEnv = _game["env"]
    if env is None:
        return jsonify({"error": "No active game."}), 400
    if env.is_game_over():
        return jsonify({"error": "Game is over."}), 400
    if env.board.turn != _game["player_color"]:
        return jsonify({"error": "Not your turn."}), 400

    data = request.get_json(force=True)
    try:
        from_idx = chess.parse_square(data["from"])
        to_idx   = chess.parse_square(data["to"])
    except (KeyError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400

    promo_map = {"q": chess.QUEEN, "r": chess.ROOK, "b": chess.BISHOP, "n": chess.KNIGHT}
    promo = promo_map.get(data.get("promotion"))

    move = None
    for candidate in [
        chess.Move(from_idx, to_idx, promotion=promo or chess.QUEEN),
        chess.Move(from_idx, to_idx),
    ]:
        if candidate in env.board.legal_moves:
            move = candidate
            break

    if move is None:
        return jsonify({"error": "Illegal move."}), 400

    san = env.board.san(move)
    _game["last_move"] = {"from": data["from"], "to": data["to"]}
    env.step(move.from_square * 64 + move.to_square)
    _game["move_history"].append(san)

    state = _build_state()
    if state["status"] == "over":
        return jsonify(state)
    return jsonify(_do_bot_move())


@app.route("/api/hint")
def hint():
    env = _game["env"]
    if env is None or env.is_game_over():
        return jsonify({"error": "No active game."}), 400
    if env.board.turn != _game["player_color"]:
        return jsonify({"error": "Not your turn."}), 400

    action, _, top = _mcts().get_best_move(env, add_noise=False)
    move = _action_to_move(action, env.board)
    san  = env.board.san(move) if move else "?"
    return jsonify({
        "san":        san,
        "from":       chess.square_name(move.from_square) if move else None,
        "to":         chess.square_name(move.to_square) if move else None,
        "candidates": [{"move": m, "prob": float(p)} for m, p in top],
    })


@app.route("/api/state")
def state():
    return jsonify(_build_state())


@app.route("/api/learn_from_game", methods=["POST"])
def learn_from_game():
    global _human_games, _pos_buffer

    if not _pos_buffer:
        return jsonify({"steps": 0, "games": _human_games})

    data   = request.get_json(force=True)
    winner = data.get("winner")   # "player" | "bot" | "draw"
    pc     = _game["player_color"]

    # Build value targets: +1 if that side won, -1 if lost, 0 draw
    states, policies, values = [], [], []
    for obs, pol, mover in _pos_buffer:
        if winner == "draw":
            z = 0.0
        elif (winner == "bot"    and mover != pc) or \
             (winner == "player" and mover == pc):
            z = 1.0
        else:
            z = -1.0
        states.append(obs)
        policies.append(pol)
        values.append(z)

    n  = len(states)
    st = torch.tensor(np.stack(states),   dtype=torch.float32, device=_args.device)
    po = torch.tensor(np.stack(policies), dtype=torch.float32, device=_args.device)
    va = torch.tensor(values,             dtype=torch.float32, device=_args.device)

    # Mini gradient loop — 30 steps max, batch up to 32
    n_steps    = min(30, n * 3)
    total_loss = 0.0
    _model.train()
    for _ in range(n_steps):
        idx = torch.randperm(n, device=_args.device)[:min(32, n)]
        log_pol, val = _model(st[idx])
        loss = (-(po[idx] * log_pol).sum(dim=1).mean()
                + F.mse_loss(val.squeeze(-1), va[idx]))
        _online_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(_model.parameters(), 1.0)
        _online_opt.step()
        total_loss += loss.item()
    _model.eval()

    # Persist the updated weights back to the same file we loaded
    torch.save({
        "model_state_dict":     _model.state_dict(),
        "optimizer_state_dict": _online_opt.state_dict(),
        "elo":                  INITIAL_ELO,
        "iteration":            0,
    }, str(Path(_args.model)))

    _human_games += 1
    _pos_buffer   = []

    return jsonify({
        "steps":     n_steps,
        "avg_loss":  round(total_loss / max(n_steps, 1), 4),
        "positions": n,
        "games":     _human_games,
    })


@app.route("/api/elo")
def elo():
    csv_path = ROOT / "evaluation" / "elo_history.csv"
    if not csv_path.exists():
        return jsonify({
            "current":    INITIAL_ELO,
            "history":    [],
            "humanGames": _human_games,
        })

    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "pit":      int(row["pit"]),
                "elo":      float(row["elo"]),
                "win_rate": float(row["win_rate"]),
                "wins":     int(row["wins"]),
                "draws":    int(row["draws"]),
                "losses":   int(row["losses"]),
            })

    return jsonify({
        "current":    rows[-1]["elo"] if rows else INITIAL_ELO,
        "history":    rows,
        "humanGames": _human_games,
    })


# ── Entry point ────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Marvin Chess Bot — Web UI")
    p.add_argument("--model",       default=str(BEST_MODEL_PATH))
    p.add_argument("--simulations", type=int, default=100)
    p.add_argument("--device",      default=DEVICE)
    p.add_argument("--port",        type=int, default=5000)
    p.add_argument("--host",        default="127.0.0.1")
    return p.parse_args()


if __name__ == "__main__":
    _args = parse_args()
    model_path = Path(_args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Try: cp checkpoints/pretrained.pt checkpoints/best.pt")
        sys.exit(1)

    print(f"Loading Marvin from {model_path} ...")
    _model = build_model(device=_args.device)
    ckpt   = torch.load(str(model_path), map_location=_args.device, weights_only=False)
    _model.load_state_dict(ckpt["model_state_dict"])
    _model.eval()

    # Separate low-LR optimizer for online human-game learning
    _online_opt = torch.optim.Adam(_model.parameters(), lr=1e-4)

    print(f"Ready. Open http://{_args.host}:{_args.port}")
    app.run(host=_args.host, port=_args.port, debug=False)
