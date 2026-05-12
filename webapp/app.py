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
import sys
import torch
from pathlib import Path
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from chess_rl.config import BEST_MODEL_PATH, DEVICE
from chess_rl.environment.chess_env import ChessEnv
from chess_rl.mcts.mcts import MCTS
from chess_rl.model.chess_net import build_model

app = Flask(__name__, template_folder="templates")

_model = None
_args = None

_game: dict = {
    "env": None,
    "player_color": chess.WHITE,
    "move_history": [],
    "last_move": None,
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
        "fen": board.fen(),
        "turn": "white" if board.turn == chess.WHITE else "black",
        "playerColor": "white" if _game["player_color"] == chess.WHITE else "black",
        "status": status,
        "result": result,
        "evaluation": white_eval,
        "moveHistory": _game["move_history"],
        "lastMove": _game["last_move"],
        "topMoves": [],
        "isCheck": board.is_check(),
    }


def _do_bot_move() -> dict:
    env: ChessEnv = _game["env"]
    board = env.board

    action, _, top = _mcts().get_best_move(env, add_noise=False)
    move = _action_to_move(action, board)
    if move is None:
        return _build_state()

    san = board.san(move)
    _game["last_move"] = {
        "from": chess.square_name(move.from_square),
        "to": chess.square_name(move.to_square),
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
    data = request.get_json(force=True)
    color = data.get("color", "white")

    _game["env"] = ChessEnv()
    _game["env"].reset()
    _game["player_color"] = chess.WHITE if color == "white" else chess.BLACK
    _game["move_history"] = []
    _game["last_move"] = None

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
        to_idx = chess.parse_square(data["to"])
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
    san = env.board.san(move) if move else "?"
    return jsonify({
        "san": san,
        "from": chess.square_name(move.from_square) if move else None,
        "to": chess.square_name(move.to_square) if move else None,
        "candidates": [{"move": m, "prob": float(p)} for m, p in top],
    })


@app.route("/api/state")
def state():
    return jsonify(_build_state())


# ── Entry point ────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Marvin Chess Bot — Web UI")
    p.add_argument("--model", default=str(BEST_MODEL_PATH))
    p.add_argument("--simulations", type=int, default=100)
    p.add_argument("--device", default=DEVICE)
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--host", default="127.0.0.1")
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
    ckpt = torch.load(str(model_path), map_location=_args.device, weights_only=False)
    _model.load_state_dict(ckpt["model_state_dict"])
    _model.eval()
    print(f"Ready. Open http://{_args.host}:{_args.port}")
    app.run(host=_args.host, port=_args.port, debug=False)
