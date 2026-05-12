"""RL training entry point.

Runs the AlphaZero self-play / train / evaluate loop for a given number
of iterations.  Each iteration:

  1. Self-play   - generate games with the current model + MCTS
  2. Fill buffer - push (state, policy, value) triples into the replay buffer
  3. Train       - gradient steps on random batches from the buffer
  4. Pit         - new model vs best model (40 games)
  5. Promote     - keep new model only if win-rate > 55 %%
  6. Checkpoint  - save best.pt and latest.pt

Usage
-----
    python train.py --iterations 100
    python train.py --iterations 100 --start-from pretrained
    python train.py --iterations 50  --simulations 200 --device cuda
"""

import argparse
import copy
import sys
import time
from pathlib import Path

import torch

from chess_rl.config import (
    BEST_MODEL_PATH,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DEVICE,
    NUM_SIMULATIONS,
    NUM_TRAINING_STEPS,
    PIT_GAMES,
    PRETRAINED_MODEL_PATH,
    SELF_PLAY_GAMES,
    WIN_THRESHOLD,
)
from chess_rl.evaluation.evaluator import Evaluator
from chess_rl.model.chess_net import ChessNet, build_model
from chess_rl.training.replay_buffer import ReplayBuffer
from chess_rl.training.self_play import generate_games
from chess_rl.training.trainer import Trainer


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AlphaZero-style RL training for chess.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of self-play / train / pit cycles to run.",
    )
    parser.add_argument(
        "--start-from",
        choices=["scratch", "pretrained", "latest"],
        default="scratch",
        help=(
            "scratch    - random init.\n"
            "pretrained - load checkpoints/pretrained.pt.\n"
            "latest     - resume from checkpoints/latest.pt."
        ),
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=NUM_SIMULATIONS,
        help="MCTS rollouts per move during self-play.",
    )
    parser.add_argument(
        "--self-play-games",
        type=int,
        default=SELF_PLAY_GAMES,
        help="Games to generate per iteration.",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=NUM_TRAINING_STEPS,
        help="Gradient steps per iteration.",
    )
    parser.add_argument(
        "--pit-games",
        type=int,
        default=PIT_GAMES,
        help="Games in the pit match between new and best model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Positions per gradient step.",
    )
    parser.add_argument(
        "--device",
        default=DEVICE,
        help="Torch device: 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=CHECKPOINT_DIR,
        help="Directory for checkpoints.",
    )
    return parser.parse_args()


# ── Checkpoint helpers ─────────────────────────────────────────────────────

def _save(
    model: ChessNet,
    path: Path,
    iteration: int,
    trainer: Trainer,
    elo: float,
) -> None:
    """Save model + optimiser state to *path*."""
    torch.save(
        {
            "iteration":            iteration,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "elo":                  elo,
        },
        path,
    )


def _load_weights(model: ChessNet, path: Path, device: str) -> None:
    """Load only the model weights from a checkpoint file."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded weights from {path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("  Chess RL - Self-Play Training")
    print("=" * 62)
    print(f"  Iterations    : {args.iterations}")
    print(f"  Simulations   : {args.simulations}")
    print(f"  Self-play games: {args.self_play_games}")
    print(f"  Train steps   : {args.train_steps}")
    print(f"  Pit games     : {args.pit_games}")
    print(f"  Device        : {args.device}")
    print(f"  Start from    : {args.start_from}")
    print()

    # ── Build model ────────────────────────────────────────────────────────
    model = build_model(device=args.device)

    if args.start_from == "pretrained":
        if not PRETRAINED_MODEL_PATH.exists():
            print(f"Error: {PRETRAINED_MODEL_PATH} not found.")
            print("Run:  python pretrain.py --pgn <file.pgn>  first.")
            sys.exit(1)
        _load_weights(model, PRETRAINED_MODEL_PATH, args.device)

    elif args.start_from == "latest":
        latest = args.checkpoint_dir / "latest.pt"
        if not latest.exists():
            print(f"Error: {latest} not found. Use --start-from scratch.")
            sys.exit(1)
        _load_weights(model, latest, args.device)

    # ── Initialise components ──────────────────────────────────────────────
    replay_buffer = ReplayBuffer()
    trainer = Trainer(
        model,
        num_steps=args.train_steps,
        batch_size=args.batch_size,
        device=args.device,
    )
    evaluator = Evaluator(
        num_simulations=args.simulations,
        device=args.device,
    )

    # Best model starts as a copy of the initial model
    best_model = copy.deepcopy(model)
    best_path = args.checkpoint_dir / "best.pt"
    latest_path = args.checkpoint_dir / "latest.pt"

    # ── Training loop ──────────────────────────────────────────────────────
    for iteration in range(1, args.iterations + 1):
        t0 = time.time()
        print(f"{'='*62}")
        print(f"  Iteration {iteration}/{args.iterations}")
        print(f"{'='*62}")

        # ── 1. Self-play ───────────────────────────────────────────────────
        print(f"[1/4] Self-play ({args.self_play_games} games, "
              f"{args.simulations} sims/move)...")
        samples = generate_games(
            model,
            n_games=args.self_play_games,
            num_simulations=args.simulations,
            device=args.device,
            add_noise=True,
        )
        replay_buffer.push_many(samples)
        print(f"      Generated {len(samples)} positions  "
              f"(buffer: {len(replay_buffer):,})")

        # ── 2. Train ───────────────────────────────────────────────────────
        if not replay_buffer.is_ready(args.batch_size):
            print(f"[2/4] Buffer too small ({len(replay_buffer)} < "
                  f"{args.batch_size}), skipping training this iteration.")
        else:
            print(f"[2/4] Training ({args.train_steps} steps)...")
            train_metrics = trainer.train(replay_buffer, args.train_steps)
            print(
                f"      loss={train_metrics['loss']:.4f}  "
                f"pol={train_metrics['policy_loss']:.4f}  "
                f"val={train_metrics['value_loss']:.4f}"
            )

        # ── 3. Pit: new model vs best model ────────────────────────────────
        print(f"[3/4] Pit match ({args.pit_games} games)...")
        win_rate, results = evaluator.pit(
            model, best_model, n=args.pit_games
        )
        print(
            f"      Win rate: {win_rate:.1%}  "
            f"(W={results['wins']}  "
            f"D={results['draws']}  "
            f"L={results['losses']})"
        )

        # ── 4. Promote or revert ───────────────────────────────────────────
        elo = evaluator.current_elo
        if win_rate > WIN_THRESHOLD:
            print(
                f"[4/4] PROMOTED  (win_rate={win_rate:.1%} > "
                f"{WIN_THRESHOLD:.0%})  Elo ~{elo:.0f}"
            )
            best_model = copy.deepcopy(model)
            _save(model, best_path, iteration, trainer, elo)
        else:
            print(
                f"[4/4] Not promoted (win_rate={win_rate:.1%} <= "
                f"{WIN_THRESHOLD:.0%}).  Reverting to best model."
            )
            model.load_state_dict(best_model.state_dict())

        # Always persist the latest checkpoint for resuming
        _save(model, latest_path, iteration, trainer, elo)

        elapsed = time.time() - t0
        print(f"\n  Iteration {iteration} done in {elapsed/60:.1f} min  "
              f"| Elo ~{elo:.0f}")

    # ── Done ───────────────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("Training complete.")
    print(f"  Best model : {best_path}")
    print(f"  Latest Elo : {evaluator.current_elo:.0f}")
    print()
    print("Play against it:")
    print("  python play.py --model checkpoints/best.pt")
    print("=" * 62)


if __name__ == "__main__":
    main()
