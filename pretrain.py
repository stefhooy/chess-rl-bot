"""Supervised pretraining entry point.

Learns chess from grandmaster games before any self-play begins.
Training on a large GM dataset (e.g. Lichess 2500+ games) for a few
epochs gives the network a strong prior, dramatically speeding up RL.

Usage
-----
    python pretrain.py --pgn games/lichess_gm.pgn
    python pretrain.py --pgn games/lichess_gm.pgn --min-elo 2500 --epochs 10
    python pretrain.py --pgn games/lichess_gm.pgn --batch-size 256 --device cuda


After training, run:
    python train.py --start-from pretrained
"""

import argparse
import sys
import time
from pathlib import Path

from chess_rl.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DEVICE,
    MIN_ELO,
    PRETRAIN_EPOCHS,
    PRETRAIN_LR,
    PRETRAIN_VAL_SPLIT,
    PRETRAIN_WEIGHT_DECAY,
)
from chess_rl.model.chess_net import build_model
from chess_rl.pretrain.pgn_loader import scan_games
from chess_rl.pretrain.supervised_trainer import SupervisedTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain ChessNet on grandmaster PGN games.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pgn",
        required=True,
        help="Path to the .pgn file (plain text, not compressed).",
    )
    parser.add_argument(
        "--min-elo",
        type=int,
        default=MIN_ELO,
        help="Minimum Elo for both players; lower-rated games are skipped.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=PRETRAIN_EPOCHS,
        help="Number of full passes through the data.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Positions per gradient step.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=PRETRAIN_LR,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=PRETRAIN_WEIGHT_DECAY,
        help="Adam L2 weight decay.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=PRETRAIN_VAL_SPLIT,
        help="Fraction of games held out for validation (e.g. 0.1 = 10%%).",
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
        help="Directory where pretrained.pt will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pgn_path = Path(args.pgn)

    # ── Validate input file ────────────────────────────────────────────────
    if not pgn_path.exists():
        print(f"Error: PGN file not found: {pgn_path}")
        print()
        print("Download free grandmaster games from:")
        print("  Lichess open database : https://database.lichess.org")
        print("  PGN Mentor (top GMs)  : https://www.pgnmentor.com/files.html")
        print("  KingBase              : https://www.kingbase-chess.net")
        print()
        print("Place the .pgn file in chess_rl/games/ and re-run.")
        sys.exit(1)

    # ── Scan file for stats (reads headers only, no tensors) ──────────────
    print("=" * 60)
    print("  Chess RL - Supervised Pretraining")
    print("=" * 60)
    print(f"  PGN file  : {pgn_path}")
    print(f"  Min Elo   : {args.min_elo}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device    : {args.device}")
    print()
    print("Scanning PGN file for qualifying games...")
    t0 = time.time()
    n_games, n_positions = scan_games(str(pgn_path), min_elo=args.min_elo)
    elapsed = time.time() - t0

    if n_games == 0:
        print(
            f"No qualifying games found (min Elo {args.min_elo}). "
            "Try lowering --min-elo."
        )
        sys.exit(1)

    print(f"  Found {n_games:,} qualifying games "
          f"({n_positions:,} positions)  [{elapsed:.1f}s]")
    print(
        f"  Train / val split: "
        f"~{int((1 - args.val_split) * n_games):,} / "
        f"~{int(args.val_split * n_games):,} games"
    )
    print()

    # ── Build model ────────────────────────────────────────────────────────
    print("Building ChessNet...")
    model = build_model(device=args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print()

    # ── Run supervised training ────────────────────────────────────────────
    trainer = SupervisedTrainer(
        model=model,
        pgn_path=str(pgn_path),
        min_elo=args.min_elo,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )

    trainer.train()

    # ── Done ───────────────────────────────────────────────────────────────
    ckpt = args.checkpoint_dir / "pretrained.pt"
    print("=" * 60)
    print("Pretraining done.")
    print(f"  Best checkpoint : {ckpt}")
    print(f"  Best val top-1  : {trainer.best_val_top1:.4f}")
    print()
    print("Run:  python train.py --start-from pretrained")
    print("=" * 60)


if __name__ == "__main__":
    main()
