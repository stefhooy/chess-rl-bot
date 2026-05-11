"""Central configuration — all hyperparameters and path constants.

Import this module everywhere instead of hardcoding values.
Override any field at runtime via environment variables or argparse.
"""

from dataclasses import dataclass, field
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent          # chess-rl-bot/
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
EVAL_DIR = ROOT_DIR / "evaluation"
GAMES_DIR = ROOT_DIR / "chess_rl" / "games"

BEST_MODEL_PATH = CHECKPOINT_DIR / "best.pt"
PRETRAINED_MODEL_PATH = CHECKPOINT_DIR / "pretrained.pt"
ELO_HISTORY_PATH = EVAL_DIR / "elo_history.csv"


# ── Neural network ─────────────────────────────────────────────────────────
NUM_RES_BLOCKS: int = 10        # depth of ResNet tower
NUM_CHANNELS: int = 256         # convolutional filters per layer
INPUT_CHANNELS: int = 119       # AlphaZero 119-plane board encoding
BOARD_SIZE: int = 8
ACTION_SPACE: int = 4096        # 64 × 64 (from_square × to_square)


# ── MCTS ───────────────────────────────────────────────────────────────────
NUM_SIMULATIONS: int = 800      # rollouts per move (lower = faster but weaker)
C_PUCT: float = 1.5             # exploration constant in PUCT formula
DIRICHLET_ALPHA: float = 0.3    # noise added at root for exploration
DIRICHLET_EPSILON: float = 0.25 # weight of Dirichlet noise vs. prior
TEMPERATURE_THRESHOLD: int = 30 # moves before temperature drops to ~0


# ── Self-play / RL training ────────────────────────────────────────────────
SELF_PLAY_GAMES: int = 100      # games generated per RL iteration
BATCH_SIZE: int = 512
REPLAY_BUFFER_SIZE: int = 500_000
NUM_TRAINING_STEPS: int = 1000  # gradient steps per RL iteration
LEARNING_RATE: float = 1e-3
LR_DECAY_RATE: float = 1e-4     # final LR after decay
LR_DECAY_THRESHOLD: float = 0.6 # fraction of iterations before LR drops
WEIGHT_DECAY: float = 1e-4
GRAD_CLIP: float = 1.0
WIN_THRESHOLD: float = 0.55     # new model must win >55% to be promoted


# ── Evaluation ─────────────────────────────────────────────────────────────
PIT_GAMES: int = 40             # games in each evaluation pit match
ELO_K: float = 32.0             # Elo K-factor
INITIAL_ELO: float = 1200.0


# ── Supervised pretraining ─────────────────────────────────────────────────
PRETRAIN_LR: float = 1e-3
PRETRAIN_WEIGHT_DECAY: float = 1e-4
PRETRAIN_EPOCHS: int = 5
MIN_ELO: int = 2200             # minimum player Elo to include a game
PRETRAIN_VAL_SPLIT: float = 0.1 # fraction of games held out for validation


# ── Device ─────────────────────────────────────────────────────────────────
import torch
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
