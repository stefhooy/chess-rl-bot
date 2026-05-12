"""Self-play game generation for RL training.

Each call to play_one_game() runs a full game of the bot against itself,
collecting (state, policy, value_target) triples for the replay buffer.

Value-target convention
-----------------------
z = +1.0  the player who was to move at this position eventually won
z = -1.0  they eventually lost
z =  0.0  the game was drawn (or truncated at the move cap)

This matches the reward convention used in ChessEnv and the pretraining
result_value, so the same network and loss functions serve all three phases.
"""

from __future__ import annotations

import chess
import numpy as np
from tqdm import tqdm
from typing import List, Tuple

from chess_rl.config import ACTION_SPACE, DEVICE, NUM_SIMULATIONS, SELF_PLAY_GAMES
from chess_rl.environment.chess_env import ChessEnv
from chess_rl.mcts.mcts import MCTS
from chess_rl.model.chess_net import ChessNet

# One training sample: (board_tensor, mcts_policy, value_target)
Sample = Tuple[np.ndarray, np.ndarray, float]

# Hard cap to prevent runaway games; python-chess handles normal draw rules
MAX_GAME_MOVES = 512


def play_one_game(
    model: ChessNet,
    num_simulations: int = NUM_SIMULATIONS,
    device: str = DEVICE,
    add_noise: bool = True,
) -> List[Sample]:
    """Play one full game of self-play and return training samples.

    Args:
        model:           ChessNet used for both sides (shared weights).
        num_simulations: MCTS rollouts per move. Lower = faster but weaker.
        device:          Torch device string.
        add_noise:       Add Dirichlet noise at the root for exploration.
                         Should be True during self-play, False during eval.

    Returns:
        List of (state_tensor, policy_vector, value_target) triples,
        one per move played.  Empty list if the game ends immediately.
    """
    env = ChessEnv()
    mcts = MCTS(model, num_simulations=num_simulations, device=device)

    env.reset()

    # Accumulate (state, policy, color_to_move) — value assigned after game
    history: List[Tuple[np.ndarray, np.ndarray, bool]] = []

    for _ in range(MAX_GAME_MOVES):
        if env.is_game_over():
            break

        state = env.get_observation()          # (119, 8, 8) before the move
        color = env.board.turn                 # chess.WHITE or chess.BLACK

        policy = mcts.search(env, add_noise=add_noise)   # (4096,) visit dist

        # Sample action proportionally to visit counts (temperature in policy)
        action = int(np.random.choice(ACTION_SPACE, p=policy))

        history.append((state, policy, color))

        _, _, done, _ = env.step(action)
        if done:
            break

    # ── Assign value targets retroactively ────────────────────────────────
    outcome = env.board.outcome()
    winner: chess.Color | None = outcome.winner if outcome is not None else None

    samples: List[Sample] = []
    for state, policy, color in history:
        if winner is None:
            z = 0.0                   # draw or truncated game
        elif winner == color:
            z = 1.0                   # this position's player won
        else:
            z = -1.0                  # this position's player lost
        samples.append((state, policy, z))

    return samples


def generate_games(
    model: ChessNet,
    n_games: int = SELF_PLAY_GAMES,
    num_simulations: int = NUM_SIMULATIONS,
    device: str = DEVICE,
    add_noise: bool = True,
) -> List[Sample]:
    """Play n_games self-play games and return all collected samples.

    Args:
        model:           ChessNet used for both sides.
        n_games:         Number of complete games to play.
        num_simulations: MCTS rollouts per move.
        device:          Torch device string.
        add_noise:       Dirichlet noise at root (True for training).

    Returns:
        Flat list of all (state, policy, value) samples from all games.
    """
    all_samples: List[Sample] = []

    pbar = tqdm(range(n_games), desc="Self-play", unit="game")
    for game_idx in pbar:
        samples = play_one_game(
            model,
            num_simulations=num_simulations,
            device=device,
            add_noise=add_noise,
        )
        all_samples.extend(samples)
        pbar.set_postfix({
            "positions": len(all_samples),
            "last_game": len(samples),
        })

    return all_samples
