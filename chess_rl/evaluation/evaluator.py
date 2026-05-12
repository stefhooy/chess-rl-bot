"""Pit-match evaluator with Elo tracking.

Runs a fixed number of games between two ChessNet models (alternating
colours), computes the win rate, updates an Elo estimate, and logs
results to a CSV file.

Elo convention
--------------
current_elo tracks the best model's strength over the training run.
Both models in a pit start with the same Elo, so expected_score = 0.5.
The update rule is:
    new_elo = current_elo + K * (win_rate - expected_score)
"""

from __future__ import annotations

import chess
import csv
from pathlib import Path
from typing import Dict, Tuple

from tqdm import tqdm

from chess_rl.config import (
    DEVICE,
    EVAL_DIR,
    ELO_HISTORY_PATH,
    ELO_K,
    INITIAL_ELO,
    NUM_SIMULATIONS,
    PIT_GAMES,
)
from chess_rl.environment.chess_env import ChessEnv
from chess_rl.mcts.mcts import MCTS
from chess_rl.model.chess_net import ChessNet

_MAX_MOVES = 512        # hard cap per game to prevent infinite loops
_CSV_HEADER = ["pit", "wins", "draws", "losses", "win_rate", "elo"]


class Evaluator:
    """Evaluates a new model against the current best via pit matches.

    Args:
        num_simulations: MCTS rollouts per move during pit games.
                         Lower = faster evaluation but noisier results.
        device:          Torch device string.
        elo_history_path: CSV file to append pit results to.
        initial_elo:     Starting Elo rating (default 1200).
        k_factor:        Elo K-factor controlling rating sensitivity.
    """

    def __init__(
        self,
        num_simulations: int = NUM_SIMULATIONS,
        device: str = DEVICE,
        elo_history_path: Path = ELO_HISTORY_PATH,
        initial_elo: float = INITIAL_ELO,
        k_factor: float = ELO_K,
    ) -> None:
        self.num_simulations = num_simulations
        self.device = device
        self.current_elo = initial_elo
        self.k_factor = k_factor
        self._pit_count = 0

        self._csv_path = Path(elo_history_path)
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_csv()

    # ── Public API ─────────────────────────────────────────────────────────

    def pit(
        self,
        new_model: ChessNet,
        old_model: ChessNet,
        n: int = PIT_GAMES,
    ) -> Tuple[float, Dict[str, int]]:
        """Play n games between new_model and old_model.

        Colors alternate every game so neither model has a systematic
        colour advantage.  No Dirichlet noise is added (evaluation mode).

        Args:
            new_model: Candidate model (the one we just trained).
            old_model: Current best model (the benchmark).
            n:         Total games to play (split evenly across colours).

        Returns:
            win_rate: (wins + 0.5 * draws) / n  in [0, 1].
            results:  Dict with 'wins', 'draws', 'losses' for new_model.
        """
        self._pit_count += 1
        wins = draws = losses = 0

        pbar = tqdm(range(n), desc="Pit match", unit="game", leave=False)

        for game_idx in pbar:
            new_plays_white = (game_idx % 2 == 0)
            outcome = self._play_one_game(new_model, old_model, new_plays_white)

            if outcome == 1:
                wins += 1
            elif outcome == 0:
                draws += 1
            else:
                losses += 1

            pbar.set_postfix({
                "W": wins, "D": draws, "L": losses,
                "wr": f"{(wins + 0.5 * draws) / (game_idx + 1):.2f}",
            })

        win_rate = (wins + 0.5 * draws) / max(n, 1)

        # ── Elo update ────────────────────────────────────────────────────
        # Both models have the same Elo, so expected = 0.5
        expected = self._expected_score(self.current_elo, self.current_elo)
        self.current_elo += self.k_factor * (win_rate - expected)

        results = {"wins": wins, "draws": draws, "losses": losses}
        self._append_csv(results, win_rate)

        print(
            f"  Pit {self._pit_count}: "
            f"W={wins}  D={draws}  L={losses}  "
            f"win_rate={win_rate:.1%}  Elo={self.current_elo:.0f}"
        )

        return win_rate, results

    # ── Game play ──────────────────────────────────────────────────────────

    def _play_one_game(
        self,
        new_model: ChessNet,
        old_model: ChessNet,
        new_plays_white: bool,
    ) -> int:
        """Play one game with MCTS (no noise).

        Returns:
            +1  new model won
             0  draw
            -1  old model won
        """
        env = ChessEnv()
        env.reset()

        white_model = new_model if new_plays_white else old_model
        black_model = old_model if new_plays_white else new_model

        white_mcts = MCTS(
            white_model,
            num_simulations=self.num_simulations,
            device=self.device,
        )
        black_mcts = MCTS(
            black_model,
            num_simulations=self.num_simulations,
            device=self.device,
        )

        for _ in range(_MAX_MOVES):
            if env.is_game_over():
                break

            if env.board.turn == chess.WHITE:
                action, _, _ = white_mcts.get_best_move(env, add_noise=False)
            else:
                action, _, _ = black_mcts.get_best_move(env, add_noise=False)

            env.step(action)

        return self._game_result(env, new_plays_white)

    @staticmethod
    def _game_result(env: ChessEnv, new_plays_white: bool) -> int:
        """Convert board outcome to +1 / 0 / -1 from new model's perspective."""
        outcome = env.board.outcome()
        if outcome is None or outcome.winner is None:
            return 0  # draw or move-cap truncation

        new_is_winner = (
            (outcome.winner == chess.WHITE and new_plays_white)
            or (outcome.winner == chess.BLACK and not new_plays_white)
        )
        return 1 if new_is_winner else -1

    # ── Elo helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _expected_score(rating_a: float, rating_b: float) -> float:
        """Elo expected score for player A facing player B."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    # ── CSV logging ────────────────────────────────────────────────────────

    def _init_csv(self) -> None:
        """Write header row if the file doesn't already exist."""
        if not self._csv_path.exists():
            with open(self._csv_path, "w", newline="") as f:
                csv.writer(f).writerow(_CSV_HEADER)

    def _append_csv(
        self, results: Dict[str, int], win_rate: float
    ) -> None:
        """Append one row to the Elo history CSV."""
        with open(self._csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                self._pit_count,
                results["wins"],
                results["draws"],
                results["losses"],
                f"{win_rate:.4f}",
                f"{self.current_elo:.1f}",
            ])
