"""Monte Carlo Tree Search with PUCT selection and neural-network evaluation.

Each call to search() runs self.num_simulations rollouts and returns a
policy vector (visit-count distribution over the 4096 action space).

Backup sign convention
----------------------
The neural network returns *value* from the CURRENT player's perspective.
Before calling node.backup(), we negate it so that node.W accumulates
value from the PARENT player's perspective — matching Q's definition.
node.backup() then continues to negate at each ancestor level.
"""

import copy
from typing import Dict, List, Tuple

import numpy as np

from chess_rl.config import (
    ACTION_SPACE,
    C_PUCT,
    DEVICE,
    DIRICHLET_ALPHA,
    DIRICHLET_EPSILON,
    NUM_SIMULATIONS,
    TEMPERATURE_THRESHOLD,
)
from chess_rl.environment.chess_env import ChessEnv
from chess_rl.mcts.node import Node
from chess_rl.model.chess_net import ChessNet


class MCTS:
    """AlphaZero-style Monte Carlo Tree Search.

    Args:
        model:             ChessNet used to evaluate positions.
        num_simulations:   Rollouts per move (higher = stronger but slower).
        c_puct:            Exploration constant in the PUCT formula.
        dirichlet_alpha:   Dirichlet noise concentration at the root.
        dirichlet_epsilon: Weight of Dirichlet noise vs. network prior.
        device:            Torch device string ('cpu' or 'cuda').
    """

    def __init__(
        self,
        model: ChessNet,
        num_simulations: int = NUM_SIMULATIONS,
        c_puct: float = C_PUCT,
        dirichlet_alpha: float = DIRICHLET_ALPHA,
        dirichlet_epsilon: float = DIRICHLET_EPSILON,
        device: str = DEVICE,
    ) -> None:
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.device = device

    # ── Public API ─────────────────────────────────────────────────────────

    def search(self, env: ChessEnv, add_noise: bool = True) -> np.ndarray:
        """Run MCTS from the current position and return a policy vector.

        Args:
            env:        Current game state (not modified).
            add_noise:  Mix Dirichlet noise into root priors for exploration.
                        Should be True during self-play, False during evaluation.

        Returns:
            policy: np.ndarray shape (4096,), visit-count distribution
                    with temperature applied.
        """
        root = Node(prior=1.0)
        self._expand(root, env)

        if add_noise and root.children:
            self._add_dirichlet_noise(root)

        for _ in range(self.num_simulations):
            # Deep-copy so each simulation starts from the same position
            sim_env = copy.deepcopy(env)
            node = root

            # ── Selection ─────────────────────────────────────────────────
            # Walk down the tree choosing the highest PUCT child until we
            # reach a leaf (unexpanded node or terminal position).
            while node.is_expanded and not sim_env.is_game_over():
                node = node.select_child(self.c_puct)
                sim_env.step(node.action)

            # ── Expansion / evaluation ────────────────────────────────────
            if sim_env.is_game_over():
                value = self._terminal_value(sim_env)
            else:
                value = self._expand(node, sim_env)

            # ── Backup ────────────────────────────────────────────────────
            # value is from the LEAF player's perspective.
            # Negate so node.backup() receives it from the PARENT's perspective.
            node.backup(-value)

        move_count = len(env.board.move_stack)
        return self._build_policy(root, move_count)

    def get_best_move(
        self,
        env: ChessEnv,
        add_noise: bool = False,
    ) -> Tuple[int, np.ndarray, List[Tuple[str, float]]]:
        """Run MCTS and return the best action plus human-readable metadata.

        Args:
            env:        Current game state.
            add_noise:  Whether to add Dirichlet noise (False for play/eval).

        Returns:
            best_action: Flat action index with the highest visit count.
            policy:      Full (4096,) visit-count policy vector.
            top_moves:   [(uci, visit_pct), ...] for the top 3 moves.
        """
        policy = self.search(env, add_noise=add_noise)
        best_action = int(np.argmax(policy))

        top_indices = np.argsort(policy)[::-1][:3]
        top_moves: List[Tuple[str, float]] = [
            (ChessEnv.action_to_uci(int(a)), float(policy[a]))
            for a in top_indices
            if policy[a] > 0
        ]

        return best_action, policy, top_moves

    # ── Private helpers ────────────────────────────────────────────────────

    def _expand(self, node: Node, env: ChessEnv) -> float:
        """Query the network, create child nodes, and return value.

        Returns:
            value: float in [-1, +1] from the current player's perspective.
        """
        obs = env.get_observation()
        probs, value = self.model.predict(obs)

        # Zero-out illegal moves, then renormalize
        mask = env.legal_moves_mask()
        masked_probs = probs * mask
        total = masked_probs.sum()

        if total > 1e-8:
            masked_probs /= total
        else:
            # Network assigns near-zero mass to all legal moves → uniform fallback
            masked_probs = mask.astype(np.float32) / mask.sum()

        action_priors: Dict[int, float] = {
            int(a): float(masked_probs[a])
            for a in np.where(mask)[0]
        }
        node.expand(action_priors)
        return value

    def _terminal_value(self, env: ChessEnv) -> float:
        """Value at a terminal position from the current player's perspective.

        In checkmate the player-to-move is the loser → -1.
        Stalemate / any draw variant                 →  0.
        """
        outcome = env.board.outcome()
        if outcome is None or outcome.winner is None:
            return 0.0
        return -1.0  # current player (board.turn) is checkmated

    def _add_dirichlet_noise(self, root: Node) -> None:
        """Mix Dirichlet noise into root-node priors.

        Ensures the search explores all legal moves at least occasionally,
        preventing the bot from getting trapped in a single line.
        """
        actions = list(root.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        eps = self.dirichlet_epsilon
        for action, n in zip(actions, noise):
            child = root.children[action]
            child.P = (1 - eps) * child.P + eps * n

    def _build_policy(self, root: Node, move_count: int) -> np.ndarray:
        """Build the final policy vector from visit counts.

        Temperature schedule:
          τ = 1  (move_count < TEMPERATURE_THRESHOLD) → proportional to visits,
                  preserves diversity for the first 30 half-moves.
          τ → 0  afterwards → argmax, fully deterministic play.
        """
        policy = np.zeros(ACTION_SPACE, dtype=np.float32)

        if move_count < TEMPERATURE_THRESHOLD:
            for action, child in root.children.items():
                policy[action] = child.N
        else:
            best_action = max(root.children, key=lambda a: root.children[a].N)
            policy[best_action] = 1.0

        total = policy.sum()
        if total > 0:
            policy /= total

        return policy
