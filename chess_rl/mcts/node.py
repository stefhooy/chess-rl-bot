"""MCTS tree node.

W stores total value from the PARENT player's perspective.
Q = W / N  is the exploitation term in the PUCT formula.

Convention
----------
When backup() is first called from MCTS, *value* must already be negated
(i.e. from the parent's perspective).  backup() then continues to negate
at each level as it climbs the tree, alternating between players.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional


class Node:
    """A single node in the Monte Carlo search tree.

    Attributes:
        N:        Visit count.
        W:        Total value accumulated (parent player's perspective).
        P:        Prior probability assigned by the policy network.
        parent:   Parent node, or None for the root.
        action:   The action (flat index) that led to this node.
        children: Mapping from action index to child Node.
    """

    __slots__ = ("N", "W", "P", "parent", "action", "children")

    def __init__(
        self,
        prior: float,
        parent: Optional[Node] = None,
        action: Optional[int] = None,
    ) -> None:
        self.N: int = 0
        self.W: float = 0.0
        self.P: float = prior
        self.parent: Optional[Node] = parent
        self.action: Optional[int] = action
        self.children: Dict[int, Node] = {}

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def Q(self) -> float:
        """Mean value from the parent player's perspective (0 if unvisited)."""
        return self.W / self.N if self.N > 0 else 0.0

    @property
    def is_expanded(self) -> bool:
        """True once child nodes have been created."""
        return bool(self.children)

    # ── Core MCTS operations ────────────────────────────────────────────────

    def puct_score(self, c_puct: float) -> float:
        """PUCT selection score used by the parent to rank children.

        score = Q  +  c_puct × P × √N_parent / (1 + N)

        Higher Q  → exploitation of known-good moves.
        Higher U  → exploration of under-visited / high-prior moves.
        """
        parent_n = self.parent.N if self.parent else 1
        U = c_puct * self.P * np.sqrt(parent_n) / (1 + self.N)
        return self.Q + U

    def select_child(self, c_puct: float) -> Node:
        """Return the child with the highest PUCT score."""
        return max(self.children.values(), key=lambda n: n.puct_score(c_puct))

    def expand(self, action_priors: Dict[int, float]) -> None:
        """Create one child node per legal action with its prior probability.

        Args:
            action_priors: Mapping from flat action index to prior probability.
        """
        for action, prior in action_priors.items():
            if action not in self.children:
                self.children[action] = Node(
                    prior=prior, parent=self, action=action
                )

    def backup(self, value: float) -> None:
        """Propagate *value* up to the root, negating at each level.

        Args:
            value: From the PARENT player's perspective for this node.
                   backup() negates it before passing to the grandparent,
                   which restores the original player's perspective there.
        """
        self.N += 1
        self.W += value
        if self.parent is not None:
            self.parent.backup(-value)
