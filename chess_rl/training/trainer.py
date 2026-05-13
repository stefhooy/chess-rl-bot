"""RL training loop for ChessNet.

Each call to train() draws random batches from the replay buffer and
updates the network with a combined policy + value loss.

Loss
----
    L = cross_entropy(log_pi_pred, pi_mcts) + mse(v_pred, z)

The policy loss uses the soft cross-entropy formula
    -(pi_mcts * log_pi_pred).sum(dim=1).mean()
because pi_mcts is a full probability distribution (not a class index).

LR schedule
-----------
    Steps  0 – 59%   : lr = LEARNING_RATE  (1e-3)
    Steps 60% – end  : lr = LR_DECAY_RATE  (1e-4)
Implemented via MultiStepLR with gamma = LR_DECAY_RATE / LEARNING_RATE = 0.1.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from chess_rl.config import (
    BATCH_SIZE,
    DEVICE,
    GRAD_CLIP,
    LEARNING_RATE,
    LR_DECAY_RATE,
    NUM_TRAINING_STEPS,
    WEIGHT_DECAY,
)
from chess_rl.model.chess_net import ChessNet
from chess_rl.training.replay_buffer import ReplayBuffer


class Trainer:
    """Runs gradient-descent updates on ChessNet using replay buffer samples.

    Args:
        model:       ChessNet to train (modified in-place).
        num_steps:   Gradient steps per RL iteration.
        batch_size:  Positions per gradient step.
        lr:          Initial Adam learning rate.
        weight_decay: Adam L2 regularisation coefficient.
        grad_clip:   Max gradient norm for clipping (prevents explosions).
        device:      Torch device string.
    """

    def __init__(
        self,
        model: ChessNet,
        num_steps: int = NUM_TRAINING_STEPS,
        batch_size: int = BATCH_SIZE,
        lr: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        grad_clip: float = GRAD_CLIP,
        device: str = DEVICE,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.grad_clip = grad_clip

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Drop LR to 1e-4 once we pass 60% of the training steps
        decay_at = max(1, int(0.6 * num_steps))
        gamma = LR_DECAY_RATE / lr          # = 0.1 with default values
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[decay_at], gamma=gamma
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def train(
        self,
        replay_buffer: ReplayBuffer,
        num_steps: int | None = None,
    ) -> Dict[str, float]:
        """Run gradient steps, sampling randomly from the replay buffer.

        Args:
            replay_buffer: Experience store to draw batches from.
            num_steps:     Override the default step count for this call.

        Returns:
            Dict with averaged 'loss', 'policy_loss', 'value_loss' over all steps.
        """
        n_steps = num_steps if num_steps is not None else self.num_steps

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        steps_done = 0

        pbar = tqdm(range(n_steps), desc="Training", leave=False)

        for step in pbar:
            if not replay_buffer.is_ready(self.batch_size):
                break

            states, policies, values = replay_buffer.sample(
                self.batch_size, self.device
            )

            metrics = self._train_step(states, policies, values)
            self.scheduler.step()

            total_loss += metrics["loss"]
            total_policy_loss += metrics["policy_loss"]
            total_value_loss += metrics["value_loss"]
            steps_done += 1

            if step % max(1, n_steps // 10) == 0:
                lr_now = self.optimizer.param_groups[0]["lr"]
                pbar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "pol":  f"{metrics['policy_loss']:.4f}",
                    "val":  f"{metrics['value_loss']:.4f}",
                    "lr":   f"{lr_now:.2e}",
                })

        n = max(steps_done, 1)
        return {
            "loss":        total_loss / n,
            "policy_loss": total_policy_loss / n,
            "value_loss":  total_value_loss / n,
        }

    def train_step(
        self,
        states: torch.Tensor,
        mcts_policies: torch.Tensor,
        value_targets: torch.Tensor,
    ) -> Dict[str, float]:
        """Public single-step wrapper (useful for unit tests).

        Args:
            states:        (B, 119, 8, 8) float32.
            mcts_policies: (B, 4096) float32 probability distributions.
            value_targets: (B,) float32 game outcomes in [-1, +1].

        Returns:
            Dict with 'loss', 'policy_loss', 'value_loss'.
        """
        return self._train_step(states, mcts_policies, value_targets)

    # ── Private ────────────────────────────────────────────────────────────

    def _train_step(
        self,
        states: torch.Tensor,
        mcts_policies: torch.Tensor,
        value_targets: torch.Tensor,
    ) -> Dict[str, float]:
        """One forward + backward pass with gradient clipping."""
        self.model.train()

        states = states.to(self.device)
        mcts_policies = mcts_policies.to(self.device)
        value_targets = value_targets.to(self.device)

        log_policy, value = self.model(states)

        # Soft cross-entropy: works when targets are full distributions
        policy_loss = -(mcts_policies * log_policy).sum(dim=1).mean()

        # Value head: predict game outcome
        value_loss = F.huber_loss(value.squeeze(-1), value_targets)

        loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_clip
        )
        self.optimizer.step()

        return {
            "loss":        loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss":  value_loss.item(),
        }
