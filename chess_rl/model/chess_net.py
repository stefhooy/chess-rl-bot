"""AlphaZero-style chess neural network.

Architecture
------------
Stem      : Conv(119 → C, 3×3) → BN → ReLU
Tower     : N × ResBlock(C)
Policy    : Conv(C → 2, 1×1) → BN → ReLU → Linear(128 → 4096) → LogSoftmax
Value     : Conv(C → 1, 1×1) → BN → ReLU → Linear(64 → 256) → ReLU → Linear(256 → 1) → Tanh

C = NUM_CHANNELS (default 256), N = NUM_RES_BLOCKS (default 10).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

from chess_rl.config import (
    NUM_RES_BLOCKS,
    NUM_CHANNELS,
    INPUT_CHANNELS,
    ACTION_SPACE,
    DEVICE,
)


class ResBlock(nn.Module):
    """Single residual block: Conv→BN→ReLU→Conv→BN + skip → ReLU."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class ChessNet(nn.Module):
    """Dual-headed ResNet for chess policy and value estimation.

    Args:
        num_res_blocks: Number of residual blocks in the shared tower.
        num_channels:   Number of convolutional filters throughout the tower.
        input_channels: Depth of the board encoding (119 for AlphaZero spec).
        action_space:   Size of the flat move space (4096 = 64 × 64).

    Inputs:
        x : Tensor of shape (B, input_channels, 8, 8)

    Outputs:
        policy : Tensor of shape (B, action_space) — log-probabilities.
        value  : Tensor of shape (B, 1)            — scalar in [-1, +1].
    """

    def __init__(
        self,
        num_res_blocks: int = NUM_RES_BLOCKS,
        num_channels: int = NUM_CHANNELS,
        input_channels: int = INPUT_CHANNELS,
        action_space: int = ACTION_SPACE,
    ) -> None:
        super().__init__()

        # ── Stem ──────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
        )

        # ── Residual tower ────────────────────────────────────────────────
        self.tower = nn.Sequential(
            *[ResBlock(num_channels) for _ in range(num_res_blocks)]
        )

        # ── Policy head ───────────────────────────────────────────────────
        # 1×1 conv reduces to 2 planes, then flatten → linear → log-softmax
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.policy_bn   = nn.BatchNorm2d(2)
        self.policy_fc   = nn.Linear(2 * 8 * 8, action_space)

        # ── Value head ────────────────────────────────────────────────────
        # 1×1 conv reduces to 1 plane, then two linear layers → tanh
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.value_bn   = nn.BatchNorm2d(1)
        self.value_fc1  = nn.Linear(1 * 8 * 8, 256)
        self.value_fc2  = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            policy: (B, 4096) log-probabilities (use with NLLLoss or sum with mask).
            value:  (B, 1) in [-1, +1].
        """
        x = self.stem(x)
        x = self.tower(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)          # (B, 128)
        p = self.policy_fc(p)              # (B, 4096)
        policy = F.log_softmax(p, dim=1)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)          # (B, 64)
        v = F.relu(self.value_fc1(v))      # (B, 256)
        v = self.value_fc2(v)              # (B, 1)
        value = torch.tanh(v)

        return policy, value

    @torch.no_grad()
    def predict(self, board_tensor: np.ndarray) -> Tuple[np.ndarray, float]:
        """Inference helper for a single board (no batch dimension needed).

        Args:
            board_tensor: np.ndarray of shape (119, 8, 8).

        Returns:
            policy_probs: np.ndarray of shape (4096,) — probabilities (not log).
            value:        float in [-1, +1].
        """
        self.eval()
        x = torch.tensor(board_tensor, dtype=torch.float32, device=next(self.parameters()).device)
        x = x.unsqueeze(0)                         # (1, 119, 8, 8)
        log_policy, value_t = self(x)
        policy_probs = torch.exp(log_policy).squeeze(0).cpu().numpy()
        value = value_t.item()
        return policy_probs, value


def build_model(device: str = DEVICE) -> ChessNet:
    """Instantiate a ChessNet with default config and move it to *device*."""
    model = ChessNet()
    model.to(device)
    return model
