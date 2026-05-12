"""Supervised pretraining on grandmaster PGN games.

Trains the policy head to predict GM moves (NLLLoss on log-softmax output)
and the value head to predict game outcomes (MSELoss).
Saves a checkpoint whenever validation top-1 accuracy improves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

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
from chess_rl.model.chess_net import ChessNet
from chess_rl.pretrain.pgn_loader import PGNDataset


class SupervisedTrainer:
    """Trains a ChessNet on grandmaster games from a PGN file.

    Args:
        model:          ChessNet instance to train (modified in-place).
        pgn_path:       Path to the PGN file to learn from.
        min_elo:        Minimum Elo for both players.
        epochs:         Number of full passes through the data.
        batch_size:     Positions per gradient step.
        lr:             Adam learning rate.
        weight_decay:   Adam L2 regularisation.
        val_split:      Fraction of games held out for validation.
        device:         'cpu' or 'cuda'.
        checkpoint_dir: Directory to write pretrained.pt into.
    """

    def __init__(
        self,
        model: ChessNet,
        pgn_path: str,
        min_elo: int = MIN_ELO,
        epochs: int = PRETRAIN_EPOCHS,
        batch_size: int = BATCH_SIZE,
        lr: float = PRETRAIN_LR,
        weight_decay: float = PRETRAIN_WEIGHT_DECAY,
        val_split: float = PRETRAIN_VAL_SPLIT,
        device: str = DEVICE,
        checkpoint_dir: Path = CHECKPOINT_DIR,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.train_loader = DataLoader(
            PGNDataset(
                pgn_path,
                min_elo=min_elo,
                split="train",
                val_split=val_split,
            ),
            batch_size=batch_size,
            num_workers=0,
        )
        self.val_loader = DataLoader(
            PGNDataset(
                pgn_path,
                min_elo=min_elo,
                split="val",
                val_split=val_split,
            ),
            batch_size=batch_size,
            num_workers=0,
        )

        # -1 ensures epoch 1 always triggers a checkpoint save
        self.best_val_top1: float = -1.0

    # ── Public API ─────────────────────────────────────────────────────────

    def train(self) -> None:
        """Run the full pretraining loop for self.epochs epochs."""
        print(f"Pretraining on device: {self.device}")
        print(
            f"Epochs: {self.epochs}  |  "
            f"Batch size: {self.train_loader.batch_size}"
        )
        print("-" * 60)

        for epoch in range(1, self.epochs + 1):
            train_metrics = self._train_epoch(epoch)
            val_metrics = self._validate(epoch)

            if val_metrics["top1"] > self.best_val_top1:
                self.best_val_top1 = val_metrics["top1"]
                self._save_checkpoint(epoch, val_metrics)
                print(
                    f"  [*] Checkpoint saved  "
                    f"(val top-1: {self.best_val_top1:.4f})"
                )

            print(
                f"  Epoch {epoch:>3}  "
                f"train_loss={train_metrics['loss']:.4f}  "
                f"val_top1={val_metrics['top1']:.4f}  "
                f"val_top5={val_metrics['top5']:.4f}"
            )
            print()

    # ── Training pass ──────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """One full pass over the training split."""
        self.model.train()

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        n_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.epochs} [train]",
            leave=False,
        )

        for board_tensors, move_indices, result_values in pbar:
            board_tensors = board_tensors.float().to(self.device)
            move_indices = move_indices.long().to(self.device)
            result_values = result_values.float().to(self.device)

            log_policy, value = self.model(board_tensors)

            # NLLLoss expects log-probs; our network outputs log_softmax
            policy_loss = F.nll_loss(log_policy, move_indices)
            value_loss = F.mse_loss(value.squeeze(-1), result_values)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            n_batches += 1

            pbar.set_postfix({
                "loss": f"{total_loss / n_batches:.4f}",
                "pol": f"{total_policy_loss / n_batches:.4f}",
                "val_mse": f"{total_value_loss / n_batches:.4f}",
            })

        n = max(n_batches, 1)
        return {
            "loss": total_loss / n,
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
        }

    # ── Validation pass ────────────────────────────────────────────────────

    def _validate(self, epoch: int) -> Dict[str, float]:
        """Compute top-1, top-5 move accuracy and value MSE on the val split."""
        self.model.eval()

        top1_correct = 0
        top5_correct = 0
        total_value_loss = 0.0
        n_samples = 0
        n_batches = 0

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch}/{self.epochs} [val]  ",
            leave=False,
        )

        with torch.no_grad():
            for board_tensors, move_indices, result_values in pbar:
                board_tensors = board_tensors.float().to(self.device)
                move_indices = move_indices.long().to(self.device)
                result_values = result_values.float().to(self.device)

                log_policy, value = self.model(board_tensors)

                # Top-1: model's best move matches the GM move
                top1_correct += (
                    log_policy.argmax(dim=1) == move_indices
                ).sum().item()

                # Top-5: GM move appears in model's top-5 candidates
                top5_idx = log_policy.topk(5, dim=1).indices
                targets_exp = move_indices.unsqueeze(1).expand_as(top5_idx)
                top5_correct += (
                    (top5_idx == targets_exp).any(dim=1).sum().item()
                )

                value_loss = F.mse_loss(value.squeeze(-1), result_values)
                total_value_loss += value_loss.item()

                n_samples += board_tensors.size(0)
                n_batches += 1

                pbar.set_postfix({
                    "top1": f"{top1_correct / max(n_samples, 1):.4f}",
                    "top5": f"{top5_correct / max(n_samples, 1):.4f}",
                })

        top1 = top1_correct / max(n_samples, 1)
        top5 = top5_correct / max(n_samples, 1)
        val_loss = total_value_loss / max(n_batches, 1)

        print(
            f"  [val]  top-1: {top1:.4f}  "
            f"top-5: {top5:.4f}  "
            f"value_mse: {val_loss:.4f}  "
            f"({n_samples} positions)"
        )
        return {"top1": top1, "top5": top5, "val_loss": val_loss}

    # ── Checkpoint ─────────────────────────────────────────────────────────

    def _save_checkpoint(
        self, epoch: int, metrics: Dict[str, float]
    ) -> None:
        """Persist model + optimiser state to checkpoints/pretrained.pt."""
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_top1": metrics["top1"],
            "val_top5": metrics["top5"],
        }
        path = self.checkpoint_dir / "pretrained.pt"
        torch.save(ckpt, path)
