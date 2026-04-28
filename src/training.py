from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

try:
    from configuration import TrainingConfig, load_config
except ImportError:  # pragma: no cover
    from .configuration import TrainingConfig, load_config


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cross_entropy = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-cross_entropy)
        focal_loss = ((1 - pt) ** self.gamma) * cross_entropy

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


@dataclass(slots=True)
class EpochResult:
    train_loss: float
    val_loss: float


class LobTrainer:
    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or load_config().training
        self.device = torch.device(self.config.device)
        self.amp_enabled = self.config.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device=self.device.type, enabled=self.amp_enabled)

    def _criterion(self) -> FocalLoss:
        alpha = None
        if self.config.class_weights is not None:
            alpha = torch.tensor(self.config.class_weights, dtype=torch.float32, device=self.device)
        return FocalLoss(alpha=alpha, gamma=self.config.focal_gamma).to(self.device)

    def _autocast_context(self):
        return torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled)

    def fit(
        self,
        model: nn.Module,
        train_loader: Iterable,
        val_loader: Iterable,
    ) -> tuple[nn.Module, list[EpochResult]]:
        model = model.to(self.device)
        criterion = self._criterion()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.epochs)
        best_val_loss = float("inf")
        history: list[EpochResult] = []

        for epoch in range(self.config.epochs):
            train_loss = self._run_epoch(
                model=model,
                data_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                description=f"Epoch {epoch + 1}/{self.config.epochs} [Train]",
            )
            val_loss = self.evaluate(
                model=model,
                data_loader=val_loader,
                criterion=criterion,
                description=f"Epoch {epoch + 1}/{self.config.epochs} [Val]",
            )
            scheduler.step()
            history.append(EpochResult(train_loss=train_loss, val_loss=val_loss))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = Path(self.config.best_model_path)
                best_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), best_path)

        return model, history

    def _run_epoch(
        self,
        model: nn.Module,
        data_loader: Iterable,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        description: str,
    ) -> float:
        model.train()
        total_loss = 0.0
        batch_count = 0

        for x_batch, t_batch, y_batch in tqdm(data_loader, desc=description):
            x_batch = x_batch.to(self.device, non_blocking=True)
            t_batch = t_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with self._autocast_context():
                logits = model(x_batch, t_batch)
                loss = criterion(logits, y_batch)
                moe_loss = getattr(model, "moe_load_balancing_loss", None)  # load-balancing for MoE training
                if moe_loss is not None:  # add MoE auxiliary loss only when the model exposes it
                    loss = loss + moe_loss

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.config.grad_clip_norm)
            self.scaler.step(optimizer)
            self.scaler.update()

            total_loss += float(loss.item())
            batch_count += 1

        return total_loss / max(batch_count, 1)

    def evaluate(
        self,
        model: nn.Module,
        data_loader: Iterable,
        criterion: nn.Module | None = None,
        description: str = "Validation",
    ) -> float:
        model.eval()
        criterion = criterion or self._criterion()
        total_loss = 0.0
        batch_count = 0

        with torch.no_grad():
            for x_batch, t_batch, y_batch in tqdm(data_loader, desc=description):
                x_batch = x_batch.to(self.device, non_blocking=True)
                t_batch = t_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)

                logits = model(x_batch, t_batch)
                total_loss += float(criterion(logits, y_batch).item())
                batch_count += 1

        return total_loss / max(batch_count, 1)


def train_lob_transformer(
    model: nn.Module,
    train_loader: Iterable,
    val_loader: Iterable,
    device: str = "cuda",
    epochs: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    gamma: float = 2.0,
    class_weights: torch.Tensor | None = None,
) -> nn.Module:
    config = load_config().training
    config.device = device
    config.epochs = epochs
    config.learning_rate = lr
    config.weight_decay = weight_decay
    config.focal_gamma = gamma
    config.class_weights = None if class_weights is None else class_weights.detach().cpu().tolist()
    trainer = LobTrainer(config)
    trained_model, _ = trainer.fit(model, train_loader, val_loader)
    return trained_model
