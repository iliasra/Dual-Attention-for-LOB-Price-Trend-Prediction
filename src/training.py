from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

try:
    from configuration import TrainingConfig, load_config
    from run_logging import format_duration
except ImportError:  # pragma: no cover
    from .configuration import TrainingConfig, load_config
    from .run_logging import format_duration


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


def class_weights_from_sequence_labels(
    y: np.ndarray,
    num_classes: int = 3,
    gamma_mode: bool = True,
) -> tuple[list[float], list[int]]:
    """Compute clipped balanced class weights from sequence-level train labels."""
    labels = np.asarray(y, dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes)[:num_classes]
    total = int(counts.sum())
    if total <= 0:
        raise ValueError("Cannot compute class weights from an empty label array.")

    weights = total / (num_classes * np.maximum(counts, 1))
    if gamma_mode:
        weights = np.sqrt(weights) # the sqrt allows the weights to be "less agressive" when focal loss already deals with imbalanced classes
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.5, 3.0)
    return weights.astype(float).tolist(), counts.astype(int).tolist()


@dataclass(slots=True)
class ClassificationMetrics:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    balanced_accuracy: float
    expected_calibration_error: float
    per_class_precision: list[float]
    per_class_recall: list[float]
    per_class_f1: list[float]
    confusion_matrix: list[list[int]]
    normalized_confusion_matrix: list[list[float]]


@dataclass(slots=True)
class EvaluationResult:
    loss: float
    metrics: ClassificationMetrics


class ClassificationMetricAccumulator:
    def __init__(self, device: torch.device, num_calibration_bins: int = 15) -> None:
        self.device = device
        self.num_calibration_bins = num_calibration_bins
        self.num_classes: int | None = None
        self.confusion_matrix: torch.Tensor | None = None
        self.calibration_bin_counts: torch.Tensor | None = None
        self.calibration_confidence_sums: torch.Tensor | None = None
        self.calibration_correct_sums: torch.Tensor | None = None

    @staticmethod
    def _zero_metrics(num_classes: int = 0) -> ClassificationMetrics:
        return ClassificationMetrics(
            accuracy=0.0,
            macro_precision=0.0,
            macro_recall=0.0,
            macro_f1=0.0,
            weighted_f1=0.0,
            balanced_accuracy=0.0,
            expected_calibration_error=0.0,
            per_class_precision=[0.0] * num_classes,
            per_class_recall=[0.0] * num_classes,
            per_class_f1=[0.0] * num_classes,
            confusion_matrix=[[0] * num_classes for _ in range(num_classes)],
            normalized_confusion_matrix=[[0.0] * num_classes for _ in range(num_classes)],
        )

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        if self.num_classes is None:
            self.num_classes = int(logits.shape[-1])
            self.confusion_matrix = torch.zeros(
                (self.num_classes, self.num_classes),
                dtype=torch.long,
                device=self.device,
            )
            self.calibration_bin_counts = torch.zeros(
                self.num_calibration_bins,
                dtype=torch.float64,
                device=self.device,
            )
            self.calibration_confidence_sums = torch.zeros_like(self.calibration_bin_counts)
            self.calibration_correct_sums = torch.zeros_like(self.calibration_bin_counts)
        if self.confusion_matrix is None:
            raise RuntimeError("Classification metric accumulator was not initialized.")
        if (
            self.calibration_bin_counts is None
            or self.calibration_confidence_sums is None
            or self.calibration_correct_sums is None
        ):
            raise RuntimeError("Calibration metric accumulator was not initialized.")

        probabilities = torch.softmax(logits.detach().float(), dim=-1)
        confidences, predictions = torch.max(probabilities, dim=-1)
        targets = targets.detach()
        valid_mask = (
            (targets >= 0)
            & (targets < self.num_classes)
            & (predictions >= 0)
            & (predictions < self.num_classes)
        )
        if not bool(valid_mask.any()):
            return

        flat_indices = targets[valid_mask] * self.num_classes + predictions[valid_mask]
        counts = torch.bincount(flat_indices, minlength=self.num_classes * self.num_classes)
        self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes)

        valid_confidences = confidences[valid_mask].to(torch.float64)
        correctness = (predictions[valid_mask] == targets[valid_mask]).to(torch.float64)
        bin_indices = torch.clamp(
            (valid_confidences * self.num_calibration_bins).long(),
            max=self.num_calibration_bins - 1,
        )
        self.calibration_bin_counts += torch.bincount(
            bin_indices,
            minlength=self.num_calibration_bins,
        ).to(torch.float64)
        self.calibration_confidence_sums += torch.bincount(
            bin_indices,
            weights=valid_confidences,
            minlength=self.num_calibration_bins,
        )
        self.calibration_correct_sums += torch.bincount(
            bin_indices,
            weights=correctness,
            minlength=self.num_calibration_bins,
        )

    def compute(self) -> ClassificationMetrics:
        if self.confusion_matrix is None:
            return self._zero_metrics()

        confusion = self.confusion_matrix.to(torch.float64)
        total = confusion.sum()
        if float(total.item()) == 0.0:
            return self._zero_metrics(int(confusion.shape[0]))

        true_positives = torch.diag(confusion)
        support = confusion.sum(dim=1)
        predicted = confusion.sum(dim=0)
        eps = torch.finfo(confusion.dtype).eps

        precision = true_positives / predicted.clamp_min(1.0)
        recall = true_positives / support.clamp_min(1.0)
        f1 = 2.0 * precision * recall / (precision + recall).clamp_min(eps)
        macro_precision = precision.mean()
        macro_recall = recall.mean()
        macro_f1 = f1.mean()
        weighted_f1 = (f1 * support).sum() / support.sum().clamp_min(1.0)
        accuracy = true_positives.sum() / total
        normalized_confusion = confusion / support.clamp_min(1.0).unsqueeze(1)
        expected_calibration_error = torch.zeros((), dtype=torch.float64, device=self.device)
        if (
            self.calibration_bin_counts is not None
            and self.calibration_confidence_sums is not None
            and self.calibration_correct_sums is not None
        ):
            non_empty_bins = self.calibration_bin_counts > 0
            bin_accuracy = torch.zeros_like(self.calibration_bin_counts)
            bin_confidence = torch.zeros_like(self.calibration_bin_counts)
            bin_accuracy[non_empty_bins] = (
                self.calibration_correct_sums[non_empty_bins]
                / self.calibration_bin_counts[non_empty_bins]
            )
            bin_confidence[non_empty_bins] = (
                self.calibration_confidence_sums[non_empty_bins]
                / self.calibration_bin_counts[non_empty_bins]
            )
            bin_weights = self.calibration_bin_counts / self.calibration_bin_counts.sum().clamp_min(1.0)
            expected_calibration_error = torch.sum(bin_weights * torch.abs(bin_accuracy - bin_confidence))

        return ClassificationMetrics(
            accuracy=float(accuracy.item()),
            macro_precision=float(macro_precision.item()),
            macro_recall=float(macro_recall.item()),
            macro_f1=float(macro_f1.item()),
            weighted_f1=float(weighted_f1.item()),
            balanced_accuracy=float(macro_recall.item()),
            expected_calibration_error=float(expected_calibration_error.item()),
            per_class_precision=[float(value) for value in precision.detach().cpu().tolist()],
            per_class_recall=[float(value) for value in recall.detach().cpu().tolist()],
            per_class_f1=[float(value) for value in f1.detach().cpu().tolist()],
            confusion_matrix=[
                [int(value) for value in row]
                for row in self.confusion_matrix.detach().cpu().tolist()
            ],
            normalized_confusion_matrix=[
                [float(value) for value in row]
                for row in normalized_confusion.detach().cpu().tolist()
            ],
        )


@dataclass(slots=True)
class EpochResult:
    train_loss: float
    val_loss: float
    test_loss: float | None = None
    train_metrics: ClassificationMetrics | None = None
    val_metrics: ClassificationMetrics | None = None
    test_metrics: ClassificationMetrics | None = None


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

    def _amp_context(self):
        return torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled)

    def fit(
        self,
        model: nn.Module,
        train_loader: Iterable,
        val_loader: Iterable,
    ) -> tuple[nn.Module, list[EpochResult]]:
        """Train using only train/validation data and reload the best validation model.

        The held-out test split is intentionally excluded from this method. Evaluate
        the returned best model on test data after training has finished.
        """
        fit_start = perf_counter()
        model = model.to(self.device)
        criterion = self._criterion()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.epochs)
        best_val_loss = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        history: list[EpochResult] = []

        print(f"Starting training for {self.config.epochs} epoch(s) on {self.device}.")
        for epoch in range(self.config.epochs):
            print(f"Starting epoch {epoch + 1}/{self.config.epochs}.")
            train_result = self._run_epoch(
                model=model,
                data_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                description=f"Epoch {epoch + 1}/{self.config.epochs} [Train]",
            )
            val_result = self.evaluate(
                model=model,
                data_loader=val_loader,
                criterion=criterion,
                description=f"Epoch {epoch + 1}/{self.config.epochs} [Val]",
            )
            scheduler.step()
            history.append(
                EpochResult(
                    train_loss=train_result.loss,
                    val_loss=val_result.loss,
                    train_metrics=train_result.metrics,
                    val_metrics=val_result.metrics,
                )
            )

            if val_result.loss < best_val_loss:
                best_val_loss = val_result.loss
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                best_path = Path(self.config.best_model_path)
                best_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), best_path)
                print(
                    f"Saved new best model to {best_path} from epoch {best_epoch} "
                    f"with val_loss={best_val_loss:.6f}."
                )
            else:
                epochs_without_improvement += 1

            print(
                f"Epoch {epoch + 1}/{self.config.epochs} completed: "
                f"train_loss={train_result.loss:.6f}, val_loss={val_result.loss:.6f}, "
                f"train_acc={train_result.metrics.accuracy:.4f}, "
                f"val_acc={val_result.metrics.accuracy:.4f}, "
                f"val_macro_f1={val_result.metrics.macro_f1:.4f}, "
                f"val_ece={val_result.metrics.expected_calibration_error:.4f}."
            )
            if (
                self.config.early_stopping_patience > 0
                and epochs_without_improvement >= self.config.early_stopping_patience
            ):
                print(
                    "Early stopping triggered after "
                    f"{epochs_without_improvement} epoch(s) without validation loss improvement. "
                    f"Best val_loss={best_val_loss:.6f}."
                )
                break

        if best_epoch:
            best_path = Path(self.config.best_model_path)
            model.load_state_dict(torch.load(best_path, map_location=self.device, weights_only=True))
            print(
                f"Best model selected from epoch {best_epoch}: "
                f"val_loss={best_val_loss:.6f}."
            )
        print(f"Training finished ({format_duration(perf_counter() - fit_start)}).")
        return model, history

    def _run_epoch(
        self,
        model: nn.Module,
        data_loader: Iterable,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        description: str,
    ) -> EvaluationResult:
        model.train()
        total_loss = 0.0
        batch_count = 0
        metrics = ClassificationMetricAccumulator(device=self.device)

        for x_batch, t_batch, y_batch in tqdm(data_loader, desc=description):
            x_batch = x_batch.to(self.device, non_blocking=True)
            t_batch = t_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with self._amp_context():
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

            metrics.update(logits, y_batch)
            total_loss += float(loss.item())
            batch_count += 1

        return EvaluationResult(loss=total_loss / max(batch_count, 1), metrics=metrics.compute())

    def evaluate(
        self,
        model: nn.Module,
        data_loader: Iterable,
        criterion: nn.Module | None = None,
        description: str = "Validation",
    ) -> EvaluationResult:
        model.eval()
        criterion = criterion or self._criterion()
        total_loss = 0.0
        batch_count = 0
        metrics = ClassificationMetricAccumulator(device=self.device)

        with torch.no_grad():
            for x_batch, t_batch, y_batch in tqdm(data_loader, desc=description):
                x_batch = x_batch.to(self.device, non_blocking=True)
                t_batch = t_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)

                logits = model(x_batch, t_batch)
                total_loss += float(criterion(logits, y_batch).item())
                metrics.update(logits, y_batch)
                batch_count += 1

        return EvaluationResult(loss=total_loss / max(batch_count, 1), metrics=metrics.compute())


def train_lob_transformer(
    model: nn.Module,
    train_loader: Iterable,
    val_loader: Iterable,
    device: str = "cuda",
    epochs: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    gamma: float = 2.0,
) -> nn.Module:
    config = load_config().training
    config.device = device
    config.epochs = epochs
    config.learning_rate = lr
    config.weight_decay = weight_decay
    config.focal_gamma = gamma
    trainer = LobTrainer(config)
    trained_model, _ = trainer.fit(model, train_loader, val_loader)
    return trained_model
