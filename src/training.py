from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

try:
    from configuration import TrainingConfig, load_config
    from monitoring import monitor_value
    from pr_metrics import per_class_ranking_metrics
    from run_logging import format_duration
except ImportError:  # pragma: no cover
    from .configuration import TrainingConfig, load_config
    from .monitoring import monitor_value
    from .pr_metrics import per_class_ranking_metrics
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
    beta: float = 0.25,
    min_weight: float = 0.5,
    max_weight: float = 3.0,
) -> tuple[list[float], list[int]]:
    """Compute clipped balanced class weights from sequence-level train labels."""
    labels = np.asarray(y, dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes)[:num_classes]
    return class_weights_from_class_counts(
        counts,
        beta=beta,
        min_weight=min_weight,
        max_weight=max_weight,
    )


def class_weights_from_class_counts(
    counts: np.ndarray | list[int],
    beta: float = 0.25,
    min_weight: float = 0.5,
    max_weight: float = 3.0,
) -> tuple[list[float], list[int]]:
    """Compute clipped balanced class weights from per-class counts."""
    counts = np.asarray(counts, dtype=np.int64)
    beta = float(beta)
    min_weight = float(min_weight)
    max_weight = float(max_weight)
    if beta < 0.0:
        raise ValueError("class weight beta must be >= 0.")
    if min_weight <= 0.0:
        raise ValueError("class weight minimum must be > 0.")
    if max_weight < min_weight:
        raise ValueError("class weight maximum must be >= minimum.")

    total = int(counts.sum())
    if total <= 0:
        raise ValueError("Cannot compute class weights from empty class counts.")

    weights = total / (counts.size * np.maximum(counts, 1))
    weights = weights ** beta
    weights = weights / weights.mean()
    weights = np.clip(weights, min_weight, max_weight)
    return weights.astype(float).tolist(), counts.astype(int).tolist()


@dataclass(slots=True)
class ClassificationMetrics:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    directional_macro_f1: float
    weighted_f1: float
    balanced_accuracy: float
    expected_calibration_error: float
    per_class_expected_calibration_error: list[float]
    per_class_pr_ap: list[float] | None
    per_class_pr_auc: list[float] | None
    per_class_roc_auc: list[float] | None
    per_class_precision: list[float]
    per_class_recall: list[float]
    per_class_f1: list[float]
    confusion_matrix: list[list[int]]
    normalized_confusion_matrix: list[list[float]]


def classification_metrics_from_predictions(
    targets: np.ndarray,
    predictions: np.ndarray,
    *,
    num_classes: int,
    probabilities: np.ndarray | None = None,
    num_calibration_bins: int = 15,
) -> ClassificationMetrics:
    """Compute metrics from fixed predictions and optional class probabilities."""
    target_array = np.asarray(targets, dtype=np.int64).reshape(-1)
    prediction_array = np.asarray(predictions, dtype=np.int64).reshape(-1)
    if target_array.shape[0] != prediction_array.shape[0]:
        raise ValueError("targets and predictions must have the same length.")

    probability_array = None
    if probabilities is not None:
        probability_array = np.asarray(probabilities, dtype=np.float32)
        if probability_array.ndim == 1 and probability_array.size == 0:
            probability_array = np.empty((0, int(num_classes)), dtype=np.float32)
        if probability_array.ndim != 2:
            raise ValueError("probabilities must be a 2D array.")
        if probability_array.shape[0] != target_array.shape[0]:
            raise ValueError("probabilities and targets must have the same number of rows.")
        if probability_array.shape[1] < int(num_classes):
            raise ValueError("probabilities has fewer columns than num_classes.")

    num_classes = int(num_classes)
    valid_mask = (
        (target_array >= 0)
        & (target_array < num_classes)
        & (prediction_array >= 0)
        & (prediction_array < num_classes)
    )
    if not bool(np.any(valid_mask)):
        return ClassificationMetricAccumulator._zero_metrics(num_classes)

    valid_targets = target_array[valid_mask]
    valid_predictions = prediction_array[valid_mask]
    flat_indices = valid_targets * num_classes + valid_predictions
    confusion = np.bincount(flat_indices, minlength=num_classes * num_classes).reshape(
        num_classes,
        num_classes,
    )
    confusion_float = confusion.astype(np.float64, copy=False)
    total = float(confusion_float.sum())
    true_positives = np.diag(confusion_float)
    support = confusion_float.sum(axis=1)
    predicted = confusion_float.sum(axis=0)

    precision = np.divide(true_positives, predicted, out=np.zeros(num_classes, dtype=np.float64), where=predicted > 0)
    recall = np.divide(true_positives, support, out=np.zeros(num_classes, dtype=np.float64), where=support > 0)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(num_classes, dtype=np.float64),
        where=(precision + recall) > 0,
    )
    normalized_confusion = np.divide(
        confusion_float,
        support[:, None],
        out=np.zeros_like(confusion_float),
        where=support[:, None] > 0,
    )
    directional_class_indices = [index for index in (0, 2) if index < num_classes]
    directional_macro_f1 = float(np.mean(f1[directional_class_indices])) if directional_class_indices else 0.0

    expected_calibration_error = 0.0
    per_class_expected_calibration_error = [0.0] * num_classes
    per_class_pr_ap = None
    per_class_pr_auc = None
    per_class_roc_auc = None
    if probability_array is not None:
        valid_probabilities = probability_array[valid_mask, :num_classes]
        chosen_confidences = valid_probabilities[np.arange(valid_predictions.shape[0]), valid_predictions]
        bin_indices = np.minimum((chosen_confidences * num_calibration_bins).astype(np.int64), num_calibration_bins - 1)
        bin_counts = np.bincount(bin_indices, minlength=num_calibration_bins).astype(np.float64)
        bin_confidence_sums = np.bincount(
            bin_indices,
            weights=chosen_confidences.astype(np.float64),
            minlength=num_calibration_bins,
        )
        correctness = (valid_predictions == valid_targets).astype(np.float64)
        bin_correct_sums = np.bincount(bin_indices, weights=correctness, minlength=num_calibration_bins)
        non_empty_bins = bin_counts > 0
        bin_accuracy = np.zeros(num_calibration_bins, dtype=np.float64)
        bin_confidence = np.zeros(num_calibration_bins, dtype=np.float64)
        bin_accuracy[non_empty_bins] = bin_correct_sums[non_empty_bins] / bin_counts[non_empty_bins]
        bin_confidence[non_empty_bins] = bin_confidence_sums[non_empty_bins] / bin_counts[non_empty_bins]
        expected_calibration_error = float(
            np.sum((bin_counts / max(float(bin_counts.sum()), 1.0)) * np.abs(bin_accuracy - bin_confidence))
        )

        per_class_ece: list[float] = []
        for class_id in range(num_classes):
            class_scores = valid_probabilities[:, class_id]
            class_bins = np.minimum((class_scores * num_calibration_bins).astype(np.int64), num_calibration_bins - 1)
            class_bin_counts = np.bincount(class_bins, minlength=num_calibration_bins).astype(np.float64)
            class_confidence_sums = np.bincount(
                class_bins,
                weights=class_scores.astype(np.float64),
                minlength=num_calibration_bins,
            )
            class_positive_sums = np.bincount(
                class_bins,
                weights=(valid_targets == class_id).astype(np.float64),
                minlength=num_calibration_bins,
            )
            class_non_empty = class_bin_counts > 0
            class_positive_rate = np.zeros(num_calibration_bins, dtype=np.float64)
            class_confidence = np.zeros(num_calibration_bins, dtype=np.float64)
            class_positive_rate[class_non_empty] = (
                class_positive_sums[class_non_empty] / class_bin_counts[class_non_empty]
            )
            class_confidence[class_non_empty] = class_confidence_sums[class_non_empty] / class_bin_counts[class_non_empty]
            per_class_ece.append(
                float(
                    np.sum(
                        (class_bin_counts / max(float(class_bin_counts.sum()), 1.0))
                        * np.abs(class_positive_rate - class_confidence)
                    )
                )
            )
        per_class_expected_calibration_error = per_class_ece
        ranking_metrics = per_class_ranking_metrics(valid_probabilities, valid_targets, num_classes)
        per_class_pr_ap = ranking_metrics["pr_ap"]  # type: ignore[assignment]
        per_class_pr_auc = ranking_metrics["pr_auc"]  # type: ignore[assignment]
        per_class_roc_auc = ranking_metrics["roc_auc"]  # type: ignore[assignment]

    return ClassificationMetrics(
        accuracy=float(true_positives.sum() / max(total, 1.0)),
        macro_precision=float(np.mean(precision)),
        macro_recall=float(np.mean(recall)),
        macro_f1=float(np.mean(f1)),
        directional_macro_f1=directional_macro_f1,
        weighted_f1=float(np.sum(f1 * support) / max(float(support.sum()), 1.0)),
        balanced_accuracy=float(np.mean(recall)),
        expected_calibration_error=expected_calibration_error,
        per_class_expected_calibration_error=per_class_expected_calibration_error,
        per_class_pr_ap=per_class_pr_ap,
        per_class_pr_auc=per_class_pr_auc,
        per_class_roc_auc=per_class_roc_auc,
        per_class_precision=[float(value) for value in precision.tolist()],
        per_class_recall=[float(value) for value in recall.tolist()],
        per_class_f1=[float(value) for value in f1.tolist()],
        confusion_matrix=[[int(value) for value in row] for row in confusion.tolist()],
        normalized_confusion_matrix=[
            [float(value) for value in row]
            for row in normalized_confusion.tolist()
        ],
    )


@dataclass(slots=True)
class EvaluationResult:
    loss: float
    metrics: ClassificationMetrics
    expert_usage: dict[str, Any] | None = None
    prediction_outputs: dict[str, Any] | None = None


class ExpertUsageAccumulator:
    def __init__(self, num_classes: int | None = None) -> None:
        self.num_classes = num_classes
        self.num_experts: int | None = None
        self.top_k: int | None = None
        self.total_sequences = 0
        self.total_tokens = 0
        self.total_assignments = 0
        self.selected_counts: torch.Tensor | None = None
        self.primary_counts: torch.Tensor | None = None
        self.router_probability_sums: torch.Tensor | None = None
        self.selected_weight_sums: torch.Tensor | None = None
        self.class_sequence_counts: torch.Tensor | None = None
        self.class_token_counts: torch.Tensor | None = None
        self.class_assignment_counts: torch.Tensor | None = None
        self.class_selected_counts: torch.Tensor | None = None
        self.class_primary_counts: torch.Tensor | None = None
        self.class_router_probability_sums: torch.Tensor | None = None
        self.class_selected_weight_sums: torch.Tensor | None = None

    def _ensure_initialized(self, num_experts: int, top_k: int, num_classes: int) -> None:
        if self.num_experts is None:
            self.num_experts = int(num_experts)
            self.top_k = int(top_k)
            self.num_classes = int(num_classes)
            self.selected_counts = torch.zeros(self.num_experts, dtype=torch.long)
            self.primary_counts = torch.zeros(self.num_experts, dtype=torch.long)
            self.router_probability_sums = torch.zeros(self.num_experts, dtype=torch.float64)
            self.selected_weight_sums = torch.zeros(self.num_experts, dtype=torch.float64)
            self.class_sequence_counts = torch.zeros(self.num_classes, dtype=torch.long)
            self.class_token_counts = torch.zeros(self.num_classes, dtype=torch.long)
            self.class_assignment_counts = torch.zeros(self.num_classes, dtype=torch.long)
            self.class_selected_counts = torch.zeros((self.num_classes, self.num_experts), dtype=torch.long)
            self.class_primary_counts = torch.zeros((self.num_classes, self.num_experts), dtype=torch.long)
            self.class_router_probability_sums = torch.zeros(
                (self.num_classes, self.num_experts),
                dtype=torch.float64,
            )
            self.class_selected_weight_sums = torch.zeros(
                (self.num_classes, self.num_experts),
                dtype=torch.float64,
            )
            return

        if self.num_experts != int(num_experts) or self.top_k != int(top_k):
            raise ValueError("MoE routing shape changed during accumulation.")

    @staticmethod
    def _counts(values: torch.Tensor, num_experts: int) -> torch.Tensor:
        return torch.bincount(values.reshape(-1), minlength=num_experts)[:num_experts]

    def update(
        self,
        routing: dict[str, Any] | None,
        targets: torch.Tensor,
        *,
        num_classes: int,
    ) -> None:
        if not routing:
            return
        topk_indices = routing.get("topk_indices")
        topk_weights = routing.get("topk_weights")
        router_probabilities = routing.get("router_probabilities")
        if not isinstance(topk_indices, torch.Tensor):
            return
        if not isinstance(topk_weights, torch.Tensor) or not isinstance(router_probabilities, torch.Tensor):
            return

        indices = topk_indices.detach().to("cpu", dtype=torch.long)
        selected_weights = topk_weights.detach().to("cpu", dtype=torch.float64)
        probabilities = router_probabilities.detach().to("cpu", dtype=torch.float64)
        batch_size, sequence_length, top_k = indices.shape
        num_experts = int(routing.get("num_experts", probabilities.shape[-1]))
        self._ensure_initialized(num_experts, int(routing.get("top_k", top_k)), num_classes)

        if (
            self.selected_counts is None
            or self.primary_counts is None
            or self.router_probability_sums is None
            or self.selected_weight_sums is None
            or self.class_sequence_counts is None
            or self.class_token_counts is None
            or self.class_assignment_counts is None
            or self.class_selected_counts is None
            or self.class_primary_counts is None
            or self.class_router_probability_sums is None
            or self.class_selected_weight_sums is None
        ):
            raise RuntimeError("Expert usage accumulator was not initialized.")

        flat_indices = indices.reshape(-1)
        flat_weights = selected_weights.reshape(-1)
        self.total_sequences += int(batch_size)
        self.total_tokens += int(batch_size * sequence_length)
        self.total_assignments += int(batch_size * sequence_length * top_k)
        self.selected_counts += self._counts(flat_indices, num_experts)
        self.primary_counts += self._counts(indices[:, :, 0], num_experts)
        self.router_probability_sums += probabilities.sum(dim=(0, 1))
        self.selected_weight_sums.scatter_add_(0, flat_indices, flat_weights)

        targets_cpu = targets.detach().to("cpu", dtype=torch.long)
        for class_id in range(num_classes):
            class_mask = targets_cpu == class_id
            class_sequences = int(class_mask.sum().item())
            if class_sequences == 0:
                continue
            class_indices = indices[class_mask]
            class_weights = selected_weights[class_mask]
            class_probabilities = probabilities[class_mask]
            self.class_sequence_counts[class_id] += class_sequences
            self.class_token_counts[class_id] += class_sequences * sequence_length
            self.class_assignment_counts[class_id] += class_sequences * sequence_length * top_k
            self.class_selected_counts[class_id] += self._counts(class_indices, num_experts)
            self.class_primary_counts[class_id] += self._counts(class_indices[:, :, 0], num_experts)
            self.class_router_probability_sums[class_id] += class_probabilities.sum(dim=(0, 1))
            self.class_selected_weight_sums[class_id].scatter_add_(
                0,
                class_indices.reshape(-1),
                class_weights.reshape(-1),
            )

    @staticmethod
    def _percentages(counts: torch.Tensor, total: int) -> list[float]:
        if total <= 0:
            return [0.0 for _ in counts.tolist()]
        return [float(100.0 * value / total) for value in counts.tolist()]

    @staticmethod
    def _means(sums: torch.Tensor, counts: torch.Tensor) -> list[float]:
        denominators = counts.to(torch.float64).clamp_min(1.0)
        return [float(value) for value in (sums / denominators).tolist()]

    @staticmethod
    def _averages(sums: torch.Tensor, total: int) -> list[float]:
        if total <= 0:
            return [0.0 for _ in sums.tolist()]
        return [float(value / total) for value in sums.tolist()]

    def compute(self) -> dict[str, Any] | None:
        if self.num_experts is None or self.top_k is None:
            return None
        if (
            self.selected_counts is None
            or self.primary_counts is None
            or self.router_probability_sums is None
            or self.selected_weight_sums is None
            or self.class_sequence_counts is None
            or self.class_token_counts is None
            or self.class_assignment_counts is None
            or self.class_selected_counts is None
            or self.class_primary_counts is None
            or self.class_router_probability_sums is None
            or self.class_selected_weight_sums is None
        ):
            raise RuntimeError("Expert usage accumulator was not initialized.")

        by_class: dict[str, Any] = {}
        for class_id in range(int(self.num_classes or 0)):
            token_count = int(self.class_token_counts[class_id].item())
            assignment_count = int(self.class_assignment_counts[class_id].item())
            selected_counts = self.class_selected_counts[class_id]
            primary_counts = self.class_primary_counts[class_id]
            by_class[str(class_id)] = {
                "sequences": int(self.class_sequence_counts[class_id].item()),
                "tokens": token_count,
                "assignments": assignment_count,
                "selected_counts": [int(value) for value in selected_counts.tolist()],
                "selected_percentages": self._percentages(selected_counts, assignment_count),
                "primary_counts": [int(value) for value in primary_counts.tolist()],
                "primary_percentages": self._percentages(primary_counts, token_count),
                "mean_router_probability": self._averages(
                    self.class_router_probability_sums[class_id],
                    token_count,
                ),
                "selected_weight_sums": [
                    float(value) for value in self.class_selected_weight_sums[class_id].tolist()
                ],
                "mean_selected_weight": self._means(
                    self.class_selected_weight_sums[class_id],
                    selected_counts,
                ),
            }

        return {
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "sequences": self.total_sequences,
            "tokens": self.total_tokens,
            "assignments": self.total_assignments,
            "selected_counts": [int(value) for value in self.selected_counts.tolist()],
            "selected_percentages": self._percentages(self.selected_counts, self.total_assignments),
            "primary_counts": [int(value) for value in self.primary_counts.tolist()],
            "primary_percentages": self._percentages(self.primary_counts, self.total_tokens),
            "mean_router_probability": self._averages(self.router_probability_sums, self.total_tokens),
            "selected_weight_sums": [float(value) for value in self.selected_weight_sums.tolist()],
            "mean_selected_weight": self._means(self.selected_weight_sums, self.selected_counts),
            "by_true_class": by_class,
        }


class ClassificationMetricAccumulator:
    def __init__(
        self,
        device: torch.device,
        num_calibration_bins: int = 15,
        *,
        track_pr_metrics: bool = False,
    ) -> None:
        self.device = device
        self.num_calibration_bins = num_calibration_bins
        self.track_pr_metrics = bool(track_pr_metrics)
        self.num_classes: int | None = None
        self.confusion_matrix: torch.Tensor | None = None
        self.calibration_bin_counts: torch.Tensor | None = None
        self.calibration_confidence_sums: torch.Tensor | None = None
        self.calibration_correct_sums: torch.Tensor | None = None
        self.class_calibration_bin_counts: torch.Tensor | None = None
        self.class_calibration_confidence_sums: torch.Tensor | None = None
        self.class_calibration_positive_sums: torch.Tensor | None = None
        self.pr_probability_chunks: list[np.ndarray] = []
        self.pr_target_chunks: list[np.ndarray] = []

    @staticmethod
    def _zero_metrics(num_classes: int = 0) -> ClassificationMetrics:
        return ClassificationMetrics(
            accuracy=0.0,
            macro_precision=0.0,
            macro_recall=0.0,
            macro_f1=0.0,
            directional_macro_f1=0.0,
            weighted_f1=0.0,
            balanced_accuracy=0.0,
            expected_calibration_error=0.0,
            per_class_expected_calibration_error=[0.0] * num_classes,
            per_class_pr_ap=None,
            per_class_pr_auc=None,
            per_class_roc_auc=None,
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
            self.class_calibration_bin_counts = torch.zeros(
                (self.num_classes, self.num_calibration_bins),
                dtype=torch.float64,
                device=self.device,
            )
            self.class_calibration_confidence_sums = torch.zeros_like(self.class_calibration_bin_counts)
            self.class_calibration_positive_sums = torch.zeros_like(self.class_calibration_bin_counts)
        if self.confusion_matrix is None:
            raise RuntimeError("Classification metric accumulator was not initialized.")
        if (
            self.calibration_bin_counts is None
            or self.calibration_confidence_sums is None
            or self.calibration_correct_sums is None
            or self.class_calibration_bin_counts is None
            or self.class_calibration_confidence_sums is None
            or self.class_calibration_positive_sums is None
        ):
            raise RuntimeError("Calibration metric accumulator was not initialized.")

        logits = logits.detach().float()
        targets = targets.detach().reshape(-1)
        if logits.ndim != 2:
            logits = logits.reshape(-1, logits.shape[-1])
        if logits.shape[0] != targets.shape[0]:
            raise ValueError("logits and targets have inconsistent row counts.")

        finite_logits_mask = torch.isfinite(logits).all(dim=-1)
        valid_target_mask = (targets >= 0) & (targets < self.num_classes)
        valid_sample_mask = finite_logits_mask & valid_target_mask
        if not bool(valid_sample_mask.any()):
            return

        logits = logits[valid_sample_mask]
        targets = targets[valid_sample_mask].long()
        probabilities = torch.softmax(logits, dim=-1)
        finite_prob_mask = torch.isfinite(probabilities).all(dim=-1)
        if not bool(finite_prob_mask.any()):
            return

        probabilities = probabilities[finite_prob_mask]
        targets = targets[finite_prob_mask]
        confidences, predictions = torch.max(probabilities, dim=-1)

        flat_indices = targets * self.num_classes + predictions
        counts = torch.bincount(flat_indices.long(), minlength=self.num_classes * self.num_classes)
        self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes)

        valid_confidences = confidences.to(torch.float64)
        correctness = (predictions == targets).to(torch.float64)
        bin_indices = torch.clamp(
            (valid_confidences * self.num_calibration_bins).long(),
            min=0,
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

        valid_probabilities = probabilities
        valid_targets = targets
        if self.track_pr_metrics:
            self.pr_probability_chunks.append(valid_probabilities.detach().cpu().numpy().astype(np.float32, copy=False))
            self.pr_target_chunks.append(valid_targets.detach().cpu().numpy())
        valid_probabilities = valid_probabilities.to(torch.float64)
        class_bin_indices = torch.clamp(
            (valid_probabilities * self.num_calibration_bins).long(),
            min=0,
            max=self.num_calibration_bins - 1,
        )
        for class_id in range(self.num_classes):
            class_bins = class_bin_indices[:, class_id]
            class_confidences = valid_probabilities[:, class_id]
            class_positives = (valid_targets == class_id).to(torch.float64)
            self.class_calibration_bin_counts[class_id] += torch.bincount(
                class_bins,
                minlength=self.num_calibration_bins,
            ).to(torch.float64)
            self.class_calibration_confidence_sums[class_id] += torch.bincount(
                class_bins,
                weights=class_confidences,
                minlength=self.num_calibration_bins,
            )
            self.class_calibration_positive_sums[class_id] += torch.bincount(
                class_bins,
                weights=class_positives,
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
        directional_class_indices = [index for index in (0, 2) if index < len(f1)]
        directional_macro_f1 = (
            f1[directional_class_indices].mean()
            if directional_class_indices
            else torch.zeros((), dtype=torch.float64, device=self.device)
        )
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
        per_class_expected_calibration_error = torch.zeros(
            int(confusion.shape[0]),
            dtype=torch.float64,
            device=self.device,
        )
        if (
            self.class_calibration_bin_counts is not None
            and self.class_calibration_confidence_sums is not None
            and self.class_calibration_positive_sums is not None
        ):
            non_empty_class_bins = self.class_calibration_bin_counts > 0
            class_bin_positive_rate = torch.zeros_like(self.class_calibration_bin_counts)
            class_bin_confidence = torch.zeros_like(self.class_calibration_bin_counts)
            class_bin_positive_rate[non_empty_class_bins] = (
                self.class_calibration_positive_sums[non_empty_class_bins]
                / self.class_calibration_bin_counts[non_empty_class_bins]
            )
            class_bin_confidence[non_empty_class_bins] = (
                self.class_calibration_confidence_sums[non_empty_class_bins]
                / self.class_calibration_bin_counts[non_empty_class_bins]
            )
            class_bin_weights = (
                self.class_calibration_bin_counts
                / self.class_calibration_bin_counts.sum(dim=1, keepdim=True).clamp_min(1.0)
            )
            per_class_expected_calibration_error = torch.sum(
                class_bin_weights * torch.abs(class_bin_positive_rate - class_bin_confidence),
                dim=1,
            )
        per_class_pr_ap = None
        per_class_pr_auc_values = None
        per_class_roc_auc_values = None
        if self.track_pr_metrics and self.pr_probability_chunks and self.pr_target_chunks:
            pr_probabilities = np.concatenate(self.pr_probability_chunks, axis=0)
            pr_targets = np.concatenate(self.pr_target_chunks, axis=0)
            ranking_metrics = per_class_ranking_metrics(pr_probabilities, pr_targets, int(confusion.shape[0]))
            per_class_pr_ap = ranking_metrics["pr_ap"]  # type: ignore[assignment]
            per_class_pr_auc_values = ranking_metrics["pr_auc"]  # type: ignore[assignment]
            per_class_roc_auc_values = ranking_metrics["roc_auc"]  # type: ignore[assignment]

        return ClassificationMetrics(
            accuracy=float(accuracy.item()),
            macro_precision=float(macro_precision.item()),
            macro_recall=float(macro_recall.item()),
            macro_f1=float(macro_f1.item()),
            directional_macro_f1=float(directional_macro_f1.item()),
            weighted_f1=float(weighted_f1.item()),
            balanced_accuracy=float(macro_recall.item()),
            expected_calibration_error=float(expected_calibration_error.item()),
            per_class_expected_calibration_error=[
                float(value) for value in per_class_expected_calibration_error.detach().cpu().tolist()
            ],
            per_class_pr_ap=per_class_pr_ap,
            per_class_pr_auc=per_class_pr_auc_values,
            per_class_roc_auc=per_class_roc_auc_values,
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


class PredictionOutputAccumulator:
    """Collect evaluation outputs for probability, logit, and PR artifacts."""

    def __init__(self) -> None:
        self.logit_chunks: list[np.ndarray] = []
        self.probability_chunks: list[np.ndarray] = []
        self.target_chunks: list[np.ndarray] = []
        self.prediction_chunks: list[np.ndarray] = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Collect logits, post-softmax probabilities, and labels for one batch."""
        detached_logits = logits.detach().float()
        self.logit_chunks.append(detached_logits.cpu().numpy().astype(np.float32, copy=False))
        probabilities = torch.softmax(detached_logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)
        self.probability_chunks.append(probabilities.cpu().numpy().astype(np.float32, copy=False))
        self.target_chunks.append(targets.detach().to("cpu", dtype=torch.long).numpy())
        self.prediction_chunks.append(predictions.cpu().numpy())

    def compute(self) -> dict[str, Any]:
        """Return concatenated prediction outputs for artifact logging."""
        if not self.probability_chunks:
            return {
                "sample_index": np.asarray([], dtype=np.int64),
                "targets": np.asarray([], dtype=np.int64),
                "predictions": np.asarray([], dtype=np.int64),
                "probabilities": np.empty((0, 0), dtype=np.float32),
                "logits": np.empty((0, 0), dtype=np.float32),
            }
        logits = np.concatenate(self.logit_chunks, axis=0)
        probabilities = np.concatenate(self.probability_chunks, axis=0)
        targets = np.concatenate(self.target_chunks, axis=0).astype(np.int64, copy=False)
        predictions = np.concatenate(self.prediction_chunks, axis=0).astype(np.int64, copy=False)
        return {
            "sample_index": np.arange(targets.shape[0], dtype=np.int64),
            "targets": targets,
            "predictions": predictions,
            "probabilities": probabilities,
            "logits": logits,
        }


@dataclass(slots=True)
class EpochResult:
    train_loss: float
    val_loss: float
    test_loss: float | None = None
    train_metrics: ClassificationMetrics | None = None
    val_metrics: ClassificationMetrics | None = None
    test_metrics: ClassificationMetrics | None = None
    val_threshold_metrics: dict[str, float] | None = None
    test_threshold_metrics: dict[str, float] | None = None
    val_argmax_metrics: ClassificationMetrics | None = None
    test_argmax_metrics: ClassificationMetrics | None = None
    train_expert_usage: dict[str, Any] | None = None
    val_expert_usage: dict[str, Any] | None = None
    test_expert_usage: dict[str, Any] | None = None
    epoch: int | None = None
    batch_in_epoch: int | None = None
    global_step: int | None = None
    validation_index: int | None = None
    checkpoint_label: str | None = None


@dataclass(slots=True)
class CheckpointCandidate:
    epoch: int
    monitor_value: float
    path: Path
    batch_in_epoch: int | None = None
    global_step: int | None = None
    validation_index: int | None = None
    checkpoint_label: str | None = None


class LobTrainer:
    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or load_config().training
        self.device = torch.device(self.config.device)
        self.top_checkpoint_candidates: list[CheckpointCandidate] = []
        self.amp_enabled = self.config.use_amp and self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if self.amp_enabled and self._cuda_supports_bf16(self.device) else None
        self.scaler = torch.amp.GradScaler(
            device=self.device.type,
            enabled=self.amp_enabled and self.amp_dtype is None,
        )
        if self.amp_enabled:
            amp_dtype_name = "bfloat16" if self.amp_dtype is torch.bfloat16 else "default"
            print(
                "AMP enabled: "
                f"autocast_dtype={amp_dtype_name}, "
                f"grad_scaler_enabled={self.scaler.is_enabled()}."
            )

    @staticmethod
    def _cuda_supports_bf16(device: torch.device) -> bool:
        """Return whether the selected CUDA device supports bf16 autocast."""
        if device.type != "cuda" or not torch.cuda.is_available():
            return False
        if hasattr(torch.cuda, "is_bf16_supported"):
            try:
                device_index = torch.cuda.current_device() if device.index is None else int(device.index)
                with torch.cuda.device(device_index):
                    return bool(torch.cuda.is_bf16_supported())
            except (AssertionError, RuntimeError):
                return False
        return False

    def _criterion(self) -> FocalLoss:
        alpha = None
        if self.config.class_weights is not None:
            alpha = torch.tensor(self.config.class_weights, dtype=torch.float32, device=self.device)
        return FocalLoss(alpha=alpha, gamma=self.config.focal_gamma).to(self.device)

    def _amp_context(self):
        if self.amp_dtype is None:
            return torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled)
        return torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled, dtype=self.amp_dtype)

    @staticmethod
    def _set_epoch(data_loader: Iterable, epoch: int) -> None:
        """Notify epoch-aware samplers before iterating a loader."""
        sampler = getattr(data_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    def _monitor_value(self, result: EvaluationResult) -> float:
        """Return the configured validation monitor value."""
        return monitor_value(
            loss=result.loss,
            metrics=result.metrics,
            monitor=self.config.monitor,
            monitor_params=self.config.monitor_params,
        )

    def _is_improvement(self, value: float, best_value: float) -> bool:
        """Compare a monitor value against the current best value."""
        min_delta = float(self.config.early_stopping_min_delta)
        if self.config.monitor_mode == "min":
            return value < best_value - min_delta
        return value > best_value + min_delta

    def _rank_checkpoint_candidates(
        self,
        candidates: Iterable[CheckpointCandidate],
    ) -> list[CheckpointCandidate]:
        """Return checkpoint candidates sorted from best to worst."""
        reverse_sign = -1.0 if self.config.monitor_mode == "max" else 1.0
        return sorted(
            candidates,
            key=lambda item: (
                reverse_sign * float(item.monitor_value),
                int(item.validation_index if item.validation_index is not None else item.epoch),
                int(item.global_step if item.global_step is not None else 0),
            ),
        )

    def _save_top_checkpoint_candidate(
        self,
        model: nn.Module,
        *,
        epoch: int,
        batch_in_epoch: int | None = None,
        global_step: int | None = None,
        validation_index: int | None = None,
        checkpoint_label: str | None = None,
        monitor_value: float,
    ) -> None:
        """Persist this epoch if it belongs to the configured top-k candidates."""
        checkpoint_label = checkpoint_label or self.config.checkpoint_label(epoch, global_step=global_step)
        candidate = CheckpointCandidate(
            epoch=int(epoch),
            monitor_value=float(monitor_value),
            path=self.config.checkpoint_path(epoch, global_step=global_step),
            batch_in_epoch=None if batch_in_epoch is None else int(batch_in_epoch),
            global_step=None if global_step is None else int(global_step),
            validation_index=None if validation_index is None else int(validation_index),
            checkpoint_label=checkpoint_label,
        )
        ranked = self._rank_checkpoint_candidates([*self.top_checkpoint_candidates, candidate])
        kept = ranked[: self.config.top_k_checkpoints]
        dropped = ranked[self.config.top_k_checkpoints :]
        if any(item.checkpoint_label == candidate.checkpoint_label for item in kept):
            candidate.path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), candidate.path)
        for item in dropped:
            if item.path.exists():
                item.path.unlink()
        self.top_checkpoint_candidates = kept

    def _optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Build the configured optimizer."""
        optimizer_name = self.config.optimizer.lower()
        optimizer_class = torch.optim.Adam if optimizer_name == "adam" else torch.optim.AdamW
        return optimizer_class(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

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
        optimizer = self._optimizer(model)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.epochs)
        best_monitor_value = float("inf") if self.config.monitor_mode == "min" else -float("inf")
        best_epoch = 0
        best_checkpoint_label = ""
        validations_without_improvement = 0
        validation_index = 0
        global_step = 0
        history: list[EpochResult] = []
        self.top_checkpoint_candidates = []
        validate_by_epoch = bool(self.config.validates_by_epoch)
        validation_interval = None if validate_by_epoch else int(self.config.validate_every_n_batches)
        validation_unit = "epoch" if validate_by_epoch else "validation interval"

        def record_validation(
            *,
            epoch_number: int,
            batch_in_epoch: int | None,
            global_step_value: int | None,
            train_result: EvaluationResult,
            val_result: EvaluationResult,
        ) -> bool:
            nonlocal best_monitor_value
            nonlocal best_epoch
            nonlocal best_checkpoint_label
            nonlocal validations_without_improvement
            nonlocal validation_index

            validation_index += 1
            checkpoint_global_step = None if validate_by_epoch else global_step_value
            checkpoint_label = self.config.checkpoint_label(epoch_number, global_step=checkpoint_global_step)
            history.append(
                EpochResult(
                    train_loss=train_result.loss,
                    val_loss=val_result.loss,
                    train_metrics=train_result.metrics,
                    val_metrics=val_result.metrics,
                    train_expert_usage=train_result.expert_usage,
                    val_expert_usage=val_result.expert_usage,
                    epoch=epoch_number,
                    batch_in_epoch=batch_in_epoch,
                    global_step=global_step_value,
                    validation_index=validation_index,
                    checkpoint_label=checkpoint_label,
                )
            )

            monitor_value = self._monitor_value(val_result)
            self._save_top_checkpoint_candidate(
                model,
                epoch=epoch_number,
                batch_in_epoch=batch_in_epoch,
                global_step=checkpoint_global_step,
                validation_index=validation_index,
                checkpoint_label=checkpoint_label,
                monitor_value=monitor_value,
            )
            if self._is_improvement(monitor_value, best_monitor_value):
                best_monitor_value = monitor_value
                best_epoch = epoch_number
                best_checkpoint_label = checkpoint_label
                validations_without_improvement = 0
                best_path = Path(self.config.best_model_path)
                best_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), best_path)
                print(
                    f"Saved new best model to {best_path} from {checkpoint_label} "
                    f"with {self.config.monitor}={best_monitor_value:.6f}."
                )
            else:
                if validation_index > self.config.early_stopping_warmup:
                    validations_without_improvement += 1

            if validate_by_epoch:
                progress = f"Epoch {epoch_number}/{self.config.epochs}"
            else:
                progress = (
                    f"Validation {validation_index} at epoch {epoch_number}/{self.config.epochs}, "
                    f"batch {batch_in_epoch}, global_step={global_step_value}"
                )
            print(
                f"{progress} completed: "
                f"train_loss={train_result.loss:.6f}, val_loss={val_result.loss:.6f}, "
                f"train_acc={train_result.metrics.accuracy:.4f}, "
                f"val_acc={val_result.metrics.accuracy:.4f}, "
                f"val_macro_f1={val_result.metrics.macro_f1:.4f}, "
                f"val_directional_macro_f1={val_result.metrics.directional_macro_f1:.4f}, "
                f"val_ece={val_result.metrics.expected_calibration_error:.4f}."
            )
            if (
                self.config.early_stopping_patience > 0
                and validation_index > self.config.early_stopping_warmup
                and validations_without_improvement >= self.config.early_stopping_patience
            ):
                print(
                    "Early stopping triggered after "
                    f"{validations_without_improvement} {validation_unit}(s) without "
                    f"{self.config.monitor} improvement. "
                    f"Best {self.config.monitor}={best_monitor_value:.6f}; "
                    f"warmup={self.config.early_stopping_warmup} {validation_unit}(s); "
                    f"min_delta={self.config.early_stopping_min_delta:.6g}."
                )
                return True
            return False

        print(
            f"Starting training for {self.config.epochs} epoch(s) on {self.device}. "
            f"Monitoring {self.config.monitor} ({self.config.monitor_mode}); "
            f"saving top_k_checkpoints={self.config.top_k_checkpoints}; "
            f"validate_every_n_batches={self.config.validate_every_n_batches}; "
            f"early stopping warmup={self.config.early_stopping_warmup} {validation_unit}(s), "
            f"min_delta={self.config.early_stopping_min_delta:.6g}; "
            f"optimizer={self.config.optimizer}."
        )
        for epoch in range(self.config.epochs):
            epoch_number = epoch + 1
            self._set_epoch(train_loader, epoch)
            print(f"Starting epoch {epoch_number}/{self.config.epochs}.")
            if validate_by_epoch:
                train_result = self._run_epoch(
                    model=model,
                    data_loader=train_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    description=f"Epoch {epoch_number}/{self.config.epochs} [Train]",
                )
                val_result = self.evaluate(
                    model=model,
                    data_loader=val_loader,
                    criterion=criterion,
                    description=f"Epoch {epoch_number}/{self.config.epochs} [Val]",
                    track_pr_metrics=True,
                )
                scheduler.step()
                if record_validation(
                    epoch_number=epoch_number,
                    batch_in_epoch=None,
                    global_step_value=None,
                    train_result=train_result,
                    val_result=val_result,
                ):
                    break
                continue

            model.train()
            interval_loss = 0.0
            interval_batches = 0
            interval_metrics = ClassificationMetricAccumulator(device=self.device)
            stopped = False
            last_batch_in_epoch = 0
            progress = tqdm(train_loader, desc=f"Epoch {epoch_number}/{self.config.epochs} [Train]")
            for batch_in_epoch, (x_batch, t_batch, y_batch) in enumerate(progress, start=1):
                last_batch_in_epoch = batch_in_epoch
                x_batch = x_batch.to(self.device, non_blocking=True)
                t_batch = t_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with self._amp_context():
                    logits = model(x_batch, t_batch)
                    loss = criterion(logits, y_batch)
                    moe_loss = getattr(model, "moe_load_balancing_loss", None)
                    if moe_loss is not None:
                        loss = loss + moe_loss

                if not bool(torch.isfinite(logits).all()):
                    finite_summary = logits.detach().float().nan_to_num()
                    raise FloatingPointError(
                        f"Non-finite logits at batch {batch_in_epoch}: "
                        f"logits min={finite_summary.min().item()}, "
                        f"max={finite_summary.max().item()}"
                    )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Non-finite loss at batch {batch_in_epoch}: {loss.item()}")

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.config.grad_clip_norm)
                self.scaler.step(optimizer)
                self.scaler.update()

                interval_metrics.update(logits, y_batch)
                interval_loss += float(loss.item())
                interval_batches += 1
                global_step += 1

                if validation_interval is not None and global_step % validation_interval == 0:
                    train_result = EvaluationResult(
                        loss=interval_loss / max(interval_batches, 1),
                        metrics=interval_metrics.compute(),
                    )
                    val_result = self.evaluate(
                        model=model,
                        data_loader=val_loader,
                        criterion=criterion,
                        description=f"{self.config.checkpoint_label(epoch_number, global_step=global_step)} [Val]",
                        track_pr_metrics=True,
                    )
                    stopped = record_validation(
                        epoch_number=epoch_number,
                        batch_in_epoch=batch_in_epoch,
                        global_step_value=global_step,
                        train_result=train_result,
                        val_result=val_result,
                    )
                    interval_loss = 0.0
                    interval_batches = 0
                    interval_metrics = ClassificationMetricAccumulator(device=self.device)
                    model.train()
                    if stopped:
                        break
            if stopped:
                break

            if interval_batches > 0:
                train_result = EvaluationResult(
                    loss=interval_loss / max(interval_batches, 1),
                    metrics=interval_metrics.compute(),
                )
                val_result = self.evaluate(
                    model=model,
                    data_loader=val_loader,
                    criterion=criterion,
                    description=f"{self.config.checkpoint_label(epoch_number, global_step=global_step)} [Val]",
                    track_pr_metrics=True,
                )
                if record_validation(
                    epoch_number=epoch_number,
                    batch_in_epoch=last_batch_in_epoch,
                    global_step_value=global_step,
                    train_result=train_result,
                    val_result=val_result,
                ):
                    break
            scheduler.step()

        if best_epoch:
            best_path = Path(self.config.best_model_path)
            model.load_state_dict(torch.load(best_path, map_location=self.device, weights_only=True))
            print(
                f"Best model selected from {best_checkpoint_label or f'epoch {best_epoch}'}: "
                f"{self.config.monitor}={best_monitor_value:.6f}."
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

            current_batch = batch_count + 1
            if not bool(torch.isfinite(logits).all()):
                finite_summary = logits.detach().float().nan_to_num()
                raise FloatingPointError(
                    f"Non-finite logits at batch {current_batch}: "
                    f"logits min={finite_summary.min().item()}, "
                    f"max={finite_summary.max().item()}"
                )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite loss at batch {current_batch}: {loss.item()}")

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
        *,
        collect_outputs: bool = False,
        track_pr_metrics: bool = False,
        track_expert_usage: bool = False,
    ) -> EvaluationResult:
        model.eval()
        criterion = criterion or self._criterion()
        total_loss = 0.0
        batch_count = 0
        metrics = ClassificationMetricAccumulator(device=self.device, track_pr_metrics=track_pr_metrics)
        expert_usage = ExpertUsageAccumulator() if track_expert_usage else None
        prediction_outputs = PredictionOutputAccumulator() if collect_outputs else None

        with torch.no_grad():
            for x_batch, t_batch, y_batch in tqdm(data_loader, desc=description):
                x_batch = x_batch.to(self.device, non_blocking=True)
                t_batch = t_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)

                logits = model(x_batch, t_batch)
                total_loss += float(criterion(logits, y_batch).item())
                metrics.update(logits, y_batch)
                if prediction_outputs is not None:
                    prediction_outputs.update(logits, y_batch)
                if expert_usage is not None:
                    expert_usage.update(
                        getattr(model, "moe_routing", None),
                        y_batch,
                        num_classes=int(logits.shape[-1]),
                    )
                batch_count += 1

        return EvaluationResult(
            loss=total_loss / max(batch_count, 1),
            metrics=metrics.compute(),
            expert_usage=None if expert_usage is None else expert_usage.compute(),
            prediction_outputs=None if prediction_outputs is None else prediction_outputs.compute(),
        )


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
