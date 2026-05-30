from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class DirectionalThresholdSelection:
    """Store the selected directional thresholds and validation score."""

    threshold_down: float
    threshold_up: float
    score: float
    n_candidates: int


def threshold_candidates(min_threshold: float, max_threshold: float, step: float) -> np.ndarray:
    """Build an inclusive floating-point threshold grid."""
    if step <= 0.0:
        raise ValueError("threshold step must be > 0.")
    if min_threshold > max_threshold:
        raise ValueError("threshold min must be <= max.")
    values: list[float] = []
    current = float(min_threshold)
    while current <= float(max_threshold) + 1e-12:
        values.append(round(current, 12))
        current += float(step)
    return np.asarray(values, dtype=np.float64)


def apply_directional_threshold_policy(
    probabilities: np.ndarray,
    *,
    threshold_down: float,
    threshold_up: float,
    down_id: int,
    neutral_id: int,
    up_id: int,
) -> np.ndarray:
    """Classify samples with fixed down/up probability thresholds."""
    probs = np.asarray(probabilities, dtype=np.float32)
    if probs.ndim != 2:
        raise ValueError("probabilities must be a 2D array.")
    max_id = max(down_id, neutral_id, up_id)
    if probs.shape[1] <= max_id:
        raise ValueError("probabilities has fewer columns than the requested class ids.")

    p_down = probs[:, down_id]
    p_up = probs[:, up_id]
    down_hit = p_down >= float(threshold_down)
    up_hit = p_up >= float(threshold_up)

    predictions = np.full(probs.shape[0], int(neutral_id), dtype=np.int64)
    predictions[down_hit & ~up_hit] = int(down_id)
    predictions[up_hit & ~down_hit] = int(up_id)
    both_hit = down_hit & up_hit
    down_margin = p_down - float(threshold_down)
    up_margin = p_up - float(threshold_up)
    predictions[both_hit & (down_margin >= up_margin)] = int(down_id)
    predictions[both_hit & (up_margin > down_margin)] = int(up_id)
    return predictions


def _precision_recall_f1(targets: np.ndarray, predictions: np.ndarray, class_id: int) -> tuple[float, float, float]:
    """Return one-vs-rest precision, recall, and F1 for one class id."""
    true_positive = int(np.sum((targets == class_id) & (predictions == class_id)))
    false_positive = int(np.sum((targets != class_id) & (predictions == class_id)))
    false_negative = int(np.sum((targets == class_id) & (predictions != class_id)))
    precision = 0.0 if true_positive + false_positive == 0 else true_positive / (true_positive + false_positive)
    recall = 0.0 if true_positive + false_negative == 0 else true_positive / (true_positive + false_negative)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return float(precision), float(recall), float(f1)


def directional_macro_f1_from_predictions(
    targets: np.ndarray,
    predictions: np.ndarray,
    *,
    down_id: int,
    up_id: int,
) -> float:
    """Return macro F1 over down/up classes only."""
    target_array = np.asarray(targets, dtype=np.int64).reshape(-1)
    prediction_array = np.asarray(predictions, dtype=np.int64).reshape(-1)
    if target_array.shape[0] != prediction_array.shape[0]:
        raise ValueError("targets and predictions must have the same length.")
    down_f1 = _precision_recall_f1(target_array, prediction_array, int(down_id))[2]
    up_f1 = _precision_recall_f1(target_array, prediction_array, int(up_id))[2]
    return float((down_f1 + up_f1) / 2.0)


def optimize_directional_thresholds(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    down_candidates: np.ndarray,
    up_candidates: np.ndarray,
    down_id: int,
    neutral_id: int,
    up_id: int,
) -> DirectionalThresholdSelection:
    """Select down/up thresholds maximizing validation directional macro F1."""
    best: DirectionalThresholdSelection | None = None
    n_candidates = int(len(down_candidates) * len(up_candidates))
    for threshold_down in down_candidates:
        for threshold_up in up_candidates:
            predictions = apply_directional_threshold_policy(
                probabilities,
                threshold_down=float(threshold_down),
                threshold_up=float(threshold_up),
                down_id=down_id,
                neutral_id=neutral_id,
                up_id=up_id,
            )
            score = directional_macro_f1_from_predictions(
                targets,
                predictions,
                down_id=down_id,
                up_id=up_id,
            )
            if best is None or score > best.score:
                best = DirectionalThresholdSelection(
                    threshold_down=float(threshold_down),
                    threshold_up=float(threshold_up),
                    score=float(score),
                    n_candidates=n_candidates,
                )
    if best is None:
        raise ValueError("threshold grid is empty.")
    return best


def thresholded_metric_summary(
    targets: np.ndarray,
    predictions: np.ndarray,
    *,
    down_id: int,
    neutral_id: int,
    up_id: int,
) -> dict[str, float]:
    """Summarize thresholded classification metrics for logging."""
    target_array = np.asarray(targets, dtype=np.int64).reshape(-1)
    prediction_array = np.asarray(predictions, dtype=np.int64).reshape(-1)
    if target_array.shape[0] != prediction_array.shape[0]:
        raise ValueError("targets and predictions must have the same length.")
    total = int(target_array.shape[0])
    class_ids = [int(down_id), int(neutral_id), int(up_id)]
    per_class = {
        class_id: _precision_recall_f1(target_array, prediction_array, class_id)
        for class_id in class_ids
    }
    down_precision, down_recall, down_f1 = per_class[int(down_id)]
    up_precision, up_recall, up_f1 = per_class[int(up_id)]
    f1_values = [per_class[class_id][2] for class_id in class_ids]
    denominator = max(total, 1)
    return {
        "directional_macro_f1": float((down_f1 + up_f1) / 2.0),
        "macro_f1": float(np.mean(f1_values)),
        "accuracy": float(np.sum(target_array == prediction_array) / denominator),
        "down_precision": down_precision,
        "down_recall": down_recall,
        "down_f1": down_f1,
        "up_precision": up_precision,
        "up_recall": up_recall,
        "up_f1": up_f1,
        "pred_rate_down": float(np.sum(prediction_array == down_id) / denominator),
        "pred_rate_up": float(np.sum(prediction_array == up_id) / denominator),
        "pred_rate_neutral": float(np.sum(prediction_array == neutral_id) / denominator),
        "true_rate_down": float(np.sum(target_array == down_id) / denominator),
        "true_rate_up": float(np.sum(target_array == up_id) / denominator),
        "true_rate_neutral": float(np.sum(target_array == neutral_id) / denominator),
    }


def apply_thresholds_and_summarize(
    outputs: dict[str, Any],
    *,
    threshold_down: float,
    threshold_up: float,
    down_id: int,
    neutral_id: int,
    up_id: int,
) -> dict[str, float]:
    """Apply directional thresholds to collected outputs and summarize metrics."""
    probabilities = np.asarray(outputs.get("probabilities", []), dtype=np.float32)
    targets = np.asarray(outputs.get("targets", []), dtype=np.int64).reshape(-1)
    predictions = apply_directional_threshold_policy(
        probabilities,
        threshold_down=threshold_down,
        threshold_up=threshold_up,
        down_id=down_id,
        neutral_id=neutral_id,
        up_id=up_id,
    )
    return thresholded_metric_summary(
        targets,
        predictions,
        down_id=down_id,
        neutral_id=neutral_id,
        up_id=up_id,
    )
