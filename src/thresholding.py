from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class DirectionalThresholdSelection:
    """Store the selected directional thresholds and validation score."""

    threshold_down: float
    threshold_up: float
    score: float
    rate_penalty: float
    min_directional_precision: float
    n_candidates: int
    stage_summaries: tuple[dict[str, Any], ...] = ()


_TIE_TOLERANCE = 1e-12


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


def _inferred_grid_step(candidates: np.ndarray) -> float | None:
    """Infer the smallest positive spacing in a threshold grid."""
    values = np.unique(np.asarray(candidates, dtype=np.float64))
    if values.size < 2:
        return None
    diffs = np.diff(values)
    positive_diffs = diffs[diffs > _TIE_TOLERANCE]
    if positive_diffs.size == 0:
        return None
    return float(np.min(positive_diffs))


def _refined_threshold_candidates(
    center: float,
    *,
    radius: float,
    step: float,
    min_threshold: float,
    max_threshold: float,
) -> np.ndarray:
    """Build a clipped local threshold grid around a selected center."""
    lower = max(float(min_threshold), float(center) - float(radius))
    upper = min(float(max_threshold), float(center) + float(radius))
    return threshold_candidates(lower, upper, step)


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


def _directional_selection_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    *,
    down_id: int,
    up_id: int,
) -> tuple[float, float, float]:
    """Return threshold selection score and tie-breaker metrics."""
    target_array = np.asarray(targets, dtype=np.int64).reshape(-1)
    prediction_array = np.asarray(predictions, dtype=np.int64).reshape(-1)
    if target_array.shape[0] != prediction_array.shape[0]:
        raise ValueError("targets and predictions must have the same length.")

    down_precision, _down_recall, down_f1 = _precision_recall_f1(target_array, prediction_array, int(down_id))
    up_precision, _up_recall, up_f1 = _precision_recall_f1(target_array, prediction_array, int(up_id))
    denominator = max(int(target_array.shape[0]), 1)
    pred_rate_down = float(np.sum(prediction_array == down_id) / denominator)
    pred_rate_up = float(np.sum(prediction_array == up_id) / denominator)
    true_rate_down = float(np.sum(target_array == down_id) / denominator)
    true_rate_up = float(np.sum(target_array == up_id) / denominator)
    rate_penalty = abs(pred_rate_down - true_rate_down) + abs(pred_rate_up - true_rate_up)
    return float((down_f1 + up_f1) / 2.0), float(rate_penalty), float(min(down_precision, up_precision))


def _threshold_key(selection: DirectionalThresholdSelection) -> tuple[float, float, float]:
    """Return the final tie-breaker key favoring higher thresholds."""
    return (
        float(selection.threshold_down + selection.threshold_up),
        float(selection.threshold_down),
        float(selection.threshold_up),
    )


def _is_better_selection(
    candidate: DirectionalThresholdSelection,
    best: DirectionalThresholdSelection | None,
) -> bool:
    """Compare threshold candidates using the configured validation tie-breakers."""
    if best is None:
        return True
    if candidate.score > best.score + _TIE_TOLERANCE:
        return True
    if abs(candidate.score - best.score) > _TIE_TOLERANCE:
        return False
    if candidate.rate_penalty < best.rate_penalty - _TIE_TOLERANCE:
        return True
    if abs(candidate.rate_penalty - best.rate_penalty) > _TIE_TOLERANCE:
        return False
    if candidate.min_directional_precision > best.min_directional_precision + _TIE_TOLERANCE:
        return True
    if abs(candidate.min_directional_precision - best.min_directional_precision) > _TIE_TOLERANCE:
        return False
    return _threshold_key(candidate) > _threshold_key(best)


def _select_best_on_grid(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    down_candidates: np.ndarray,
    up_candidates: np.ndarray,
    down_id: int,
    neutral_id: int,
    up_id: int,
) -> DirectionalThresholdSelection:
    """Search one down/up grid and return its best threshold pair."""
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
            score, rate_penalty, min_directional_precision = _directional_selection_metrics(
                targets,
                predictions,
                down_id=down_id,
                up_id=up_id,
            )
            candidate = DirectionalThresholdSelection(
                threshold_down=float(threshold_down),
                threshold_up=float(threshold_up),
                score=float(score),
                rate_penalty=float(rate_penalty),
                min_directional_precision=float(min_directional_precision),
                n_candidates=n_candidates,
            )
            if _is_better_selection(candidate, best):
                best = candidate
    if best is None:
        raise ValueError("threshold grid is empty.")
    return best


def _stage_summary(
    *,
    stage_index: int,
    stage_name: str,
    down_candidates: np.ndarray,
    up_candidates: np.ndarray,
    best: DirectionalThresholdSelection,
) -> dict[str, Any]:
    """Return a compact summary for one threshold-search stage."""
    return {
        "stage": int(stage_index),
        "name": stage_name,
        "step_down": _inferred_grid_step(down_candidates),
        "step_up": _inferred_grid_step(up_candidates),
        "down_min": float(down_candidates[0]),
        "down_max": float(down_candidates[-1]),
        "up_min": float(up_candidates[0]),
        "up_max": float(up_candidates[-1]),
        "down_candidates": int(len(down_candidates)),
        "up_candidates": int(len(up_candidates)),
        "n_candidates": int(len(down_candidates) * len(up_candidates)),
        "best_threshold_down": float(best.threshold_down),
        "best_threshold_up": float(best.threshold_up),
        "best_score": float(best.score),
        "best_rate_penalty": float(best.rate_penalty),
        "best_min_directional_precision": float(best.min_directional_precision),
    }


def optimize_directional_thresholds(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    down_candidates: np.ndarray,
    up_candidates: np.ndarray,
    down_id: int,
    neutral_id: int,
    up_id: int,
    refinement_steps: tuple[float, ...] = (),
) -> DirectionalThresholdSelection:
    """Select thresholds using F1, rate, precision, then high-threshold tie-breaks."""
    down_grid = np.asarray(down_candidates, dtype=np.float64)
    up_grid = np.asarray(up_candidates, dtype=np.float64)
    if down_grid.size == 0 or up_grid.size == 0:
        raise ValueError("threshold grid is empty.")

    min_down_threshold = float(np.min(down_grid))
    max_down_threshold = float(np.max(down_grid))
    min_up_threshold = float(np.min(up_grid))
    max_up_threshold = float(np.max(up_grid))
    current_radius = max(
        _inferred_grid_step(down_grid) or 0.0,
        _inferred_grid_step(up_grid) or 0.0,
    )
    best = _select_best_on_grid(
        probabilities,
        targets,
        down_candidates=down_grid,
        up_candidates=up_grid,
        down_id=down_id,
        neutral_id=neutral_id,
        up_id=up_id,
    )
    stage_summaries = [
        _stage_summary(
            stage_index=0,
            stage_name="coarse",
            down_candidates=down_grid,
            up_candidates=up_grid,
            best=best,
        )
    ]
    total_candidates = int(best.n_candidates)

    for stage_index, step in enumerate(refinement_steps, start=1):
        if step <= 0.0:
            raise ValueError("refinement threshold steps must be > 0.")
        if current_radius <= 0.0:
            break
        down_grid = _refined_threshold_candidates(
            best.threshold_down,
            radius=current_radius,
            step=float(step),
            min_threshold=min_down_threshold,
            max_threshold=max_down_threshold,
        )
        up_grid = _refined_threshold_candidates(
            best.threshold_up,
            radius=current_radius,
            step=float(step),
            min_threshold=min_up_threshold,
            max_threshold=max_up_threshold,
        )
        stage_best = _select_best_on_grid(
            probabilities,
            targets,
            down_candidates=down_grid,
            up_candidates=up_grid,
            down_id=down_id,
            neutral_id=neutral_id,
            up_id=up_id,
        )
        total_candidates += int(stage_best.n_candidates)
        if _is_better_selection(stage_best, best):
            best = stage_best
        stage_summaries.append(
            _stage_summary(
                stage_index=stage_index,
                stage_name=f"refine_{step:.6g}",
                down_candidates=down_grid,
                up_candidates=up_grid,
                best=stage_best,
            )
        )
        current_radius = float(step)

    return replace(
        best,
        n_candidates=total_candidates,
        stage_summaries=tuple(stage_summaries),
    )


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
