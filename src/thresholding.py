from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class DirectionalThresholdSelection:
    """Store the selected directional thresholds and validation score."""

    threshold_down: float | None
    threshold_up: float | None
    score: float
    rate_penalty: float
    min_directional_precision: float
    n_candidates: int
    down_enabled: bool = True
    up_enabled: bool = True
    stage_summaries: tuple[dict[str, Any], ...] = ()
    selection_details: dict[str, Any] = field(default_factory=dict)


_TIE_TOLERANCE = 1e-12
_PROBABILITY_EPS = 1e-7


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


def _clipped_logit(values: np.ndarray | float) -> np.ndarray:
    """Return numerically stable logits from probabilities."""
    probabilities = np.asarray(values, dtype=np.float64)
    clipped = np.clip(probabilities, _PROBABILITY_EPS, 1.0 - _PROBABILITY_EPS)
    return np.log(clipped / (1.0 - clipped))


def apply_directional_threshold_policy(
    probabilities: np.ndarray,
    *,
    threshold_down: float | None,
    threshold_up: float | None,
    down_id: int,
    neutral_id: int,
    up_id: int,
    delta: float = 0.0,
    down_enabled: bool = True,
    up_enabled: bool = True,
) -> np.ndarray:
    """Classify samples with fixed down/up probability thresholds."""
    if delta < 0.0:
        raise ValueError("delta must be >= 0.")
    probs = np.asarray(probabilities, dtype=np.float32)
    if probs.ndim != 2:
        raise ValueError("probabilities must be a 2D array.")
    max_id = max(down_id, neutral_id, up_id)
    if probs.shape[1] <= max_id:
        raise ValueError("probabilities has fewer columns than the requested class ids.")

    p_down = probs[:, down_id]
    p_up = probs[:, up_id]
    down_active = bool(down_enabled) and threshold_down is not None
    up_active = bool(up_enabled) and threshold_up is not None
    down_hit = np.zeros(probs.shape[0], dtype=bool)
    up_hit = np.zeros(probs.shape[0], dtype=bool)
    if down_active:
        down_hit = p_down >= float(threshold_down)
    if up_active:
        up_hit = p_up >= float(threshold_up)

    predictions = np.full(probs.shape[0], int(neutral_id), dtype=np.int64)
    predictions[down_hit & ~up_hit] = int(down_id)
    predictions[up_hit & ~down_hit] = int(up_id)
    both_hit = down_hit & up_hit
    if np.any(both_hit):
        down_margin = _clipped_logit(p_down) - _clipped_logit(float(threshold_down))
        up_margin = _clipped_logit(p_up) - _clipped_logit(float(threshold_up))
        predictions[both_hit & (down_margin > up_margin + float(delta))] = int(down_id)
        predictions[both_hit & (up_margin > down_margin + float(delta))] = int(up_id)
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


def _selection_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    *,
    down_id: int,
    neutral_id: int,
    up_id: int,
    score: str = "directional_macro_f1",
) -> tuple[float, float, float]:
    """Return threshold selection score and tie-breaker metrics."""
    target_array = np.asarray(targets, dtype=np.int64).reshape(-1)
    prediction_array = np.asarray(predictions, dtype=np.int64).reshape(-1)
    if target_array.shape[0] != prediction_array.shape[0]:
        raise ValueError("targets and predictions must have the same length.")

    score_name = str(score).strip().lower()
    if score_name not in {"macro_f1", "directional_macro_f1"}:
        raise ValueError("threshold score must be 'macro_f1' or 'directional_macro_f1'.")
    down_precision, _down_recall, down_f1 = _precision_recall_f1(target_array, prediction_array, int(down_id))
    _neutral_precision, _neutral_recall, neutral_f1 = _precision_recall_f1(
        target_array,
        prediction_array,
        int(neutral_id),
    )
    up_precision, _up_recall, up_f1 = _precision_recall_f1(target_array, prediction_array, int(up_id))
    denominator = max(int(target_array.shape[0]), 1)
    pred_rate_down = float(np.sum(prediction_array == down_id) / denominator)
    pred_rate_up = float(np.sum(prediction_array == up_id) / denominator)
    true_rate_down = float(np.sum(target_array == down_id) / denominator)
    true_rate_up = float(np.sum(target_array == up_id) / denominator)
    rate_penalty = abs(pred_rate_down - true_rate_down) + abs(pred_rate_up - true_rate_up)
    if score_name == "macro_f1":
        selection_score = float((down_f1 + neutral_f1 + up_f1) / 3.0)
    else:
        selection_score = float((down_f1 + up_f1) / 2.0)
    return selection_score, float(rate_penalty), float(min(down_precision, up_precision))


def _threshold_key(selection: DirectionalThresholdSelection) -> tuple[float, float, float]:
    """Return the final tie-breaker key favoring higher thresholds."""
    threshold_down = -float("inf") if selection.threshold_down is None else float(selection.threshold_down)
    threshold_up = -float("inf") if selection.threshold_up is None else float(selection.threshold_up)
    return (
        float(threshold_down + threshold_up),
        threshold_down,
        threshold_up,
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
    delta: float = 0.0,
    score: str = "directional_macro_f1",
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
                delta=delta,
            )
            selection_score, rate_penalty, min_directional_precision = _selection_metrics(
                targets,
                predictions,
                down_id=down_id,
                neutral_id=neutral_id,
                up_id=up_id,
                score=score,
            )
            candidate = DirectionalThresholdSelection(
                threshold_down=float(threshold_down),
                threshold_up=float(threshold_up),
                score=float(selection_score),
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
    delta: float = 0.0,
    score: str = "directional_macro_f1",
) -> DirectionalThresholdSelection:
    """Select thresholds using F1, rate, precision, then high-threshold tie-breaks."""
    if delta < 0.0:
        raise ValueError("delta must be >= 0.")
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
        delta=delta,
        score=score,
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
            delta=delta,
            score=score,
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


def _binary_threshold_metrics(
    scores: np.ndarray,
    targets: np.ndarray,
    *,
    threshold: float,
    class_id: int,
) -> tuple[float, float, float]:
    """Return precision, recall, and F1 for one class threshold."""
    score_array = np.asarray(scores, dtype=np.float32).reshape(-1)
    target_array = np.asarray(targets, dtype=np.int64).reshape(-1)
    if score_array.shape[0] != target_array.shape[0]:
        raise ValueError("scores and targets must have the same length.")
    positive_predictions = score_array >= float(threshold)
    positive_targets = target_array == int(class_id)
    true_positive = int(np.sum(positive_predictions & positive_targets))
    false_positive = int(np.sum(positive_predictions & ~positive_targets))
    false_negative = int(np.sum(~positive_predictions & positive_targets))
    precision = 0.0 if true_positive + false_positive == 0 else true_positive / (true_positive + false_positive)
    recall = 0.0 if true_positive + false_negative == 0 else true_positive / (true_positive + false_negative)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return float(precision), float(recall), float(f1)


def _is_better_precision_floor_candidate(
    candidate: dict[str, Any],
    best: dict[str, Any] | None,
) -> bool:
    """Compare candidates by recall, precision, then higher threshold."""
    if best is None:
        return True
    if candidate["recall"] > best["recall"] + _TIE_TOLERANCE:
        return True
    if abs(candidate["recall"] - best["recall"]) > _TIE_TOLERANCE:
        return False
    if candidate["precision"] > best["precision"] + _TIE_TOLERANCE:
        return True
    if abs(candidate["precision"] - best["precision"]) > _TIE_TOLERANCE:
        return False
    return candidate["threshold"] > best["threshold"] + _TIE_TOLERANCE


def _select_precision_floor_threshold(
    scores: np.ndarray,
    targets: np.ndarray,
    *,
    candidates: np.ndarray,
    class_id: int,
    precision_floor: float,
) -> dict[str, Any]:
    """Select one class threshold maximizing recall under a precision floor."""
    if not 0.0 <= precision_floor <= 1.0:
        raise ValueError("precision floor must be in [0, 1].")
    best: dict[str, Any] | None = None
    meeting_floor = 0
    for threshold in np.asarray(candidates, dtype=np.float64):
        precision, recall, f1 = _binary_threshold_metrics(
            scores,
            targets,
            threshold=float(threshold),
            class_id=int(class_id),
        )
        if precision + _TIE_TOLERANCE < precision_floor:
            continue
        meeting_floor += 1
        candidate = {
            "enabled": True,
            "threshold": float(threshold),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        if _is_better_precision_floor_candidate(candidate, best):
            best = candidate
    if best is None:
        return {
            "enabled": False,
            "threshold": None,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "precision_floor": float(precision_floor),
            "n_candidates": int(len(candidates)),
            "n_candidates_meeting_floor": 0,
            "fallback": "disabled_no_candidate_meets_precision_floor",
        }
    best["precision_floor"] = float(precision_floor)
    best["n_candidates"] = int(len(candidates))
    best["n_candidates_meeting_floor"] = int(meeting_floor)
    return best


def optimize_precision_floor_thresholds(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    down_candidates: np.ndarray,
    up_candidates: np.ndarray,
    down_precision_floor: float,
    up_precision_floor: float,
    down_id: int,
    neutral_id: int,
    up_id: int,
    delta: float = 0.0,
    score: str = "directional_macro_f1",
) -> DirectionalThresholdSelection:
    """Select independent down/up thresholds under precision floors."""
    probs = np.asarray(probabilities, dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.int64).reshape(-1)
    if probs.ndim != 2:
        raise ValueError("probabilities must be a 2D array.")
    if probs.shape[0] != target_array.shape[0]:
        raise ValueError("probabilities and targets must have the same number of rows.")
    max_id = max(down_id, neutral_id, up_id)
    if probs.shape[1] <= max_id:
        raise ValueError("probabilities has fewer columns than the requested class ids.")

    down_selection = _select_precision_floor_threshold(
        probs[:, int(down_id)],
        target_array,
        candidates=down_candidates,
        class_id=int(down_id),
        precision_floor=float(down_precision_floor),
    )
    up_selection = _select_precision_floor_threshold(
        probs[:, int(up_id)],
        target_array,
        candidates=up_candidates,
        class_id=int(up_id),
        precision_floor=float(up_precision_floor),
    )
    predictions = apply_directional_threshold_policy(
        probs,
        threshold_down=down_selection["threshold"],
        threshold_up=up_selection["threshold"],
        down_id=down_id,
        neutral_id=neutral_id,
        up_id=up_id,
        delta=delta,
        down_enabled=bool(down_selection["enabled"]),
        up_enabled=bool(up_selection["enabled"]),
    )
    selection_score, rate_penalty, min_directional_precision = _selection_metrics(
        target_array,
        predictions,
        down_id=down_id,
        neutral_id=neutral_id,
        up_id=up_id,
        score=score,
    )
    return DirectionalThresholdSelection(
        threshold_down=down_selection["threshold"],
        threshold_up=up_selection["threshold"],
        score=selection_score,
        rate_penalty=rate_penalty,
        min_directional_precision=min_directional_precision,
        n_candidates=int(len(down_candidates) + len(up_candidates)),
        down_enabled=bool(down_selection["enabled"]),
        up_enabled=bool(up_selection["enabled"]),
        selection_details={
            "down": down_selection,
            "up": up_selection,
        },
    )


def _top_quantile_threshold(
    scores: np.ndarray,
    *,
    quantile: float,
) -> dict[str, Any]:
    """Return the threshold selecting the top quantile of scores."""
    score_array = np.asarray(scores, dtype=np.float32).reshape(-1)
    if score_array.size == 0:
        raise ValueError("top quantile thresholding requires at least one score.")
    if not 0.0 < float(quantile) <= 1.0:
        raise ValueError("top quantile must be in (0, 1].")
    requested_count = max(1, int(np.ceil(float(quantile) * score_array.size)))
    sorted_scores = np.sort(score_array)[::-1]
    threshold = float(sorted_scores[requested_count - 1])
    selected_count = int(np.sum(score_array >= threshold))
    return {
        "enabled": True,
        "threshold": threshold,
        "quantile": float(quantile),
        "requested_count": int(requested_count),
        "selected_count": selected_count,
        "n_scores": int(score_array.size),
        "selection_rule": "threshold_at_ceil_quantile_top_score",
    }


def optimize_top_quantile_thresholds(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    down_quantile: float,
    up_quantile: float,
    down_id: int,
    neutral_id: int,
    up_id: int,
    delta: float = 0.0,
    score: str = "directional_macro_f1",
) -> DirectionalThresholdSelection:
    """Select down/up thresholds from top probability quantiles."""
    probs = np.asarray(probabilities, dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.int64).reshape(-1)
    if probs.ndim != 2:
        raise ValueError("probabilities must be a 2D array.")
    if probs.shape[0] != target_array.shape[0]:
        raise ValueError("probabilities and targets must have the same number of rows.")
    max_id = max(down_id, neutral_id, up_id)
    if probs.shape[1] <= max_id:
        raise ValueError("probabilities has fewer columns than the requested class ids.")
    if delta < 0.0:
        raise ValueError("delta must be >= 0.")

    down_selection = _top_quantile_threshold(probs[:, int(down_id)], quantile=float(down_quantile))
    up_selection = _top_quantile_threshold(probs[:, int(up_id)], quantile=float(up_quantile))
    predictions = apply_directional_threshold_policy(
        probs,
        threshold_down=down_selection["threshold"],
        threshold_up=up_selection["threshold"],
        down_id=down_id,
        neutral_id=neutral_id,
        up_id=up_id,
        delta=delta,
    )
    selection_score, rate_penalty, min_directional_precision = _selection_metrics(
        target_array,
        predictions,
        down_id=down_id,
        neutral_id=neutral_id,
        up_id=up_id,
        score=score,
    )
    return DirectionalThresholdSelection(
        threshold_down=down_selection["threshold"],
        threshold_up=up_selection["threshold"],
        score=selection_score,
        rate_penalty=rate_penalty,
        min_directional_precision=min_directional_precision,
        n_candidates=2,
        down_enabled=True,
        up_enabled=True,
        selection_details={
            "down": down_selection,
            "up": up_selection,
        },
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
    threshold_down: float | None,
    threshold_up: float | None,
    down_id: int,
    neutral_id: int,
    up_id: int,
    delta: float = 0.0,
    down_enabled: bool = True,
    up_enabled: bool = True,
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
        delta=delta,
        down_enabled=down_enabled,
        up_enabled=up_enabled,
    )
    return thresholded_metric_summary(
        targets,
        predictions,
        down_id=down_id,
        neutral_id=neutral_id,
        up_id=up_id,
    )
