from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


PR_CURVE_COLUMNS = ("threshold", "precision", "recall", "f1")
ROC_CURVE_COLUMNS = ("threshold", "fpr", "tpr")


def _as_1d_array(values: Iterable[float] | np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    """Return values as a flattened NumPy array."""
    return np.asarray(values, dtype=dtype).reshape(-1)


def _trapezoid_area(x_values: np.ndarray, y_values: np.ndarray) -> float:
    """Integrate y(x) after prepending the origin point."""
    finite_mask = np.isfinite(x_values) & np.isfinite(y_values)
    if not bool(finite_mask.any()):
        return 0.0
    x_values = x_values[finite_mask]
    y_values = y_values[finite_mask]
    x_values = np.concatenate(([0.0], x_values))
    y_values = np.concatenate(([0.0], y_values))
    return float(np.clip(np.trapezoid(y_values, x_values), 0.0, 1.0))


def _precision_recall_auc_from_arrays(precision: np.ndarray, recall: np.ndarray) -> float:
    """Integrate a monotonic recall-precision curve with the initial precision point."""
    finite_mask = np.isfinite(recall) & np.isfinite(precision)
    if not bool(finite_mask.any()):
        return 0.0
    recall = recall[finite_mask]
    precision = precision[finite_mask]
    recall = np.concatenate(([0.0], recall))
    precision = np.concatenate(([precision[0]], precision))
    return float(np.clip(np.trapezoid(precision, recall), 0.0, 1.0))


def binary_ranking_metrics(
    scores: Iterable[float] | np.ndarray,
    positives: Iterable[bool] | np.ndarray,
    *,
    include_curves: bool = False,
) -> dict[str, float | pd.DataFrame | None]:
    """Compute PR curve, PR-AP, PR-AUC, and ROC-AUC from one score sort."""
    score_array = _as_1d_array(scores, dtype=np.float32)
    positive_array = _as_1d_array(positives, dtype=bool)
    if score_array.shape[0] != positive_array.shape[0]:
        raise ValueError("scores and positives must have the same length.")
    if score_array.size == 0:
        return {
            "pr_curve": pd.DataFrame(columns=PR_CURVE_COLUMNS) if include_curves else None,
            "roc_curve": pd.DataFrame(columns=ROC_CURVE_COLUMNS) if include_curves else None,
            "pr_ap": 0.0,
            "pr_auc": 0.0,
            "roc_auc": 0.0,
        }

    score_array = np.nan_to_num(score_array, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    order = np.argsort(-score_array, kind="mergesort")
    sorted_scores = score_array[order]
    sorted_positives = positive_array[order].astype(np.int64)
    last_for_threshold = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    threshold_indices = np.flatnonzero(last_for_threshold)

    true_positives = np.cumsum(sorted_positives)[threshold_indices].astype(np.float64)
    selected_counts = (threshold_indices + 1).astype(np.float64)
    false_positives = selected_counts - true_positives
    positive_count = float(sorted_positives.sum())
    negative_count = float(sorted_positives.size - positive_count)

    precision = np.divide(
        true_positives,
        selected_counts,
        out=np.zeros_like(true_positives, dtype=np.float64),
        where=selected_counts > 0,
    )
    recall = (
        np.zeros_like(true_positives, dtype=np.float64)
        if positive_count == 0.0
        else true_positives / positive_count
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision, dtype=np.float64),
        where=(precision + recall) > 0,
    )
    fpr = (
        np.zeros_like(false_positives, dtype=np.float64)
        if negative_count == 0.0
        else false_positives / negative_count
    )
    tpr = recall.copy()

    pr_curve = None
    roc_curve_frame = None
    if include_curves:
        pr_curve = pd.DataFrame(
            {
                "threshold": sorted_scores[threshold_indices],
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
        roc_curve_frame = pd.DataFrame(
            {
                "threshold": sorted_scores[threshold_indices],
                "fpr": fpr,
                "tpr": tpr,
            }
        )

    if positive_count == 0.0:
        pr_ap = 0.0
        pr_auc = 0.0
    else:
        recall_delta = np.diff(np.concatenate(([0.0], recall)))
        pr_ap = float(np.sum(np.maximum(recall_delta, 0.0) * precision))
        pr_auc = _precision_recall_auc_from_arrays(precision, recall)
    roc_auc_value = 0.0 if positive_count == 0.0 or negative_count == 0.0 else _trapezoid_area(fpr, tpr)

    return {
        "pr_curve": pr_curve,
        "roc_curve": roc_curve_frame,
        "pr_ap": pr_ap,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc_value,
    }


def average_precision(scores: Iterable[float] | np.ndarray, positives: Iterable[bool] | np.ndarray) -> float:
    """Compute non-interpolated average precision for binary one-vs-rest scores."""
    return float(binary_ranking_metrics(scores, positives)["pr_ap"])


def precision_recall_auc(curve: pd.DataFrame) -> float:
    """Compute trapezoidal PR-AUC from a thresholded PR curve."""
    if curve.empty:
        return 0.0
    missing = set(PR_CURVE_COLUMNS).difference(curve.columns)
    if missing:
        raise ValueError(f"PR curve is missing columns: {sorted(missing)}")

    recall = curve["recall"].to_numpy(dtype=np.float64)
    precision = curve["precision"].to_numpy(dtype=np.float64)
    finite_mask = np.isfinite(recall) & np.isfinite(precision)
    if not bool(finite_mask.any()):
        return 0.0

    recall = recall[finite_mask]
    precision = precision[finite_mask]
    order = np.argsort(recall, kind="mergesort")
    recall = recall[order]
    precision = precision[order]
    recall = np.concatenate(([0.0], recall))
    precision = np.concatenate(([precision[0]], precision))
    auc = float(np.trapezoid(precision, recall))
    return float(np.clip(auc, 0.0, 1.0))


def roc_curve(scores: Iterable[float] | np.ndarray, positives: Iterable[bool] | np.ndarray) -> pd.DataFrame:
    """Build a thresholded ROC curve for binary one-vs-rest scores."""
    return binary_ranking_metrics(scores, positives, include_curves=True)["roc_curve"]  # type: ignore[return-value]


def roc_auc(scores: Iterable[float] | np.ndarray, positives: Iterable[bool] | np.ndarray) -> float:
    """Compute trapezoidal ROC-AUC for binary one-vs-rest scores."""
    return float(binary_ranking_metrics(scores, positives)["roc_auc"])


def precision_recall_curve(
    scores: Iterable[float] | np.ndarray,
    positives: Iterable[bool] | np.ndarray,
) -> pd.DataFrame:
    """Build a thresholded precision-recall curve for binary one-vs-rest scores."""
    return binary_ranking_metrics(scores, positives, include_curves=True)["pr_curve"]  # type: ignore[return-value]


def best_f1_threshold(curve: pd.DataFrame) -> dict[str, float]:
    """Return the threshold row with the highest F1 score."""
    if curve.empty:
        return {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    missing = set(PR_CURVE_COLUMNS).difference(curve.columns)
    if missing:
        raise ValueError(f"PR curve is missing columns: {sorted(missing)}")

    f1_values = np.nan_to_num(curve["f1"].to_numpy(dtype=np.float64), nan=-np.inf)
    best_index = int(np.argmax(f1_values))
    row = curve.iloc[best_index]
    return {
        "threshold": float(row["threshold"]),
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "f1": float(row["f1"]),
    }


def _validated_multiclass_arrays(
    probabilities: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return validated multiclass probabilities and targets."""
    probability_array = np.asarray(probabilities, dtype=np.float32)
    target_array = _as_1d_array(targets, dtype=np.int64)
    if probability_array.ndim != 2:
        raise ValueError("probabilities must be a 2D array.")
    if probability_array.shape[0] != target_array.shape[0]:
        raise ValueError("probabilities and targets must have matching rows.")
    if probability_array.shape[1] < num_classes:
        raise ValueError("probabilities has fewer columns than num_classes.")
    return probability_array, target_array


def per_class_ranking_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
    *,
    class_names: list[str] | None = None,
    include_curves: bool = False,
) -> dict[str, list[float] | dict[str, pd.DataFrame]]:
    """Compute all one-vs-rest ranking metrics with one sort per class."""
    probability_array, target_array = _validated_multiclass_arrays(probabilities, targets, num_classes)
    if class_names is not None and len(class_names) != num_classes:
        raise ValueError("class_names length must match num_classes.")

    pr_ap_values: list[float] = []
    pr_auc_values: list[float] = []
    roc_auc_values: list[float] = []
    curves: dict[str, pd.DataFrame] = {}
    for class_id in range(num_classes):
        metrics = binary_ranking_metrics(
            probability_array[:, class_id],
            target_array == class_id,
            include_curves=include_curves,
        )
        pr_ap_values.append(float(metrics["pr_ap"]))
        pr_auc_values.append(float(metrics["pr_auc"]))
        roc_auc_values.append(float(metrics["roc_auc"]))
        if include_curves:
            class_name = class_names[class_id] if class_names is not None else f"class_{class_id}"
            curves[class_name] = metrics["pr_curve"]  # type: ignore[assignment]

    return {
        "pr_ap": pr_ap_values,
        "pr_auc": pr_auc_values,
        "roc_auc": roc_auc_values,
        "pr_curves": curves,
    }


def per_class_average_precision(
    probabilities: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> list[float]:
    """Compute one-vs-rest average precision for each class."""
    return per_class_ranking_metrics(probabilities, targets, num_classes)["pr_ap"]  # type: ignore[return-value]


def per_class_pr_auc(
    probabilities: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> list[float]:
    """Compute trapezoidal one-vs-rest PR-AUC for each class."""
    return per_class_ranking_metrics(probabilities, targets, num_classes)["pr_auc"]  # type: ignore[return-value]


def per_class_roc_auc(
    probabilities: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> list[float]:
    """Compute one-vs-rest ROC-AUC for each class."""
    return per_class_ranking_metrics(probabilities, targets, num_classes)["roc_auc"]  # type: ignore[return-value]


def per_class_precision_recall_curves(
    probabilities: np.ndarray,
    targets: np.ndarray,
    class_names: list[str],
) -> dict[str, pd.DataFrame]:
    """Build one-vs-rest precision-recall curves keyed by class name."""
    metrics = per_class_ranking_metrics(
        probabilities,
        targets,
        len(class_names),
        class_names=class_names,
        include_curves=True,
    )
    return metrics["pr_curves"]  # type: ignore[return-value]
