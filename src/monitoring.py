from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


TAILORED_SCORE = "tailored_score"
PRECISION_AT_FIXED_RATE = "precision_at_fixed_rate"
TAILORED_BASE_METRICS = {"val_macro_f1", "val_directional_macro_f1"}


@dataclass(frozen=True, slots=True)
class TailoredScoreComponents:
    """Store the scalar pieces used by the tailored validation monitor."""

    score: float
    base_metric: str
    base_value: float
    ece_dir: float
    rate_penalty: float
    pred_rate_down: float
    true_rate_down: float
    pred_rate_up: float
    true_rate_up: float

    def prefixed(self, prefix: str) -> dict[str, float | str]:
        """Return fields named for CSV/logging output."""
        return {
            f"{prefix}_tailored_score": self.score,
            f"{prefix}_tailored_base_metric": self.base_metric,
            f"{prefix}_tailored_base_value": self.base_value,
            f"{prefix}_tailored_ece_dir": self.ece_dir,
            f"{prefix}_tailored_rate_penalty": self.rate_penalty,
            f"{prefix}_pred_rate_down": self.pred_rate_down,
            f"{prefix}_true_rate_down": self.true_rate_down,
            f"{prefix}_pred_rate_up": self.pred_rate_up,
            f"{prefix}_true_rate_up": self.true_rate_up,
        }


@dataclass(frozen=True, slots=True)
class DirectionalPrecisionAtFixedRate:
    """Store precision over per-side top down/up probability scores."""

    precision: float
    fixed_rate: float
    actual_rate: float
    k: int
    n: int

    def prefixed(self, prefix: str) -> dict[str, float | int]:
        """Return fields named for CSV/logging output."""
        return {
            f"{prefix}_directional_precision_at_fixed_rate": self.precision,
            f"{prefix}_directional_precision_at_fixed_rate_k": self.k,
            f"{prefix}_directional_precision_at_fixed_rate_actual_rate": self.actual_rate,
        }


def _param_value(params: Any, name: str) -> float:
    """Read one monitor parameter from a config object or mapping."""
    if isinstance(params, Mapping):
        value = params.get(name)
    else:
        value = getattr(params, name, None)
    if value is None:
        raise ValueError(f"tailored_score requires training.monitor_params.{name}.")
    return float(value)


def _param_string(params: Any, name: str, default: str) -> str:
    """Read one string monitor parameter from a config object or mapping."""
    if isinstance(params, Mapping):
        value = params.get(name, default)
    else:
        value = getattr(params, name, default)
    if value is None:
        value = default
    return str(value).strip().lower()


def _fixed_rate_param(params: Any) -> float:
    """Read and validate the fixed-rate precision monitor parameter."""
    if isinstance(params, Mapping):
        value = params.get("fixed_rate")
    else:
        value = getattr(params, "fixed_rate", None)
    if value is None:
        raise ValueError("precision_at_fixed_rate requires training.monitor_params.fixed_rate.")
    fixed_rate = float(value)
    if not 0.0 < fixed_rate <= 1.0:
        raise ValueError("training.monitor_params.fixed_rate must be in (0, 1].")
    return fixed_rate


def tailored_base_value(metrics: Any, base_metric: str) -> float:
    """Return the F1 quantity used as the tailored score base."""
    metric_name = str(base_metric).strip().lower()
    if metric_name == "val_macro_f1":
        return float(metrics.macro_f1)
    if metric_name == "val_directional_macro_f1":
        return float(metrics.directional_macro_f1)
    raise ValueError(
        "tailored_score base_metric must be 'val_macro_f1' or 'val_directional_macro_f1'."
    )


def directional_class_ids(
    label_mapping: Mapping[int, int] | None = None,
    *,
    num_classes: int | None = None,
) -> tuple[int, int]:
    """Resolve model class ids for down/up labels."""
    if label_mapping is None:
        down_id, up_id = 0, 2
    else:
        if -1 not in label_mapping or 1 not in label_mapping:
            raise ValueError("tailored_score requires label_mapping entries for raw labels -1 and 1.")
        down_id, up_id = int(label_mapping[-1]), int(label_mapping[1])
    if num_classes is not None:
        for label, class_id in (("down", down_id), ("up", up_id)):
            if class_id < 0 or class_id >= num_classes:
                raise ValueError(f"tailored_score {label} class id {class_id} is outside [0, {num_classes}).")
    return down_id, up_id


def directional_precision_at_fixed_rate(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    fixed_rate: float,
    down_id: int = 0,
    up_id: int = 2,
) -> DirectionalPrecisionAtFixedRate:
    """Compute precision on top fixed-rate down scores and top fixed-rate up scores."""
    fixed_rate = float(fixed_rate)
    if not 0.0 < fixed_rate <= 1.0:
        raise ValueError("fixed_rate must be in (0, 1].")

    probability_array = np.asarray(probabilities, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.int64).reshape(-1)
    if probability_array.ndim != 2:
        raise ValueError("probabilities must be a 2D array.")
    if probability_array.shape[0] != target_array.shape[0]:
        raise ValueError("probabilities and targets must have the same number of rows.")

    num_classes = int(probability_array.shape[1])
    for label, class_id in (("down", int(down_id)), ("up", int(up_id))):
        if class_id < 0 or class_id >= num_classes:
            raise ValueError(f"{label} class id {class_id} is outside [0, {num_classes}).")

    finite_mask = np.isfinite(probability_array).all(axis=1)
    valid_target_mask = (target_array >= 0) & (target_array < num_classes)
    valid_mask = finite_mask & valid_target_mask
    if not bool(np.any(valid_mask)):
        return DirectionalPrecisionAtFixedRate(
            precision=0.0,
            fixed_rate=fixed_rate,
            actual_rate=0.0,
            k=0,
            n=0,
        )

    valid_probabilities = probability_array[valid_mask]
    valid_targets = target_array[valid_mask]
    down_scores = valid_probabilities[:, int(down_id)]
    up_scores = valid_probabilities[:, int(up_id)]

    n_samples = int(valid_targets.shape[0])
    k = min(n_samples, max(1, int(np.ceil(fixed_rate * n_samples))))
    top_down_indices = np.argsort(-down_scores, kind="mergesort")[:k]
    top_up_indices = np.argsort(-up_scores, kind="mergesort")[:k]
    correct_down = int(np.sum(valid_targets[top_down_indices] == int(down_id)))
    correct_up = int(np.sum(valid_targets[top_up_indices] == int(up_id)))
    precision = float((correct_down + correct_up) / max(2 * k, 1))
    return DirectionalPrecisionAtFixedRate(
        precision=precision,
        fixed_rate=fixed_rate,
        actual_rate=float(k / n_samples),
        k=int(k),
        n=n_samples,
    )


def tailored_score_components(
    metrics: Any,
    *,
    lambda_ece: float,
    lambda_rate: float,
    base_metric: str = "val_directional_macro_f1",
    label_mapping: Mapping[int, int] | None = None,
) -> TailoredScoreComponents:
    """Compute the custom validation monitor from classification metrics."""
    confusion = np.asarray(getattr(metrics, "confusion_matrix", []), dtype=np.float64)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError("tailored_score requires a square confusion_matrix.")
    num_classes = int(confusion.shape[0])
    down_id, up_id = directional_class_ids(label_mapping, num_classes=num_classes)

    ece_values = getattr(metrics, "per_class_expected_calibration_error", None)
    if ece_values is None or max(down_id, up_id) >= len(ece_values):
        raise ValueError("tailored_score requires per-class ECE for down and up classes.")
    ece_dir = float((float(ece_values[down_id]) + float(ece_values[up_id])) / 2.0)

    total = float(confusion.sum())
    if total <= 0.0:
        pred_rate_down = true_rate_down = pred_rate_up = true_rate_up = 0.0
    else:
        true_rate_down = float(confusion[down_id, :].sum() / total)
        true_rate_up = float(confusion[up_id, :].sum() / total)
        pred_rate_down = float(confusion[:, down_id].sum() / total)
        pred_rate_up = float(confusion[:, up_id].sum() / total)
    rate_penalty = abs(pred_rate_down - true_rate_down) + abs(pred_rate_up - true_rate_up)
    base_name = str(base_metric).strip().lower()
    base_value = tailored_base_value(metrics, base_name)
    score = base_value - float(lambda_ece) * ece_dir - float(lambda_rate) * rate_penalty
    return TailoredScoreComponents(
        score=score,
        base_metric=base_name,
        base_value=base_value,
        ece_dir=ece_dir,
        rate_penalty=float(rate_penalty),
        pred_rate_down=pred_rate_down,
        true_rate_down=true_rate_down,
        pred_rate_up=pred_rate_up,
        true_rate_up=true_rate_up,
    )


def tailored_score_from_params(
    metrics: Any,
    monitor_params: Any,
    *,
    label_mapping: Mapping[int, int] | None = None,
) -> TailoredScoreComponents:
    """Compute tailored score using monitor_params from configuration."""
    return tailored_score_components(
        metrics,
        lambda_ece=_param_value(monitor_params, "lambda_ece"),
        lambda_rate=_param_value(monitor_params, "lambda_rate"),
        base_metric=_param_string(monitor_params, "base_metric", "val_directional_macro_f1"),
        label_mapping=label_mapping,
    )


def monitor_value(
    *,
    loss: float,
    metrics: Any | None,
    monitor: str,
    monitor_params: Any = None,
    label_mapping: Mapping[int, int] | None = None,
) -> float:
    """Return a configured monitor value from validation results."""
    monitor_name = monitor.lower()
    if monitor_name == "val_loss":
        return float(loss)
    if metrics is None:
        raise ValueError(f"Cannot compute {monitor_name} because validation metrics are unavailable.")
    if monitor_name == "val_macro_f1":
        return float(metrics.macro_f1)
    if monitor_name == "val_directional_macro_f1":
        return float(metrics.directional_macro_f1)
    if monitor_name == PRECISION_AT_FIXED_RATE:
        value = getattr(metrics, "directional_precision_at_fixed_rate", None)
        if value is None:
            raise ValueError(
                "precision_at_fixed_rate requires validation metrics computed with "
                "training.monitor_params.fixed_rate."
            )
        _fixed_rate_param(monitor_params)
        return float(value)
    if monitor_name == TAILORED_SCORE:
        return tailored_score_from_params(
            metrics,
            monitor_params,
            label_mapping=label_mapping,
        ).score
    raise ValueError(f"Unsupported monitor: {monitor}")


def epoch_monitor_value(
    result: Any,
    *,
    monitor: str,
    monitor_params: Any = None,
    label_mapping: Mapping[int, int] | None = None,
) -> float:
    """Return a configured monitor value from a saved epoch result."""
    return monitor_value(
        loss=float(result.val_loss),
        metrics=getattr(result, "val_metrics", None),
        monitor=monitor,
        monitor_params=monitor_params,
        label_mapping=label_mapping,
    )
