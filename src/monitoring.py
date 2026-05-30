from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


TAILORED_SCORE = "tailored_score"


@dataclass(frozen=True, slots=True)
class TailoredScoreComponents:
    """Store the scalar pieces used by the tailored validation monitor."""

    score: float
    ece_dir: float
    rate_penalty: float
    pred_rate_down: float
    true_rate_down: float
    pred_rate_up: float
    true_rate_up: float

    def prefixed(self, prefix: str) -> dict[str, float]:
        """Return fields named for CSV/logging output."""
        return {
            f"{prefix}_tailored_score": self.score,
            f"{prefix}_tailored_ece_dir": self.ece_dir,
            f"{prefix}_tailored_rate_penalty": self.rate_penalty,
            f"{prefix}_pred_rate_down": self.pred_rate_down,
            f"{prefix}_true_rate_down": self.true_rate_down,
            f"{prefix}_pred_rate_up": self.pred_rate_up,
            f"{prefix}_true_rate_up": self.true_rate_up,
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


def tailored_score_components(
    metrics: Any,
    *,
    lambda_ece: float,
    lambda_rate: float,
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
    score = float(metrics.directional_macro_f1) - float(lambda_ece) * ece_dir - float(lambda_rate) * rate_penalty
    return TailoredScoreComponents(
        score=score,
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
