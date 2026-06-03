from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    from configuration import ExperimentConfig
    from monitoring import epoch_monitor_value, tailored_score_from_params
    from pr_metrics import (
        best_f1_threshold,
        per_class_ranking_metrics,
    )
except ImportError:  # pragma: no cover
    from .configuration import ExperimentConfig
    from .monitoring import epoch_monitor_value, tailored_score_from_params
    from .pr_metrics import (
        best_f1_threshold,
        per_class_ranking_metrics,
    )


RUN_FILE_PATTERN = re.compile(r"^run_(\d+)(?:[._]|$)")
RUN_STEM_TOKEN_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
PREPROCESSING_METADATA_FILENAME = "preprocessing_metadata.yaml"
METRIC_NAMES = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "directional_macro_f1",
    "weighted_f1",
    "balanced_accuracy",
    "expected_calibration_error",
)
SPLIT_METRIC_PREFIXES = ("train", "val", "test")


def format_duration(seconds: float) -> str:
    """Return a compact human-readable duration string."""
    total_seconds = max(0.0, float(seconds))
    hours, remainder = divmod(total_seconds, 3600.0)
    minutes, seconds = divmod(remainder, 60.0)
    if hours >= 1:
        return f"{int(hours)}h {int(minutes):02d}m {seconds:05.2f}s"
    if minutes >= 1:
        return f"{int(minutes)}m {seconds:05.2f}s"
    return f"{seconds:.2f}s"


def resolve_config_path(config: ExperimentConfig, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (config.path.parent / candidate).resolve()


def next_run_stem(logs_dir: Path) -> str:
    """
    Return the next available run filename stem based on existing log files.
    For example, if logs_dir contains files matching run_1.*, run_2.*,
    and run_5.*, this returns "run_6".
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    max_run_id = 0
    for path in logs_dir.iterdir():
        match = RUN_FILE_PATTERN.match(path.name)
        if match:
            max_run_id = max(max_run_id, int(match.group(1)))
    return f"run_{max_run_id + 1}"


def sanitize_run_stem_token(value: str) -> str:
    """Return a filesystem-friendly token for run directory names."""
    token = RUN_STEM_TOKEN_PATTERN.sub("_", value.strip())
    token = re.sub(r"_+", "_", token).strip("._-")
    return token or "experiment"


def timestamped_run_stem(experiment_name: str, launch_time: datetime | None = None) -> str:
    """Build a readable run stem from an experiment name and launch timestamp."""
    timestamp = (launch_time or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{sanitize_run_stem_token(experiment_name)}_{timestamp}"


def load_config_snapshot(config: ExperimentConfig) -> dict[str, Any]:
    with config.path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def preprocessing_metadata_path(sequence_dir: Path) -> Path:
    return sequence_dir / PREPROCESSING_METADATA_FILENAME


def load_preprocessing_metadata(sequence_dir: Path) -> dict[str, Any]:
    path = preprocessing_metadata_path(sequence_dir)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _fast_stream_summary(config: ExperimentConfig, kind: str) -> dict[str, Any]:
    stream_config = (
        config.preprocessing.price_kinematic
        if kind == "price"
        else config.preprocessing.volume_kinematic
    )
    fast_config = stream_config.fast
    return {
        "enabled": bool(stream_config.enabled),
        "n_basis": int(fast_config.n_basis),
        "target_df": float(fast_config.df),
        "eval_at": float(fast_config.eval_at),
        "selected_smoothing_lambda": fast_config.selected_smoothing_lambda,
    }


def fast_smoothing_lambda_summary(
    config: ExperimentConfig,
    preprocessing_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    method = config.preprocessing.kinematic_tokenization.method
    if method != "fast":
        return {
            "method": method,
            "status": "not_applicable",
        }

    summary = {
        "method": method,
        "status": "available",
        "price": _fast_stream_summary(config, "price"),
        "volume": _fast_stream_summary(config, "volume"),
    }

    metadata_lambdas = (preprocessing_metadata or {}).get("fast_smoothing_lambdas")
    if isinstance(metadata_lambdas, dict):
        for kind in ("price", "volume"):
            if isinstance(metadata_lambdas.get(kind), dict):
                summary[kind].update(metadata_lambdas[kind])
    return summary


def save_preprocessing_metadata(
    config: ExperimentConfig,
    sequence_dir: Path,
    *,
    lambda_results: dict[str, dict[str, float]] | None = None,
    label_distribution: dict[str, Any] | None = None,
    smoothing_threshold: dict[str, Any] | None = None,
    price_static_plgs: dict[str, Any] | None = None,
    volume_static_exp: dict[str, Any] | None = None,
    volume_bar_scaling: dict[str, Any] | None = None,
    sample_clock: dict[str, Any] | None = None,
    sample_clock_counts: dict[str, Any] | None = None,
    timing: dict[str, Any] | None = None,
) -> Path:
    lambdas = fast_smoothing_lambda_summary(config)
    if lambda_results:
        for kind, result in lambda_results.items():
            if kind in lambdas:
                lambdas[kind].update(result)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config.path),
        "sequence_data_dir": str(sequence_dir),
        "save_processed_dataframes": bool(config.preprocessing.save_processed_dataframes),
        "sample_clock": sample_clock
        or {
            "mode": config.preprocessing.sample_clock.mode,
            "volume_step_shares": config.preprocessing.sample_clock.volume_step_shares,
            "volume_source": config.preprocessing.sample_clock.volume_source,
            "trade_type_values": list(config.preprocessing.sample_clock.trade_type_values),
        },
        "fast_smoothing_lambdas": lambdas,
    }
    if label_distribution is not None:
        payload["label_distribution"] = label_distribution
    if smoothing_threshold is not None:
        payload["smoothing_threshold"] = smoothing_threshold
    if price_static_plgs is not None:
        payload["price_static_plgs"] = price_static_plgs
    if volume_static_exp is not None:
        payload["volume_static_exp"] = volume_static_exp
    if volume_bar_scaling is not None:
        payload["volume_bar_scaling"] = volume_bar_scaling
    if sample_clock_counts is not None:
        payload["sample_clock_counts"] = sample_clock_counts
    if timing is not None:
        payload["timing"] = timing
    target = preprocessing_metadata_path(sequence_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    return target


def model_parameter_summary(model: Any) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total": int(total),
        "trainable": int(trainable),
        "non_trainable": int(total - trainable),
    }


def _effective_labels(dataset: Any) -> np.ndarray:
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        base_labels = _effective_labels(dataset.dataset)
        indices = np.asarray(list(dataset.indices), dtype=np.int64)
        return base_labels[indices]

    if hasattr(dataset, "y_data") and hasattr(dataset, "sequence_window"):
        sequence_window = int(dataset.sequence_window)
        labels_by_day = [
            np.asarray(labels, dtype=np.int64)[sequence_window - 1 :]
            for labels in dataset.y_data
        ]
        return np.concatenate(labels_by_day) if labels_by_day else np.asarray([], dtype=np.int64)

    labels: list[int] = []
    for _, _, label in dataset:
        labels.append(int(label.item() if hasattr(label, "item") else label))
    return np.asarray(labels, dtype=np.int64)


def class_distribution(dataset: Any, num_classes: int | None = None) -> dict[str, Any]:
    labels = _effective_labels(dataset)
    if num_classes is None:
        num_classes = int(labels.max()) + 1 if len(labels) else 0
    counts = np.bincount(labels, minlength=num_classes)[:num_classes] if num_classes > 0 else np.asarray([], dtype=int)
    total = int(counts.sum())
    return {
        "total": total,
        "classes": {
            str(class_id): {
                "count": int(count),
                "percentage": 0.0 if total == 0 else float(100.0 * count / total),
            }
            for class_id, count in enumerate(counts)
        },
    }


def save_run_config_snapshot(
    config: ExperimentConfig,
    target: Path,
    *,
    fold_id: str | None = None,
    model_parameters: dict[str, int] | None = None,
    preprocessing_metadata: dict[str, Any] | None = None,
    sampling_summary: dict[str, Any] | None = None,
) -> None:
    payload = load_config_snapshot(config)
    payload.setdefault("run_metadata", {})
    payload["run_metadata"].update(
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "config_path": str(config.path),
            "fold_id": fold_id,
            "resolved_model_dir": str(config.training.model_dir),
            "resolved_best_model_path": str(config.training.best_model_path),
            "model_max_dt": {
                "quantile": float(config.model.max_dt_quantile),
                "resolved_max_dt": config.model.max_dt,
            },
            "class_weights": config.training.class_weights,
            "monitor": {
                "name": config.training.monitor,
                "mode": config.training.monitor_mode,
                "early_stopping_patience": config.training.early_stopping_patience,
                "early_stopping_warmup": config.training.early_stopping_warmup,
                "params": {
                    "lambda_ece": config.training.monitor_params.lambda_ece,
                    "lambda_rate": config.training.monitor_params.lambda_rate,
                },
            },
            "training_sampling": sampling_summary or {"enabled": False},
            "temperature_scaling": {
                "enabled": config.training.temperature_scaling.enabled,
            },
            "directional_thresholds": {
                "enabled": config.training.directional_thresholds.enabled,
                "method": config.training.directional_thresholds.method,
                "min": config.training.directional_thresholds.min_threshold,
                "max": config.training.directional_thresholds.max_threshold,
                "step": config.training.directional_thresholds.step,
                "delta": config.training.directional_thresholds.delta,
                "up_precision_floor": config.training.directional_thresholds.up_precision_floor,
                "down_precision_floor": config.training.directional_thresholds.down_precision_floor,
                "up_quantile": config.training.directional_thresholds.up_quantile,
                "down_quantile": config.training.directional_thresholds.down_quantile,
            },
            "model_parameters": model_parameters or {},
            "fast_smoothing_lambdas": fast_smoothing_lambda_summary(config, preprocessing_metadata),
        }
    )
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def _metric_value(metrics: Any, name: str) -> str:
    if metrics is None:
        return ""
    value = getattr(metrics, name, None)
    return "" if value is None else f"{float(value):.10g}"


def _mapped_class_id(config: ExperimentConfig, raw_label: int) -> int | None:
    mapped = config.data.label_mapping.get(raw_label)
    return None if mapped is None else int(mapped)


def _class_metric_value(metrics: Any, metric_name: str, class_id: int | None) -> str:
    if metrics is None or class_id is None:
        return ""
    values = getattr(metrics, metric_name, None)
    if values is None or class_id < 0 or class_id >= len(values):
        return ""
    return f"{float(values[class_id]):.10g}"


def _tailored_metric_values(metrics: Any, config: ExperimentConfig, prefix: str) -> dict[str, str]:
    """Return tailored monitor audit fields for one split."""
    keys = (
        "tailored_score",
        "tailored_ece_dir",
        "tailored_rate_penalty",
        "pred_rate_down",
        "true_rate_down",
        "pred_rate_up",
        "true_rate_up",
    )
    empty = {f"{prefix}_{key}": "" for key in keys}
    if metrics is None or not config.training.monitor_params.complete:
        return empty
    try:
        components = tailored_score_from_params(
            metrics,
            config.training.monitor_params,
            label_mapping=config.data.label_mapping,
        )
    except ValueError:
        return empty
    return {
        key: f"{float(value):.10g}"
        for key, value in components.prefixed(prefix).items()
    }


THRESHOLD_METRIC_KEYS = (
    "directional_macro_f1",
    "macro_f1",
    "accuracy",
    "down_precision",
    "down_recall",
    "down_f1",
    "up_precision",
    "up_recall",
    "up_f1",
    "pred_rate_down",
    "pred_rate_up",
    "pred_rate_neutral",
    "true_rate_down",
    "true_rate_up",
    "true_rate_neutral",
)


ARGMAX_ABLATION_METRIC_KEYS = (
    "accuracy",
    "macro_f1",
    "directional_macro_f1",
    "expected_calibration_error",
    "down_precision",
    "down_recall",
    "down_f1",
    "up_precision",
    "up_recall",
    "up_f1",
    "pred_rate_down",
    "pred_rate_up",
    "pred_rate_neutral",
    "true_rate_down",
    "true_rate_up",
    "true_rate_neutral",
)


def _threshold_metric_values(values: Any, prefix: str) -> dict[str, str]:
    """Return thresholded metric fields for CSV/log output."""
    empty = {f"{prefix}_threshold_{key}": "" for key in THRESHOLD_METRIC_KEYS}
    if not isinstance(values, dict):
        return empty
    return {
        f"{prefix}_threshold_{key}": "" if values.get(key) is None else f"{float(values[key]):.10g}"
        for key in THRESHOLD_METRIC_KEYS
    }


def _argmax_ablation_metric_values(
    metrics: Any,
    prefix: str,
    *,
    down_class_id: int | None,
    neutral_class_id: int | None,
    up_class_id: int | None,
) -> dict[str, str]:
    """Return argmax ablation metric fields for CSV/log output."""
    empty = {f"{prefix}_argmax_{key}": "" for key in ARGMAX_ABLATION_METRIC_KEYS}
    if metrics is None:
        return empty
    values = {
        "accuracy": _metric_value(metrics, "accuracy"),
        "macro_f1": _metric_value(metrics, "macro_f1"),
        "directional_macro_f1": _metric_value(metrics, "directional_macro_f1"),
        "expected_calibration_error": _metric_value(metrics, "expected_calibration_error"),
        "down_precision": _class_metric_value(metrics, "per_class_precision", down_class_id),
        "down_recall": _class_metric_value(metrics, "per_class_recall", down_class_id),
        "down_f1": _class_metric_value(metrics, "per_class_f1", down_class_id),
        "up_precision": _class_metric_value(metrics, "per_class_precision", up_class_id),
        "up_recall": _class_metric_value(metrics, "per_class_recall", up_class_id),
        "up_f1": _class_metric_value(metrics, "per_class_f1", up_class_id),
    }
    confusion = getattr(metrics, "confusion_matrix", None)
    if isinstance(confusion, list):
        matrix = np.asarray(confusion, dtype=np.float64)
        total = float(matrix.sum())
        if matrix.ndim == 2 and total > 0.0:
            for label, class_id in (
                ("down", down_class_id),
                ("up", up_class_id),
                ("neutral", neutral_class_id),
            ):
                if class_id is None or not 0 <= int(class_id) < matrix.shape[0]:
                    values[f"pred_rate_{label}"] = ""
                    values[f"true_rate_{label}"] = ""
                    continue
                values[f"pred_rate_{label}"] = f"{float(matrix[:, int(class_id)].sum() / total):.10g}"
                values[f"true_rate_{label}"] = f"{float(matrix[int(class_id), :].sum() / total):.10g}"
    return {f"{prefix}_argmax_{key}": values.get(key, "") for key in ARGMAX_ABLATION_METRIC_KEYS}


def _epoch_row(
    epoch_index: int,
    result: Any,
    fold: str,
    config: ExperimentConfig,
) -> dict[str, str | int]:
    row: dict[str, str | int] = {
        "fold": fold,
        "epoch": epoch_index,
        "train_loss": f"{result.train_loss:.10g}",
        "val_loss": f"{result.val_loss:.10g}",
        "test_loss": "" if result.test_loss is None else f"{result.test_loss:.10g}",
    }
    down_class_id = _mapped_class_id(config, -1)
    neutral_class_id = _mapped_class_id(config, 0)
    up_class_id = _mapped_class_id(config, 1)
    for split, metrics in (
        ("train", getattr(result, "train_metrics", None)),
        ("val", getattr(result, "val_metrics", None)),
        ("test", getattr(result, "test_metrics", None)),
    ):
        for metric_name in METRIC_NAMES:
            row[f"{split}_{metric_name}"] = _metric_value(metrics, metric_name)
        for class_id in range(config.model.num_classes):
            row[f"{split}_class_{class_id}_f1"] = _class_metric_value(metrics, "per_class_f1", class_id)
        row[f"{split}_down_precision"] = _class_metric_value(metrics, "per_class_precision", down_class_id)
        row[f"{split}_down_recall"] = _class_metric_value(metrics, "per_class_recall", down_class_id)
        row[f"{split}_up_precision"] = _class_metric_value(metrics, "per_class_precision", up_class_id)
        row[f"{split}_up_recall"] = _class_metric_value(metrics, "per_class_recall", up_class_id)
        row[f"{split}_ece_down"] = _class_metric_value(
            metrics,
            "per_class_expected_calibration_error",
            down_class_id,
        )
        row[f"{split}_ece_neutral"] = _class_metric_value(
            metrics,
            "per_class_expected_calibration_error",
            neutral_class_id,
        )
        row[f"{split}_ece_up"] = _class_metric_value(
            metrics,
            "per_class_expected_calibration_error",
            up_class_id,
        )
        row[f"{split}_pr_ap_down"] = _class_metric_value(metrics, "per_class_pr_ap", down_class_id)
        row[f"{split}_pr_ap_neutral"] = _class_metric_value(metrics, "per_class_pr_ap", neutral_class_id)
        row[f"{split}_pr_ap_up"] = _class_metric_value(metrics, "per_class_pr_ap", up_class_id)
        row[f"{split}_pr_auc_down"] = _class_metric_value(metrics, "per_class_pr_auc", down_class_id)
        row[f"{split}_pr_auc_neutral"] = _class_metric_value(metrics, "per_class_pr_auc", neutral_class_id)
        row[f"{split}_pr_auc_up"] = _class_metric_value(metrics, "per_class_pr_auc", up_class_id)
        row[f"{split}_roc_auc_down"] = _class_metric_value(metrics, "per_class_roc_auc", down_class_id)
        row[f"{split}_roc_auc_neutral"] = _class_metric_value(metrics, "per_class_roc_auc", neutral_class_id)
        row[f"{split}_roc_auc_up"] = _class_metric_value(metrics, "per_class_roc_auc", up_class_id)
        if split == "val":
            row.update(_tailored_metric_values(metrics, config, "val"))
            row.update(_threshold_metric_values(getattr(result, "val_threshold_metrics", None), "val"))
            row.update(
                _argmax_ablation_metric_values(
                    getattr(result, "val_argmax_metrics", None),
                    "val",
                    down_class_id=down_class_id,
                    neutral_class_id=neutral_class_id,
                    up_class_id=up_class_id,
                )
            )
        if split == "test":
            row.update(_threshold_metric_values(getattr(result, "test_threshold_metrics", None), "test"))
            row.update(
                _argmax_ablation_metric_values(
                    getattr(result, "test_argmax_metrics", None),
                    "test",
                    down_class_id=down_class_id,
                    neutral_class_id=neutral_class_id,
                    up_class_id=up_class_id,
                )
            )
    return row


def _epoch_fieldnames(config: ExperimentConfig) -> list[str]:
    fieldnames = ["fold", "epoch", "train_loss", "val_loss", "test_loss"]
    for split in SPLIT_METRIC_PREFIXES:
        fieldnames.extend(f"{split}_{metric}" for metric in METRIC_NAMES)
        fieldnames.extend(f"{split}_class_{class_id}_f1" for class_id in range(config.model.num_classes))
        fieldnames.extend(
            [
                f"{split}_down_precision",
                f"{split}_down_recall",
                f"{split}_up_precision",
                f"{split}_up_recall",
                f"{split}_ece_down",
                f"{split}_ece_neutral",
                f"{split}_ece_up",
                f"{split}_pr_ap_down",
                f"{split}_pr_ap_neutral",
                f"{split}_pr_ap_up",
                f"{split}_pr_auc_down",
                f"{split}_pr_auc_neutral",
                f"{split}_pr_auc_up",
                f"{split}_roc_auc_down",
                f"{split}_roc_auc_neutral",
                f"{split}_roc_auc_up",
            ]
        )
        if split == "val":
            fieldnames.extend(
                [
                    "val_tailored_score",
                    "val_tailored_ece_dir",
                    "val_tailored_rate_penalty",
                    "val_pred_rate_down",
                    "val_true_rate_down",
                    "val_pred_rate_up",
                    "val_true_rate_up",
                ]
            )
        if split in {"val", "test"}:
            fieldnames.extend(f"{split}_threshold_{key}" for key in THRESHOLD_METRIC_KEYS)
            fieldnames.extend(f"{split}_argmax_{key}" for key in ARGMAX_ABLATION_METRIC_KEYS)
    return fieldnames


def save_epoch_history(
    history: list[Any],
    target: Path,
    *,
    config: ExperimentConfig,
    fold: str = "single",
) -> None:
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_epoch_fieldnames(config))
        writer.writeheader()
        for epoch_index, result in enumerate(history, start=1):
            writer.writerow(_epoch_row(epoch_index, result, fold, config))


def _confusion_payload(metrics: Any) -> dict[str, Any] | None:
    if metrics is None:
        return None
    return {
        "raw": getattr(metrics, "confusion_matrix", []),
        "normalized_by_true_class": getattr(metrics, "normalized_confusion_matrix", []),
    }


def save_confusion_matrices(
    history: list[Any],
    target: Path,
    *,
    fold: str = "single",
) -> None:
    payload = {
        "normalization": (
            "Rows are true classes; columns are predicted classes. Normalized rows sum to 1 when support > 0. "
            "When directional thresholds are enabled, validation/test entries for the best epoch use thresholded "
            "decisions; *_argmax_ablation entries keep the original softmax argmax decisions for comparison."
        ),
        "folds": {
            fold: {
                f"epoch_{epoch_index}": {
                    "train": _confusion_payload(getattr(result, "train_metrics", None)),
                    "validation": _confusion_payload(getattr(result, "val_metrics", None)),
                    "test": _confusion_payload(getattr(result, "test_metrics", None)),
                    "validation_argmax_ablation": _confusion_payload(
                        getattr(result, "val_argmax_metrics", None),
                    ),
                    "test_argmax_ablation": _confusion_payload(
                        getattr(result, "test_argmax_metrics", None),
                    ),
                }
                for epoch_index, result in enumerate(history, start=1)
            }
        },
    }
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def _class_label_names(config: ExperimentConfig) -> dict[str, str]:
    raw_label_names = {-1: "down", 0: "neutral", 1: "up"}
    labels = {str(class_id): f"class_{class_id}" for class_id in range(config.model.num_classes)}
    for raw_label, mapped_label in config.data.label_mapping.items():
        mapped_id = int(mapped_label)
        if 0 <= mapped_id < config.model.num_classes:
            labels[str(mapped_id)] = raw_label_names.get(int(raw_label), f"raw_{raw_label}")
    return labels


def _ordered_class_labels(config: ExperimentConfig) -> list[str]:
    """Return class labels ordered by model class id."""
    labels = _class_label_names(config)
    return [labels.get(str(class_id), f"class_{class_id}") for class_id in range(config.model.num_classes)]


def _safe_artifact_label(label: str) -> str:
    """Return a filesystem-safe class label for artifact filenames."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "class"


def prediction_outputs_to_frame(outputs: dict[str, Any], config: ExperimentConfig) -> pd.DataFrame:
    """Convert collected model outputs to a probability CSV frame."""
    probabilities = np.asarray(outputs.get("probabilities", []), dtype=np.float32)
    targets = np.asarray(outputs.get("targets", []), dtype=np.int64).reshape(-1)
    predictions = np.asarray(outputs.get("predictions", []), dtype=np.int64).reshape(-1)
    sample_index = np.asarray(
        outputs.get("sample_index", np.arange(targets.shape[0])),
        dtype=np.int64,
    ).reshape(-1)

    if probabilities.ndim == 1 and probabilities.size == 0:
        probabilities = np.empty((0, config.model.num_classes), dtype=np.float32)
    if probabilities.ndim != 2:
        raise ValueError("prediction probabilities must be a 2D array.")
    if probabilities.shape[0] != targets.shape[0] or predictions.shape[0] != targets.shape[0]:
        raise ValueError("prediction outputs have inconsistent row counts.")
    if sample_index.shape[0] != targets.shape[0]:
        raise ValueError("sample indices and targets have inconsistent row counts.")

    labels = _ordered_class_labels(config)
    frame_payload: dict[str, Any] = {
        "sample_index": sample_index,
        "true_label": [
            labels[class_id] if 0 <= class_id < len(labels) else f"class_{class_id}"
            for class_id in targets
        ],
        "pred_label": [
            labels[class_id] if 0 <= class_id < len(labels) else f"class_{class_id}"
            for class_id in predictions
        ],
    }
    argmax_predictions = outputs.get("argmax_predictions")
    if argmax_predictions is not None:
        argmax_predictions_array = np.asarray(argmax_predictions, dtype=np.int64).reshape(-1)
        if argmax_predictions_array.shape[0] != targets.shape[0]:
            raise ValueError("argmax predictions and targets have inconsistent row counts.")
        frame_payload["argmax_pred_label"] = [
            labels[class_id] if 0 <= class_id < len(labels) else f"class_{class_id}"
            for class_id in argmax_predictions_array
        ]
    for class_id, label in enumerate(labels):
        column = f"p_{_safe_artifact_label(label)}"
        values = probabilities[:, class_id] if class_id < probabilities.shape[1] else np.zeros(targets.shape[0])
        frame_payload[column] = values
    return pd.DataFrame(frame_payload)


def save_probability_outputs(outputs: dict[str, Any], target: Path, config: ExperimentConfig) -> None:
    """Write collected post-softmax probabilities to CSV."""
    target.parent.mkdir(parents=True, exist_ok=True)
    prediction_outputs_to_frame(outputs, config).to_csv(target, index=False)


def save_best_pr_artifacts(
    validation_outputs: dict[str, Any],
    *,
    curves_dir: Path,
    thresholds_path: Path,
    config: ExperimentConfig,
    best_epoch: int,
    fold: str = "single",
) -> dict[str, Any]:
    """Write validation PR curves and max-F1 thresholds for the best epoch."""
    probabilities = np.asarray(validation_outputs.get("probabilities", []), dtype=np.float32)
    targets = np.asarray(validation_outputs.get("targets", []), dtype=np.int64).reshape(-1)
    if probabilities.ndim == 1 and probabilities.size == 0:
        probabilities = np.empty((0, config.model.num_classes), dtype=np.float32)
    if probabilities.ndim != 2:
        raise ValueError("validation probabilities must be a 2D array.")
    if probabilities.shape[0] != targets.shape[0]:
        raise ValueError("validation probabilities and targets have inconsistent row counts.")

    labels = _ordered_class_labels(config)
    ranking_metrics = per_class_ranking_metrics(
        probabilities,
        targets,
        config.model.num_classes,
        class_names=labels,
        include_curves=True,
    )
    curves = ranking_metrics["pr_curves"]
    pr_ap_values = ranking_metrics["pr_ap"]
    pr_auc_values = ranking_metrics["pr_auc"]
    roc_auc_values = ranking_metrics["roc_auc"]
    curves_dir.mkdir(parents=True, exist_ok=True)
    threshold_payload: dict[str, Any] = {
        "description": "One-vs-rest thresholds selected on validation by maximizing F1.",
        "fold": fold,
        "split": "validation",
        "best_epoch": int(best_epoch),
        "selection_rule": "max_f1",
        "classes": {},
    }
    curve_paths: dict[str, str] = {}
    for class_id, label in enumerate(labels):
        safe_label = _safe_artifact_label(label)
        curve = curves[label]
        curve_path = curves_dir / f"validation_best_epoch_{best_epoch}_{safe_label}.csv"
        curve.to_csv(curve_path, index=False)
        threshold = best_f1_threshold(curve)
        threshold_payload["classes"][label] = {
            "class_id": int(class_id),
            "threshold": float(threshold["threshold"]),
            "precision": float(threshold["precision"]),
            "recall": float(threshold["recall"]),
            "f1": float(threshold["f1"]),
            "pr_ap": float(pr_ap_values[class_id]),
            "pr_auc": float(pr_auc_values[class_id]),
            "roc_auc": float(roc_auc_values[class_id]),
            "curve_csv": str(curve_path),
        }
        curve_paths[label] = str(curve_path)

    thresholds_path.parent.mkdir(parents=True, exist_ok=True)
    with thresholds_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(threshold_payload, handle, sort_keys=False, allow_unicode=True)
    return {
        "thresholds": threshold_payload,
        "thresholds_path": str(thresholds_path),
        "curve_paths": curve_paths,
    }


def save_directional_threshold_artifact(
    payload: dict[str, Any],
    target: Path,
) -> None:
    """Write selected directional thresholds to YAML."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def _expert_usage_payload(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if not isinstance(usage, dict):
        return None
    return usage


def save_expert_usage(
    history: list[Any],
    target: Path,
    *,
    config: ExperimentConfig,
    fold: str = "single",
) -> None:
    payload = {
        "description": (
            "MoE routing usage. selected_* counts include every top-k assignment; "
            "primary_* counts include only the first/top expert per token. "
            "by_true_class applies each sequence label to all tokens in that sequence."
        ),
        "class_labels": _class_label_names(config),
        "folds": {
            fold: {
                f"epoch_{epoch_index}": {
                    "train": _expert_usage_payload(getattr(result, "train_expert_usage", None)),
                    "validation": _expert_usage_payload(getattr(result, "val_expert_usage", None)),
                    "test": _expert_usage_payload(getattr(result, "test_expert_usage", None)),
                }
                for epoch_index, result in enumerate(history, start=1)
            }
        },
    }
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def _write_class_distribution(handle: Any, split: str, distribution: dict[str, Any]) -> None:
    handle.write(f"{split}_total: {distribution['total']}\n")
    for class_id, values in distribution["classes"].items():
        handle.write(
            f"{split}_class_{class_id}: "
            f"{values['count']} ({values['percentage']:.2f}%)\n"
        )


def _write_lambda_summary(handle: Any, summary: dict[str, Any]) -> None:
    handle.write("\nFast smoothing lambdas\n")
    handle.write(f"method: {summary['method']}\n")
    if summary.get("status") == "not_applicable":
        handle.write("status: not_applicable\n")
        return

    for kind in ("price", "volume"):
        values = summary[kind]
        selected = values.get("selected_smoothing_lambda")
        selected_text = "not_available" if selected is None else f"{float(selected):.10g}"
        handle.write(
            f"{kind}: enabled={values.get('enabled')}, n_basis={values.get('n_basis')}, "
            f"target_df={values.get('target_df')}, eval_at={values.get('eval_at')}, "
            f"lambda={selected_text}"
        )
        if values.get("effective_df") is not None:
            handle.write(f", effective_df={float(values['effective_df']):.10g}")
        if values.get("mean_gcv") is not None:
            handle.write(f", mean_gcv={float(values['mean_gcv']):.10g}")
        handle.write("\n")


def _monitor_value(result: Any, config: ExperimentConfig) -> float:
    return epoch_monitor_value(
        result,
        monitor=config.training.monitor,
        monitor_params=config.training.monitor_params,
        label_mapping=config.data.label_mapping,
    )


def _write_best_epoch_summary(
    handle: Any,
    history: list[Any],
    config: ExperimentConfig,
    *,
    selected_epoch: int | None = None,
    selected_monitor_value: float | None = None,
) -> None:
    handle.write("\nBest epoch\n")
    if not history:
        handle.write("status: unavailable\n")
        return
    if selected_epoch is None:
        reverse = config.training.monitor_mode == "max"
        best_index, best_result = sorted(
            enumerate(history, start=1),
            key=lambda item: _monitor_value(item[1], config),
            reverse=reverse,
        )[0]
        best_monitor_value = _monitor_value(best_result, config)
    else:
        best_index = int(selected_epoch)
        if not 1 <= best_index <= len(history):
            raise ValueError("selected best epoch is outside the epoch history.")
        best_result = history[best_index - 1]
        best_monitor_value = (
            float(selected_monitor_value)
            if selected_monitor_value is not None
            else _monitor_value(best_result, config)
        )
    handle.write(f"monitor: {config.training.monitor}\n")
    handle.write(f"monitor_mode: {config.training.monitor_mode}\n")
    if config.training.monitor_params.complete:
        handle.write(f"monitor_lambda_ece: {config.training.monitor_params.lambda_ece:.10g}\n")
        handle.write(f"monitor_lambda_rate: {config.training.monitor_params.lambda_rate:.10g}\n")
    handle.write(f"monitor_value: {best_monitor_value:.10g}\n")
    monitor_metrics = getattr(best_result, "val_argmax_metrics", None) or getattr(best_result, "val_metrics", None)
    if getattr(best_result, "val_argmax_metrics", None) is not None:
        handle.write("monitor_metric_source: validation_argmax_before_thresholding\n")
    if config.training.monitor == "tailored_score" and monitor_metrics is not None:
        components = tailored_score_from_params(
            monitor_metrics,
            config.training.monitor_params,
            label_mapping=config.data.label_mapping,
        )
        for key, value in components.prefixed("val").items():
            handle.write(f"{key}: {value:.10g}\n")
    handle.write(f"epoch: {best_index}\n")
    handle.write(f"train_loss: {best_result.train_loss:.10g}\n")
    handle.write(f"val_loss: {best_result.val_loss:.10g}\n")
    if best_result.test_loss is not None:
        handle.write(f"test_loss: {best_result.test_loss:.10g}\n")
    for split, metrics in (
        ("train", getattr(best_result, "train_metrics", None)),
        ("validation", getattr(best_result, "val_metrics", None)),
        ("test", getattr(best_result, "test_metrics", None)),
    ):
        if metrics is None:
            continue
        handle.write(f"{split}_accuracy: {metrics.accuracy:.10g}\n")
        handle.write(f"{split}_macro_f1: {metrics.macro_f1:.10g}\n")
        handle.write(f"{split}_directional_macro_f1: {metrics.directional_macro_f1:.10g}\n")
        handle.write(f"{split}_ece: {metrics.expected_calibration_error:.10g}\n")
        down_class_id = _mapped_class_id(config, -1)
        neutral_class_id = _mapped_class_id(config, 0)
        up_class_id = _mapped_class_id(config, 1)
        for label, class_id in (("down", down_class_id), ("neutral", neutral_class_id), ("up", up_class_id)):
            value = _class_metric_value(metrics, "per_class_expected_calibration_error", class_id)
            if value:
                handle.write(f"{split}_ece_{label}: {value}\n")
            pr_ap = _class_metric_value(metrics, "per_class_pr_ap", class_id)
            if pr_ap:
                handle.write(f"{split}_pr_ap_{label}: {pr_ap}\n")
            pr_auc = _class_metric_value(metrics, "per_class_pr_auc", class_id)
            if pr_auc:
                handle.write(f"{split}_pr_auc_{label}: {pr_auc}\n")
            roc_auc_value = _class_metric_value(metrics, "per_class_roc_auc", class_id)
            if roc_auc_value:
                handle.write(f"{split}_roc_auc_{label}: {roc_auc_value}\n")


def _write_sampling_summary(handle: Any, summary: dict[str, Any]) -> None:
    handle.write(f"enabled: {summary.get('enabled', False)}\n")
    if not summary.get("enabled"):
        return
    handle.write(f"method: {summary.get('method')}\n")
    handle.write(f"neutral_to_directional_ratio: {summary.get('neutral_to_directional_ratio')}\n")
    handle.write(f"base_seed: {summary.get('base_seed')}\n")
    handle.write(f"epoch_seed_rule: {summary.get('epoch_seed_rule')}\n")
    handle.write(
        "train_metric_scope: train_* metrics are computed on the sampled train windows for each epoch; "
        "validation/test metrics use complete splits.\n"
    )
    for section in ("full_counts", "sampled_counts_per_epoch"):
        values = summary.get(section)
        if isinstance(values, dict):
            handle.write(f"{section}: {values}\n")


def save_run_log(
    *,
    target: Path,
    config: ExperimentConfig,
    run_stem: str,
    dataset_sizes: dict[str, int],
    class_distributions: dict[str, dict[str, Any]],
    history: list[Any],
    losses_path: Path,
    confusion_matrices_path: Path,
    expert_usage_path: Path,
    config_snapshot_path: Path,
    model_parameters: dict[str, int],
    pr_thresholds_path: Path | None = None,
    pr_curves_dir: Path | None = None,
    probabilities_dir: Path | None = None,
    temperature_scaling_path: Path | None = None,
    temperature_scaling_summary: dict[str, Any] | None = None,
    directional_thresholds_path: Path | None = None,
    directional_threshold_summary: dict[str, Any] | None = None,
    selected_best_epoch: int | None = None,
    selected_monitor_value: float | None = None,
    preprocessing_metadata: dict[str, Any] | None = None,
    sampling_summary: dict[str, Any] | None = None,
    timing: dict[str, Any] | None = None,
    fold: str = "single",
) -> None:
    lambda_summary = fast_smoothing_lambda_summary(config, preprocessing_metadata)

    with target.open("w", encoding="utf-8") as handle:
        handle.write(f"Run: {run_stem}\n")
        handle.write(f"Experiment: {config.experiment.name}\n")
        handle.write(f"Fold: {fold}\n")
        handle.write(f"Created at: {datetime.now().isoformat(timespec='seconds')}\n")
        handle.write(f"Config: {config.path}\n")
        handle.write(f"Config snapshot: {config_snapshot_path}\n")
        handle.write(f"Loss and metrics CSV: {losses_path}\n")
        handle.write(f"Confusion matrices: {confusion_matrices_path}\n")
        handle.write(f"Expert usage: {expert_usage_path}\n")
        if pr_thresholds_path is not None:
            handle.write(f"PR thresholds: {pr_thresholds_path}\n")
        if pr_curves_dir is not None:
            handle.write(f"PR curves directory: {pr_curves_dir}\n")
        if probabilities_dir is not None:
            handle.write(f"Probability outputs directory: {probabilities_dir}\n")
        if temperature_scaling_path is not None:
            handle.write(f"Temperature scaling: {temperature_scaling_path}\n")
        if directional_thresholds_path is not None:
            handle.write(f"Directional thresholds: {directional_thresholds_path}\n")
        handle.write(f"Model directory: {config.training.model_dir}\n")
        handle.write(f"Best model path: {config.training.best_model_path}\n")
        if timing:
            handle.write("\nTiming\n")
            for key, value in timing.items():
                handle.write(f"{key}: {value}\n")
        handle.write("\nModel temporal window\n")
        handle.write(f"max_dt_quantile: {config.model.max_dt_quantile}\n")
        handle.write(f"resolved_max_dt: {config.model.max_dt}\n")
        handle.write("\nTraining class weights\n")
        handle.write(f"class_weights: {config.training.class_weights}\n")
        handle.write("\nTraining monitor\n")
        handle.write(f"monitor: {config.training.monitor}\n")
        handle.write(f"monitor_mode: {config.training.monitor_mode}\n")
        handle.write(f"early_stopping_patience: {config.training.early_stopping_patience}\n")
        handle.write(f"early_stopping_warmup: {config.training.early_stopping_warmup}\n")
        if config.training.monitor_params.complete:
            handle.write(f"monitor_lambda_ece: {config.training.monitor_params.lambda_ece:.10g}\n")
            handle.write(f"monitor_lambda_rate: {config.training.monitor_params.lambda_rate:.10g}\n")
        handle.write("\nTemperature scaling\n")
        calibration_summary = temperature_scaling_summary or {"enabled": False}
        handle.write(f"enabled: {calibration_summary.get('enabled', False)}\n")
        if calibration_summary.get("enabled"):
            handle.write("probability_source: temperature_scaled_logits\n")
            handle.write("calibration_loss: unweighted_cross_entropy\n")
            handle.write("selection_split: validation\n")
            for key in (
                "temperature",
                "validation_nll_before",
                "validation_nll_after",
                "n_samples",
                "n_classes",
                "optimizer_evaluations",
                "fit_seconds",
                "fit_duration",
            ):
                if key in calibration_summary:
                    handle.write(f"{key}: {calibration_summary[key]}\n")
            if temperature_scaling_path is not None:
                handle.write(f"artifact: {temperature_scaling_path}\n")
        handle.write("\nDirectional thresholds\n")
        threshold_summary = directional_threshold_summary or {"enabled": False}
        handle.write(f"enabled: {threshold_summary.get('enabled', False)}\n")
        if threshold_summary.get("enabled"):
            handle.write("classification_mode: directional_thresholds\n")
            handle.write("argmax_ablation_columns: val_argmax_* and test_argmax_*\n")
            handle.write(
                "decision_scope: validation/test decision metrics, pred_label outputs, and confusion matrices "
                "use thresholded decisions; PR/ROC/AP stay probability-ranking metrics.\n"
            )
            for key in (
                "threshold_down",
                "threshold_up",
                "down_enabled",
                "up_enabled",
                "method",
                "score",
                "rate_penalty",
                "min_directional_precision",
                "selection_split",
                "selection_metric",
                "decision_tie_break",
                "selection_final_tie_break",
                "grid_min",
                "grid_max",
                "grid_step",
                "refinement_steps",
                "delta",
                "down_precision_floor",
                "up_precision_floor",
                "down_quantile",
                "up_quantile",
                "n_candidates",
            ):
                if key in threshold_summary:
                    handle.write(f"{key}: {threshold_summary[key]}\n")
            stages = threshold_summary.get("optimization_stages")
            if isinstance(stages, list):
                handle.write(f"optimization_stages: {len(stages)}\n")
            details = threshold_summary.get("selection_details")
            if isinstance(details, dict) and details:
                handle.write(f"selection_details: {details}\n")
            if directional_thresholds_path is not None:
                handle.write(f"artifact: {directional_thresholds_path}\n")
        handle.write("\nTraining sampling\n")
        _write_sampling_summary(handle, sampling_summary or {"enabled": False})
        handle.write("\nModel parameters\n")
        for key, value in model_parameters.items():
            handle.write(f"{key}: {value}\n")

        _write_lambda_summary(handle, lambda_summary)

        handle.write("\nDatasets\n")
        for split, size in dataset_sizes.items():
            handle.write(f"{split}_sequences: {size}\n")

        handle.write("\nClass distributions\n")
        for split, distribution in class_distributions.items():
            _write_class_distribution(handle, split, distribution)

        _write_best_epoch_summary(
            handle,
            history,
            config,
            selected_epoch=selected_best_epoch,
            selected_monitor_value=selected_monitor_value,
        )

        handle.write("\nEpoch history\n")
        epoch_fields = _epoch_fieldnames(config)
        handle.write(",".join(epoch_fields) + "\n" if history else "")
        for epoch_index, result in enumerate(history, start=1):
            row = _epoch_row(epoch_index, result, fold, config)
            handle.write(",".join(str(row.get(field, "")) for field in epoch_fields) + "\n")


def save_run_summary(summary: dict[str, Any], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False, allow_unicode=True)
