from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    from configuration import ExperimentConfig
except ImportError:  # pragma: no cover
    from .configuration import ExperimentConfig


RUN_FILE_PATTERN = re.compile(r"^run_(\d+)(?:[._]|$)")
PREPROCESSING_METADATA_FILENAME = "preprocessing_metadata.yaml"
METRIC_NAMES = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
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
    price_static_plgs: dict[str, Any] | None = None,
    volume_static_exp: dict[str, Any] | None = None,
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
        "fast_smoothing_lambdas": lambdas,
    }
    if label_distribution is not None:
        payload["label_distribution"] = label_distribution
    if price_static_plgs is not None:
        payload["price_static_plgs"] = price_static_plgs
    if volume_static_exp is not None:
        payload["volume_static_exp"] = volume_static_exp
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
            ]
        )
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
        "normalization": "Rows are true classes; columns are predicted classes. Normalized rows sum to 1 when support > 0.",
        "folds": {
            fold: {
                f"epoch_{epoch_index}": {
                    "train": _confusion_payload(getattr(result, "train_metrics", None)),
                    "validation": _confusion_payload(getattr(result, "val_metrics", None)),
                    "test": _confusion_payload(getattr(result, "test_metrics", None)),
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


def _write_best_epoch_summary(handle: Any, history: list[Any]) -> None:
    handle.write("\nBest epoch\n")
    if not history:
        handle.write("status: unavailable\n")
        return
    best_index, best_result = min(enumerate(history, start=1), key=lambda item: item[1].val_loss)
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
        handle.write(f"{split}_ece: {metrics.expected_calibration_error:.10g}\n")


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
    config_snapshot_path: Path,
    model_parameters: dict[str, int],
    preprocessing_metadata: dict[str, Any] | None = None,
    timing: dict[str, Any] | None = None,
    fold: str = "single",
) -> None:
    lambda_summary = fast_smoothing_lambda_summary(config, preprocessing_metadata)

    with target.open("w", encoding="utf-8") as handle:
        handle.write(f"Run: {run_stem}\n")
        handle.write(f"Fold: {fold}\n")
        handle.write(f"Created at: {datetime.now().isoformat(timespec='seconds')}\n")
        handle.write(f"Config: {config.path}\n")
        handle.write(f"Config snapshot: {config_snapshot_path}\n")
        handle.write(f"Loss and metrics CSV: {losses_path}\n")
        handle.write(f"Confusion matrices: {confusion_matrices_path}\n")
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

        _write_best_epoch_summary(handle, history)

        handle.write("\nEpoch history\n")
        handle.write(",".join(_epoch_fieldnames(config)) + "\n" if history else "")
        for epoch_index, result in enumerate(history, start=1):
            row = _epoch_row(epoch_index, result, fold, config)
            handle.write(",".join(str(value) for value in row.values()) + "\n")


def save_run_summary(summary: dict[str, Any], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False, allow_unicode=True)
