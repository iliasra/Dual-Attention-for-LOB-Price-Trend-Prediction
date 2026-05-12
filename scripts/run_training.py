from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from configuration import ExperimentConfig, load_config
from datasets import LOBDataset
from model import build_model
from run_logging import (
    class_distribution,
    load_preprocessing_metadata,
    model_parameter_summary,
    next_run_stem,
    resolve_config_path,
    save_confusion_matrices,
    save_epoch_history,
    save_run_config_snapshot,
    save_run_log,
    save_run_summary,
)
from training import LobTrainer, class_weights_from_sequence_labels


def sequence_paths(sequence_dir: Path, split: str) -> tuple[list[str], list[str], list[str]]:
    split_dir = sequence_dir / split
    x_paths: list[str] = []
    t_paths: list[str] = []
    y_paths: list[str] = []

    for x_path in sorted(split_dir.glob("*_features.npy")):
        prefix = x_path.name.removesuffix("_features.npy")
        t_path = x_path.with_name(f"{prefix}_times.npy")
        y_path = x_path.with_name(f"{prefix}_labels.npy")
        if not t_path.exists() or not y_path.exists():
            raise FileNotFoundError(f"Missing matching times/labels files for {x_path}")
        x_paths.append(str(x_path))
        t_paths.append(str(t_path))
        y_paths.append(str(y_path))

    return x_paths, t_paths, y_paths


def build_dataset(sequence_dir: Path, split: str, sequence_window: int) -> LOBDataset:
    x_paths, t_paths, y_paths = sequence_paths(sequence_dir, split)
    return LOBDataset(x_paths, t_paths, y_paths, sequence_window=sequence_window)


def sequence_time_span_quantile(
    time_arrays: list[np.ndarray],
    *,
    sequence_window: int,
    quantile: float,
) -> dict[str, float | int]:
    if sequence_window <= 0:
        raise ValueError("sequence_window must be > 0 to compute model.max_dt.")
    if not 0.0 <= quantile <= 100.0:
        raise ValueError("model.max_dt_quantile must be in [0, 100].")

    spans_by_day: list[np.ndarray] = []
    for times in time_arrays:
        values = np.asarray(times, dtype=np.float64)
        if values.shape[0] < sequence_window:
            continue
        if np.any(np.diff(values) < 0.0):
            raise ValueError("Cannot compute model.max_dt because train timestamps are not non-decreasing.")
        spans = values[sequence_window - 1 :] - values[: values.shape[0] - sequence_window + 1]
        spans_by_day.append(spans)

    if not spans_by_day:
        raise ValueError("Cannot compute model.max_dt because the train split has no full sequence windows.")

    all_spans = np.concatenate(spans_by_day)
    all_spans = all_spans[np.isfinite(all_spans)]
    if all_spans.size == 0:
        raise ValueError("Cannot compute model.max_dt because all train sequence time spans are non-finite.")
    if np.any(all_spans < 0.0):
        raise ValueError("Cannot compute model.max_dt because train timestamps are not non-decreasing.")

    max_dt = float(np.quantile(all_spans, quantile / 100.0))
    return {
        "quantile": float(quantile),
        "max_dt": max_dt,
        "n_windows": int(all_spans.size),
        "min_span": float(np.min(all_spans)),
        "max_span": float(np.max(all_spans)),
    }


def resolve_model_max_dt(config: ExperimentConfig, train_dataset: LOBDataset) -> dict[str, float | int]:
    summary = sequence_time_span_quantile(
        train_dataset.T_data,
        sequence_window=config.data.sequence_window,
        quantile=config.model.max_dt_quantile,
    )
    config.model.max_dt = float(summary["max_dt"])
    return summary


def sequence_label_values(dataset: LOBDataset) -> np.ndarray:
    """Return the label attached to each sequence window in a compact LOBDataset."""
    labels_by_day = [
        np.asarray(labels, dtype=np.int64)[dataset.sequence_window - 1 :]
        for labels in dataset.y_data
        if len(labels) >= dataset.sequence_window
    ]
    return np.concatenate(labels_by_day) if labels_by_day else np.asarray([], dtype=np.int64)


def resolve_class_weights(config: ExperimentConfig, train_dataset: LOBDataset) -> dict[str, list[float] | list[int] | bool]:
    """Fit runtime loss class weights from train sequence labels for one fold."""
    labels = sequence_label_values(train_dataset)
    weights, counts = class_weights_from_sequence_labels(
        labels,
        num_classes=config.model.num_classes,
        gamma_mode=config.training.focal_gamma > 0.0,
    )
    config.training.class_weights = weights
    return {
        "weights": weights,
        "counts": counts,
        "gamma_mode": config.training.focal_gamma > 0.0,
    }


def fold_artifact_paths(
    *,
    sequence_dir: Path,
    run_log_dir: Path,
    run_result_dir: Path,
    fold_id: str,
) -> dict[str, Path]:
    return {
        "sequence_dir": sequence_dir / fold_id,
        "log_dir": run_log_dir / fold_id,
        "result_dir": run_result_dir / fold_id,
    }


def train_fold(
    *,
    config: ExperimentConfig,
    fold_id: str,
    fold_sequence_dir: Path,
    fold_log_dir: Path,
    fold_result_dir: Path,
    run_stem: str,
) -> dict:
    print(f"Starting training fold '{fold_id}'.")
    print(f"Fold '{fold_id}' sequence directory: {fold_sequence_dir}")
    print(f"Fold '{fold_id}' log directory: {fold_log_dir}")
    print(f"Fold '{fold_id}' result directory: {fold_result_dir}")
    fold_log_dir.mkdir(parents=True, exist_ok=True)
    fold_result_dir.mkdir(parents=True, exist_ok=True)

    run_log_path = fold_log_dir / "run.log"
    run_config_path = fold_log_dir / "config.yaml"
    run_losses_path = fold_log_dir / "metrics.csv"
    run_confusion_matrices_path = fold_log_dir / "confusion_matrices.yaml"
    config.training.model_dir = str(fold_result_dir)
    preprocessing_metadata = load_preprocessing_metadata(fold_sequence_dir)

    train_dataset = build_dataset(fold_sequence_dir, "train", config.data.sequence_window)
    if len(train_dataset) == 0:
        raise ValueError(
            f"No training sequences found in {fold_sequence_dir / 'train'}. "
            "Run scripts/process_data.py first."
        )

    validation_dataset = build_dataset(fold_sequence_dir, "validation", config.data.sequence_window)
    if len(validation_dataset) == 0:
        raise ValueError(
            f"No validation sequences found in {fold_sequence_dir / 'validation'}. "
            "Validation dates must be configured and preprocessed explicitly."
        )

    test_dataset = build_dataset(fold_sequence_dir, "test", config.data.sequence_window)
    if len(test_dataset) == 0:
        raise ValueError(
            f"No test sequences found in {fold_sequence_dir / 'test'}. "
            "Test dates must be configured and preprocessed explicitly."
        )

    max_dt_summary = resolve_model_max_dt(config, train_dataset)
    print(
        f"Fold '{fold_id}' model max_dt selected from train spans: "
        f"q{max_dt_summary['quantile']:.4g}={max_dt_summary['max_dt']:.10g} "
        f"over {max_dt_summary['n_windows']} windows."
    )
    class_weight_summary = resolve_class_weights(config, train_dataset)
    print(
        f"Fold '{fold_id}' class weights selected from train sequence labels: "
        f"counts={class_weight_summary['counts']}, weights={class_weight_summary['weights']}."
    )

    loader_kwargs = config.training.data_loader_kwargs()
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    x_sample, _, _ = train_dataset[0]
    model = build_model(config.model, d_input=x_sample.shape[-1])
    model_parameters = model_parameter_summary(model)
    class_distributions = {
        "train": class_distribution(train_dataset, config.model.num_classes),
        "validation": class_distribution(validation_dataset, config.model.num_classes),
        "test": class_distribution(test_dataset, config.model.num_classes),
    }
    dataset_sizes = {
        "train": len(train_dataset),
        "validation": len(validation_dataset),
        "test": len(test_dataset),
    }
    print(
        f"Fold '{fold_id}' dataset sizes: "
        f"train={dataset_sizes['train']}, "
        f"validation={dataset_sizes['validation']}, "
        f"test={dataset_sizes['test']}."
    )
    save_run_config_snapshot(
        config,
        run_config_path,
        fold_id=fold_id,
        model_parameters=model_parameters,
        preprocessing_metadata=preprocessing_metadata,
    )
    trainer = LobTrainer(config.training)
    _, history = trainer.fit(model, train_loader, validation_loader, test_loader=test_loader)

    save_epoch_history(history, run_losses_path, config=config, fold=fold_id)
    save_confusion_matrices(history, run_confusion_matrices_path, fold=fold_id)
    save_run_log(
        target=run_log_path,
        config=config,
        run_stem=run_stem,
        dataset_sizes=dataset_sizes,
        class_distributions=class_distributions,
        history=history,
        losses_path=run_losses_path,
        confusion_matrices_path=run_confusion_matrices_path,
        config_snapshot_path=run_config_path,
        model_parameters=model_parameters,
        preprocessing_metadata=preprocessing_metadata,
        fold=fold_id,
    )
    print(f"Fold {fold_id} training complete. Best model saved to: {config.training.best_model_path}")
    print(f"Fold {fold_id} run log saved to: {run_log_path}")
    for epoch_index, result in enumerate(history, start=1):
        test_suffix = "" if result.test_loss is None else f", test_loss={result.test_loss:.6f}"
        val_metrics = result.val_metrics
        metric_suffix = "" if val_metrics is None else (
            f", val_acc={val_metrics.accuracy:.4f}, val_macro_f1={val_metrics.macro_f1:.4f}, "
            f"val_ece={val_metrics.expected_calibration_error:.4f}"
        )
        print(
            f"epoch {epoch_index}: train_loss={result.train_loss:.6f}, "
            f"val_loss={result.val_loss:.6f}{test_suffix}{metric_suffix}"
        )

    return {
        "sequence_dir": str(fold_sequence_dir),
        "log_dir": str(fold_log_dir),
        "result_dir": str(fold_result_dir),
        "best_model_path": str(config.training.best_model_path),
        "model_max_dt": max_dt_summary,
        "class_weights": class_weight_summary,
        "dataset_sizes": dataset_sizes,
        "logs": {
            "run_log": str(run_log_path),
            "config": str(run_config_path),
            "metrics": str(run_losses_path),
            "confusion_matrices": str(run_confusion_matrices_path),
        },
    }


def main() -> None:
    config = load_config()
    sequence_dir = resolve_config_path(config, config.data.sequence_data_dir)
    logs_dir = resolve_config_path(config, config.data.logs_dir)
    run_stem = next_run_stem(logs_dir)
    run_log_dir = logs_dir / run_stem
    resolved_model_dir = resolve_config_path(config, config.training.model_dir)
    run_result_dir = resolved_model_dir / run_stem

    summary = {
        "run": run_stem,
        "config": str(config.path),
        "log_dir": str(run_log_dir),
        "result_dir": str(run_result_dir),
        "folds": {},
    }
    total_folds = len(config.folds)
    for fold_index, fold in enumerate(config.folds, start=1):
        print(f"Starting fold {fold_index}/{total_folds}: {fold.id}")
        paths = fold_artifact_paths(
            sequence_dir=sequence_dir,
            run_log_dir=run_log_dir,
            run_result_dir=run_result_dir,
            fold_id=fold.id,
        )
        fold_summary = train_fold(
            config=config,
            fold_id=fold.id,
            fold_sequence_dir=paths["sequence_dir"],
            fold_log_dir=paths["log_dir"],
            fold_result_dir=paths["result_dir"],
            run_stem=run_stem,
        )
        summary["folds"][fold.id] = fold_summary

    summary_path = run_log_dir / "summary.yaml"
    save_run_summary(summary, summary_path)
    print(f"Run summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
