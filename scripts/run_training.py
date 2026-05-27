from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter
from typing import Any
import argparse
import copy
import os

import numpy as np
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from configuration import ExperimentConfig, load_config
from datasets import EpochNeutralDownsamplingSampler, LOBDataset, sequence_window_labels
from model import build_model
from run_logging import (
    class_distribution,
    format_duration,
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
from training import EpochResult, LobTrainer, class_weights_from_class_counts, class_weights_from_sequence_labels
from utils import seed_torch_worker, set_global_seed, torch_generator_from_seed

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LOB model on one or more folds.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--fold-id", type=str, default=None)
    parser.add_argument("--fold-index", type=int, default=None)
    parser.add_argument(
        "--run-stem",
        type=str,
        default=None,
        help="Shared run directory name. Useful for PBS array jobs.",
    )
    return parser.parse_args()

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
    return sequence_window_labels(dataset)


def class_distribution_from_counts(counts: list[int]) -> dict[str, Any]:
    """Format per-class counts like dataset class distributions."""
    counts_array = np.asarray(counts, dtype=np.int64)
    total = int(counts_array.sum())
    return {
        "total": total,
        "classes": {
            str(class_id): {
                "count": int(count),
                "percentage": 0.0 if total == 0 else float(100.0 * count / total),
            }
            for class_id, count in enumerate(counts_array)
        },
    }


def build_train_sampler(
    config: ExperimentConfig,
    train_dataset: LOBDataset,
    *,
    seed: int,
) -> tuple[EpochNeutralDownsamplingSampler | None, dict[str, Any]]:
    """Create the optional train-only neutral downsampling sampler."""
    ratio = config.training.sampling.neutral_to_directional_ratio
    if ratio is None:
        return None, {"enabled": False, "neutral_to_directional_ratio": None}

    sampler = EpochNeutralDownsamplingSampler(
        train_dataset,
        label_mapping=config.data.label_mapping,
        neutral_to_directional_ratio=ratio,
        base_seed=seed,
    )
    return sampler, sampler.summary(config.model.num_classes)


def resolve_class_weights(
    config: ExperimentConfig,
    train_dataset: LOBDataset,
    *,
    sampled_class_counts: list[int] | None = None,
) -> dict[str, list[float] | list[int] | bool | str]:
    """Fit runtime loss class weights from train sequence labels for one fold."""
    if sampled_class_counts is None:
        labels = sequence_label_values(train_dataset)
        weights, counts = class_weights_from_sequence_labels(
            labels,
            num_classes=config.model.num_classes,
            beta=config.training.class_weight_beta,
            min_weight=config.training.class_weight_min,
            max_weight=config.training.class_weight_max,
        )
        source = "full_train"
    else:
        weights, counts = class_weights_from_class_counts(
            sampled_class_counts,
            beta=config.training.class_weight_beta,
            min_weight=config.training.class_weight_min,
            max_weight=config.training.class_weight_max,
        )
        source = "sampled_train_per_epoch"
    config.training.class_weights = weights
    return {
        "weights": weights,
        "counts": counts,
        "beta": config.training.class_weight_beta,
        "min": config.training.class_weight_min,
        "max": config.training.class_weight_max,
        "source": source,
    }


def epoch_monitor_value(result: EpochResult, monitor: str) -> float:
    """Return a saved epoch's validation monitor value."""
    if monitor == "val_loss":
        return result.val_loss
    if result.val_metrics is None:
        raise ValueError(f"Cannot compute {monitor} because validation metrics are unavailable.")
    if monitor == "val_macro_f1":
        return result.val_metrics.macro_f1
    if monitor == "val_directional_macro_f1":
        return result.val_metrics.directional_macro_f1
    raise ValueError(f"Unsupported monitor: {monitor}")


def best_epoch_from_history(config: ExperimentConfig, history: list[EpochResult]) -> tuple[int, EpochResult, float]:
    """Select the best epoch using the configured validation monitor."""
    if not history:
        raise ValueError("Cannot select a best epoch because training produced no epoch history.")
    reverse = config.training.monitor_mode == "max"
    best_epoch, best_history = sorted(
        enumerate(history, start=1),
        key=lambda item: epoch_monitor_value(item[1], config.training.monitor),
        reverse=reverse,
    )[0]
    return best_epoch, best_history, epoch_monitor_value(best_history, config.training.monitor)


def evaluate_best_model_on_test_split(
    *,
    config: ExperimentConfig,
    trainer: LobTrainer,
    model: Any,
    test_loader: DataLoader,
    history: list[EpochResult],
) -> float:
    """Evaluate the validation-selected model once on the held-out test split."""
    best_epoch, best_history, best_monitor_value = best_epoch_from_history(config, history)
    print(
        "Training uses train/validation splits only. "
        f"Evaluating best validation epoch {best_epoch} "
        f"({config.training.monitor}={best_monitor_value:.6f}) on the held-out test split."
    )
    evaluation_start = perf_counter()
    test_result = trainer.evaluate(
        model=model,
        data_loader=test_loader,
        description=f"Best epoch {best_epoch} [Test]",
    )
    evaluation_duration_seconds = perf_counter() - evaluation_start
    best_history.test_loss = test_result.loss
    best_history.test_metrics = test_result.metrics
    print(
        f"Best epoch {best_epoch} test evaluation: "
        f"test_loss={test_result.loss:.6f}, "
        f"test_acc={test_result.metrics.accuracy:.4f}, "
        f"test_macro_f1={test_result.metrics.macro_f1:.4f}, "
        f"test_directional_macro_f1={test_result.metrics.directional_macro_f1:.4f}, "
        f"test_ece={test_result.metrics.expected_calibration_error:.4f} "
        f"({format_duration(evaluation_duration_seconds)})."
    )
    return evaluation_duration_seconds


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
    seed: int | None = None,
) -> dict:
    fold_start = perf_counter()
    fold_seed = config.seed if seed is None else int(seed)
    set_global_seed(fold_seed, deterministic_torch=config.training.deterministic_torch)
    print(f"Starting training fold '{fold_id}'.")
    print(f"Fold '{fold_id}' global seed: {fold_seed}.")
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
    train_sampler, sampling_summary = build_train_sampler(config, train_dataset, seed=fold_seed)
    if sampling_summary.get("enabled"):
        sampled_counts = sampling_summary["sampled_counts_per_epoch"]
        full_counts = sampling_summary["full_counts"]
        print(
            f"Fold '{fold_id}' train sampling enabled: "
            f"neutral_to_directional_ratio={sampling_summary['neutral_to_directional_ratio']}, "
            f"full_counts={full_counts}, sampled_counts_per_epoch={sampled_counts}."
        )
    else:
        print(f"Fold '{fold_id}' train sampling disabled.")

    sampled_class_counts = (
        None
        if train_sampler is None
        else train_sampler.sampled_class_counts(config.model.num_classes)
    )
    class_weight_summary = resolve_class_weights(
        config,
        train_dataset,
        sampled_class_counts=sampled_class_counts,
    )
    print(
        f"Fold '{fold_id}' class weights selected for loss: "
        f"source={class_weight_summary['source']}, counts={class_weight_summary['counts']}, "
        f"weights={class_weight_summary['weights']}."
    )

    loader_kwargs = config.training.data_loader_kwargs()
    loader_kwargs["worker_init_fn"] = seed_torch_worker
    if train_sampler is None:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            generator=torch_generator_from_seed(fold_seed),
            **loader_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            sampler=train_sampler,
            shuffle=False, # already shuffled during sampling
            generator=torch_generator_from_seed(fold_seed),
            **loader_kwargs,
        )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.training.eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.eval_batch_size,
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
    if sampled_class_counts is not None:
        class_distributions["train_sampled_per_epoch"] = class_distribution_from_counts(sampled_class_counts)
    dataset_sizes = {
        "train": len(train_dataset),
        "validation": len(validation_dataset),
        "test": len(test_dataset),
    }
    if train_sampler is not None:
        dataset_sizes["train_sampled_per_epoch"] = len(train_sampler)
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
        sampling_summary=sampling_summary,
    )
    trainer = LobTrainer(config.training)
    fit_start = perf_counter()
    model, history = trainer.fit(model, train_loader, validation_loader)
    fit_duration_seconds = perf_counter() - fit_start
    test_evaluation_duration_seconds = evaluate_best_model_on_test_split(
        trainer=trainer,
        config=config,
        model=model,
        test_loader=test_loader,
        history=history,
    )
    fold_duration_seconds = perf_counter() - fold_start
    timing = {
        "fold_training_seconds": round(fold_duration_seconds, 6),
        "fold_training_duration": format_duration(fold_duration_seconds),
        "model_fit_seconds": round(fit_duration_seconds, 6),
        "model_fit_duration": format_duration(fit_duration_seconds),
        "test_evaluation_seconds": round(test_evaluation_duration_seconds, 6),
        "test_evaluation_duration": format_duration(test_evaluation_duration_seconds),
    }

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
        sampling_summary=sampling_summary,
        timing=timing,
        fold=fold_id,
    )
    print(
        f"Fold {fold_id} training complete "
        f"(fit {timing['model_fit_duration']}, test {timing['test_evaluation_duration']}, "
        f"total {timing['fold_training_duration']}). "
        f"Best model saved to: {config.training.best_model_path}"
    )
    print(f"Fold {fold_id} run log saved to: {run_log_path}")
    for epoch_index, result in enumerate(history, start=1):
        test_suffix = "" if result.test_loss is None else f", test_loss={result.test_loss:.6f}"
        val_metrics = result.val_metrics
        metric_suffix = "" if val_metrics is None else (
            f", val_acc={val_metrics.accuracy:.4f}, val_macro_f1={val_metrics.macro_f1:.4f}, "
            f"val_directional_macro_f1={val_metrics.directional_macro_f1:.4f}, "
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
        "sampling": sampling_summary,
        "dataset_sizes": dataset_sizes,
        "timing": timing,
        "logs": {
            "run_log": str(run_log_path),
            "config": str(run_config_path),
            "metrics": str(run_losses_path),
            "confusion_matrices": str(run_confusion_matrices_path),
        },
    }


def main() -> None:
    args = parse_args()
    if args.fold_id is not None and args.fold_index is not None:
        raise ValueError("Use either --fold-id or --fold-index, not both.")

    training_start = perf_counter()
    config = load_config(args.config)
    set_global_seed(config.seed, deterministic_torch=config.training.deterministic_torch)
    print(f"Global seed set to {config.seed}.")

    selected_folds = config.folds
    if args.fold_id is not None:
        selected_folds = [fold for fold in config.folds if fold.id == args.fold_id]
        if not selected_folds:
            raise ValueError(f"Unknown fold id: {args.fold_id}")
    elif args.fold_index is not None:
        if args.fold_index < 1 or args.fold_index > len(config.folds):
            raise ValueError(f"--fold-index must be in [1, {len(config.folds)}].")
        selected_folds = [config.folds[args.fold_index - 1]]

    sequence_dir = resolve_config_path(config, config.data.sequence_data_dir)
    logs_dir = resolve_config_path(config, config.data.logs_dir)

    run_stem = (
        args.run_stem
        or os.environ.get("TRAINING_RUN_STEM")
        or next_run_stem(logs_dir)
    )

    run_log_dir = logs_dir / run_stem
    resolved_model_dir = resolve_config_path(config, config.training.model_dir)
    run_result_dir = resolved_model_dir / run_stem

    summary = {
        "run": run_stem,
        "seed": config.seed,
        "config": str(config.path),
        "log_dir": str(run_log_dir),
        "result_dir": str(run_result_dir),
        "folds": {},
    }
    total_folds = len(selected_folds)
    for fold_index, fold in enumerate(selected_folds, start=1):
        print(f"Starting selected fold {fold_index}/{total_folds}: {fold.id}")

        fold_config = copy.deepcopy(config)

        paths = fold_artifact_paths(
            sequence_dir=sequence_dir,
            run_log_dir=run_log_dir,
            run_result_dir=run_result_dir,
            fold_id=fold.id,
        )

        fold_summary = train_fold(
            config=fold_config,
            fold_id=fold.id,
            fold_sequence_dir=paths["sequence_dir"],
            fold_log_dir=paths["log_dir"],
            fold_result_dir=paths["result_dir"],
            run_stem=run_stem,
            seed=fold_config.seed,
        )
        summary["folds"][fold.id] = fold_summary

    training_duration_seconds = perf_counter() - training_start
    summary["timing"] = {
        "training_pipeline_seconds": round(training_duration_seconds, 6),
        "training_pipeline_duration": format_duration(training_duration_seconds),
    }

    if len(selected_folds) == 1:
        summary_path = run_log_dir / f"summary_{selected_folds[0].id}.yaml"
    else:
        summary_path = run_log_dir / "summary.yaml"

    save_run_summary(summary, summary_path)
    print(f"Run summary saved to: {summary_path}")
    print(f"Training pipeline finished ({format_duration(training_duration_seconds)}).")


if __name__ == "__main__":
    main()
