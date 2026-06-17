from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter
from typing import Any
import argparse
import copy
import os
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from configuration import ExperimentConfig, load_config
from calibration import (
    apply_logit_calibration_to_outputs,
    fit_logit_calibration,
    save_temperature_scaling_artifact,
)
from datasets import EpochNeutralDownsamplingSampler, LOBDataset, sequence_window_labels
from model import build_model
from monitoring import epoch_monitor_value as configured_epoch_monitor_value
from run_logging import (
    class_distribution,
    format_duration,
    load_preprocessing_metadata,
    model_parameter_summary,
    resolve_config_path,
    save_confusion_matrices,
    save_directional_threshold_artifact,
    save_epoch_history,
    save_expert_usage,
    save_best_pr_artifacts,
    save_probability_outputs,
    save_run_config_snapshot,
    save_run_log,
    save_run_summary,
    timestamped_run_stem,
)
from thresholding import (
    apply_directional_threshold_policy,
    apply_thresholds_and_summarize,
    optimize_directional_thresholds,
    optimize_precision_floor_thresholds,
    optimize_top_quantile_thresholds,
    threshold_candidates,
)
from training import (
    CheckpointCandidate,
    ClassificationMetrics,
    EpochResult,
    LobTrainer,
    MovementDirectionAuxiliaryLoss,
    class_weights_from_class_counts,
    class_weights_from_sequence_labels,
    classification_metrics_from_predictions,
)
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


def build_auxiliary_criterion(
    config: ExperimentConfig,
    *,
    class_counts: list[int],
) -> tuple[MovementDirectionAuxiliaryLoss | None, dict[str, Any]]:
    """Create the optional movement/direction auxiliary loss for training."""
    aux_heads = config.model.auxiliary_heads
    aux_losses = config.training.auxiliary_losses
    required_labels = {-1: "down", 0: "neutral", 1: "up"}
    missing = [name for raw_label, name in required_labels.items() if raw_label not in config.data.label_mapping]
    if missing:
        raise ValueError(f"Auxiliary heads require label_mapping entries for: {', '.join(missing)}.")

    down_id = int(config.data.label_mapping[-1])
    neutral_id = int(config.data.label_mapping[0])
    up_id = int(config.data.label_mapping[1])
    counts = np.asarray(class_counts, dtype=np.int64)
    if counts.size < config.model.num_classes:
        counts = np.pad(counts, (0, config.model.num_classes - counts.size))

    neutral_count = int(counts[neutral_id]) if 0 <= neutral_id < counts.size else 0
    down_count = int(counts[down_id]) if 0 <= down_id < counts.size else 0
    up_count = int(counts[up_id]) if 0 <= up_id < counts.size else 0
    directional_count = int(down_count + up_count)

    movement_pos_weight: float | None = None
    if aux_losses.movement_pos_weight == "auto_clipped":
        raw_ratio = float("inf") if directional_count == 0 else float(neutral_count / max(directional_count, 1))
        movement_pos_weight = float(
            np.clip(raw_ratio, aux_losses.movement_pos_weight_min, aux_losses.movement_pos_weight_max)
        )
    elif aux_losses.movement_pos_weight is not None:
        movement_pos_weight = float(aux_losses.movement_pos_weight)

    direction_weights: list[float] | None = None
    direction_weights_tensor: torch.Tensor | None = None
    if aux_losses.direction_weight > 0.0 and directional_count > 0:
        direction_weights, _ = class_weights_from_class_counts(
            [up_count, down_count],
            beta=aux_losses.direction_class_weight_beta,
            min_weight=config.training.class_weight_min,
            max_weight=config.training.class_weight_max,
        )
        direction_weights_tensor = torch.tensor(direction_weights, dtype=torch.float32)

    enabled = bool(aux_heads.enabled and aux_losses.enabled)
    criterion = None
    if enabled:
        criterion = MovementDirectionAuxiliaryLoss(
            up_id=up_id,
            neutral_id=neutral_id,
            down_id=down_id,
            movement_weight=aux_losses.movement_weight if aux_heads.movement else 0.0,
            direction_weight=aux_losses.direction_weight if aux_heads.direction else 0.0,
            consistency_weight=aux_losses.consistency_weight if aux_heads.movement else 0.0,
            movement_pos_weight=movement_pos_weight,
            direction_class_weights=direction_weights_tensor,
        )

    return criterion, {
        "enabled": enabled,
        "heads_enabled": bool(aux_heads.enabled),
        "movement_head": bool(aux_heads.enabled and aux_heads.movement),
        "direction_head": bool(aux_heads.enabled and aux_heads.direction),
        "movement_weight": float(aux_losses.movement_weight),
        "direction_weight": float(aux_losses.direction_weight),
        "consistency_weight": float(aux_losses.consistency_weight),
        "movement_pos_weight": movement_pos_weight,
        "movement_pos_weight_mode": aux_losses.movement_pos_weight,
        "movement_pos_weight_min": float(aux_losses.movement_pos_weight_min),
        "movement_pos_weight_max": float(aux_losses.movement_pos_weight_max),
        "direction_class_weight_beta": float(aux_losses.direction_class_weight_beta),
        "direction_class_weights": direction_weights,
        "label_ids": {"down": down_id, "neutral": neutral_id, "up": up_id},
        "counts": {
            "down": down_count,
            "neutral": neutral_count,
            "up": up_count,
            "directional": directional_count,
        },
    }


def epoch_monitor_value(config: ExperimentConfig, result: EpochResult) -> float:
    """Return a saved epoch's validation monitor value."""
    return configured_epoch_monitor_value(
        result,
        monitor=config.training.monitor,
        monitor_params=config.training.monitor_params,
        label_mapping=config.data.label_mapping,
    )


def validation_index_for_history_item(result: EpochResult, fallback_index: int) -> int:
    """Return the stable validation index for a history item."""
    value = getattr(result, "validation_index", None)
    return int(fallback_index if value is None else value)


def checkpoint_label_for_history_item(result: EpochResult, fallback_index: int) -> str:
    """Return a display/filesystem label for a history item."""
    label = getattr(result, "checkpoint_label", None)
    if label:
        return str(label)
    epoch = int(getattr(result, "epoch", fallback_index) or fallback_index)
    return f"epoch_{epoch:04d}"


def history_item_by_validation_index(history: list[EpochResult], validation_index: int) -> EpochResult:
    """Find a training history entry by its validation index."""
    for fallback_index, result in enumerate(history, start=1):
        if validation_index_for_history_item(result, fallback_index) == int(validation_index):
            return result
    raise ValueError(f"Selected validation index {validation_index} is outside the training history.")


def candidate_checkpoint_label(candidate: CheckpointCandidate) -> str:
    """Return a stable label for a checkpoint candidate."""
    if candidate.checkpoint_label:
        return str(candidate.checkpoint_label)
    if candidate.global_step is not None:
        return f"epoch_{int(candidate.epoch):04d}_step_{int(candidate.global_step):08d}"
    return f"epoch_{int(candidate.epoch):04d}"


def mapped_class_metric(config: ExperimentConfig, metrics: Any, raw_label: int, metric_name: str) -> float | None:
    """Return a per-class metric using the configured raw-to-class label mapping."""
    mapped_label = config.data.label_mapping.get(raw_label)
    if mapped_label is None:
        return None
    values = getattr(metrics, metric_name, None)
    class_id = int(mapped_label)
    if values is None or class_id < 0 or class_id >= len(values):
        return None
    return float(values[class_id])


def format_optional_metric(value: float | None) -> str:
    """Format optional metrics for console logs."""
    return "nan" if value is None else f"{value:.4f}"


def format_optional_threshold(value: float | None) -> str:
    """Format optional thresholds for console logs."""
    return "disabled" if value is None else f"{float(value):.4f}"


def pr_ap_summary(config: ExperimentConfig, metrics: Any, prefix: str) -> str:
    """Format one-vs-rest PR-AP values for console logs."""
    values = {
        "down": mapped_class_metric(config, metrics, -1, "per_class_pr_ap"),
        "neutral": mapped_class_metric(config, metrics, 0, "per_class_pr_ap"),
        "up": mapped_class_metric(config, metrics, 1, "per_class_pr_ap"),
    }
    return ", ".join(f"{prefix}_pr_ap_{label}={format_optional_metric(value)}" for label, value in values.items())


def pr_auc_summary(config: ExperimentConfig, metrics: Any, prefix: str) -> str:
    """Format one-vs-rest PR-AUC values for console logs."""
    values = {
        "down": mapped_class_metric(config, metrics, -1, "per_class_pr_auc"),
        "neutral": mapped_class_metric(config, metrics, 0, "per_class_pr_auc"),
        "up": mapped_class_metric(config, metrics, 1, "per_class_pr_auc"),
    }
    return ", ".join(f"{prefix}_pr_auc_{label}={format_optional_metric(value)}" for label, value in values.items())


def roc_auc_summary(config: ExperimentConfig, metrics: Any, prefix: str) -> str:
    """Format one-vs-rest ROC-AUC values for console logs."""
    values = {
        "down": mapped_class_metric(config, metrics, -1, "per_class_roc_auc"),
        "neutral": mapped_class_metric(config, metrics, 0, "per_class_roc_auc"),
        "up": mapped_class_metric(config, metrics, 1, "per_class_roc_auc"),
    }
    return ", ".join(f"{prefix}_roc_auc_{label}={format_optional_metric(value)}" for label, value in values.items())


def directional_threshold_class_ids(config: ExperimentConfig) -> tuple[int, int, int]:
    """Resolve down/neutral/up class ids for thresholded decisions."""
    required = {-1: "down", 0: "neutral", 1: "up"}
    missing = [name for raw_label, name in required.items() if raw_label not in config.data.label_mapping]
    if missing:
        raise ValueError(f"Directional thresholds require label_mapping entries for: {', '.join(missing)}.")
    return (
        int(config.data.label_mapping[-1]),
        int(config.data.label_mapping[0]),
        int(config.data.label_mapping[1]),
    )


def prediction_outputs_with_decisions(outputs: dict[str, Any], predictions: np.ndarray) -> dict[str, Any]:
    """Return collected outputs with replacement decisions and saved argmax decisions."""
    updated = dict(outputs)
    if "predictions" in outputs and "argmax_predictions" not in updated:
        updated["argmax_predictions"] = np.asarray(outputs["predictions"], dtype=np.int64).reshape(-1)
    updated["predictions"] = np.asarray(predictions, dtype=np.int64).reshape(-1)
    return updated


def metrics_from_prediction_outputs(
    config: ExperimentConfig,
    outputs: dict[str, Any],
    predictions: np.ndarray,
) -> ClassificationMetrics:
    """Build full classification metrics for fixed predictions."""
    return classification_metrics_from_predictions(
        np.asarray(outputs["targets"], dtype=np.int64),
        np.asarray(predictions, dtype=np.int64),
        num_classes=config.model.num_classes,
        probabilities=np.asarray(outputs["probabilities"], dtype=np.float32),
    )


def fit_and_apply_temperature_scaling(
    *,
    config: ExperimentConfig,
    validation_outputs: dict[str, Any],
    test_outputs: dict[str, Any] | None,
    best_epoch: int,
    checkpoint_label: str | None = None,
    batch_in_epoch: int | None = None,
    global_step: int | None = None,
    validation_index: int | None = None,
    fold: str,
    target_path: Path,
) -> tuple[
    dict[str, Any],
    ClassificationMetrics,
    ClassificationMetrics | None,
    dict[str, Any],
    dict[str, Any] | None,
]:
    """Fit validation logit calibration and return calibrated artifacts."""
    calibration_start = perf_counter()
    class_bias_calibration = bool(config.training.temperature_scaling.class_bias_calibration)
    result = fit_logit_calibration(
        np.asarray(validation_outputs["logits"], dtype=np.float32),
        np.asarray(validation_outputs["targets"], dtype=np.int64),
        device=config.training.device,
        class_bias_calibration=class_bias_calibration,
    )
    fit_seconds = perf_counter() - calibration_start

    validation_outputs_calibrated = apply_logit_calibration_to_outputs(
        validation_outputs,
        result.temperature,
        class_biases=result.class_biases if class_bias_calibration else None,
    )
    validation_metrics = metrics_from_prediction_outputs(
        config,
        validation_outputs_calibrated,
        np.asarray(validation_outputs_calibrated["predictions"], dtype=np.int64),
    )
    test_outputs_calibrated = None
    test_metrics = None
    if test_outputs is not None:
        test_outputs_calibrated = apply_logit_calibration_to_outputs(
            test_outputs,
            result.temperature,
            class_biases=result.class_biases if class_bias_calibration else None,
        )
        test_metrics = metrics_from_prediction_outputs(
            config,
            test_outputs_calibrated,
            np.asarray(test_outputs_calibrated["predictions"], dtype=np.int64),
        )

    payload = {
        "enabled": True,
        "fold": fold,
        "best_epoch": int(best_epoch),
        "selection_split": "validation",
        "note": (
            "Temperature and optional class biases are fitted with unweighted cross-entropy "
            "on the natural validation distribution."
        ),
        **result.to_dict(),
        "fit_seconds": round(fit_seconds, 6),
        "fit_duration": format_duration(fit_seconds),
    }
    if checkpoint_label is not None:
        payload["checkpoint_label"] = checkpoint_label
    if batch_in_epoch is not None:
        payload["batch_in_epoch"] = int(batch_in_epoch)
    if global_step is not None:
        payload["global_step"] = int(global_step)
    if validation_index is not None:
        payload["validation_index"] = int(validation_index)
    save_temperature_scaling_artifact(payload, target_path)
    return payload, validation_metrics, test_metrics, validation_outputs_calibrated, test_outputs_calibrated


def fit_and_apply_directional_thresholds(
    *,
    config: ExperimentConfig,
    validation_outputs: dict[str, Any],
    test_outputs: dict[str, Any] | None,
    best_epoch: int,
    checkpoint_label: str | None = None,
    batch_in_epoch: int | None = None,
    global_step: int | None = None,
    validation_index: int | None = None,
    fold: str,
    target_path: Path,
) -> tuple[
    dict[str, Any],
    ClassificationMetrics,
    ClassificationMetrics | None,
    dict[str, Any],
    dict[str, Any] | None,
]:
    """Fit validation thresholds and return thresholded val/test artifacts."""
    threshold_config = config.training.directional_thresholds
    down_id, neutral_id, up_id = directional_threshold_class_ids(config)
    down_candidates = threshold_candidates(
        threshold_config.min_threshold,
        threshold_config.max_threshold,
        threshold_config.step,
    )
    up_candidates = threshold_candidates(
        threshold_config.min_threshold,
        threshold_config.max_threshold,
        threshold_config.step,
    )
    refinement_steps = tuple(
        step
        for step in (0.01, 0.005, 0.002, 0.001)
        if step < float(threshold_config.step) - 1e-12
    )
    probabilities_validation = np.asarray(validation_outputs["probabilities"], dtype=np.float32)
    targets_validation = np.asarray(validation_outputs["targets"], dtype=np.int64)
    monitor_params = config.training.monitor_params
    applied_refinement_steps: tuple[float, ...] = ()
    if threshold_config.method == "joint_up_down":
        applied_refinement_steps = refinement_steps
        selection = optimize_directional_thresholds(
            probabilities_validation,
            targets_validation,
            down_candidates=down_candidates,
            up_candidates=up_candidates,
            down_id=down_id,
            neutral_id=neutral_id,
            up_id=up_id,
            refinement_steps=applied_refinement_steps,
            delta=threshold_config.delta,
            score=threshold_config.score,
            monitor_params=monitor_params,
        )
        selection_metric = threshold_config.score
        tie_break_order = [
            f"maximize_{threshold_config.score}",
            "minimize_directional_rate_penalty",
            "maximize_min_down_up_precision",
            "maximize_thresholds",
        ]
        selection_final_tie_break = "highest_threshold_sum_then_down_then_up"
    elif threshold_config.method == "precision_floor":
        selection = optimize_precision_floor_thresholds(
            probabilities_validation,
            targets_validation,
            down_candidates=down_candidates,
            up_candidates=up_candidates,
            down_precision_floor=float(threshold_config.down_precision_floor),
            up_precision_floor=float(threshold_config.up_precision_floor),
            down_id=down_id,
            neutral_id=neutral_id,
            up_id=up_id,
            delta=threshold_config.delta,
            score=threshold_config.score,
            monitor_params=monitor_params,
        )
        selection_metric = "per_class_precision_floor_then_recall"
        tie_break_order = [
            "per_class_precision_at_or_above_floor",
            "maximize_per_class_recall",
            "maximize_per_class_precision",
            "maximize_per_class_threshold",
        ]
        selection_final_tie_break = "independent_class_threshold_selection"
    elif threshold_config.method == "top_x_quantile":
        selection = optimize_top_quantile_thresholds(
            probabilities_validation,
            targets_validation,
            down_quantile=float(threshold_config.down_quantile),
            up_quantile=float(threshold_config.up_quantile),
            down_id=down_id,
            neutral_id=neutral_id,
            up_id=up_id,
            delta=threshold_config.delta,
            score=threshold_config.score,
            monitor_params=monitor_params,
        )
        selection_metric = "top_probability_quantile"
        tie_break_order = [
            "threshold_down_from_top_down_quantile",
            "threshold_up_from_top_up_quantile",
            "decision_tie_break_by_logit_margin_delta",
        ]
        selection_final_tie_break = "not_applicable_fixed_quantiles"
    else:
        raise ValueError(f"Unsupported directional threshold method: {threshold_config.method}")
    validation_predictions = apply_directional_threshold_policy(
        probabilities_validation,
        threshold_down=selection.threshold_down,
        threshold_up=selection.threshold_up,
        down_id=down_id,
        neutral_id=neutral_id,
        up_id=up_id,
        delta=threshold_config.delta,
        down_enabled=selection.down_enabled,
        up_enabled=selection.up_enabled,
    )
    validation_metrics = metrics_from_prediction_outputs(config, validation_outputs, validation_predictions)
    validation_summary = apply_thresholds_and_summarize(
        validation_outputs,
        threshold_down=selection.threshold_down,
        threshold_up=selection.threshold_up,
        down_id=down_id,
        neutral_id=neutral_id,
        up_id=up_id,
        delta=threshold_config.delta,
        down_enabled=selection.down_enabled,
        up_enabled=selection.up_enabled,
    )
    validation_outputs_thresholded = prediction_outputs_with_decisions(validation_outputs, validation_predictions)
    test_metrics = None
    test_summary = None
    test_outputs_thresholded = None
    if test_outputs is not None:
        test_predictions = apply_directional_threshold_policy(
            np.asarray(test_outputs["probabilities"], dtype=np.float32),
            threshold_down=selection.threshold_down,
            threshold_up=selection.threshold_up,
            down_id=down_id,
            neutral_id=neutral_id,
            up_id=up_id,
            delta=threshold_config.delta,
            down_enabled=selection.down_enabled,
            up_enabled=selection.up_enabled,
        )
        test_metrics = metrics_from_prediction_outputs(config, test_outputs, test_predictions)
        test_summary = apply_thresholds_and_summarize(
            test_outputs,
            threshold_down=selection.threshold_down,
            threshold_up=selection.threshold_up,
            down_id=down_id,
            neutral_id=neutral_id,
            up_id=up_id,
            delta=threshold_config.delta,
            down_enabled=selection.down_enabled,
            up_enabled=selection.up_enabled,
        )
        test_outputs_thresholded = prediction_outputs_with_decisions(test_outputs, test_predictions)

    payload: dict[str, Any] = {
        "enabled": True,
        "classification_mode": "directional_thresholds",
        "method": threshold_config.method,
        "configured_score": threshold_config.score,
        "decision_scope": (
            "Validation/test decision metrics, saved pred_label values, and confusion matrices use thresholded "
            "decisions. PR/ROC/AP remain probability-ranking metrics computed from softmax scores."
        ),
        "fold": fold,
        "best_epoch": int(best_epoch),
        "selection_split": "validation",
        "selection_metric": selection_metric,
        "tie_break_order": tie_break_order,
        "threshold_down": None if selection.threshold_down is None else float(selection.threshold_down),
        "threshold_up": None if selection.threshold_up is None else float(selection.threshold_up),
        "down_enabled": bool(selection.down_enabled),
        "up_enabled": bool(selection.up_enabled),
        "score": float(selection.score),
        "rate_penalty": float(selection.rate_penalty),
        "grid_min": float(threshold_config.min_threshold),
        "grid_max": float(threshold_config.max_threshold),
        "grid_step": float(threshold_config.step),
        "refinement_steps": [float(step) for step in applied_refinement_steps],
        "delta": float(threshold_config.delta),
        "down_precision_floor": threshold_config.down_precision_floor,
        "up_precision_floor": threshold_config.up_precision_floor,
        "down_quantile": threshold_config.down_quantile,
        "up_quantile": threshold_config.up_quantile,
        "top_quantile_convention": "0.01 means the top 1% validation probabilities for that class.",
        "n_candidates": int(selection.n_candidates),
        "optimization_stages": list(selection.stage_summaries),
        "selection_details": selection.selection_details,
        "decision_tie_break": "logit_margin_over_threshold_with_delta_else_neutral",
        "selection_final_tie_break": selection_final_tie_break,
        "class_ids": {
            "down": int(down_id),
            "neutral": int(neutral_id),
            "up": int(up_id),
        },
        "validation_threshold_metrics": validation_summary,
        "test_threshold_metrics": test_summary,
    }
    if checkpoint_label is not None:
        payload["checkpoint_label"] = checkpoint_label
    if batch_in_epoch is not None:
        payload["batch_in_epoch"] = int(batch_in_epoch)
    if global_step is not None:
        payload["global_step"] = int(global_step)
    if validation_index is not None:
        payload["validation_index"] = int(validation_index)
    if threshold_config.method != "top_x_quantile":
        payload["min_directional_precision"] = float(selection.min_directional_precision)
    if selection.score_details:
        payload.update(selection.score_details)
    save_directional_threshold_artifact(payload, target_path)
    return payload, validation_metrics, test_metrics, validation_outputs_thresholded, test_outputs_thresholded


def best_epoch_from_history(config: ExperimentConfig, history: list[EpochResult]) -> tuple[int, EpochResult, float]:
    """Select the best epoch using the configured validation monitor."""
    if not history:
        raise ValueError("Cannot select a best epoch because training produced no epoch history.")
    reverse = config.training.monitor_mode == "max"
    best_epoch, best_history = sorted(
        enumerate(history, start=1),
        key=lambda item: epoch_monitor_value(config, item[1]),
        reverse=reverse,
    )[0]
    return best_epoch, best_history, epoch_monitor_value(config, best_history)


def evaluate_best_model_on_validation_and_test_splits(
    *,
    config: ExperimentConfig,
    trainer: LobTrainer,
    model: Any,
    validation_loader: DataLoader,
    test_loader: DataLoader | None,
    history: list[EpochResult],
) -> dict[str, Any]:
    """Evaluate and collect artifacts from the validation-selected model."""
    best_epoch, best_history, best_monitor_value = best_epoch_from_history(config, history)
    test_message = "and on the held-out test split." if test_loader is not None else "with no fold test split."
    print(
        "Training uses train/validation splits only. "
        f"Evaluating best validation epoch {best_epoch} "
        f"({config.training.monitor}={best_monitor_value:.6f}) for validation artifacts "
        f"{test_message}"
    )
    validation_start = perf_counter()
    validation_result = trainer.evaluate(
        model=model,
        data_loader=validation_loader,
        description=f"Best epoch {best_epoch} [Validation artifacts]",
        collect_outputs=True,
        track_pr_metrics=True,
        track_expert_usage=True,
    )
    validation_duration_seconds = perf_counter() - validation_start
    best_history.val_loss = validation_result.loss
    best_history.val_metrics = validation_result.metrics
    best_history.val_expert_usage = validation_result.expert_usage
    print(
        f"Best epoch {best_epoch} validation evaluation: "
        f"val_loss={validation_result.loss:.6f}, "
        f"val_acc={validation_result.metrics.accuracy:.4f}, "
        f"val_macro_f1={validation_result.metrics.macro_f1:.4f}, "
        f"val_directional_macro_f1={validation_result.metrics.directional_macro_f1:.4f}, "
        f"val_ece={validation_result.metrics.expected_calibration_error:.4f}, "
        f"{roc_auc_summary(config, validation_result.metrics, 'val')}, "
        f"{pr_auc_summary(config, validation_result.metrics, 'val')}, "
        f"{pr_ap_summary(config, validation_result.metrics, 'val')} "
        f"({format_duration(validation_duration_seconds)})."
    )

    test_duration_seconds: float | None = None
    test_outputs = None
    if test_loader is None:
        best_history.test_loss = None
        best_history.test_metrics = None
        best_history.test_expert_usage = None
        print(f"Best epoch {best_epoch} test evaluation skipped: fold has no configured test split.")
    else:
        test_start = perf_counter()
        test_result = trainer.evaluate(
            model=model,
            data_loader=test_loader,
            description=f"Best epoch {best_epoch} [Test]",
            collect_outputs=True,
            track_pr_metrics=True,
            track_expert_usage=True,
        )
        test_duration_seconds = perf_counter() - test_start
        best_history.test_loss = test_result.loss
        best_history.test_metrics = test_result.metrics
        best_history.test_expert_usage = test_result.expert_usage
        test_outputs = test_result.prediction_outputs
        test_ece_down = mapped_class_metric(
            config,
            test_result.metrics,
            -1,
            "per_class_expected_calibration_error",
        )
        test_ece_neutral = mapped_class_metric(
            config,
            test_result.metrics,
            0,
            "per_class_expected_calibration_error",
        )
        test_ece_up = mapped_class_metric(
            config,
            test_result.metrics,
            1,
            "per_class_expected_calibration_error",
        )
        print(
            f"Best epoch {best_epoch} test evaluation: "
            f"test_loss={test_result.loss:.6f}, "
            f"test_acc={test_result.metrics.accuracy:.4f}, "
            f"test_macro_f1={test_result.metrics.macro_f1:.4f}, "
            f"test_directional_macro_f1={test_result.metrics.directional_macro_f1:.4f}, "
            f"test_ece={test_result.metrics.expected_calibration_error:.4f}, "
            f"test_ece_down={format_optional_metric(test_ece_down)}, "
            f"test_ece_neutral={format_optional_metric(test_ece_neutral)}, "
            f"test_ece_up={format_optional_metric(test_ece_up)}, "
            f"{roc_auc_summary(config, test_result.metrics, 'test')}, "
            f"{pr_auc_summary(config, test_result.metrics, 'test')}, "
            f"{pr_ap_summary(config, test_result.metrics, 'test')} "
            f"({format_duration(test_duration_seconds)})."
        )
    total_seconds = validation_duration_seconds + (test_duration_seconds or 0.0)
    return {
        "best_epoch": best_epoch,
        "best_monitor_value": best_monitor_value,
        "validation_seconds": validation_duration_seconds,
        "test_seconds": test_duration_seconds,
        "total_seconds": total_seconds,
        "validation_outputs": validation_result.prediction_outputs,
        "test_outputs": test_outputs,
    }


def ranked_checkpoint_candidates(
    config: ExperimentConfig,
    trainer: LobTrainer,
    history: list[EpochResult],
) -> list[CheckpointCandidate]:
    """Return saved candidate checkpoints sorted by raw validation monitor."""
    candidates = list(getattr(trainer, "top_checkpoint_candidates", []))
    if not candidates:
        best_epoch, _best_history, best_monitor_value = best_epoch_from_history(config, history)
        best_validation_index = validation_index_for_history_item(_best_history, best_epoch)
        best_label = checkpoint_label_for_history_item(_best_history, best_epoch)
        candidates = [
            CheckpointCandidate(
                epoch=int(getattr(_best_history, "epoch", best_epoch) or best_epoch),
                monitor_value=best_monitor_value,
                path=config.training.best_model_path,
                batch_in_epoch=getattr(_best_history, "batch_in_epoch", None),
                global_step=getattr(_best_history, "global_step", None),
                validation_index=best_validation_index,
                checkpoint_label=best_label,
            )
        ]
    missing = [str(candidate.path) for candidate in candidates if not candidate.path.exists()]
    if missing:
        raise FileNotFoundError(f"Top-k checkpoint candidate file(s) are missing: {missing}")
    reverse_sign = -1.0 if config.training.monitor_mode == "max" else 1.0
    return sorted(
        candidates,
        key=lambda item: (
            reverse_sign * float(item.monitor_value),
            int(item.validation_index if item.validation_index is not None else item.epoch),
            int(item.global_step if item.global_step is not None else 0),
        ),
    )


def load_checkpoint_into_model(
    *,
    model: Any,
    trainer: LobTrainer,
    checkpoint_path: Path,
) -> None:
    """Load a candidate checkpoint into the shared model instance."""
    state_dict = torch.load(checkpoint_path, map_location=trainer.device, weights_only=True)
    model.load_state_dict(state_dict)


def monitor_value_after_postprocessing(
    *,
    config: ExperimentConfig,
    raw_monitor_value: float,
    val_loss: float,
    val_metrics: ClassificationMetrics,
) -> float:
    """Return the monitor used to select among postprocessed checkpoint candidates."""
    if config.training.monitor == "val_loss":
        return float(raw_monitor_value)
    return epoch_monitor_value(
        config,
        EpochResult(
            train_loss=0.0,
            val_loss=float(val_loss),
            val_metrics=val_metrics,
        ),
    )


def compact_temperature_summary(summary: dict[str, Any], artifact_path: Path | None) -> dict[str, Any]:
    """Return checkpoint-selection fields for a temperature scaling candidate."""
    payload = {
        "enabled": bool(summary.get("enabled", False)),
        "class_bias_calibration": bool(summary.get("class_bias_calibration", False)),
    }
    if artifact_path is not None and payload["enabled"]:
        payload["artifact"] = str(artifact_path)
    for key in (
        "method",
        "temperature",
        "class_bias_sum",
        "validation_nll_before",
        "validation_nll_after",
        "fit_seconds",
        "fit_duration",
    ):
        if key in summary:
            payload[key] = summary[key]
    return payload


def compact_directional_threshold_summary(summary: dict[str, Any], artifact_path: Path | None) -> dict[str, Any]:
    """Return checkpoint-selection fields for a directional threshold candidate."""
    payload = {"enabled": bool(summary.get("enabled", False))}
    if artifact_path is not None and payload["enabled"]:
        payload["artifact"] = str(artifact_path)
    for key in (
        "method",
        "configured_score",
        "score",
        "threshold_down",
        "threshold_up",
        "down_enabled",
        "up_enabled",
        "rate_penalty",
        "tailored_base_metric",
        "tailored_base_value",
        "tailored_ece_dir",
        "tailored_rate_penalty",
        "tailored_lambda_ece",
        "tailored_lambda_rate",
        "min_directional_precision",
        "n_candidates",
    ):
        if key in summary:
            payload[key] = summary[key]
    return payload


def evaluate_checkpoint_candidate_on_validation(
    *,
    config: ExperimentConfig,
    trainer: LobTrainer,
    model: Any,
    validation_loader: DataLoader,
    candidate: CheckpointCandidate,
    fold: str,
    candidate_dir: Path,
) -> dict[str, Any]:
    """Evaluate and postprocess one checkpoint candidate on validation only."""
    label = candidate_checkpoint_label(candidate)
    load_checkpoint_into_model(model=model, trainer=trainer, checkpoint_path=candidate.path)
    validation_start = perf_counter()
    validation_result = trainer.evaluate(
        model=model,
        data_loader=validation_loader,
        description=f"Candidate {label} [Validation selection]",
        collect_outputs=True,
        track_pr_metrics=True,
        track_expert_usage=True,
    )
    validation_seconds = perf_counter() - validation_start
    if validation_result.prediction_outputs is None:
        raise RuntimeError("Checkpoint candidate validation evaluation did not collect probability outputs.")

    validation_outputs = validation_result.prediction_outputs
    validation_metrics = validation_result.metrics
    temperature_scaling_summary: dict[str, Any] = {
        "enabled": bool(config.training.temperature_scaling.enabled),
        "class_bias_calibration": bool(config.training.temperature_scaling.class_bias_calibration),
    }
    temperature_path = candidate_dir / "temperature_scaling.yaml"
    directional_threshold_summary: dict[str, Any] = {
        "enabled": bool(config.training.directional_thresholds.enabled),
    }
    thresholds_path = candidate_dir / "directional_thresholds.yaml"

    postprocess_start = perf_counter()
    if config.training.temperature_scaling.enabled:
        (
            temperature_scaling_summary,
            validation_metrics,
            _test_calibrated_metrics,
            validation_outputs,
            _test_outputs,
        ) = fit_and_apply_temperature_scaling(
            config=config,
            validation_outputs=validation_outputs,
            test_outputs=None,
            best_epoch=int(candidate.epoch),
            checkpoint_label=label,
            batch_in_epoch=candidate.batch_in_epoch,
            global_step=candidate.global_step,
            validation_index=candidate.validation_index,
            fold=fold,
            target_path=temperature_path,
        )
    if config.training.directional_thresholds.enabled:
        (
            directional_threshold_summary,
            validation_metrics,
            _test_threshold_metrics,
            validation_outputs,
            _test_outputs,
        ) = fit_and_apply_directional_thresholds(
            config=config,
            validation_outputs=validation_outputs,
            test_outputs=None,
            best_epoch=int(candidate.epoch),
            checkpoint_label=label,
            batch_in_epoch=candidate.batch_in_epoch,
            global_step=candidate.global_step,
            validation_index=candidate.validation_index,
            fold=fold,
            target_path=thresholds_path,
        )
    postprocess_seconds = perf_counter() - postprocess_start
    postprocessed_monitor_value = monitor_value_after_postprocessing(
        config=config,
        raw_monitor_value=float(candidate.monitor_value),
        val_loss=validation_result.loss,
        val_metrics=validation_metrics,
    )
    return {
        "epoch": int(candidate.epoch),
        "batch_in_epoch": candidate.batch_in_epoch,
        "global_step": candidate.global_step,
        "validation_index": candidate.validation_index,
        "checkpoint_label": label,
        "checkpoint_path": candidate.path,
        "raw_monitor_value": float(candidate.monitor_value),
        "postprocessed_monitor_value": float(postprocessed_monitor_value),
        "validation_loss": float(validation_result.loss),
        "validation_metrics": validation_result.metrics,
        "validation_expert_usage": validation_result.expert_usage,
        "validation_outputs": validation_result.prediction_outputs,
        "postprocessed_validation_metrics": validation_metrics,
        "postprocessed_validation_outputs": validation_outputs,
        "validation_seconds": validation_seconds,
        "postprocessing_seconds": postprocess_seconds,
        "candidate_dir": candidate_dir,
        "temperature_scaling": temperature_scaling_summary,
        "temperature_scaling_path": temperature_path if temperature_scaling_summary.get("enabled") else None,
        "directional_thresholds": directional_threshold_summary,
        "directional_thresholds_path": thresholds_path if directional_threshold_summary.get("enabled") else None,
    }


def checkpoint_selection_sort_key(config: ExperimentConfig, candidate: dict[str, Any]) -> tuple[float, float, int]:
    """Sort postprocessed checkpoint candidates from best to worst."""
    reverse_sign = -1.0 if config.training.monitor_mode == "max" else 1.0
    return (
        reverse_sign * float(candidate["postprocessed_monitor_value"]),
        reverse_sign * float(candidate["raw_monitor_value"]),
        int(candidate.get("validation_index") or candidate["epoch"]),
    )


def checkpoint_candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    """Return a YAML-friendly checkpoint candidate summary."""
    temperature_path = candidate.get("temperature_scaling_path")
    thresholds_path = candidate.get("directional_thresholds_path")
    checkpoint_label = str(candidate.get("checkpoint_label") or f"epoch_{int(candidate['epoch']):04d}")
    return {
        "epoch": int(candidate["epoch"]),
        "batch_in_epoch": candidate.get("batch_in_epoch"),
        "global_step": candidate.get("global_step"),
        "validation_index": candidate.get("validation_index"),
        "checkpoint_label": checkpoint_label,
        "checkpoint_path": str(candidate["checkpoint_path"]),
        "candidate_dir": str(candidate["candidate_dir"]),
        "raw_monitor_value": float(candidate["raw_monitor_value"]),
        "postprocessed_monitor_value": float(candidate["postprocessed_monitor_value"]),
        "validation_loss": float(candidate["validation_loss"]),
        "validation_seconds": round(float(candidate["validation_seconds"]), 6),
        "postprocessing_seconds": round(float(candidate["postprocessing_seconds"]), 6),
        "temperature_scaling": compact_temperature_summary(
            candidate["temperature_scaling"],
            None if temperature_path is None else Path(temperature_path),
        ),
        "directional_thresholds": compact_directional_threshold_summary(
            candidate["directional_thresholds"],
            None if thresholds_path is None else Path(thresholds_path),
        ),
    }


def select_checkpoint_after_validation_postprocessing(
    *,
    config: ExperimentConfig,
    trainer: LobTrainer,
    model: Any,
    validation_loader: DataLoader,
    history: list[EpochResult],
    fold: str,
    selection_dir: Path,
) -> dict[str, Any]:
    """Select the final checkpoint from top-k candidates after validation postprocessing."""
    selection_start = perf_counter()
    candidates = ranked_checkpoint_candidates(config, trainer, history)
    selection_dir.mkdir(parents=True, exist_ok=True)
    evaluated_candidates = [
        evaluate_checkpoint_candidate_on_validation(
            config=config,
            trainer=trainer,
            model=model,
            validation_loader=validation_loader,
            candidate=candidate,
            fold=fold,
            candidate_dir=selection_dir / candidate_checkpoint_label(candidate),
        )
        for candidate in candidates
    ]
    selected = sorted(evaluated_candidates, key=lambda item: checkpoint_selection_sort_key(config, item))[0]
    selected_checkpoint_label = str(selected.get("checkpoint_label") or f"epoch_{int(selected['epoch']):04d}")
    selected["checkpoint_label"] = selected_checkpoint_label
    total_seconds = perf_counter() - selection_start
    summary = {
        "enabled": True,
        "top_k": int(config.training.top_k_checkpoints),
        "monitor": config.training.monitor,
        "monitor_mode": config.training.monitor_mode,
        "validate_every_n_batches": config.training.validate_every_n_batches,
        "selection_split": "validation",
        "tie_break_order": [
            "best_postprocessed_monitor",
            "best_raw_monitor",
            "earliest_validation_index",
        ],
        "selected_epoch": int(selected["epoch"]),
        "selected_batch_in_epoch": selected.get("batch_in_epoch"),
        "selected_global_step": selected.get("global_step"),
        "selected_validation_index": selected.get("validation_index"),
        "selected_checkpoint_label": selected_checkpoint_label,
        "selected_checkpoint_path": str(selected["checkpoint_path"]),
        "raw_monitor_value": float(selected["raw_monitor_value"]),
        "postprocessed_monitor_value": float(selected["postprocessed_monitor_value"]),
        "n_candidates": len(evaluated_candidates),
        "selection_seconds": round(total_seconds, 6),
        "selection_duration": format_duration(total_seconds),
        "candidates": [checkpoint_candidate_summary(candidate) for candidate in evaluated_candidates],
    }
    print(
        f"Selected checkpoint {selected_checkpoint_label} after validation postprocessing: "
        f"raw_{config.training.monitor}={selected['raw_monitor_value']:.6f}, "
        f"postprocessed_{config.training.monitor}={selected['postprocessed_monitor_value']:.6f} "
        f"from {len(evaluated_candidates)} candidate(s)."
    )
    return {
        "selected": selected,
        "candidates": evaluated_candidates,
        "summary": summary,
        "selection_seconds": total_seconds,
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
    seed: int | None = None,
    fold_has_test_split: bool | None = None,
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
    run_expert_usage_path = fold_log_dir / "expert_usage.yaml"
    run_pr_thresholds_path = fold_log_dir / "pr_thresholds.yaml"
    run_directional_thresholds_path = fold_log_dir / "directional_thresholds.yaml"
    run_temperature_scaling_path = fold_log_dir / "temperature_scaling.yaml"
    run_checkpoint_selection_path = fold_log_dir / "checkpoint_selection.yaml"
    run_checkpoint_selection_dir = fold_log_dir / "checkpoint_selection"
    run_pr_curves_dir = fold_log_dir / "pr_curves"
    run_probabilities_dir = fold_log_dir / "probabilities"
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

    test_dataset: LOBDataset | None = None
    if fold_has_test_split is False:
        has_test_split = False
        print(f"Fold '{fold_id}' has no configured test split; test evaluation will be skipped.")
    else:
        test_dataset = build_dataset(fold_sequence_dir, "test", config.data.sequence_window)
        has_test_split = len(test_dataset) > 0 if fold_has_test_split is None else bool(fold_has_test_split)
        if has_test_split and len(test_dataset) == 0:
            raise ValueError(
                f"No test sequences found in {fold_sequence_dir / 'test'}. "
                "Test dates must be configured and preprocessed explicitly."
            )
        if not has_test_split:
            print(f"Fold '{fold_id}' has no configured test split; test evaluation will be skipped.")

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
    auxiliary_criterion, auxiliary_loss_summary = build_auxiliary_criterion(
        config,
        class_counts=list(class_weight_summary["counts"]),  # type: ignore[arg-type]
    )
    if auxiliary_loss_summary.get("enabled"):
        print(
            f"Fold '{fold_id}' auxiliary losses enabled: "
            f"movement_weight={auxiliary_loss_summary['movement_weight']}, "
            f"direction_weight={auxiliary_loss_summary['direction_weight']}, "
            f"movement_pos_weight={auxiliary_loss_summary['movement_pos_weight']}, "
            f"direction_class_weights={auxiliary_loss_summary['direction_class_weights']}."
        )
    else:
        print(f"Fold '{fold_id}' auxiliary losses disabled.")

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
            shuffle=False,  # already shuffled during sampling
            generator=torch_generator_from_seed(fold_seed),
            **loader_kwargs,
        )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.training.eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=config.training.eval_batch_size,
            shuffle=False,
            **loader_kwargs,
        )
        if has_test_split and test_dataset is not None
        else None
    )

    x_sample, _, _ = train_dataset[0]
    model = build_model(config.model, d_input=x_sample.shape[-1])
    model_parameters = model_parameter_summary(model)
    class_distributions = {
        "train": class_distribution(train_dataset, config.model.num_classes),
        "validation": class_distribution(validation_dataset, config.model.num_classes),
    }
    if has_test_split and test_dataset is not None:
        class_distributions["test"] = class_distribution(test_dataset, config.model.num_classes)
    if sampled_class_counts is not None:
        class_distributions["train_sampled_per_epoch"] = class_distribution_from_counts(sampled_class_counts)
    dataset_sizes = {
        "train": len(train_dataset),
        "validation": len(validation_dataset),
        "test": len(test_dataset) if has_test_split and test_dataset is not None else 0,
    }
    if train_sampler is not None:
        dataset_sizes["train_sampled_per_epoch"] = len(train_sampler)
    print(
        f"Fold '{fold_id}' dataset sizes: "
        f"train={dataset_sizes['train']}, "
        f"validation={dataset_sizes['validation']}, "
        f"test={dataset_sizes['test']}{'' if has_test_split else ' (skipped)'}."
    )
    save_run_config_snapshot(
        config,
        run_config_path,
        fold_id=fold_id,
        model_parameters=model_parameters,
        preprocessing_metadata=preprocessing_metadata,
        sampling_summary=sampling_summary,
        auxiliary_loss_summary=auxiliary_loss_summary,
    )
    trainer = LobTrainer(config.training, auxiliary_criterion=auxiliary_criterion)
    fit_start = perf_counter()
    model, history = trainer.fit(model, train_loader, validation_loader)
    fit_duration_seconds = perf_counter() - fit_start
    checkpoint_selection = select_checkpoint_after_validation_postprocessing(
        trainer=trainer,
        config=config,
        model=model,
        validation_loader=validation_loader,
        history=history,
        fold=fold_id,
        selection_dir=run_checkpoint_selection_dir,
    )
    checkpoint_selection_summary = checkpoint_selection["summary"]
    save_run_summary(checkpoint_selection_summary, run_checkpoint_selection_path)
    selected_candidate = checkpoint_selection["selected"]
    best_epoch = int(selected_candidate["epoch"])
    selected_validation_index = int(selected_candidate.get("validation_index") or best_epoch)
    best_checkpoint_label = str(selected_candidate["checkpoint_label"])
    best_history = history_item_by_validation_index(history, selected_validation_index)
    best_history.val_loss = float(selected_candidate["validation_loss"])
    best_history.val_metrics = selected_candidate["validation_metrics"]
    best_history.val_expert_usage = selected_candidate["validation_expert_usage"]
    validation_outputs_for_artifacts = selected_candidate["validation_outputs"]

    selected_checkpoint_path = Path(selected_candidate["checkpoint_path"])
    best_model_path = config.training.best_model_path
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    if selected_checkpoint_path.resolve() != best_model_path.resolve():
        shutil.copy2(selected_checkpoint_path, best_model_path)
    load_checkpoint_into_model(model=model, trainer=trainer, checkpoint_path=best_model_path)

    validation_metrics = best_history.val_metrics
    if validation_metrics is None:
        raise RuntimeError("Selected checkpoint validation metrics are unavailable.")
    print(
        f"Selected checkpoint {best_checkpoint_label} validation evaluation: "
        f"val_loss={best_history.val_loss:.6f}, "
        f"val_acc={validation_metrics.accuracy:.4f}, "
        f"val_macro_f1={validation_metrics.macro_f1:.4f}, "
        f"val_directional_macro_f1={validation_metrics.directional_macro_f1:.4f}, "
        f"val_ece={validation_metrics.expected_calibration_error:.4f}, "
        f"{roc_auc_summary(config, validation_metrics, 'val')}, "
        f"{pr_auc_summary(config, validation_metrics, 'val')}, "
        f"{pr_ap_summary(config, validation_metrics, 'val')}."
    )

    test_duration_seconds: float | None = None
    test_outputs_for_artifacts = None
    if test_loader is None:
        best_history.test_loss = None
        best_history.test_metrics = None
        best_history.test_expert_usage = None
        print(f"Selected checkpoint {best_checkpoint_label} test evaluation skipped: fold has no configured test split.")
    else:
        test_start = perf_counter()
        test_result = trainer.evaluate(
            model=model,
            data_loader=test_loader,
            description=f"Selected {best_checkpoint_label} [Test]",
            collect_outputs=True,
            track_pr_metrics=True,
            track_expert_usage=True,
        )
        test_duration_seconds = perf_counter() - test_start
        best_history.test_loss = test_result.loss
        best_history.test_metrics = test_result.metrics
        best_history.test_expert_usage = test_result.expert_usage
        test_outputs_for_artifacts = test_result.prediction_outputs
        test_ece_down = mapped_class_metric(
            config,
            test_result.metrics,
            -1,
            "per_class_expected_calibration_error",
        )
        test_ece_neutral = mapped_class_metric(
            config,
            test_result.metrics,
            0,
            "per_class_expected_calibration_error",
        )
        test_ece_up = mapped_class_metric(
            config,
            test_result.metrics,
            1,
            "per_class_expected_calibration_error",
        )
        print(
            f"Selected checkpoint {best_checkpoint_label} test evaluation: "
            f"test_loss={test_result.loss:.6f}, "
            f"test_acc={test_result.metrics.accuracy:.4f}, "
            f"test_macro_f1={test_result.metrics.macro_f1:.4f}, "
            f"test_directional_macro_f1={test_result.metrics.directional_macro_f1:.4f}, "
            f"test_ece={test_result.metrics.expected_calibration_error:.4f}, "
            f"test_ece_down={format_optional_metric(test_ece_down)}, "
            f"test_ece_neutral={format_optional_metric(test_ece_neutral)}, "
            f"test_ece_up={format_optional_metric(test_ece_up)}, "
            f"{roc_auc_summary(config, test_result.metrics, 'test')}, "
            f"{pr_auc_summary(config, test_result.metrics, 'test')}, "
            f"{pr_ap_summary(config, test_result.metrics, 'test')} "
            f"({format_duration(test_duration_seconds)})."
        )

    validation_probabilities_path = run_probabilities_dir / f"validation_best_{best_checkpoint_label}.csv"
    test_probabilities_path = (
        run_probabilities_dir / f"test_best_{best_checkpoint_label}.csv"
        if test_outputs_for_artifacts is not None
        else None
    )
    temperature_scaling_summary: dict[str, Any] = {
        "enabled": bool(config.training.temperature_scaling.enabled),
        "class_bias_calibration": bool(config.training.temperature_scaling.class_bias_calibration),
    }
    if config.training.temperature_scaling.enabled:
        (
            temperature_scaling_summary,
            val_calibrated_metrics,
            test_calibrated_metrics,
            validation_outputs_for_artifacts,
            test_outputs_for_artifacts,
        ) = fit_and_apply_temperature_scaling(
            config=config,
            validation_outputs=validation_outputs_for_artifacts,
            test_outputs=test_outputs_for_artifacts,
            best_epoch=best_epoch,
            checkpoint_label=best_checkpoint_label,
            batch_in_epoch=selected_candidate.get("batch_in_epoch"),
            global_step=selected_candidate.get("global_step"),
            validation_index=selected_validation_index,
            fold=fold_id,
            target_path=run_temperature_scaling_path,
        )
        best_history.val_metrics = val_calibrated_metrics
        best_history.test_metrics = test_calibrated_metrics
        print(
            f"Fold {fold_id} temperature scaling fitted on validation: "
            f"T={temperature_scaling_summary['temperature']:.6g}, "
            f"class_bias_calibration={temperature_scaling_summary.get('class_bias_calibration', False)}, "
            f"val_unweighted_ce_before={temperature_scaling_summary['validation_nll_before']:.6f}, "
            f"val_unweighted_ce_after={temperature_scaling_summary['validation_nll_after']:.6f}."
        )
    directional_threshold_summary: dict[str, Any] = {
        "enabled": bool(config.training.directional_thresholds.enabled),
    }
    if config.training.directional_thresholds.enabled:
        best_history.val_argmax_metrics = best_history.val_metrics
        best_history.test_argmax_metrics = best_history.test_metrics
        (
            directional_threshold_summary,
            val_threshold_metrics,
            test_threshold_metrics,
            validation_outputs_for_artifacts,
            test_outputs_for_artifacts,
        ) = fit_and_apply_directional_thresholds(
            config=config,
            validation_outputs=validation_outputs_for_artifacts,
            test_outputs=test_outputs_for_artifacts,
            best_epoch=best_epoch,
            checkpoint_label=best_checkpoint_label,
            batch_in_epoch=selected_candidate.get("batch_in_epoch"),
            global_step=selected_candidate.get("global_step"),
            validation_index=selected_validation_index,
            fold=fold_id,
            target_path=run_directional_thresholds_path,
        )
        best_history.val_metrics = val_threshold_metrics
        best_history.test_metrics = test_threshold_metrics
        best_history.val_threshold_metrics = directional_threshold_summary["validation_threshold_metrics"]
        best_history.test_threshold_metrics = directional_threshold_summary["test_threshold_metrics"]
        print(
            f"Fold {fold_id} directional thresholds selected on validation: "
            f"method={directional_threshold_summary['method']}, "
            f"down={format_optional_threshold(directional_threshold_summary['threshold_down'])}, "
            f"up={format_optional_threshold(directional_threshold_summary['threshold_up'])}, "
            f"val_threshold_{directional_threshold_summary['configured_score']}="
            f"{directional_threshold_summary['score']:.4f}."
        )
    save_probability_outputs(validation_outputs_for_artifacts, validation_probabilities_path, config)
    if test_outputs_for_artifacts is not None and test_probabilities_path is not None:
        save_probability_outputs(test_outputs_for_artifacts, test_probabilities_path, config)
    pr_artifacts = save_best_pr_artifacts(
        validation_outputs_for_artifacts,
        curves_dir=run_pr_curves_dir,
        thresholds_path=run_pr_thresholds_path,
        config=config,
        best_epoch=best_epoch,
        checkpoint_label=best_checkpoint_label,
        fold=fold_id,
    )
    final_postprocessed_monitor_value = monitor_value_after_postprocessing(
        config=config,
        raw_monitor_value=float(selected_candidate["raw_monitor_value"]),
        val_loss=float(best_history.val_loss),
        val_metrics=best_history.val_metrics,
    )
    selected_candidate["postprocessed_monitor_value"] = float(final_postprocessed_monitor_value)
    checkpoint_selection_summary["postprocessed_monitor_value"] = float(final_postprocessed_monitor_value)
    for candidate_summary in checkpoint_selection_summary.get("candidates", []):
        candidate_validation_index = int(candidate_summary.get("validation_index") or candidate_summary.get("epoch", -1))
        if candidate_validation_index == selected_validation_index:
            candidate_summary["selected"] = True
            candidate_summary["final_postprocessed_monitor_value"] = float(final_postprocessed_monitor_value)
        else:
            candidate_summary["selected"] = False
    checkpoint_selection_summary["final_artifacts"] = {
        "best_model_path": str(config.training.best_model_path),
        "temperature_scaling": (
            str(run_temperature_scaling_path) if temperature_scaling_summary.get("enabled") else None
        ),
        "directional_thresholds": (
            str(run_directional_thresholds_path) if directional_threshold_summary.get("enabled") else None
        ),
        "validation_probabilities": str(validation_probabilities_path),
        "test_probabilities": None if test_probabilities_path is None else str(test_probabilities_path),
        "pr_thresholds": str(run_pr_thresholds_path),
        "pr_curves_dir": str(run_pr_curves_dir),
    }
    save_run_summary(checkpoint_selection_summary, run_checkpoint_selection_path)
    fold_duration_seconds = perf_counter() - fold_start
    timing = {
        "fold_training_seconds": round(fold_duration_seconds, 6),
        "fold_training_duration": format_duration(fold_duration_seconds),
        "model_fit_seconds": round(fit_duration_seconds, 6),
        "model_fit_duration": format_duration(fit_duration_seconds),
        "checkpoint_selection_seconds": round(float(checkpoint_selection["selection_seconds"]), 6),
        "checkpoint_selection_duration": format_duration(float(checkpoint_selection["selection_seconds"])),
        "best_validation_evaluation_seconds": round(float(selected_candidate["validation_seconds"]), 6),
        "best_validation_evaluation_duration": format_duration(float(selected_candidate["validation_seconds"])),
        "test_evaluation_seconds": (
            None if test_duration_seconds is None else round(test_duration_seconds, 6)
        ),
        "test_evaluation_duration": (
            "skipped" if test_duration_seconds is None else format_duration(test_duration_seconds)
        ),
        "best_model_evaluation_seconds": round(
            float(selected_candidate["validation_seconds"]) + (test_duration_seconds or 0.0),
            6,
        ),
        "best_model_evaluation_duration": format_duration(
            float(selected_candidate["validation_seconds"]) + (test_duration_seconds or 0.0)
        ),
    }
    if temperature_scaling_summary.get("enabled"):
        timing["temperature_scaling_fit_seconds"] = temperature_scaling_summary.get("fit_seconds")
        timing["temperature_scaling_fit_duration"] = temperature_scaling_summary.get("fit_duration")

    save_epoch_history(history, run_losses_path, config=config, fold=fold_id)
    save_confusion_matrices(history, run_confusion_matrices_path, fold=fold_id)
    save_expert_usage(history, run_expert_usage_path, config=config, fold=fold_id)
    save_run_log(
        target=run_log_path,
        config=config,
        run_stem=run_stem,
        dataset_sizes=dataset_sizes,
        class_distributions=class_distributions,
        history=history,
        losses_path=run_losses_path,
        confusion_matrices_path=run_confusion_matrices_path,
        expert_usage_path=run_expert_usage_path,
        config_snapshot_path=run_config_path,
        model_parameters=model_parameters,
        pr_thresholds_path=run_pr_thresholds_path,
        pr_curves_dir=run_pr_curves_dir,
        probabilities_dir=run_probabilities_dir,
        temperature_scaling_path=(
            run_temperature_scaling_path if temperature_scaling_summary.get("enabled") else None
        ),
        temperature_scaling_summary=temperature_scaling_summary,
        directional_thresholds_path=(
            run_directional_thresholds_path if directional_threshold_summary.get("enabled") else None
        ),
        directional_threshold_summary=directional_threshold_summary,
        selected_best_epoch=selected_validation_index,
        selected_monitor_value=float(selected_candidate["postprocessed_monitor_value"]),
        selected_raw_monitor_value=float(selected_candidate["raw_monitor_value"]),
        selected_postprocessed_monitor_value=float(selected_candidate["postprocessed_monitor_value"]),
        checkpoint_selection_path=run_checkpoint_selection_path,
        checkpoint_selection_summary=checkpoint_selection_summary,
        preprocessing_metadata=preprocessing_metadata,
        sampling_summary=sampling_summary,
        auxiliary_loss_summary=auxiliary_loss_summary,
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
    print(f"Fold {fold_id} checkpoint selection saved to: {run_checkpoint_selection_path}")
    print(f"Fold {fold_id} PR thresholds saved to: {run_pr_thresholds_path}")
    if temperature_scaling_summary.get("enabled"):
        print(f"Fold {fold_id} temperature scaling saved to: {run_temperature_scaling_path}")
    if directional_threshold_summary.get("enabled"):
        print(f"Fold {fold_id} directional thresholds saved to: {run_directional_thresholds_path}")
    if test_probabilities_path is None:
        print(f"Fold {fold_id} validation probability outputs saved to: {validation_probabilities_path}")
    else:
        print(f"Fold {fold_id} validation/test probability outputs saved to: {run_probabilities_dir}")
    for epoch_index, result in enumerate(history, start=1):
        history_label = checkpoint_label_for_history_item(result, epoch_index)
        test_suffix = "" if result.test_loss is None else f", test_loss={result.test_loss:.6f}"
        val_metrics = result.val_metrics
        metric_suffix = ""
        if val_metrics is not None:
            val_ece_down = mapped_class_metric(
                config,
                val_metrics,
                -1,
                "per_class_expected_calibration_error",
            )
            val_ece_neutral = mapped_class_metric(
                config,
                val_metrics,
                0,
                "per_class_expected_calibration_error",
            )
            val_ece_up = mapped_class_metric(
                config,
                val_metrics,
                1,
                "per_class_expected_calibration_error",
            )
            metric_suffix = (
                f", val_acc={val_metrics.accuracy:.4f}, val_macro_f1={val_metrics.macro_f1:.4f}, "
                f"val_directional_macro_f1={val_metrics.directional_macro_f1:.4f}, "
                f"val_ece={val_metrics.expected_calibration_error:.4f}, "
                f"val_ece_down={format_optional_metric(val_ece_down)}, "
                f"val_ece_neutral={format_optional_metric(val_ece_neutral)}, "
                f"val_ece_up={format_optional_metric(val_ece_up)}, "
                f"{roc_auc_summary(config, val_metrics, 'val')}, "
                f"{pr_auc_summary(config, val_metrics, 'val')}, "
                f"{pr_ap_summary(config, val_metrics, 'val')}"
            )
        print(
            f"{history_label}: train_loss={result.train_loss:.6f}, "
            f"val_loss={result.val_loss:.6f}{test_suffix}{metric_suffix}"
        )

    return {
        "sequence_dir": str(fold_sequence_dir),
        "log_dir": str(fold_log_dir),
        "result_dir": str(fold_result_dir),
        "best_model_path": str(config.training.best_model_path),
        "model_max_dt": max_dt_summary,
        "class_weights": class_weight_summary,
        "auxiliary_losses": auxiliary_loss_summary,
        "sampling": sampling_summary,
        "dataset_sizes": dataset_sizes,
        "timing": timing,
        "logs": {
            "run_log": str(run_log_path),
            "config": str(run_config_path),
            "metrics": str(run_losses_path),
            "confusion_matrices": str(run_confusion_matrices_path),
            "expert_usage": str(run_expert_usage_path),
            "pr_thresholds": str(run_pr_thresholds_path),
            "checkpoint_selection": str(run_checkpoint_selection_path),
            "temperature_scaling": (
                str(run_temperature_scaling_path) if temperature_scaling_summary.get("enabled") else None
            ),
            "directional_thresholds": (
                str(run_directional_thresholds_path) if directional_threshold_summary.get("enabled") else None
            ),
            "pr_curves_dir": str(run_pr_curves_dir),
            "validation_probabilities": str(validation_probabilities_path),
            "test_probabilities": None if test_probabilities_path is None else str(test_probabilities_path),
        },
        "pr_artifacts": pr_artifacts,
        "checkpoint_selection": checkpoint_selection_summary,
        "temperature_scaling": temperature_scaling_summary,
        "directional_thresholds": directional_threshold_summary,
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
        or timestamped_run_stem(config.experiment.name)
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
            fold_has_test_split=fold.has_test_dates,
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
