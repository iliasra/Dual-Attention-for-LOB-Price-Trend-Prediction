from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from configuration import ExperimentConfig, load_config
from thresholding import (
    DirectionalThresholdSelection,
    apply_directional_threshold_policy,
    optimize_directional_thresholds,
    optimize_precision_floor_thresholds,
    optimize_top_quantile_thresholds,
    threshold_candidates,
    thresholded_metric_summary,
)


RAW_CLASS_NAMES = {-1: "down", 0: "neutral", 1: "up"}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for threshold replay on saved probabilities."""
    parser = argparse.ArgumentParser(
        description="Fit/apply directional thresholds from saved probability CSVs."
    )
    parser.add_argument(
        "--probabilities-dir",
        type=Path,
        required=True,
        help="Directory containing validation/test probability CSVs.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Experiment config or fold config snapshot with training.directional_thresholds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <probabilities-dir>/thresholded.",
    )
    parser.add_argument(
        "--validation-glob",
        default="validation*.csv",
        help="Glob used to select the validation CSV for threshold fitting.",
    )
    parser.add_argument(
        "--apply-glob",
        default="*.csv",
        help="Glob of CSVs to rewrite with thresholded predictions.",
    )
    return parser.parse_args()


def ordered_class_labels(config: ExperimentConfig) -> list[str]:
    """Return class labels ordered by model class id."""
    labels = [f"class_{class_id}" for class_id in range(config.model.num_classes)]
    for raw_label, class_id in config.data.label_mapping.items():
        if 0 <= int(class_id) < len(labels):
            labels[int(class_id)] = RAW_CLASS_NAMES.get(int(raw_label), f"raw_{raw_label}")
    return labels


def safe_label(label: str) -> str:
    """Return the artifact-safe label suffix used by probability CSV columns."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "class"


def directional_class_ids(config: ExperimentConfig) -> tuple[int, int, int]:
    """Resolve down/neutral/up class ids from label_mapping."""
    missing = [name for raw, name in RAW_CLASS_NAMES.items() if raw not in config.data.label_mapping]
    if missing:
        raise ValueError(f"Directional thresholding requires label_mapping entries for: {', '.join(missing)}.")
    return (
        int(config.data.label_mapping[-1]),
        int(config.data.label_mapping[0]),
        int(config.data.label_mapping[1]),
    )


def softmax(logits: np.ndarray) -> np.ndarray:
    """Convert logits to probabilities row-wise."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return (exp_values / np.sum(exp_values, axis=1, keepdims=True)).astype(np.float32)


def probability_or_logit_array(frame: pd.DataFrame, labels: list[str]) -> np.ndarray:
    """Extract probabilities, or softmax logits if probability columns are absent."""
    probability_columns = [f"p_{safe_label(label)}" for label in labels]
    if all(column in frame.columns for column in probability_columns):
        return frame[probability_columns].to_numpy(dtype=np.float32)

    logit_columns = [f"logit_{safe_label(label)}" for label in labels]
    if all(column in frame.columns for column in logit_columns):
        return softmax(frame[logit_columns].to_numpy(dtype=np.float32))

    raise ValueError(
        "CSV must contain either probability columns "
        f"{probability_columns} or logit columns {logit_columns}."
    )


def labels_to_ids(values: pd.Series, labels: list[str]) -> np.ndarray:
    """Map saved label strings or numeric class ids back to model class ids."""
    mapping = {label.lower(): class_id for class_id, label in enumerate(labels)}
    mapping.update({f"class_{class_id}": class_id for class_id in range(len(labels))})
    result: list[int] = []
    for value in values:
        key = str(value).strip().lower()
        if key in mapping:
            result.append(mapping[key])
            continue
        try:
            result.append(int(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unknown label value in probability CSV: {value!r}") from exc
    return np.asarray(result, dtype=np.int64)


def ids_to_labels(values: np.ndarray, labels: list[str]) -> list[str]:
    """Map model class ids to saved label strings."""
    return [
        labels[int(class_id)] if 0 <= int(class_id) < len(labels) else f"class_{int(class_id)}"
        for class_id in np.asarray(values, dtype=np.int64).reshape(-1)
    ]


def frame_to_arrays(frame: pd.DataFrame, config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    """Extract targets and probabilities from one logged CSV."""
    labels = ordered_class_labels(config)
    if "true_label" not in frame.columns:
        raise ValueError("Probability CSV must contain a true_label column.")
    targets = labels_to_ids(frame["true_label"], labels)
    probabilities = probability_or_logit_array(frame, labels)
    if probabilities.shape[0] != targets.shape[0]:
        raise ValueError("Probability and target row counts do not match.")
    return targets, probabilities


def refinement_steps_from_config(step: float) -> tuple[float, ...]:
    """Return fine-grid refinement steps used by the training pipeline."""
    return tuple(candidate for candidate in (0.01, 0.005, 0.002, 0.001) if candidate < float(step) - 1e-12)


def fit_thresholds(
    config: ExperimentConfig,
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[DirectionalThresholdSelection, tuple[float, ...]]:
    """Fit thresholds on validation according to the experiment config."""
    threshold_config = config.training.directional_thresholds
    if not threshold_config.enabled:
        raise ValueError("training.directional_thresholds.enabled is false in the provided config.")

    down_id, neutral_id, up_id = directional_class_ids(config)
    candidates = threshold_candidates(
        threshold_config.min_threshold,
        threshold_config.max_threshold,
        threshold_config.step,
    )
    monitor_params = config.training.monitor_params
    if threshold_config.method == "joint_up_down":
        refinements = refinement_steps_from_config(threshold_config.step)
        return (
            optimize_directional_thresholds(
                probabilities,
                targets,
                down_candidates=candidates,
                up_candidates=candidates,
                down_id=down_id,
                neutral_id=neutral_id,
                up_id=up_id,
                refinement_steps=refinements,
                delta=threshold_config.delta,
                score=threshold_config.score,
                monitor_params=monitor_params,
            ),
            refinements,
        )
    if threshold_config.method == "precision_floor":
        return (
            optimize_precision_floor_thresholds(
                probabilities,
                targets,
                down_candidates=candidates,
                up_candidates=candidates,
                down_precision_floor=float(threshold_config.down_precision_floor),
                up_precision_floor=float(threshold_config.up_precision_floor),
                down_id=down_id,
                neutral_id=neutral_id,
                up_id=up_id,
                delta=threshold_config.delta,
                score=threshold_config.score,
                monitor_params=monitor_params,
            ),
            (),
        )
    if threshold_config.method == "top_x_quantile":
        return (
            optimize_top_quantile_thresholds(
                probabilities,
                targets,
                down_quantile=float(threshold_config.down_quantile),
                up_quantile=float(threshold_config.up_quantile),
                down_id=down_id,
                neutral_id=neutral_id,
                up_id=up_id,
                delta=threshold_config.delta,
                score=threshold_config.score,
                monitor_params=monitor_params,
            ),
            (),
        )
    raise ValueError(f"Unsupported thresholding method: {threshold_config.method}")


def apply_selection(
    frame: pd.DataFrame,
    config: ExperimentConfig,
    selection: DirectionalThresholdSelection,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Apply fixed thresholds to one logged probability frame."""
    labels = ordered_class_labels(config)
    down_id, neutral_id, up_id = directional_class_ids(config)
    targets, probabilities = frame_to_arrays(frame, config)
    predictions = apply_directional_threshold_policy(
        probabilities,
        threshold_down=selection.threshold_down,
        threshold_up=selection.threshold_up,
        down_id=down_id,
        neutral_id=neutral_id,
        up_id=up_id,
        delta=config.training.directional_thresholds.delta,
        down_enabled=selection.down_enabled,
        up_enabled=selection.up_enabled,
    )
    output = frame.copy()
    output["threshold_pred_label"] = ids_to_labels(predictions, labels)
    output["threshold_pred_class_id"] = predictions
    metrics = thresholded_metric_summary(
        targets,
        predictions,
        down_id=down_id,
        neutral_id=neutral_id,
        up_id=up_id,
    )
    return output, metrics


def confusion_payload(targets: np.ndarray, predictions: np.ndarray, num_classes: int) -> dict[str, Any]:
    """Build raw and row-normalized confusion matrices."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(targets, predictions, strict=False):
        if 0 <= int(target) < num_classes and 0 <= int(prediction) < num_classes:
            matrix[int(target), int(prediction)] += 1
    denominators = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        denominators,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=denominators > 0,
    )
    return {
        "raw": matrix.tolist(),
        "normalized_by_true_class": normalized.tolist(),
    }


def to_builtin(value: Any) -> Any:
    """Convert numpy scalars/arrays into YAML-friendly Python values."""
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def threshold_artifact(
    config: ExperimentConfig,
    selection: DirectionalThresholdSelection,
    *,
    validation_file: Path,
    probabilities_dir: Path,
    refinement_steps: tuple[float, ...],
) -> dict[str, Any]:
    """Build a YAML payload describing the fitted thresholds."""
    threshold_config = config.training.directional_thresholds
    down_id, neutral_id, up_id = directional_class_ids(config)
    payload = {
        "enabled": True,
        "source": "scripts/apply_directional_thresholds.py",
        "source_config": str(config.path),
        "probabilities_dir": str(probabilities_dir),
        "selection_split": "validation",
        "selection_file": str(validation_file),
        "selection_metric": (
            threshold_config.score
            if threshold_config.method == "joint_up_down"
            else (
                "per_class_precision_floor_then_recall"
                if threshold_config.method == "precision_floor"
                else "top_probability_quantile"
            )
        ),
        "method": threshold_config.method,
        "configured_score": threshold_config.score,
        "threshold_down": selection.threshold_down,
        "threshold_up": selection.threshold_up,
        "down_enabled": selection.down_enabled,
        "up_enabled": selection.up_enabled,
        "score": selection.score,
        "rate_penalty": selection.rate_penalty,
        "grid_min": threshold_config.min_threshold,
        "grid_max": threshold_config.max_threshold,
        "grid_step": threshold_config.step,
        "refinement_steps": refinement_steps,
        "delta": threshold_config.delta,
        "down_precision_floor": threshold_config.down_precision_floor,
        "up_precision_floor": threshold_config.up_precision_floor,
        "down_quantile": threshold_config.down_quantile,
        "up_quantile": threshold_config.up_quantile,
        "top_quantile_convention": "0.01 means the top 1% validation probabilities for that class.",
        "n_candidates": selection.n_candidates,
        "decision_tie_break": "logit_margin_gap_else_neutral",
        "class_ids": {"down": down_id, "neutral": neutral_id, "up": up_id},
        "optimization_stages": selection.stage_summaries,
        "selection_details": selection.selection_details,
    }
    if threshold_config.method != "top_x_quantile":
        payload["min_directional_precision"] = selection.min_directional_precision
    if selection.score_details:
        payload.update(selection.score_details)
    return to_builtin(payload)


def resolve_validation_file(probabilities_dir: Path, pattern: str) -> Path:
    """Return the single validation CSV used for fitting."""
    matches = sorted(probabilities_dir.glob(pattern))
    matches = [path for path in matches if path.is_file()]
    if not matches:
        raise FileNotFoundError(f"No validation CSV matched {pattern!r} in {probabilities_dir}.")
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise ValueError(f"Validation glob matched multiple files; narrow --validation-glob: {names}")
    return matches[0]


def main() -> None:
    """Fit thresholds on validation and replay them on saved probability CSVs."""
    args = parse_args()
    config = load_config(args.config)
    probabilities_dir = args.probabilities_dir
    output_dir = args.output_dir or probabilities_dir / "thresholded"
    validation_file = resolve_validation_file(probabilities_dir, args.validation_glob)

    validation_frame = pd.read_csv(validation_file)
    validation_targets, validation_probabilities = frame_to_arrays(validation_frame, config)
    selection, refinement_steps = fit_thresholds(config, validation_targets, validation_probabilities)
    artifact = threshold_artifact(
        config,
        selection,
        validation_file=validation_file,
        probabilities_dir=probabilities_dir,
        refinement_steps=refinement_steps,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "directional_thresholds.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(artifact, handle, sort_keys=False, allow_unicode=True)

    metric_rows: list[dict[str, Any]] = []
    confusion_matrices: dict[str, Any] = {}
    labels = ordered_class_labels(config)
    for csv_path in sorted(path for path in probabilities_dir.glob(args.apply_glob) if path.is_file()):
        frame = pd.read_csv(csv_path)
        output_frame, metrics = apply_selection(frame, config, selection)
        output_path = output_dir / f"{csv_path.stem}_thresholded.csv"
        output_frame.to_csv(output_path, index=False)

        targets, _ = frame_to_arrays(frame, config)
        predictions = labels_to_ids(output_frame["threshold_pred_label"], labels)
        metric_rows.append(
            {
                "file": csv_path.name,
                "output_file": output_path.name,
                **metrics,
            }
        )
        confusion_matrices[csv_path.stem] = confusion_payload(
            targets,
            predictions,
            config.model.num_classes,
        )

    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(metric_rows[0].keys()) if metric_rows else ["file"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)
    with (output_dir / "confusion_matrices.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "normalization": "Rows are true classes; columns are thresholded predicted classes.",
                "class_labels": {str(index): label for index, label in enumerate(labels)},
                "files": confusion_matrices,
            },
            handle,
            sort_keys=False,
            allow_unicode=True,
        )

    print(
        "Directional thresholds applied: "
        f"method={artifact['method']}, down={artifact['threshold_down']}, "
        f"up={artifact['threshold_up']}, val_score={artifact['score']:.6f}."
    )
    print(f"Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
