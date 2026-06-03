from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from configuration import ExperimentConfig
from datasets import LOBDataset
from model import build_model
from run_logging import (
    format_duration,
    save_probability_outputs,
)
from thresholding import apply_directional_threshold_policy
from training import LobTrainer, classification_metrics_from_predictions


RAW_CLASS_NAMES = {-1: "down", 0: "neutral", 1: "up"}
CHECKPOINT_STATE_KEYS = ("state_dict", "model_state_dict")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for checkpoint evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a trained LOB checkpoint on preprocessed sequences.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a .pth checkpoint.")
    parser.add_argument("--config", type=Path, required=True, help="Training config or fold config snapshot.")
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        required=True,
        help="Sequence parent directory or direct split directory containing *_features.npy files.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Optional split subdirectory under --sequence-dir, e.g. validation, test, or holdout.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where evaluation artifacts are saved.")
    parser.add_argument("--max-dt", type=float, default=None, help="Override model.max_dt for this evaluation.")
    parser.add_argument("--device", type=str, default=None, help="Override training.device from the config.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training.eval_batch_size.")
    parser.add_argument(
        "--save-probabilities",
        action="store_true",
        help="Also write per-sample post-softmax probabilities to probabilities.csv.",
    )
    parser.add_argument(
        "--directional-thresholds",
        type=Path,
        default=None,
        help="Optional directional_thresholds.yaml file whose down/up thresholds define evaluation decisions.",
    )
    return parser.parse_args()


def load_yaml_payload(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    return payload


def resolve_snapshot_max_dt(payload: Mapping[str, Any]) -> float | None:
    """Return resolved max_dt from run metadata when available."""
    run_metadata = payload.get("run_metadata")
    if not isinstance(run_metadata, Mapping):
        return None
    model_max_dt = run_metadata.get("model_max_dt")
    if not isinstance(model_max_dt, Mapping):
        return None
    resolved = model_max_dt.get("resolved_max_dt")
    return None if resolved is None else float(resolved)


def resolve_snapshot_class_weights(payload: Mapping[str, Any]) -> list[float] | None:
    """Return loss class weights stored in run metadata when available."""
    run_metadata = payload.get("run_metadata")
    if not isinstance(run_metadata, Mapping):
        return None
    weights = run_metadata.get("class_weights")
    if weights is None:
        return None
    return [float(value) for value in weights]


def load_config_for_evaluation(config_path: Path, max_dt_override: float | None = None) -> ExperimentConfig:
    """Load config and restore runtime fields needed for checkpoint evaluation."""
    payload = load_yaml_payload(config_path)
    config = ExperimentConfig.from_yaml(config_path)

    if max_dt_override is not None:
        config.model.max_dt = float(max_dt_override)
    elif config.model.max_dt is None:
        config.model.max_dt = resolve_snapshot_max_dt(payload)
    if config.model.max_dt is None:
        raise ValueError(
            "model.max_dt is required for evaluation. Provide --max-dt, set model.max_dt in the config, "
            "or use a fold config snapshot containing run_metadata.model_max_dt.resolved_max_dt."
        )

    if config.training.class_weights is None:
        config.training.class_weights = resolve_snapshot_class_weights(payload)
    return config


def sequence_paths(sequence_dir: Path, split: str | None = None) -> tuple[list[str], list[str], list[str], Path]:
    """Resolve matching feature/time/label arrays for one evaluation split."""
    split_dir = sequence_dir / split if split else sequence_dir
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
    if not x_paths:
        raise ValueError(f"No *_features.npy sequence files found in {split_dir}.")
    return x_paths, t_paths, y_paths, split_dir


def build_evaluation_dataset(sequence_dir: Path, split: str | None, sequence_window: int) -> tuple[LOBDataset, Path]:
    """Build the evaluation dataset from a parent or direct sequence directory."""
    x_paths, t_paths, y_paths, split_dir = sequence_paths(sequence_dir, split)
    dataset = LOBDataset(x_paths, t_paths, y_paths, sequence_window=sequence_window)
    if len(dataset) == 0:
        raise ValueError(f"No full sequence windows found in {split_dir}.")
    return dataset, split_dir


def _looks_like_state_dict(value: Any) -> bool:
    """Return whether an object resembles a PyTorch state_dict."""
    return (
        isinstance(value, Mapping)
        and all(isinstance(key, str) for key in value)
        and any(torch.is_tensor(tensor) for tensor in value.values())
    )


def extract_checkpoint_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    """Extract a model state_dict from common checkpoint formats."""
    if _looks_like_state_dict(checkpoint):
        return checkpoint
    if isinstance(checkpoint, Mapping):
        for key in CHECKPOINT_STATE_KEYS:
            state_dict = checkpoint.get(key)
            if _looks_like_state_dict(state_dict):
                return state_dict
    raise ValueError("Checkpoint must be a state_dict or contain 'state_dict'/'model_state_dict'.")


def load_checkpoint_state_dict(checkpoint_path: Path, device: torch.device) -> Mapping[str, Any]:
    """Load and normalize a checkpoint from disk."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    return extract_checkpoint_state_dict(checkpoint)


def ordered_class_labels(config: ExperimentConfig) -> list[str]:
    """Return class labels ordered by model class id."""
    labels = [f"class_{class_id}" for class_id in range(config.model.num_classes)]
    for raw_label, class_id in config.data.label_mapping.items():
        class_index = int(class_id)
        if 0 <= class_index < len(labels):
            labels[class_index] = RAW_CLASS_NAMES.get(int(raw_label), f"raw_{raw_label}")
    return labels


def class_metric(metrics: Any, metric_name: str, class_id: int | None) -> float | None:
    """Return one optional per-class metric value."""
    if class_id is None:
        return None
    values = getattr(metrics, metric_name, None)
    if values is None or class_id < 0 or class_id >= len(values):
        return None
    return float(values[class_id])


def label_class_id(config: ExperimentConfig, raw_label: int) -> int | None:
    """Map a raw class label to the model class id."""
    mapped = config.data.label_mapping.get(raw_label)
    return None if mapped is None else int(mapped)


def directional_threshold_class_ids(
    config: ExperimentConfig,
    payload: Mapping[str, Any],
) -> tuple[int, int, int]:
    """Resolve down/neutral/up ids from threshold YAML or config mapping."""
    class_ids = payload.get("class_ids")
    if isinstance(class_ids, Mapping):
        try:
            return int(class_ids["down"]), int(class_ids["neutral"]), int(class_ids["up"])
        except KeyError as exc:
            raise ValueError("directional threshold class_ids must include down, neutral, and up.") from exc

    ids = (
        label_class_id(config, -1),
        label_class_id(config, 0),
        label_class_id(config, 1),
    )
    if any(class_id is None for class_id in ids):
        raise ValueError("directional thresholds require label_mapping entries for raw labels -1, 0, and 1.")
    return int(ids[0]), int(ids[1]), int(ids[2])  # type: ignore[arg-type]


def load_directional_thresholds(path: Path, config: ExperimentConfig) -> dict[str, Any]:
    """Load selected directional thresholds for external evaluation."""
    payload = load_yaml_payload(path)
    if payload.get("enabled") is False:
        raise ValueError(f"Directional threshold file is disabled: {path}")
    if "threshold_down" not in payload or "threshold_up" not in payload:
        raise ValueError("Directional threshold file must contain threshold_down and threshold_up.")
    down_id, neutral_id, up_id = directional_threshold_class_ids(config, payload)
    for label, class_id in (("down", down_id), ("neutral", neutral_id), ("up", up_id)):
        if not 0 <= int(class_id) < config.model.num_classes:
            raise ValueError(f"Directional threshold {label} class id is outside model.num_classes.")
    threshold_down = None if payload["threshold_down"] is None else float(payload["threshold_down"])
    threshold_up = None if payload["threshold_up"] is None else float(payload["threshold_up"])
    method = str(payload.get("method", "joint_up_down"))
    if method not in {"joint_up_down", "precision_floor", "top_x_quantile"}:
        raise ValueError("Directional threshold method must be joint_up_down, precision_floor, or top_x_quantile.")
    delta = float(payload.get("delta", 0.0))
    if delta < 0.0:
        raise ValueError("Directional threshold delta must be >= 0.")
    return {
        "path": str(path),
        "method": method,
        "threshold_down": threshold_down,
        "threshold_up": threshold_up,
        "down_enabled": bool(payload.get("down_enabled", threshold_down is not None)),
        "up_enabled": bool(payload.get("up_enabled", threshold_up is not None)),
        "delta": delta,
        "down_quantile": payload.get("down_quantile"),
        "up_quantile": payload.get("up_quantile"),
        "down_id": int(down_id),
        "neutral_id": int(neutral_id),
        "up_id": int(up_id),
        "source_payload": payload,
    }


def apply_directional_thresholds_to_result(
    result: Any,
    config: ExperimentConfig,
    threshold_config: Mapping[str, Any],
) -> None:
    """Replace argmax decisions and metrics with thresholded decisions."""
    if result.prediction_outputs is None:
        raise RuntimeError("Directional thresholds require collected probability outputs.")
    outputs = result.prediction_outputs
    probabilities = np.asarray(outputs.get("probabilities", []), dtype=np.float32)
    targets = np.asarray(outputs.get("targets", []), dtype=np.int64).reshape(-1)
    thresholded_predictions = apply_directional_threshold_policy(
        probabilities,
        threshold_down=None if threshold_config["threshold_down"] is None else float(threshold_config["threshold_down"]),
        threshold_up=None if threshold_config["threshold_up"] is None else float(threshold_config["threshold_up"]),
        down_id=int(threshold_config["down_id"]),
        neutral_id=int(threshold_config["neutral_id"]),
        up_id=int(threshold_config["up_id"]),
        delta=float(threshold_config.get("delta", 0.0)),
        down_enabled=bool(threshold_config.get("down_enabled", threshold_config["threshold_down"] is not None)),
        up_enabled=bool(threshold_config.get("up_enabled", threshold_config["threshold_up"] is not None)),
    )
    if "argmax_predictions" not in outputs:
        outputs["argmax_predictions"] = np.asarray(outputs.get("predictions", []), dtype=np.int64).reshape(-1)
    outputs["predictions"] = thresholded_predictions
    result.metrics = classification_metrics_from_predictions(
        targets,
        thresholded_predictions,
        num_classes=config.model.num_classes,
        probabilities=probabilities,
    )


def evaluation_metrics_row(result: Any, config: ExperimentConfig, split: str, num_samples: int) -> dict[str, Any]:
    """Flatten evaluation metrics into one CSV/YAML row."""
    metrics = result.metrics
    row: dict[str, Any] = {
        "split": split,
        "num_samples": int(num_samples),
        "loss": float(result.loss),
        "accuracy": float(metrics.accuracy),
        "macro_precision": float(metrics.macro_precision),
        "macro_recall": float(metrics.macro_recall),
        "macro_f1": float(metrics.macro_f1),
        "directional_macro_f1": float(metrics.directional_macro_f1),
        "weighted_f1": float(metrics.weighted_f1),
        "balanced_accuracy": float(metrics.balanced_accuracy),
        "ece": float(metrics.expected_calibration_error),
    }
    for label, raw_label in (("down", -1), ("neutral", 0), ("up", 1)):
        class_id = label_class_id(config, raw_label)
        row[f"ece_{label}"] = class_metric(metrics, "per_class_expected_calibration_error", class_id)
        row[f"pr_ap_{label}"] = class_metric(metrics, "per_class_pr_ap", class_id)
        row[f"pr_auc_{label}"] = class_metric(metrics, "per_class_pr_auc", class_id)
        row[f"roc_auc_{label}"] = class_metric(metrics, "per_class_roc_auc", class_id)
        row[f"precision_{label}"] = class_metric(metrics, "per_class_precision", class_id)
        row[f"recall_{label}"] = class_metric(metrics, "per_class_recall", class_id)
        row[f"f1_{label}"] = class_metric(metrics, "per_class_f1", class_id)
    for class_id, label in enumerate(ordered_class_labels(config)):
        row[f"f1_class_{class_id}_{label}"] = class_metric(metrics, "per_class_f1", class_id)
    return row


def add_directional_threshold_fields(row: dict[str, Any], threshold_config: Mapping[str, Any] | None) -> None:
    """Add optional threshold decision metadata to a metrics row."""
    if threshold_config is None:
        row["classification_mode"] = "argmax"
        return
    row.update(
        {
            "classification_mode": "directional_thresholds",
            "directional_thresholds_path": threshold_config["path"],
            "threshold_method": threshold_config.get("method", "joint_up_down"),
            "threshold_down": (
                None if threshold_config["threshold_down"] is None else float(threshold_config["threshold_down"])
            ),
            "threshold_up": None if threshold_config["threshold_up"] is None else float(threshold_config["threshold_up"]),
            "threshold_down_enabled": bool(threshold_config.get("down_enabled", True)),
            "threshold_up_enabled": bool(threshold_config.get("up_enabled", True)),
            "threshold_delta": float(threshold_config.get("delta", 0.0)),
            "threshold_down_quantile": threshold_config.get("down_quantile"),
            "threshold_up_quantile": threshold_config.get("up_quantile"),
            "threshold_down_id": int(threshold_config["down_id"]),
            "threshold_neutral_id": int(threshold_config["neutral_id"]),
            "threshold_up_id": int(threshold_config["up_id"]),
        }
    )


def write_metrics_csv(row: Mapping[str, Any], target: Path) -> None:
    """Write a single-row metrics CSV."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def write_confusion_matrix(metrics: Any, target: Path, split: str, config: ExperimentConfig) -> None:
    """Write raw and normalized confusion matrices."""
    payload = {
        "normalization": "Rows are true classes; columns are predicted classes.",
        "split": split,
        "class_labels": {
            str(class_id): label
            for class_id, label in enumerate(ordered_class_labels(config))
        },
        "raw": metrics.confusion_matrix,
        "normalized_by_true_class": metrics.normalized_confusion_matrix,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def write_expert_usage(expert_usage: Any, target: Path, split: str, config: ExperimentConfig) -> None:
    """Write MoE expert usage for one evaluated split."""
    payload = {
        "description": (
            "MoE routing usage. selected_* counts include every top-k assignment; "
            "primary_* counts include only the first/top expert per token."
        ),
        "split": split,
        "class_labels": {
            str(class_id): label
            for class_id, label in enumerate(ordered_class_labels(config))
        },
        "expert_usage": expert_usage,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def write_evaluation_log(
    *,
    target: Path,
    checkpoint: Path,
    config_path: Path,
    sequence_path: Path,
    output_dir: Path,
    split: str,
    device: str,
    batch_size: int,
    num_samples: int,
    duration_seconds: float,
    directional_thresholds_path: Path | None = None,
) -> None:
    """Write a compact text log for one evaluation run."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        handle.write(f"Created at: {datetime.now().isoformat(timespec='seconds')}\n")
        handle.write(f"Checkpoint: {checkpoint}\n")
        handle.write(f"Config: {config_path}\n")
        handle.write(f"Sequence path: {sequence_path}\n")
        handle.write(f"Output directory: {output_dir}\n")
        handle.write(f"Split: {split}\n")
        handle.write(f"Device: {device}\n")
        handle.write(f"Batch size: {batch_size}\n")
        handle.write(f"Samples: {num_samples}\n")
        if directional_thresholds_path is not None:
            handle.write("Classification mode: directional_thresholds\n")
            handle.write(f"Directional thresholds: {directional_thresholds_path}\n")
        else:
            handle.write("Classification mode: argmax\n")
        handle.write(f"Duration seconds: {duration_seconds:.6f}\n")
        handle.write(f"Duration: {format_duration(duration_seconds)}\n")


def write_evaluation_outputs(
    *,
    result: Any,
    config: ExperimentConfig,
    output_dir: Path,
    split: str,
    num_samples: int,
    checkpoint: Path,
    config_path: Path,
    sequence_path: Path,
    device: str,
    batch_size: int,
    duration_seconds: float,
    save_probabilities: bool,
    directional_thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write metrics and optional probability artifacts."""
    row = evaluation_metrics_row(result, config, split, num_samples)
    add_directional_threshold_fields(row, directional_thresholds)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_csv(row, output_dir / "metrics.csv")
    with (output_dir / "metrics.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(row, handle, sort_keys=False, allow_unicode=True)
    write_confusion_matrix(result.metrics, output_dir / "confusion_matrix.yaml", split, config)
    write_expert_usage(result.expert_usage, output_dir / "expert_usage.yaml", split, config)
    write_evaluation_log(
        target=output_dir / "evaluation.log",
        checkpoint=checkpoint,
        config_path=config_path,
        sequence_path=sequence_path,
        output_dir=output_dir,
        split=split,
        device=device,
        batch_size=batch_size,
        num_samples=num_samples,
        duration_seconds=duration_seconds,
        directional_thresholds_path=None if directional_thresholds is None else Path(str(directional_thresholds["path"])),
    )
    if save_probabilities:
        if result.prediction_outputs is None:
            raise RuntimeError("Probability outputs were not collected.")
        save_probability_outputs(result.prediction_outputs, output_dir / "probabilities.csv", config)
    return row


def main() -> None:
    """Run checkpoint evaluation from the CLI."""
    args = parse_args()
    config = load_config_for_evaluation(args.config, max_dt_override=args.max_dt)
    if args.device is not None:
        config.training.device = args.device.lower()
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be > 0.")
        config.training.eval_batch_size = int(args.batch_size)
    directional_thresholds = (
        None
        if args.directional_thresholds is None
        else load_directional_thresholds(args.directional_thresholds, config)
    )

    dataset, resolved_sequence_path = build_evaluation_dataset(
        args.sequence_dir,
        args.split,
        config.data.sequence_window,
    )
    x_sample, _, _ = dataset[0]
    model = build_model(config.model, d_input=x_sample.shape[-1])

    trainer = LobTrainer(config.training)
    device = torch.device(config.training.device)
    state_dict = load_checkpoint_state_dict(args.checkpoint, device)
    model.load_state_dict(state_dict)
    model = model.to(device)

    loader_kwargs = config.training.data_loader_kwargs()
    data_loader = DataLoader(
        dataset,
        batch_size=config.training.eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    split_name = args.split or resolved_sequence_path.name or "evaluation"
    start = perf_counter()
    result = trainer.evaluate(
        model=model,
        data_loader=data_loader,
        description=f"Evaluate [{split_name}]",
        collect_outputs=args.save_probabilities or directional_thresholds is not None,
        track_pr_metrics=True,
        track_expert_usage=True,
    )
    if directional_thresholds is not None:
        apply_directional_thresholds_to_result(result, config, directional_thresholds)
    duration_seconds = perf_counter() - start

    row = write_evaluation_outputs(
        result=result,
        config=config,
        output_dir=args.output_dir,
        split=split_name,
        num_samples=len(dataset),
        checkpoint=args.checkpoint,
        config_path=args.config,
        sequence_path=resolved_sequence_path,
        device=config.training.device,
        batch_size=config.training.eval_batch_size,
        duration_seconds=duration_seconds,
        save_probabilities=args.save_probabilities,
        directional_thresholds=directional_thresholds,
    )
    print(
        f"Evaluation complete: split={split_name}, loss={row['loss']:.6f}, "
        f"accuracy={row['accuracy']:.4f}, macro_f1={row['macro_f1']:.4f}, "
        f"directional_macro_f1={row['directional_macro_f1']:.4f}, ece={row['ece']:.4f} "
        f"({format_duration(duration_seconds)})."
    )
    print(f"Artifacts saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
