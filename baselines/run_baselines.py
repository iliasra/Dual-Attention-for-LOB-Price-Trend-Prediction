from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from action_value import action_value_metrics, action_value_policy_frontier
from baselines.artifacts import save_baseline_artifact
from baselines.models import (
    BaselineHead,
    LSTMBaseline,
    RecurrentBaseline,
    context_features,
    momentum_signal,
    resolve_mlp_hidden_dim,
)
from configuration import WandbTrackingConfig, load_config
from datasets import supervision_mask_path
from run_logging import timestamped_run_stem
from training import classification_metrics_from_predictions
from wandb_tracking import WandbTracker


def flatten_numeric_metrics(prefix: str, value: object) -> dict[str, float]:
    """Flatten nested metric payloads while excluding non-finite values."""
    if isinstance(value, dict):
        flattened: dict[str, float] = {}
        for key, item in value.items():
            child = f"{prefix}_{key}" if prefix else str(key)
            flattened.update(flatten_numeric_metrics(child, item))
        return flattened
    if isinstance(value, (list, tuple, np.ndarray)):
        flattened = {}
        for index, item in enumerate(value):
            flattened.update(flatten_numeric_metrics(f"{prefix}_{index}", item))
        return flattened
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        return {prefix: numeric} if np.isfinite(numeric) else {}
    return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run compute-cheap LOB baselines on prepared shards.")
    parser.add_argument("--sequence-dir", type=Path, required=True, help="One prepared fold directory.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-output",
        type=Path,
        default=None,
        help="Optional fitted MLP/XGBoost artifact for later inference-only test/holdout evaluation.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional experiment YAML supplying tracking.wandb.")
    parser.add_argument("--run-stem", default=None, help="W&B group shared by a baseline comparison.")
    parser.add_argument(
        "--evaluation-split",
        choices=("validation", "test"),
        default="validation",
        help="Evaluate the train-fitted baseline on validation or on the frozen test split.",
    )
    parser.add_argument(
        "--model",
        choices=(
            "no_skill",
            "time_of_day",
            "momentum",
            "momentum_ma",
            "label_persistence",
            "linear",
            "mlp",
            "lstm",
            "rnn",
            "gru",
            "random_forest",
            "xgboost",
        ),
        default="linear",
    )
    parser.add_argument("--context", choices=("last", "last_mean"), default="last")
    parser.add_argument(
        "--endpoint-support",
        choices=("common", "broad", "exec"),
        default="common",
        help="Endpoint mask used for fitted/evaluated samples; context rows remain untouched.",
    )
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-classes", type=int, default=3, help="Configured class count; never inferred from held-out labels.")
    parser.add_argument("--regression-loss", choices=("huber", "mse"), default="huber")
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--mlp-layers", type=int, default=1)
    parser.add_argument("--mlp-dropout", type=float, default=0.0)
    parser.add_argument(
        "--target-parameters",
        type=int,
        default=None,
        help="For --model mlp, override hidden-dim with the closest equal-width parameter budget.",
    )
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--lstm-steps", type=int, default=32, help="Causal time points sampled from each input window.")
    parser.add_argument("--lstm-dropout", type=float, default=0.0)
    parser.add_argument(
        "--feature-schema",
        type=Path,
        default=None,
        help="feature_schema.yaml; defaults to <sequence-dir>/feature_schema.yaml.",
    )
    parser.add_argument(
        "--momentum-feature-name",
        action="append",
        default=None,
        help="Causal feature name resolved from the schema. Repeat twice for midpoint MA crossover.",
    )
    parser.add_argument(
        "--momentum-feature-index",
        type=int,
        default=None,
        help="Deprecated explicit positional override; prefer --momentum-feature-name.",
    )
    parser.add_argument("--momentum-lookback", type=int, default=10)
    parser.add_argument("--momentum-short-window", type=int, default=5)
    parser.add_argument("--momentum-long-window", type=int, default=20)
    parser.add_argument(
        "--momentum-neutral-quantile",
        default="auto",
        help="Train absolute-signal quantile classified neutral, or 'auto' for the train neutral-class share.",
    )
    parser.add_argument("--up-class", type=int, default=None)
    parser.add_argument("--neutral-class", type=int, default=None)
    parser.add_argument("--down-class", type=int, default=None)
    parser.add_argument(
        "--label-lag",
        type=int,
        default=None,
        help="Availability lag for label_persistence; must be at least the label horizon.",
    )
    parser.add_argument(
        "--label-horizon",
        type=int,
        default=None,
        help="Required for label_persistence; label_lag must be >= this horizon.",
    )
    parser.add_argument("--persistence-laplace-alpha", type=float, default=1.0)
    parser.add_argument("--time-bin-minutes", type=float, default=15.0)
    parser.add_argument("--time-laplace-alpha", type=float, default=1.0)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-subsample", type=float, default=0.8)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Deprecated compatibility alias: applies the same cap to train and evaluation unless overridden.",
    )
    parser.add_argument("--max-train-rows", type=int, default=None, help="Train-row cap; default 200000, 0 = all.")
    parser.add_argument("--max-eval-rows", type=int, default=None, help="Held-out row cap; default 0 = all.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed-rate", type=float, default=0.005)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def resolve_row_caps(args: argparse.Namespace) -> tuple[int, int]:
    """Resolve new split-specific caps while preserving the legacy --max-rows behavior."""
    legacy_cap = getattr(args, "max_rows", None)
    train_cap = getattr(args, "max_train_rows", None)
    eval_cap = getattr(args, "max_eval_rows", None)
    if legacy_cap is not None:
        train_cap = legacy_cap if train_cap is None else train_cap
        eval_cap = legacy_cap if eval_cap is None else eval_cap
    else:
        train_cap = 200_000 if train_cap is None else train_cap
        eval_cap = 0 if eval_cap is None else eval_cap
    train_cap = int(train_cap)
    eval_cap = int(eval_cap)
    if train_cap < 0 or eval_cap < 0:
        raise ValueError("max_train_rows and max_eval_rows must be >= 0.")
    args.max_train_rows = train_cap
    args.max_eval_rows = eval_cap
    return train_cap, eval_cap


def load_endpoint_mask(target_path: Path, row_count: int, *, support: str) -> np.ndarray:
    mask_path = supervision_mask_path(target_path, support=support)
    if not mask_path.exists():
        if support == "common":
            return np.ones(row_count, dtype=bool)
        raise FileNotFoundError(
            f"The requested {support!r} supervision mask is absent for {target_path.name}."
        )
    mask = np.asarray(np.load(mask_path, mmap_mode="r"), dtype=bool)
    if mask.ndim != 1 or len(mask) != row_count:
        raise ValueError(f"Invalid supervision mask shape in {mask_path}; expected [{row_count}].")
    return mask


def load_split(
    fold_dir: Path,
    split: str,
    *,
    window: int,
    context: str,
    max_rows: int,
    seed: int,
    sequence_steps: int | None = None,
    supervision_support: str = "common",
) -> tuple[np.ndarray, np.ndarray]:
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    split_dir = fold_dir / split
    feature_paths = sorted(split_dir.glob("*_features.npy"))
    per_day_cap = 0 if max_rows <= 0 else max(1, int(np.ceil(max_rows / max(len(feature_paths), 1))))
    for day_index, feature_path in enumerate(feature_paths):
        stem = feature_path.name.removesuffix("_features.npy")
        target_path = feature_path.with_name(f"{stem}_labels.npy")
        if not target_path.exists():
            raise FileNotFoundError(f"Missing target shard for {feature_path.name}.")
        day_features = np.load(feature_path, mmap_mode="r")
        day_targets = np.load(target_path, mmap_mode="r")
        endpoint_mask = load_endpoint_mask(target_path, len(day_targets), support=supervision_support)
        if len(day_features) < window:
            continue
        candidate_count = len(day_features) - window + 1
        day_indices = np.flatnonzero(endpoint_mask[window - 1 :]).astype(np.int64)
        if per_day_cap > 0 and len(day_indices) > per_day_cap:
            rng = np.random.default_rng(seed + day_index)
            day_indices = np.sort(rng.choice(day_indices, size=per_day_cap, replace=False))
        if not len(day_indices):
            continue
        if sequence_steps is None:
            day_inputs = context_features(day_features, window=window, mode=context)[day_indices]
        else:
            if not 2 <= sequence_steps <= window:
                raise ValueError("lstm_steps must be in [2, window].")
            offsets = np.rint(np.linspace(-(window - 1), 0, num=sequence_steps)).astype(np.int64)
            end_indices = day_indices + window - 1
            day_inputs = np.asarray(day_features[end_indices[:, None] + offsets[None, :]], dtype=np.float32).copy()
        day_output_targets = np.asarray(day_targets[window - 1 :])[day_indices].copy()
        inputs.append(day_inputs)
        targets.append(day_output_targets)
    if not inputs:
        raise ValueError(f"No usable {split} shards under {split_dir}.")
    x = np.concatenate(inputs).astype(np.float32, copy=False)
    y = np.concatenate(targets)
    if max_rows > 0 and len(x) > max_rows:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(x), size=max_rows, replace=False))
        x = x[indices]
        y = y[indices]
    return x, y


def load_target_split(
    fold_dir: Path,
    split: str,
    *,
    window: int,
    max_rows: int,
    seed: int,
    supervision_support: str = "common",
) -> np.ndarray:
    """Load targets aligned with causal windows without materializing unused features."""
    targets: list[np.ndarray] = []
    feature_paths = sorted((fold_dir / split).glob("*_features.npy"))
    per_day_cap = 0 if max_rows <= 0 else max(1, int(np.ceil(max_rows / max(len(feature_paths), 1))))
    for day_index, feature_path in enumerate(feature_paths):
        stem = feature_path.name.removesuffix("_features.npy")
        target_path = feature_path.with_name(f"{stem}_labels.npy")
        if not target_path.exists():
            raise FileNotFoundError(f"Missing target shard for {feature_path.name}.")
        features = np.load(feature_path, mmap_mode="r")
        day_targets = np.load(target_path, mmap_mode="r")
        endpoint_mask = load_endpoint_mask(target_path, len(day_targets), support=supervision_support)
        if len(features) != len(day_targets):
            raise ValueError(f"Feature/target row mismatch for {stem}.")
        if len(day_targets) < window:
            continue
        day_output_targets = np.asarray(day_targets[window - 1 :])[endpoint_mask[window - 1 :]].copy()
        if not len(day_output_targets):
            continue
        if per_day_cap > 0 and len(day_output_targets) > per_day_cap:
            rng = np.random.default_rng(seed + day_index)
            indices = np.sort(rng.choice(len(day_output_targets), size=per_day_cap, replace=False))
            day_output_targets = day_output_targets[indices]
        targets.append(day_output_targets)
    if not targets:
        raise ValueError(f"No usable {split} targets under {fold_dir / split}.")
    target = np.concatenate(targets)
    if max_rows > 0 and len(target) > max_rows:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(target), size=max_rows, replace=False))
        target = target[indices]
    return target


def load_time_target_split(
    fold_dir: Path,
    split: str,
    *,
    window: int,
    max_rows: int,
    seed: int,
    supervision_support: str = "common",
) -> tuple[np.ndarray, np.ndarray]:
    """Load aligned time-of-day seconds and targets without crossing day boundaries."""
    times: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    feature_paths = sorted((fold_dir / split).glob("*_features.npy"))
    per_day_cap = 0 if max_rows <= 0 else max(1, int(np.ceil(max_rows / max(len(feature_paths), 1))))
    for day_index, feature_path in enumerate(feature_paths):
        stem = feature_path.name.removesuffix("_features.npy")
        time_path = feature_path.with_name(f"{stem}_times.npy")
        target_path = feature_path.with_name(f"{stem}_labels.npy")
        missing = [path.name for path in (time_path, target_path) if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing aligned shard(s) for {stem}: {missing}.")
        features = np.load(feature_path, mmap_mode="r")
        day_times = np.load(time_path, mmap_mode="r")
        day_targets = np.load(target_path, mmap_mode="r")
        endpoint_mask = load_endpoint_mask(target_path, len(day_targets), support=supervision_support)
        if len(features) != len(day_times) or len(features) != len(day_targets):
            raise ValueError(f"Feature/time/target row mismatch for {stem}.")
        if len(day_targets) < window:
            continue
        active = endpoint_mask[window - 1 :]
        day_output_times = np.asarray(day_times[window - 1 :], dtype=np.float64)[active].copy()
        day_output_targets = np.asarray(day_targets[window - 1 :])[active].copy()
        if not len(day_output_targets):
            continue
        if per_day_cap > 0 and len(day_output_targets) > per_day_cap:
            rng = np.random.default_rng(seed + day_index)
            indices = np.sort(rng.choice(len(day_output_targets), size=per_day_cap, replace=False))
            day_output_times = day_output_times[indices]
            day_output_targets = day_output_targets[indices]
        times.append(day_output_times)
        targets.append(day_output_targets)
    if not targets:
        raise ValueError(f"No usable {split} time/target shards under {fold_dir / split}.")
    time = np.concatenate(times)
    target = np.concatenate(targets)
    if max_rows > 0 and len(target) > max_rows:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(target), size=max_rows, replace=False))
        time = time[indices]
        target = target[indices]
    return time, target


def standardize(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    standardized, _mean, _scale = standardize_with_stats(train, *others)
    return standardized


def standardize_with_stats(
    train: np.ndarray, *others: np.ndarray
) -> tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    reduce_axes = tuple(range(train.ndim - 1))
    mean = train.mean(axis=reduce_axes, dtype=np.float64)
    scale = train.std(axis=reduce_axes, dtype=np.float64)
    scale[scale < 1e-8] = 1.0
    arrays = tuple(((array - mean) / scale).astype(np.float32) for array in (train, *others))
    return arrays, np.asarray(mean), np.asarray(scale)


def resolve_feature_indices(
    sequence_dir: Path,
    *,
    feature_names: list[str] | None,
    feature_index: int | None,
    feature_schema: Path | None,
) -> tuple[int, ...]:
    """Resolve causal baseline inputs by persisted feature name, with an explicit legacy escape hatch."""
    if feature_names and feature_index is not None:
        raise ValueError("Use --momentum-feature-name or --momentum-feature-index, not both.")
    if feature_names:
        schema_path = feature_schema or sequence_dir / "feature_schema.yaml"
        if not schema_path.exists():
            raise FileNotFoundError(f"Feature schema not found: {schema_path}")
        payload = yaml.safe_load(schema_path.read_text(encoding="utf-8")) or {}
        columns = payload.get("ordered_feature_columns")
        if not isinstance(columns, list) or not all(isinstance(name, str) for name in columns):
            raise ValueError(f"Invalid ordered_feature_columns in {schema_path}.")
        missing = [name for name in feature_names if name not in columns]
        if missing:
            raise ValueError(f"Feature(s) {missing} absent from {schema_path}.")
        return tuple(columns.index(name) for name in feature_names)
    if feature_index is None:
        raise ValueError(
            "Momentum baselines require --momentum-feature-name resolved from feature_schema.yaml. "
            "The positional default was removed to prevent accidental leakage or momentum-of-momentum."
        )
    return (int(feature_index),)


def load_momentum_split(
    fold_dir: Path,
    split: str,
    *,
    window: int,
    mode: str,
    feature_indices: tuple[int, ...],
    lookback: int,
    short_window: int,
    long_window: int,
    max_rows: int,
    seed: int,
    supervision_support: str = "common",
) -> tuple[np.ndarray, np.ndarray]:
    """Load one causal scalar momentum signal and its aligned targets."""
    signals: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    feature_paths = sorted((fold_dir / split).glob("*_features.npy"))
    per_day_cap = 0 if max_rows <= 0 else max(1, int(np.ceil(max_rows / max(len(feature_paths), 1))))
    for day_index, feature_path in enumerate(feature_paths):
        stem = feature_path.name.removesuffix("_features.npy")
        target_path = feature_path.with_name(f"{stem}_labels.npy")
        if not target_path.exists():
            raise FileNotFoundError(f"Missing target shard for {feature_path.name}.")
        features = np.load(feature_path, mmap_mode="r")
        day_targets = np.load(target_path, mmap_mode="r")
        endpoint_mask = load_endpoint_mask(target_path, len(day_targets), support=supervision_support)
        if len(features) < window:
            continue
        if len(feature_indices) == 1:
            signal_source = np.asarray(features[:, feature_indices[0]], dtype=np.float32)[:, None]
        elif len(feature_indices) == 2:
            signal_source = np.mean(
                np.asarray(features[:, list(feature_indices)], dtype=np.float64), axis=1, keepdims=True
            ).astype(np.float32)
        else:
            raise ValueError("Momentum baselines accept one feature, or two bid/ask features for a midpoint.")
        day_signal = momentum_signal(
            signal_source,
            window=window,
            feature_index=0,
            mode=mode,
            lookback=lookback,
            short_window=short_window,
            long_window=long_window,
        )
        active = endpoint_mask[window - 1 :]
        day_signal = day_signal[active]
        day_output_targets = np.asarray(day_targets[window - 1 :])[active].copy()
        if not len(day_output_targets):
            continue
        if per_day_cap > 0 and len(day_signal) > per_day_cap:
            rng = np.random.default_rng(seed + day_index)
            indices = np.sort(rng.choice(len(day_signal), size=per_day_cap, replace=False))
            day_signal = day_signal[indices]
            day_output_targets = day_output_targets[indices]
        signals.append(day_signal)
        targets.append(day_output_targets)
    if not signals:
        raise ValueError(f"No usable {split} shards for the momentum baseline.")
    signal = np.concatenate(signals).astype(np.float32, copy=False)
    target = np.concatenate(targets)
    if max_rows > 0 and len(signal) > max_rows:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(signal), size=max_rows, replace=False))
        signal = signal[indices]
        target = target[indices]
    return signal, target


def load_delayed_target_split(
    fold_dir: Path,
    split: str,
    *,
    window: int,
    lag: int,
    max_rows: int,
    seed: int,
    supervision_support: str = "common",
) -> tuple[np.ndarray, np.ndarray]:
    """Return y[t-lag] and y[t], preserving day boundaries."""
    if lag <= 0:
        raise ValueError("label_lag must be positive and at least the economic label horizon.")
    delayed: list[np.ndarray] = []
    actual: list[np.ndarray] = []
    feature_paths = sorted((fold_dir / split).glob("*_features.npy"))
    per_day_cap = 0 if max_rows <= 0 else max(1, int(np.ceil(max_rows / max(len(feature_paths), 1))))
    for day_index, feature_path in enumerate(feature_paths):
        stem = feature_path.name.removesuffix("_features.npy")
        target_path = feature_path.with_name(f"{stem}_labels.npy")
        if not target_path.exists():
            raise FileNotFoundError(f"Missing target shard for {feature_path.name}.")
        day_targets = np.asarray(np.load(target_path, mmap_mode="r"))
        endpoint_mask = load_endpoint_mask(target_path, len(day_targets), support=supervision_support)
        start = max(window - 1, lag)
        if len(day_targets) <= start:
            continue
        indices = np.arange(start, len(day_targets), dtype=np.int64)
        indices = indices[endpoint_mask[indices] & endpoint_mask[indices - lag]]
        if not len(indices):
            continue
        if per_day_cap > 0 and len(indices) > per_day_cap:
            rng = np.random.default_rng(seed + day_index)
            indices = np.sort(rng.choice(indices, size=per_day_cap, replace=False))
        delayed.append(day_targets[indices - lag].copy())
        actual.append(day_targets[indices].copy())
    if not delayed:
        raise ValueError(f"No usable {split} targets for label_persistence.")
    predictions = np.concatenate(delayed)
    targets = np.concatenate(actual)
    if max_rows > 0 and len(predictions) > max_rows:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(predictions), size=max_rows, replace=False))
        predictions = predictions[indices]
        targets = targets[indices]
    return predictions, targets


def no_skill_predictions(
    train_targets: np.ndarray,
    *,
    evaluation_rows: int,
    output_dim: int,
) -> tuple[np.ndarray, np.ndarray | None, object]:
    """Return an unconditional train-only mean or prevalence predictor."""
    targets = np.asarray(train_targets)
    if evaluation_rows < 0:
        raise ValueError("evaluation_rows must be >= 0.")
    if targets.ndim == 2:
        if len(targets) == 0:
            raise ValueError("No-skill regression requires at least one train target.")
        mean = np.mean(targets, axis=0, dtype=np.float64).astype(np.float32)
        return np.repeat(mean[None, :], evaluation_rows, axis=0), None
    if targets.ndim != 1:
        raise ValueError("No-skill targets must be one-dimensional classes or two-dimensional values.")
    integer_targets = targets.astype(np.int64, copy=False)
    valid = (integer_targets >= 0) & (integer_targets < output_dim)
    if not bool(valid.any()):
        raise ValueError("No-skill classification found no valid train target.")
    counts = np.bincount(integer_targets[valid], minlength=output_dim).astype(np.float64)
    prevalence = (counts / counts.sum()).astype(np.float32)
    probabilities = np.repeat(prevalence[None, :], evaluation_rows, axis=0)
    predictions = np.full(evaluation_rows, int(np.argmax(prevalence)), dtype=np.int64)
    return predictions, probabilities


def time_of_day_predictions(
    train_times: np.ndarray,
    train_targets: np.ndarray,
    evaluation_times: np.ndarray,
    *,
    bin_minutes: float,
    laplace_alpha: float,
    output_dim: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Fit a train-only 15-minute-style prior and predict held-out rows."""
    if bin_minutes <= 0.0:
        raise ValueError("time_bin_minutes must be > 0.")
    if laplace_alpha <= 0.0:
        raise ValueError("time_laplace_alpha must be > 0.")
    train_time = np.asarray(train_times, dtype=np.float64).reshape(-1)
    evaluation_time = np.asarray(evaluation_times, dtype=np.float64).reshape(-1)
    targets = np.asarray(train_targets)
    if len(train_time) != len(targets):
        raise ValueError("train_times and train_targets must have the same length.")
    bin_seconds = float(bin_minutes) * 60.0

    def bin_id(seconds: float) -> int:
        return int(np.floor(np.mod(seconds, 86_400.0) / bin_seconds))

    if targets.ndim == 2:
        finite_target = np.isfinite(targets).all(axis=1)
        if not bool(finite_target.any()):
            raise ValueError("Time-of-day regression found no finite train target.")
        global_mean = np.mean(targets[finite_target], axis=0, dtype=np.float64)
        sums: dict[int, np.ndarray] = {}
        counts: dict[int, int] = {}
        for index in np.flatnonzero(np.isfinite(train_time) & finite_target):
            current_bin = bin_id(train_time[index])
            sums[current_bin] = sums.get(current_bin, np.zeros(targets.shape[1], dtype=np.float64)) + targets[index]
            counts[current_bin] = counts.get(current_bin, 0) + 1
        predictions = np.repeat(global_mean[None, :], len(evaluation_time), axis=0)
        for index in np.flatnonzero(np.isfinite(evaluation_time)):
            current_bin = bin_id(evaluation_time[index])
            if current_bin in counts:
                predictions[index] = sums[current_bin] / float(counts[current_bin])
        return predictions.astype(np.float32), None

    if targets.ndim != 1:
        raise ValueError("Time-of-day targets must be one-dimensional classes or two-dimensional values.")
    integer_targets = targets.astype(np.int64, copy=False)
    valid_class = (integer_targets >= 0) & (integer_targets < output_dim)
    valid_bin = np.isfinite(train_time) & valid_class
    bin_counts: dict[int, np.ndarray] = {}
    for index in np.flatnonzero(valid_bin):
        current_bin = bin_id(train_time[index])
        counts = bin_counts.setdefault(current_bin, np.zeros(output_dim, dtype=np.float64))
        counts[integer_targets[index]] += 1.0
    global_counts = np.bincount(integer_targets[valid_class], minlength=output_dim).astype(np.float64)
    global_probabilities = (global_counts + laplace_alpha) / (
        global_counts.sum() + laplace_alpha * output_dim
    )
    probabilities = np.repeat(global_probabilities[None, :], len(evaluation_time), axis=0)
    for index in np.flatnonzero(np.isfinite(evaluation_time)):
        current_bin = bin_id(evaluation_time[index])
        counts = bin_counts.get(current_bin)
        if counts is not None:
            probabilities[index] = (counts + laplace_alpha) / (
                counts.sum() + laplace_alpha * output_dim
            )
    return np.argmax(probabilities, axis=1).astype(np.int64), probabilities.astype(np.float32)


def label_persistence_predictions(
    train_delayed_targets: np.ndarray,
    train_targets: np.ndarray,
    evaluation_delayed_targets: np.ndarray,
    *,
    output_dim: int,
    laplace_alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit P(y[t] | y[t-lag]) on train and score held-out delayed labels."""
    if laplace_alpha <= 0.0:
        raise ValueError("persistence_laplace_alpha must be > 0.")
    delayed = np.asarray(train_delayed_targets, dtype=np.int64).reshape(-1)
    targets = np.asarray(train_targets, dtype=np.int64).reshape(-1)
    evaluation_delayed = np.asarray(evaluation_delayed_targets, dtype=np.int64).reshape(-1)
    if len(delayed) != len(targets):
        raise ValueError("Delayed and current train targets must have the same length.")
    valid = (
        (delayed >= 0)
        & (delayed < output_dim)
        & (targets >= 0)
        & (targets < output_dim)
    )
    transition_counts = np.full((output_dim, output_dim), float(laplace_alpha), dtype=np.float64)
    np.add.at(transition_counts, (delayed[valid], targets[valid]), 1.0)
    transition_probabilities = transition_counts / transition_counts.sum(axis=1, keepdims=True)
    global_counts = np.bincount(targets[(targets >= 0) & (targets < output_dim)], minlength=output_dim).astype(np.float64)
    global_probabilities = (global_counts + laplace_alpha) / (
        global_counts.sum() + laplace_alpha * output_dim
    )
    probabilities = np.repeat(global_probabilities[None, :], len(evaluation_delayed), axis=0)
    valid_evaluation = (evaluation_delayed >= 0) & (evaluation_delayed < output_dim)
    probabilities[valid_evaluation] = transition_probabilities[evaluation_delayed[valid_evaluation]]
    return np.argmax(probabilities, axis=1).astype(np.int64), probabilities.astype(np.float32)


def momentum_predictions(
    train_signal: np.ndarray,
    train_targets: np.ndarray,
    validation_signal: np.ndarray,
    *,
    neutral_quantile: str,
    up_class: int,
    neutral_class: int,
    down_class: int,
    output_dim: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Fit only a train threshold/calibration and predict from signed momentum."""
    finite_train = np.isfinite(train_signal)
    if not bool(finite_train.any()):
        raise ValueError("Momentum signal contains no finite train value.")
    train_signal = np.nan_to_num(train_signal, nan=0.0, posinf=0.0, neginf=0.0)
    validation_signal = np.nan_to_num(validation_signal, nan=0.0, posinf=0.0, neginf=0.0)
    if train_targets.ndim == 2:
        design = np.column_stack([np.ones(len(train_signal)), train_signal]).astype(np.float64)
        coefficients, *_ = np.linalg.lstsq(design, train_targets.astype(np.float64), rcond=None)
        validation_design = np.column_stack([np.ones(len(validation_signal)), validation_signal])
        return (validation_design @ coefficients).astype(np.float32), None

    class_ids = (up_class, neutral_class, down_class)
    if len(set(class_ids)) != 3 or min(class_ids) < 0 or max(class_ids) >= output_dim:
        raise ValueError("up_class, neutral_class and down_class must be distinct valid class IDs.")
    if neutral_quantile.strip().lower() == "auto":
        quantile = float(np.mean(train_targets.astype(np.int64) == neutral_class))
    else:
        quantile = float(neutral_quantile)
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("momentum_neutral_quantile must be in [0, 1] or 'auto'.")
    threshold = float(np.quantile(np.abs(train_signal[finite_train]), quantile))
    predictions = np.full(len(validation_signal), neutral_class, dtype=np.int64)
    predictions[validation_signal > threshold] = up_class
    predictions[validation_signal < -threshold] = down_class

    scale = float(np.median(np.abs(train_signal[finite_train])))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = float(np.std(train_signal[finite_train]))
    scale = max(scale, 1e-8)
    normalized = validation_signal / scale
    logits = np.zeros((len(validation_signal), output_dim), dtype=np.float64)
    logits[:, up_class] = normalized
    logits[:, down_class] = -normalized
    logits[:, neutral_class] = (threshold - np.abs(validation_signal)) / scale
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return predictions, probabilities.astype(np.float32)


def fit_classical_model(
    name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    regression: bool,
    args: argparse.Namespace,
    wandb_tracker: WandbTracker,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Fit a lazy-imported scikit-learn or XGBoost baseline."""
    if train_x.ndim != 2:
        raise ValueError(f"{name} expects a flattened last or last_mean context.")
    try:
        if name == "random_forest":
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

            common = {
                "n_estimators": args.n_estimators,
                "max_depth": None if args.max_depth <= 0 else args.max_depth,
                "min_samples_leaf": args.min_samples_leaf,
                "n_jobs": args.n_jobs,
                "random_state": args.seed,
            }
            if regression:
                model = RandomForestRegressor(**common)
            else:
                model = RandomForestClassifier(class_weight="balanced_subsample", **common)
        else:
            from xgboost import XGBClassifier, XGBRegressor

            common = {
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "learning_rate": args.xgb_learning_rate,
                "subsample": args.xgb_subsample,
                "colsample_bytree": args.xgb_colsample_bytree,
                "n_jobs": args.n_jobs,
                "random_state": args.seed,
                "tree_method": "hist",
            }
            if regression and train_y.ndim == 2:
                model = None  # fitted action-wise below to expose every boosting round
            elif regression:
                model = XGBRegressor(objective="reg:squarederror", **common)
            else:
                model = XGBClassifier(objective="multi:softprob", **common)
    except ImportError as exc:
        package = "scikit-learn" if name == "random_forest" else "xgboost and scikit-learn"
        raise RuntimeError(f"{name} requires {package}; install the project requirements first.") from exc

    if name == "xgboost" and regression and train_y.ndim == 2:
        action_predictions = []
        action_models = []
        for action_index, action_name in enumerate(("long", "short")):
            action_model = XGBRegressor(objective="reg:squarederror", **{**common, "n_jobs": 1})
            action_model.fit(
                train_x,
                train_y[:, action_index],
                eval_set=[
                    (train_x, train_y[:, action_index]),
                    (validation_x, validation_y[:, action_index]),
                ],
                verbose=False,
            )
            results = action_model.evals_result()
            datasets = list(results)
            metric_names = list(results[datasets[0]]) if datasets else []
            rounds = len(results[datasets[0]][metric_names[0]]) if datasets and metric_names else 0
            for round_index in range(rounds):
                payload: dict[str, object] = {"global_step": round_index + 1, "epoch": round_index + 1}
                for dataset_name, namespace in zip(datasets, ("train", "validation")):
                    for metric_name, values in results[dataset_name].items():
                        payload[f"{namespace}/xgboost_{action_name}_{metric_name}"] = float(values[round_index])
                if wandb_tracker.log_training_steps_enabled:
                    wandb_tracker.log_metrics(payload)
            action_predictions.append(action_model.predict(validation_x))
            action_models.append(action_model)
        return np.column_stack(action_predictions).astype(np.float32), None, action_models
    if name == "xgboost":
        model.fit(
            train_x,
            train_y,
            eval_set=[(train_x, train_y), (validation_x, validation_y)],
            verbose=False,
        )
        results = model.evals_result()
        datasets = list(results)
        metric_names = list(results[datasets[0]]) if datasets else []
        rounds = len(results[datasets[0]][metric_names[0]]) if datasets and metric_names else 0
        for round_index in range(rounds):
            payload: dict[str, object] = {"global_step": round_index + 1, "epoch": round_index + 1}
            for dataset_name, namespace in zip(datasets, ("train", "validation")):
                for metric_name, values in results[dataset_name].items():
                    payload[f"{namespace}/xgboost_{metric_name}"] = float(values[round_index])
            if wandb_tracker.log_training_steps_enabled:
                wandb_tracker.log_metrics(payload)
    else:
        if model is None:  # pragma: no cover - guarded by the action-wise XGBoost return above.
            raise RuntimeError("Internal baseline model construction failed.")
        model.fit(train_x, train_y)
    if model is None:  # pragma: no cover - defensive type narrowing.
        raise RuntimeError("Internal baseline model construction failed.")
    if regression:
        return np.asarray(model.predict(validation_x), dtype=np.float32), None, model
    probabilities = np.asarray(model.predict_proba(validation_x), dtype=np.float32)
    return np.argmax(probabilities, axis=1).astype(np.int64), probabilities, model


def baseline_regression_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    loss_name: str,
    huber_delta: float,
) -> torch.Tensor:
    """Apply the configured neural action-value loss."""
    normalized = str(loss_name).strip().lower()
    if normalized == "huber":
        if huber_delta <= 0.0:
            raise ValueError("huber_delta must be > 0.")
        return F.huber_loss(outputs, targets, delta=float(huber_delta))
    if normalized == "mse":
        return F.mse_loss(outputs, targets)
    raise ValueError("regression_loss must be 'huber' or 'mse'.")


def train_head(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    *,
    regression: bool,
    args: argparse.Namespace,
    wandb_tracker: WandbTracker,
) -> int:
    device = torch.device(args.device)
    model.to(device)
    target_tensor = torch.from_numpy(y.astype(np.float32 if regression else np.int64, copy=False))
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), target_tensor),
        batch_size=args.batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_in_epoch, (inputs, targets) in enumerate(loader, start=1):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = (
                baseline_regression_loss(
                    outputs,
                    targets,
                    loss_name=getattr(args, "regression_loss", "huber"),
                    huber_delta=float(getattr(args, "huber_delta", 1.0)),
                )
                if regression
                else F.cross_entropy(outputs, targets.long())
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            global_step += 1
            wandb_tracker.log_training_step(
                {
                    "epoch": epoch,
                    "batch_in_epoch": batch_in_epoch,
                    "global_step": global_step,
                    "optimizer_step": global_step,
                    "train_loss_step": float(loss.detach().item()),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "optimizer_step_completed": True,
                    "optimizer_step_applied": True,
                    "supervised_tokens_per_step": int(targets.shape[0]),
                    "chunks_per_step": int(inputs.shape[0]),
                }
            )
    return global_step


@torch.inference_mode()
def predict(model: torch.nn.Module, inputs: np.ndarray, *, batch_size: int, device: str) -> np.ndarray:
    model.eval()
    chunks = []
    for (batch,) in DataLoader(TensorDataset(torch.from_numpy(inputs)), batch_size=batch_size):
        chunks.append(model(batch.to(device)).float().cpu().numpy())
    return np.concatenate(chunks)


def infer_baseline_task(
    train_targets: np.ndarray,
    evaluation_targets: np.ndarray,
    *,
    num_classes: int,
) -> tuple[bool, int]:
    """Validate target rank and return (is_action_value_regression, output_dim)."""
    train = np.asarray(train_targets)
    evaluation = np.asarray(evaluation_targets)
    if train.ndim != evaluation.ndim:
        raise ValueError("Train and evaluation targets must have the same rank.")
    if train.ndim == 2:
        if train.shape[1] != 2 or evaluation.shape[1] != 2:
            raise ValueError("Action-value targets must have shape [rows, 2].")
        return True, 2
    if train.ndim != 1:
        raise ValueError("Targets must be 1D classes or 2D long/short action values.")
    if num_classes <= 1:
        raise ValueError("num_classes must be > 1 for classification.")
    for split_name, values in (("train", train), ("evaluation", evaluation)):
        if not np.isfinite(values).all() or not np.equal(values, values.astype(np.int64)).all():
            raise ValueError(f"{split_name} classification targets must contain finite integer class IDs.")
        class_ids = values.astype(np.int64, copy=False)
        if np.any(class_ids < 0) or np.any(class_ids >= num_classes):
            raise ValueError(
                f"{split_name} classification target is outside configured [0, {num_classes}) classes."
            )
    return False, int(num_classes)


def main() -> None:
    args = parse_args()
    max_train_rows, max_eval_rows = resolve_row_caps(args)
    if args.target_parameters is not None and args.model != "mlp":
        raise ValueError("target_parameters is supported only with --model mlp.")
    started = perf_counter()
    torch.manual_seed(args.seed)
    wandb_config = (
        load_config(args.config).tracking.wandb
        if args.config is not None
        else WandbTrackingConfig(enabled=False)
    )
    run_stem = args.run_stem or timestamped_run_stem(f"baseline-{args.model}")
    tracker_fold_id = f"{args.sequence_dir.name}-{args.model}-{args.evaluation_split}"
    wandb_tracker = WandbTracker.init(
        wandb_config,
        run_stem=run_stem,
        fold_id=tracker_fold_id,
        fold_log_dir=args.output.parent,
        config_payload={
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    )
    recurrent_models = {"lstm", "rnn", "gru"}
    if args.model in recurrent_models and (max_train_rows <= 0 or max_train_rows > 50_000):
        raise ValueError(
            "Recurrent baselines materialize sampled train sequences. Set --max-train-rows to a positive "
            "value <= 50000 "
            "and increase it only after measuring host RAM."
        )
    probabilities: np.ndarray | None = None
    final_global_step = 0
    resolved_hidden_dim: int | None = None
    model_parameter_count: int | None = None
    trainable_parameter_count: int | None = None
    standardizer_mean: np.ndarray | None = None
    standardizer_scale: np.ndarray | None = None
    fitted_estimator: object | None = None
    fitted_torch_model: torch.nn.Module | None = None

    if args.model == "no_skill":
        train_y = load_target_split(
            args.sequence_dir,
            "train",
            window=args.window,
            max_rows=max_train_rows,
            seed=args.seed,
            supervision_support=args.endpoint_support,
        )
        validation_y = load_target_split(
            args.sequence_dir,
            args.evaluation_split,
            window=args.window,
            max_rows=max_eval_rows,
            seed=args.seed + 1,
            supervision_support=args.endpoint_support,
        )
        regression, output_dim = infer_baseline_task(train_y, validation_y, num_classes=args.num_classes)
        predictions, probabilities = no_skill_predictions(
            train_y,
            evaluation_rows=len(validation_y),
            output_dim=output_dim,
        )
    elif args.model == "time_of_day":
        train_times, train_y = load_time_target_split(
            args.sequence_dir,
            "train",
            window=args.window,
            max_rows=max_train_rows,
            seed=args.seed,
            supervision_support=args.endpoint_support,
        )
        validation_times, validation_y = load_time_target_split(
            args.sequence_dir,
            args.evaluation_split,
            window=args.window,
            max_rows=max_eval_rows,
            seed=args.seed + 1,
            supervision_support=args.endpoint_support,
        )
        regression, output_dim = infer_baseline_task(train_y, validation_y, num_classes=args.num_classes)
        predictions, probabilities = time_of_day_predictions(
            train_times,
            train_y,
            validation_times,
            bin_minutes=args.time_bin_minutes,
            laplace_alpha=args.time_laplace_alpha,
            output_dim=output_dim,
        )
    elif args.model in {"momentum", "momentum_ma"}:
        feature_indices = resolve_feature_indices(
            args.sequence_dir,
            feature_names=args.momentum_feature_name,
            feature_index=args.momentum_feature_index,
            feature_schema=args.feature_schema,
        )
        if args.model == "momentum" and len(feature_indices) != 1:
            raise ValueError("momentum expects exactly one precomputed causal momentum feature.")
        if args.model == "momentum_ma" and len(feature_indices) != 2:
            raise ValueError("momentum_ma expects bid_price_1 and ask_price_1 to construct a causal midpoint.")
        momentum_mode = "direct" if args.model == "momentum" else "ma_crossover"
        train_signal, momentum_train_y = load_momentum_split(
            args.sequence_dir,
            "train",
            window=args.window,
            mode=momentum_mode,
            feature_indices=feature_indices,
            lookback=args.momentum_lookback,
            short_window=args.momentum_short_window,
            long_window=args.momentum_long_window,
            max_rows=max_train_rows,
            seed=args.seed,
            supervision_support=args.endpoint_support,
        )
        validation_signal, momentum_validation_y = load_momentum_split(
            args.sequence_dir,
            args.evaluation_split,
            window=args.window,
            mode=momentum_mode,
            feature_indices=feature_indices,
            lookback=args.momentum_lookback,
            short_window=args.momentum_short_window,
            long_window=args.momentum_long_window,
            max_rows=max_eval_rows,
            seed=args.seed + 1,
            supervision_support=args.endpoint_support,
        )
        train_y = momentum_train_y
        validation_y = momentum_validation_y
        regression, output_dim = infer_baseline_task(train_y, validation_y, num_classes=args.num_classes)
        if not regression and None in (args.up_class, args.neutral_class, args.down_class):
            raise ValueError(
                "Momentum classification requires --up-class, --neutral-class and --down-class; "
                "their encoding differs between INTC and FI-2010."
            )
        predictions, probabilities = momentum_predictions(
            train_signal,
            train_y,
            validation_signal,
            neutral_quantile=args.momentum_neutral_quantile,
            up_class=0 if regression else int(args.up_class),
            neutral_class=1 if regression else int(args.neutral_class),
            down_class=2 if regression else int(args.down_class),
            output_dim=output_dim,
        )
    elif args.model == "label_persistence":
        if args.label_lag is None or args.label_horizon is None:
            raise ValueError(
                "label_persistence requires --label-lag and --label-horizon. Set lag to at least the full horizon "
                "so y[t-lag] is observable at t."
            )
        if args.label_lag < args.label_horizon:
            raise ValueError("label_lag must be greater than or equal to label_horizon.")
        train_delayed_y, train_y = load_delayed_target_split(
            args.sequence_dir,
            "train",
            window=args.window,
            lag=args.label_lag,
            max_rows=max_train_rows,
            seed=args.seed,
            supervision_support=args.endpoint_support,
        )
        validation_delayed_y, validation_y = load_delayed_target_split(
            args.sequence_dir,
            args.evaluation_split,
            window=args.window,
            lag=args.label_lag,
            max_rows=max_eval_rows,
            seed=args.seed + 1,
            supervision_support=args.endpoint_support,
        )
        regression, output_dim = infer_baseline_task(train_y, validation_y, num_classes=args.num_classes)
        if regression:
            predictions = validation_delayed_y
        else:
            predictions, probabilities = label_persistence_predictions(
                train_delayed_y,
                train_y,
                validation_delayed_y,
                output_dim=output_dim,
                laplace_alpha=args.persistence_laplace_alpha,
            )
    else:
        sequence_steps = args.lstm_steps if args.model in recurrent_models else None
        train_x, train_y = load_split(
            args.sequence_dir,
            "train",
            window=args.window,
            context=args.context,
            max_rows=max_train_rows,
            seed=args.seed,
            sequence_steps=sequence_steps,
            supervision_support=args.endpoint_support,
        )
        validation_x, validation_y = load_split(
            args.sequence_dir,
            args.evaluation_split,
            window=args.window,
            context=args.context,
            max_rows=max_eval_rows,
            seed=args.seed + 1,
            sequence_steps=sequence_steps,
            supervision_support=args.endpoint_support,
        )
        (train_x, validation_x), standardizer_mean, standardizer_scale = standardize_with_stats(
            train_x, validation_x
        )
        regression, output_dim = infer_baseline_task(train_y, validation_y, num_classes=args.num_classes)

        if args.model in {"random_forest", "xgboost"}:
            predictions, probabilities, fitted_estimator = fit_classical_model(
                args.model,
                train_x,
                train_y,
                validation_x,
                validation_y,
                regression=regression,
                args=args,
                wandb_tracker=wandb_tracker,
            )
            if args.model == "xgboost":
                final_global_step = int(args.n_estimators)
        else:
            if args.model == "lstm":
                model = LSTMBaseline(
                    train_x.shape[-1],
                    output_dim,
                    hidden_dim=args.hidden_dim,
                    num_layers=args.lstm_layers,
                    dropout=args.lstm_dropout,
                )
            elif args.model in {"rnn", "gru"}:
                model = RecurrentBaseline(
                    train_x.shape[-1],
                    output_dim,
                    cell=args.model,
                    hidden_dim=args.hidden_dim,
                    num_layers=args.lstm_layers,
                    dropout=args.lstm_dropout,
                )
            else:
                if args.model == "mlp":
                    resolved_hidden_dim = (
                        resolve_mlp_hidden_dim(
                            train_x.shape[1],
                            output_dim,
                            hidden_layers=args.mlp_layers,
                            target_parameters=args.target_parameters,
                        )
                        if args.target_parameters is not None
                        else int(args.hidden_dim)
                    )
                model = BaselineHead(
                    train_x.shape[1],
                    output_dim,
                    hidden_dim=resolved_hidden_dim,
                    hidden_layers=args.mlp_layers,
                    dropout=args.mlp_dropout,
                )
            model_parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
            trainable_parameter_count = int(
                sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            )
            final_global_step = train_head(
                model,
                train_x,
                train_y,
                regression=regression,
                args=args,
                wandb_tracker=wandb_tracker,
            )
            fitted_torch_model = model
            raw_predictions = predict(model, validation_x, batch_size=args.batch_size, device=args.device)
            if regression:
                predictions = raw_predictions
            else:
                shifted = raw_predictions - raw_predictions.max(axis=1, keepdims=True)
                exp = np.exp(shifted)
                probabilities = exp / exp.sum(axis=1, keepdims=True)
                predictions = np.argmax(probabilities, axis=1)

    ranking_pnl_frontier = None
    if regression:
        metrics = action_value_metrics(predictions, validation_y, fixed_rate=args.fixed_rate).to_dict()
        ranking_pnl_frontier = action_value_policy_frontier(predictions, validation_y)
    else:
        metrics = asdict(classification_metrics_from_predictions(
            validation_y,
            predictions,
            num_classes=output_dim,
            probabilities=probabilities,
            directional_precision_fixed_rate=args.fixed_rate if probabilities is not None and output_dim >= 3 else None,
        ))

    payload = {
        "model": args.model,
        "context": args.context,
        "endpoint_support": args.endpoint_support,
        "task": "action_value_regression" if regression else "classification",
        "evaluation_split": args.evaluation_split,
        "train_rows": int(len(train_y)),
        "evaluation_rows": int(len(validation_y)),
        # Legacy key retained for consumers written before evaluation_split was configurable.
        "validation_rows": int(len(validation_y)),
        "parameters": {
            "window": int(args.window),
            "max_rows": None if args.max_rows is None else int(args.max_rows),
            "max_train_rows": int(max_train_rows),
            "max_eval_rows": int(max_eval_rows),
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "num_classes": int(args.num_classes),
            "regression_loss": args.regression_loss,
            "huber_delta": float(args.huber_delta),
            "hidden_dim": int(args.hidden_dim),
            "resolved_hidden_dim": resolved_hidden_dim,
            "mlp_layers": int(args.mlp_layers),
            "mlp_dropout": float(args.mlp_dropout),
            "target_parameters": args.target_parameters,
            "actual_parameter_count": model_parameter_count,
            "momentum_feature_index": args.momentum_feature_index,
            "momentum_feature_name": args.momentum_feature_name,
            "feature_schema": None if args.feature_schema is None else str(args.feature_schema),
            "momentum_lookback": int(args.momentum_lookback),
            "momentum_short_window": int(args.momentum_short_window),
            "momentum_long_window": int(args.momentum_long_window),
            "momentum_neutral_quantile": args.momentum_neutral_quantile,
            "up_class": args.up_class,
            "neutral_class": args.neutral_class,
            "down_class": args.down_class,
            "label_lag": args.label_lag,
            "label_horizon": args.label_horizon,
            "persistence_laplace_alpha": float(args.persistence_laplace_alpha),
            "time_bin_minutes": float(args.time_bin_minutes),
            "time_laplace_alpha": float(args.time_laplace_alpha),
            "lstm_steps": int(args.lstm_steps),
            "lstm_layers": int(args.lstm_layers),
            "n_estimators": int(args.n_estimators),
            "max_depth": int(args.max_depth),
        },
        "model_parameters": {
            "total": model_parameter_count,
            "trainable": trainable_parameter_count,
        },
        "metrics": metrics,
        "ranking_pnl_frontier": ranking_pnl_frontier,
        "duration_seconds": float(perf_counter() - started),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.model_output is not None:
        if args.model not in {"mlp", "xgboost"}:
            raise ValueError("--model-output currently supports the campaign ML baselines: mlp and xgboost.")
        if standardizer_mean is None or standardizer_scale is None:
            raise RuntimeError("Cannot save an ML artifact without fitted train-only normalization statistics.")
        artifact: dict[str, object] = {
            "model": args.model,
            "task": "action_value_regression" if regression else "classification",
            "output_dim": int(output_dim),
            "window": int(args.window),
            "context": args.context,
            "endpoint_support": args.endpoint_support,
            "input_dim": int(train_x.shape[-1]),
            "standardizer_mean": standardizer_mean,
            "standardizer_scale": standardizer_scale,
            "training_evaluation_split": args.evaluation_split,
        }
        if args.model == "xgboost":
            if fitted_estimator is None:
                raise RuntimeError("Missing fitted XGBoost estimator.")
            artifact["estimator"] = fitted_estimator
        else:
            if fitted_torch_model is None:
                raise RuntimeError("Missing fitted MLP model.")
            artifact.update(
                {
                    "state_dict": {
                        name: tensor.detach().cpu() for name, tensor in fitted_torch_model.state_dict().items()
                    },
                    "hidden_dim": resolved_hidden_dim,
                    "hidden_layers": int(args.mlp_layers),
                    "dropout": float(args.mlp_dropout),
                }
            )
        save_baseline_artifact(args.model_output, artifact)
    final_metrics: dict[str, object] = {"global_step": int(final_global_step)}
    for key, value in flatten_numeric_metrics(args.evaluation_split, metrics).items():
        final_metrics[f"selected/{key}"] = value
    if model_parameter_count is not None:
        final_metrics["selected/model_parameters_total"] = model_parameter_count
        final_metrics["selected/model_parameters_trainable"] = trainable_parameter_count
    final_metrics["selected/duration_seconds"] = float(payload["duration_seconds"])
    wandb_tracker.log_metrics(final_metrics)
    wandb_tracker.log_artifact_files(
        name=f"{run_stem}-{tracker_fold_id}-result",
        artifact_type="baseline-result",
        paths=[args.output] + ([args.model_output] if args.model_output is not None else []),
    )
    wandb_tracker.finish(exit_code=0)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
