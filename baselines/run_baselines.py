from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from action_value import action_value_metrics
from baselines.models import BaselineHead, LSTMBaseline, context_features, momentum_signal
from training import classification_metrics_from_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run compute-cheap LOB baselines on prepared shards.")
    parser.add_argument("--sequence-dir", type=Path, required=True, help="One prepared fold directory.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=(
            "no_skill",
            "momentum",
            "momentum_ma",
            "label_persistence",
            "linear",
            "mlp",
            "lstm",
            "random_forest",
            "xgboost",
        ),
        default="linear",
    )
    parser.add_argument("--context", choices=("last", "last_mean"), default="last")
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--lstm-steps", type=int, default=32, help="Causal time points sampled from each input window.")
    parser.add_argument("--lstm-dropout", type=float, default=0.0)
    parser.add_argument("--momentum-feature-index", type=int, default=0)
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
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-subsample", type=float, default=0.8)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    parser.add_argument("--max-rows", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed-rate", type=float, default=0.005)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_split(
    fold_dir: Path,
    split: str,
    *,
    window: int,
    context: str,
    max_rows: int,
    seed: int,
    sequence_steps: int | None = None,
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
        if len(day_features) < window:
            continue
        candidate_count = len(day_features) - window + 1
        day_indices = np.arange(candidate_count, dtype=np.int64)
        if per_day_cap > 0 and candidate_count > per_day_cap:
            rng = np.random.default_rng(seed + day_index)
            day_indices = np.sort(rng.choice(candidate_count, size=per_day_cap, replace=False))
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


def standardize(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    reduce_axes = tuple(range(train.ndim - 1))
    mean = train.mean(axis=reduce_axes, dtype=np.float64)
    scale = train.std(axis=reduce_axes, dtype=np.float64)
    scale[scale < 1e-8] = 1.0
    return tuple(((array - mean) / scale).astype(np.float32) for array in (train, *others))


def load_momentum_split(
    fold_dir: Path,
    split: str,
    *,
    window: int,
    mode: str,
    feature_index: int,
    lookback: int,
    short_window: int,
    long_window: int,
    max_rows: int,
    seed: int,
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
        if len(features) < window:
            continue
        day_signal = momentum_signal(
            features,
            window=window,
            feature_index=feature_index,
            mode=mode,
            lookback=lookback,
            short_window=short_window,
            long_window=long_window,
        )
        day_output_targets = np.asarray(day_targets[window - 1 :]).copy()
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
        start = max(window - 1, lag)
        if len(day_targets) <= start:
            continue
        indices = np.arange(start, len(day_targets), dtype=np.int64)
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
    *,
    regression: bool,
    args: argparse.Namespace,
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
                from sklearn.multioutput import MultiOutputRegressor

                inner_common = {**common, "n_jobs": 1}
                model = MultiOutputRegressor(
                    XGBRegressor(objective="reg:squarederror", **inner_common),
                    n_jobs=args.n_jobs,
                )
            elif regression:
                model = XGBRegressor(objective="reg:squarederror", **common)
            else:
                model = XGBClassifier(objective="multi:softprob", **common)
    except ImportError as exc:
        package = "scikit-learn" if name == "random_forest" else "xgboost and scikit-learn"
        raise RuntimeError(f"{name} requires {package}; install the project requirements first.") from exc

    model.fit(train_x, train_y)
    if regression:
        return np.asarray(model.predict(validation_x), dtype=np.float32), None
    probabilities = np.asarray(model.predict_proba(validation_x), dtype=np.float32)
    return np.argmax(probabilities, axis=1).astype(np.int64), probabilities


def train_head(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    *,
    regression: bool,
    args: argparse.Namespace,
) -> None:
    device = torch.device(args.device)
    model.to(device)
    target_tensor = torch.from_numpy(y.astype(np.float32 if regression else np.int64, copy=False))
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), target_tensor),
        batch_size=args.batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    for _epoch in range(args.epochs):
        model.train()
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = F.huber_loss(outputs, targets) if regression else F.cross_entropy(outputs, targets.long())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


@torch.inference_mode()
def predict(model: torch.nn.Module, inputs: np.ndarray, *, batch_size: int, device: str) -> np.ndarray:
    model.eval()
    chunks = []
    for (batch,) in DataLoader(TensorDataset(torch.from_numpy(inputs)), batch_size=batch_size):
        chunks.append(model(batch.to(device)).float().cpu().numpy())
    return np.concatenate(chunks)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.model == "lstm" and (args.max_rows <= 0 or args.max_rows > 50_000):
        raise ValueError(
            "The LSTM materializes sampled sequences. Set --max-rows to a positive value <= 50000 "
            "and increase it only after measuring host RAM."
        )
    sequence_steps = args.lstm_steps if args.model == "lstm" else None
    train_x, train_y = load_split(
        args.sequence_dir,
        "train",
        window=args.window,
        context=args.context,
        max_rows=args.max_rows,
        seed=args.seed,
        sequence_steps=sequence_steps,
    )
    validation_x, validation_y = load_split(
        args.sequence_dir,
        "validation",
        window=args.window,
        context=args.context,
        max_rows=args.max_rows,
        seed=args.seed + 1,
        sequence_steps=sequence_steps,
    )
    train_x, validation_x = standardize(train_x, validation_x)
    regression = train_y.ndim == 2
    output_dim = 2 if regression else int(max(np.max(train_y), np.max(validation_y)) + 1)
    probabilities: np.ndarray | None = None

    if args.model == "no_skill":
        if regression:
            predictions = np.repeat(train_y.mean(axis=0, keepdims=True), len(validation_y), axis=0)
        else:
            majority = int(np.bincount(train_y.astype(np.int64)).argmax())
            predictions = np.full(len(validation_y), majority, dtype=np.int64)
    elif args.model in {"momentum", "momentum_ma"}:
        momentum_mode = "difference" if args.model == "momentum" else "ma_crossover"
        train_signal, momentum_train_y = load_momentum_split(
            args.sequence_dir,
            "train",
            window=args.window,
            mode=momentum_mode,
            feature_index=args.momentum_feature_index,
            lookback=args.momentum_lookback,
            short_window=args.momentum_short_window,
            long_window=args.momentum_long_window,
            max_rows=args.max_rows,
            seed=args.seed,
        )
        validation_signal, momentum_validation_y = load_momentum_split(
            args.sequence_dir,
            "validation",
            window=args.window,
            mode=momentum_mode,
            feature_index=args.momentum_feature_index,
            lookback=args.momentum_lookback,
            short_window=args.momentum_short_window,
            long_window=args.momentum_long_window,
            max_rows=args.max_rows,
            seed=args.seed + 1,
        )
        train_y = momentum_train_y
        validation_y = momentum_validation_y
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
        if args.label_lag is None:
            raise ValueError(
                "label_persistence requires --label-lag. Set it to at least the full label horizon "
                "so y[t-lag] is observable at t."
            )
        predictions, validation_y = load_delayed_target_split(
            args.sequence_dir,
            "validation",
            window=args.window,
            lag=args.label_lag,
            max_rows=args.max_rows,
            seed=args.seed + 1,
        )
    elif args.model in {"random_forest", "xgboost"}:
        predictions, probabilities = fit_classical_model(
            args.model,
            train_x,
            train_y,
            validation_x,
            regression=regression,
            args=args,
        )
    else:
        if args.model == "lstm":
            model = LSTMBaseline(
                train_x.shape[-1],
                output_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.lstm_layers,
                dropout=args.lstm_dropout,
            )
        else:
            hidden_dim = args.hidden_dim if args.model == "mlp" else None
            model = BaselineHead(train_x.shape[1], output_dim, hidden_dim=hidden_dim)
        train_head(model, train_x, train_y, regression=regression, args=args)
        raw_predictions = predict(model, validation_x, batch_size=args.batch_size, device=args.device)
        if regression:
            predictions = raw_predictions
        else:
            shifted = raw_predictions - raw_predictions.max(axis=1, keepdims=True)
            exp = np.exp(shifted)
            probabilities = exp / exp.sum(axis=1, keepdims=True)
            predictions = np.argmax(probabilities, axis=1)

    if regression:
        metrics = action_value_metrics(predictions, validation_y, fixed_rate=args.fixed_rate).to_dict()
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
        "task": "action_value_regression" if regression else "classification",
        "train_rows": int(len(train_y)),
        "validation_rows": int(len(validation_y)),
        "parameters": {
            "window": int(args.window),
            "max_rows": int(args.max_rows),
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "hidden_dim": int(args.hidden_dim),
            "momentum_feature_index": int(args.momentum_feature_index),
            "momentum_lookback": int(args.momentum_lookback),
            "momentum_short_window": int(args.momentum_short_window),
            "momentum_long_window": int(args.momentum_long_window),
            "momentum_neutral_quantile": args.momentum_neutral_quantile,
            "up_class": args.up_class,
            "neutral_class": args.neutral_class,
            "down_class": args.down_class,
            "label_lag": args.label_lag,
            "lstm_steps": int(args.lstm_steps),
            "lstm_layers": int(args.lstm_layers),
            "n_estimators": int(args.n_estimators),
            "max_depth": int(args.max_depth),
        },
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
