from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from action_value import action_value_metrics, action_value_policy_frontier
from baselines.artifacts import load_baseline_artifact
from baselines.models import BaselineHead
from baselines.run_baselines import load_split, predict
from training import classification_metrics_from_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference-only evaluation of a frozen MLP/XGBoost baseline artifact."
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test", "holdout"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path, default=None)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--fixed-rate", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def infer_frozen_baseline(
    artifact: dict[str, object], inputs: np.ndarray, *, batch_size: int, device: str
) -> tuple[np.ndarray, np.ndarray | None]:
    mean = np.asarray(artifact["standardizer_mean"], dtype=np.float64)
    scale = np.asarray(artifact["standardizer_scale"], dtype=np.float64)
    standardized = ((inputs - mean) / scale).astype(np.float32)
    regression = artifact["task"] == "action_value_regression"
    if artifact["model"] == "xgboost":
        estimator = artifact["estimator"]
        if isinstance(estimator, list):
            raw = np.column_stack([model.predict(standardized) for model in estimator]).astype(np.float32)
            return raw, None
        if regression:
            return np.asarray(estimator.predict(standardized), dtype=np.float32), None
        probabilities = np.asarray(estimator.predict_proba(standardized), dtype=np.float32)
        return np.argmax(probabilities, axis=1).astype(np.int64), probabilities
    if artifact["model"] != "mlp":
        raise ValueError(f"Unsupported frozen baseline model: {artifact['model']!r}.")
    model = BaselineHead(
        int(artifact["input_dim"]),
        int(artifact["output_dim"]),
        hidden_dim=artifact.get("hidden_dim"),
        hidden_layers=int(artifact["hidden_layers"]),
        dropout=float(artifact["dropout"]),
    )
    model.load_state_dict(artifact["state_dict"])
    model.to(device)
    raw = predict(model, standardized, batch_size=batch_size, device=device)
    if regression:
        return raw, None
    shifted = raw - raw.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return np.argmax(probabilities, axis=1).astype(np.int64), probabilities.astype(np.float32)


def main() -> None:
    args = parse_args()
    if args.max_eval_rows < 0:
        raise ValueError("max_eval_rows must be >= 0.")
    artifact = load_baseline_artifact(args.artifact)
    inputs, targets = load_split(
        args.sequence_dir,
        args.split,
        window=int(artifact["window"]),
        context=str(artifact["context"]),
        max_rows=args.max_eval_rows,
        seed=args.seed,
        supervision_support=str(artifact.get("endpoint_support", "common")),
    )
    predictions, probabilities = infer_frozen_baseline(
        artifact, inputs, batch_size=args.batch_size, device=args.device
    )
    regression = artifact["task"] == "action_value_regression"
    if regression:
        metrics = action_value_metrics(predictions, targets, fixed_rate=args.fixed_rate).to_dict()
        frontier = action_value_policy_frontier(predictions, targets)
    else:
        metrics = asdict(
            classification_metrics_from_predictions(
                targets,
                predictions,
                num_classes=int(artifact["output_dim"]),
                probabilities=probabilities,
                directional_precision_fixed_rate=(
                    args.fixed_rate if probabilities is not None and int(artifact["output_dim"]) >= 3 else None
                ),
            )
        )
        frontier = None
    payload = {
        "inference_only": True,
        "artifact": str(args.artifact),
        "model": artifact["model"],
        "task": artifact["task"],
        "split": args.split,
        "rows": int(len(targets)),
        "metrics": metrics,
        "ranking_pnl_frontier": frontier,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.predictions_output is not None:
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"predictions": predictions, "targets": targets}
        if probabilities is not None:
            arrays["probabilities"] = probabilities
        np.savez_compressed(args.predictions_output, **arrays)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
