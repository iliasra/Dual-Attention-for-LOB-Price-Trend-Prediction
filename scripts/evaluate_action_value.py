from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from action_value_training import ActionValueTrainer  # noqa: E402
from configuration import ExperimentConfig  # noqa: E402
from datasets import attach_evaluation_metadata  # noqa: E402
from evaluate_model import load_checkpoint_state_dict  # noqa: E402
from model import build_model  # noqa: E402
from run_training import build_dataset, resolve_model_max_dt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a frozen action-value checkpoint without fitting.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sequence-dir", type=Path, required=True, help="Fold sequence directory with manifest.")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--endpoint-support",
        choices=("common", "exec"),
        default="common",
        help="Use the matched intersection or the natural executable support E.",
    )
    parser.add_argument("--supervision-start-seconds", type=float, default=None)
    parser.add_argument("--supervision-end-seconds", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    if not config.training.objective.is_regression:
        raise ValueError("evaluate_action_value.py requires objective.type=action_value_regression.")
    if args.device is not None:
        config.training.device = str(args.device).lower()
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be > 0.")
        config.training.eval_batch_size = int(args.batch_size)

    override_values = (args.supervision_start_seconds, args.supervision_end_seconds)
    if (override_values[0] is None) != (override_values[1] is None):
        raise ValueError("Set both --supervision-start-seconds and --supervision-end-seconds, or neither.")
    supervision_time_window = (
        None
        if override_values[0] is None
        else (float(override_values[0]), float(override_values[1]))
    )
    dataset = build_dataset(
        args.sequence_dir,
        args.split,
        config,
        preload_to_memory=False,
        supervision_support=args.endpoint_support,
        supervision_time_window=supervision_time_window,
    )
    resolve_model_max_dt(config, dataset)
    sample = dataset[0]
    model = build_model(config.model, d_input=int(sample[0].shape[-1]))
    device = torch.device(config.training.device)
    model.load_state_dict(load_checkpoint_state_dict(args.checkpoint, device))
    model = model.to(device)
    loader_kwargs = config.training.data_loader_kwargs()
    data_loader = DataLoader(
        dataset,
        batch_size=config.training.eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    trainer = ActionValueTrainer(config.training)
    loss, metrics = trainer.evaluate(
        model,
        data_loader,
        description=f"Frozen action values [{args.split}]",
    )
    if trainer.last_evaluation_outputs is None:
        raise RuntimeError("Action-value evaluation produced no outputs.")
    outputs = attach_evaluation_metadata(trainer.last_evaluation_outputs, dataset)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs_path = args.output_dir / f"{args.split}_action_values.npz"
    metrics_path = args.output_dir / f"{args.split}_action_value_metrics.yaml"
    np.savez_compressed(outputs_path, **outputs)
    metrics_path.write_text(
        yaml.safe_dump(
            {
                "objective": "action_value_regression",
                "split": args.split,
                "loss": float(loss),
                "metrics": metrics.to_dict(),
                "outputs": str(outputs_path),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    print(f"Saved frozen action-value outputs to {outputs_path}.")
    print(f"Saved action-value metrics to {metrics_path}.")


if __name__ == "__main__":
    main()
