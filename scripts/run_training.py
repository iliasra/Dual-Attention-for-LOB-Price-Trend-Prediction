from __future__ import annotations

import sys
from pathlib import Path

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
)
from training import LobTrainer


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


def main() -> None:
    config = load_config()
    sequence_dir = resolve_config_path(config, config.data.sequence_data_dir)
    logs_dir = resolve_config_path(config, config.data.logs_dir)
    run_stem = next_run_stem(logs_dir)
    run_log_path = logs_dir / f"{run_stem}.log"
    run_config_path = logs_dir / f"{run_stem}_config.yaml"
    run_losses_path = logs_dir / f"{run_stem}_metrics.csv"
    run_confusion_matrices_path = logs_dir / f"{run_stem}_confusion_matrices.yaml"
    config.training.best_model_path = str(resolve_config_path(config, config.training.best_model_path))
    config.training.last_model_path = str(resolve_config_path(config, config.training.last_model_path))
    preprocessing_metadata = load_preprocessing_metadata(sequence_dir)

    train_dataset = build_dataset(sequence_dir, "train", config.data.sequence_window)
    if len(train_dataset) == 0:
        raise ValueError(
            f"No training sequences found in {sequence_dir / 'train'}. "
            "Run scripts/process_data.py first."
        )

    validation_dataset = build_dataset(sequence_dir, "validation", config.data.sequence_window)
    validation_split_dataset = validation_dataset
    if len(validation_dataset) == 0:
        print("No validation sequences found; using the training dataset for validation.")
        validation_dataset = train_dataset

    test_dataset = build_dataset(sequence_dir, "test", config.data.sequence_window)
    test_loader = None
    if len(test_dataset) == 0:
        print("No test sequences found; test loss will not be logged.")

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
    if len(test_dataset) > 0:
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
        "validation": class_distribution(validation_split_dataset, config.model.num_classes),
        "test": class_distribution(test_dataset, config.model.num_classes),
    }
    dataset_sizes = {
        "train": len(train_dataset),
        "validation": len(validation_split_dataset),
        "validation_evaluation": len(validation_dataset),
        "test": len(test_dataset),
    }
    save_run_config_snapshot(
        config,
        run_config_path,
        model_parameters=model_parameters,
        preprocessing_metadata=preprocessing_metadata,
    )
    trainer = LobTrainer(config.training)
    _, history = trainer.fit(model, train_loader, validation_loader, test_loader=test_loader)

    save_epoch_history(history, run_losses_path, config=config)
    save_confusion_matrices(history, run_confusion_matrices_path)
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
    )
    print(f"Training complete. Best model saved to: {config.training.best_model_path}")
    print(f"Last model saved to: {config.training.last_model_path}")
    print(f"Run log saved to: {run_log_path}")
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


if __name__ == "__main__":
    main()
