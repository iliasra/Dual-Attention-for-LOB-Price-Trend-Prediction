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
from training import LobTrainer


def resolve_config_path(config: ExperimentConfig, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (config.path.parent / candidate).resolve()


def sequence_paths(sequence_dir: Path, split: str) -> tuple[list[str], list[str], list[str]]:
    split_dir = sequence_dir / split
    x_paths: list[str] = []
    t_paths: list[str] = []
    y_paths: list[str] = []

    for x_path in sorted(split_dir.glob("*_X.npy")):
        prefix = x_path.name.removesuffix("_X.npy")
        t_path = x_path.with_name(f"{prefix}_T.npy")
        y_path = x_path.with_name(f"{prefix}_y.npy")
        if not t_path.exists() or not y_path.exists():
            raise FileNotFoundError(f"Missing matching T/y files for {x_path}")
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
    config.training.best_model_path = str(resolve_config_path(config, config.training.best_model_path))

    train_dataset = build_dataset(sequence_dir, "train", config.data.sequence_window)
    if len(train_dataset) == 0:
        raise ValueError(
            f"No training sequences found in {sequence_dir / 'train'}. "
            "Run scripts/process_data.py first."
        )

    validation_dataset = build_dataset(sequence_dir, "validation", config.data.sequence_window)
    if len(validation_dataset) == 0:
        print("No validation sequences found; using the training dataset for validation.")
        validation_dataset = train_dataset

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

    x_sample, _, _ = train_dataset[0]
    model = build_model(config.model, d_input=x_sample.shape[-1])
    trainer = LobTrainer(config.training)
    _, history = trainer.fit(model, train_loader, validation_loader)

    print(f"Training complete. Best model saved to: {config.training.best_model_path}")
    for epoch_index, result in enumerate(history, start=1):
        print(f"epoch {epoch_index}: train_loss={result.train_loss:.6f}, val_loss={result.val_loss:.6f}")


if __name__ == "__main__":
    main()
