from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset

try:
    from configuration import load_config
    from datasets import DailySequenceBuilder, LOBDataset
    from horizon import TargetLabelPipeline
    from kinematic_preprocessing import (
        DerivativeNormalizer,
        MessageFeatureProcessor,
        MessageOrderbookJoiner,
        SnapshotBatchProcessor,
        handle_abnormal_prices,
    )
    from model import build_model
    from training import LobTrainer
except ImportError:  # pragma: no cover
    from .configuration import load_config
    from .datasets import DailySequenceBuilder, LOBDataset
    from .horizon import TargetLabelPipeline
    from .kinematic_preprocessing import (
        DerivativeNormalizer,
        MessageFeatureProcessor,
        MessageOrderbookJoiner,
        SnapshotBatchProcessor,
        handle_abnormal_prices,
    )
    from .model import build_model
    from .training import LobTrainer


RAW_ROWS_FOR_SMOKE = 128
DATAFRAME_PREVIEW_ROWS = 8
TRAIN_WINDOWS_LIMIT = 16


def cleanup_smoke_dir(smoke_dir: Path) -> None:
    patterns = [
        "sample_*.npy",
        "lobster_smoke_*.npy",
        "*_preview.csv",
        "normalized_columns.txt",
        "training_summary.yaml",
        "derivatives_stats.yaml",
        "best_lob_transformer.pth",
        "last_lob_transformer.pth",
    ]
    for pattern in patterns:
        for path in smoke_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def find_lobster_pair(data_dir: Path) -> tuple[Path, Path]:
    message_files = sorted(data_dir.glob("*_message_*.csv"))
    if not message_files:
        raise FileNotFoundError(f"No LOBSTER message files found in {data_dir}")

    for message_path in message_files:
        orderbook_name = message_path.name.replace("_message_", "_orderbook_")
        orderbook_path = message_path.with_name(orderbook_name)
        if orderbook_path.exists():
            return message_path, orderbook_path

    raise FileNotFoundError(f"No matching orderbook file found for files in {data_dir}")


def save_preview(df: pd.DataFrame, target: Path, rows: int = DATAFRAME_PREVIEW_ROWS) -> None:
    df.head(rows).to_csv(target, index=False)


def save_columns(columns: list[str], target: Path) -> None:
    target.write_text("\n".join(columns), encoding="utf-8")


def save_yaml(payload: dict, target: Path) -> None:
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def main() -> None:
    smoke_test_dir = Path(__file__).resolve().parent
    config = load_config(smoke_test_dir / "pipeline_smoke_test.yaml")
    smoke_dir = smoke_test_dir / ".smoke"
    config.preprocessing.normalization.derivatives_stats_path = str(smoke_dir / "derivatives_stats.yaml")
    config.training.best_model_path = str(smoke_dir / "best_lob_transformer.pth")
    config.training.last_model_path = str(smoke_dir / "last_lob_transformer.pth")
    smoke_dir.mkdir(parents=True, exist_ok=True)
    cleanup_smoke_dir(smoke_dir)

    lobster_dir = REPO_ROOT / "data" / "LOBSTER"
    message_path, orderbook_path = find_lobster_pair(lobster_dir)

    message_df = pd.read_csv(message_path, nrows=RAW_ROWS_FOR_SMOKE)
    orderbook_df = pd.read_csv(orderbook_path, nrows=RAW_ROWS_FOR_SMOKE)
    handle_abnormal_prices([message_df, orderbook_df])

    save_preview(message_df, smoke_dir / "raw_message_preview.csv")
    save_preview(orderbook_df, smoke_dir / "raw_orderbook_preview.csv")

    joiner = MessageOrderbookJoiner(time_column=config.data.time_column)
    labeler = TargetLabelPipeline(config.preprocessing.labels)
    message_processor = MessageFeatureProcessor(config.data.time_column, config.preprocessing.message)
    snapshot_processor = SnapshotBatchProcessor(config.data, config.preprocessing)
    normalizer = DerivativeNormalizer(smoke_dir / "derivatives_stats.yaml")

    joined = joiner.transform(message_df, orderbook_df)
    labeled = labeler.transform(joined)
    enriched = message_processor.transform(labeled)
    processed = snapshot_processor.transform(enriched)
    normalizer.fit([processed])
    normalized = normalizer.transform(processed)
    assert not normalized.empty, "Snapshot preprocessing returned an empty dataframe."

    save_preview(joined, smoke_dir / "joined_preview.csv")
    save_preview(labeled, smoke_dir / "labeled_preview.csv")
    save_preview(enriched, smoke_dir / "message_features_preview.csv")
    save_preview(processed, smoke_dir / "processed_preview.csv")
    save_preview(normalized, smoke_dir / "normalized_preview.csv")
    normalized.tail(DATAFRAME_PREVIEW_ROWS).to_csv(smoke_dir / "normalized_tail_preview.csv", index=False)
    save_columns(normalized.columns.tolist(), smoke_dir / "normalized_columns.txt")

    dataset_input = normalized.iloc[: max(config.data.sequence_window + 1, 24)].copy()
    dataset_input.to_csv(smoke_dir / "normalized_dataset_input_preview.csv", index=False)

    builder = DailySequenceBuilder(config.data)
    x_path, t_path, y_path = builder.save(dataset_input, smoke_dir / "lobster_smoke")
    dataset = LOBDataset([str(x_path)], [str(t_path)], [str(y_path)], sequence_window=config.data.sequence_window)
    subset_length = min(TRAIN_WINDOWS_LIMIT, len(dataset))
    if subset_length < 2:
        raise ValueError("Not enough sequence windows were generated for the smoke test.")

    split_index = max(1, subset_length // 2)
    train_subset = Subset(dataset, list(range(split_index)))
    val_subset = Subset(dataset, list(range(split_index, subset_length)))
    if len(val_subset) == 0:
        val_subset = Subset(dataset, list(range(subset_length)))

    loader_kwargs = config.training.data_loader_kwargs()
    train_loader = DataLoader(
        train_subset,
        batch_size=config.training.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.training.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    first_batch = next(iter(train_loader))
    x_batch, t_batch, y_batch = first_batch
    model = build_model(config.model, d_input=x_batch.shape[-1])
    logits = model(x_batch, t_batch)
    predictions = torch.argmax(logits, dim=-1)
    assert logits.shape == (x_batch.shape[0], config.model.num_classes)

    trainer = LobTrainer(config.training)
    trained_model, history = trainer.fit(model, train_loader, val_loader)
    best_model_path = Path(config.training.best_model_path)
    last_model_path = Path(config.training.last_model_path)
    assert best_model_path.exists(), "Trainer did not save the best model."
    assert last_model_path.exists(), "Trainer did not save the last model."

    trained_model.eval()
    with torch.no_grad():
        post_train_logits = trained_model(x_batch.to(trainer.device), t_batch.to(trainer.device)).cpu()
        post_train_predictions = torch.argmax(post_train_logits, dim=-1)

    training_summary = {
        "source_files": {
            "message": str(message_path),
            "orderbook": str(orderbook_path),
        },
        "raw_subset_rows": RAW_ROWS_FOR_SMOKE,
        "shapes": {
            "message": list(message_df.shape),
            "orderbook": list(orderbook_df.shape),
            "joined": list(joined.shape),
            "labeled": list(labeled.shape),
            "message_features": list(enriched.shape),
            "processed": list(processed.shape),
            "normalized": list(normalized.shape),
            "dataset_windows": len(dataset),
        },
        "dataset_artifacts": {
            "x_path": str(x_path),
            "t_path": str(t_path),
            "y_path": str(y_path),
        },
        "train_val_sizes": {
            "train_windows": len(train_subset),
            "val_windows": len(val_subset),
        },
        "first_batch": {
            "x_shape": list(x_batch.shape),
            "t_shape": list(t_batch.shape),
            "y_shape": list(y_batch.shape),
            "targets": y_batch.tolist(),
            "pre_train_predictions": predictions.tolist(),
            "post_train_predictions": post_train_predictions.tolist(),
            "post_train_logits": post_train_logits.tolist(),
        },
        "history": [
            {
                "epoch": epoch_index + 1,
                "train_loss": result.train_loss,
                "val_loss": result.val_loss,
                "train_accuracy": None if result.train_metrics is None else result.train_metrics.accuracy,
                "val_accuracy": None if result.val_metrics is None else result.val_metrics.accuracy,
                "val_macro_f1": None if result.val_metrics is None else result.val_metrics.macro_f1,
            }
            for epoch_index, result in enumerate(history)
        ],
        "columns_preview": normalized.columns[:20].tolist(),
    }
    save_yaml(training_summary, smoke_dir / "training_summary.yaml")

    print("Smoke test passed.")
    print(f"Using LOBSTER files: {message_path.name} / {orderbook_path.name}")
    print(f"Normalized dataframe shape: {normalized.shape}")
    print(f"Training artifacts saved in: {smoke_dir}")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
