from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:
    from configuration import ExperimentConfig, load_config
    from datasets import DailySequenceBuilder
    from horizon import TargetLabelPipeline
    from kinematic_preprocessing import (
        DerivativeNormalizer,
        MessageFeatureProcessor,
        MessageOrderbookJoiner,
        SnapshotBatchProcessor,
        TradingSessionFilter,
        handle_abnormal_prices,
    )
except ImportError:  # pragma: no cover
    from .configuration import ExperimentConfig, load_config
    from .datasets import DailySequenceBuilder
    from .horizon import TargetLabelPipeline
    from .kinematic_preprocessing import (
        DerivativeNormalizer,
        MessageFeatureProcessor,
        MessageOrderbookJoiner,
        SnapshotBatchProcessor,
        TradingSessionFilter,
        handle_abnormal_prices,
    )


SPLIT_NAMES = ("train", "validation", "test")


@dataclass(slots=True)
class LobFilePair:
    date: str
    message_path: Path
    orderbook_path: Path


@dataclass(slots=True)
class ProcessedDay:
    split: str
    pair: LobFilePair
    raw: pd.DataFrame
    joined: pd.DataFrame
    labeled: pd.DataFrame
    message_features: pd.DataFrame
    processed: pd.DataFrame
    normalized: pd.DataFrame | None = None
    processed_csv_path: Path | None = None
    sequence_paths: tuple[Path, Path, Path] | None = None


class LobExperimentRunner:
    def __init__(self, config: ExperimentConfig | None = None, data_dir: str | Path | None = None) -> None:
        self.config = config or load_config()
        self.root_dir = Path(__file__).resolve().parent.parent
        self.config_dir = self.config.path.parent
        configured_data_dir = Path(self.config.data.raw_data_dir)
        self.data_dir = Path(data_dir) if data_dir is not None else self._resolve_path(configured_data_dir)
        self.processed_dir = self._resolve_path(Path(self.config.data.processed_data_dir))
        self.sequence_dir = self._resolve_path(Path(self.config.data.sequence_data_dir))
        self.derivatives_stats_path = self._resolve_path(
            Path(self.config.preprocessing.normalization.derivatives_stats_path)
        )

        self.joiner = MessageOrderbookJoiner(
            time_column=self.config.data.time_column,
            method=self.config.preprocessing.join.method,
        ) #joins the message/orderbook datasets for a particular day, including timestamp column
        self.session_filter = TradingSessionFilter(
            time_column=self.config.data.time_column,
            market_open_seconds=self.config.preprocessing.temporal_features.market_open_seconds,
            market_close_seconds=self.config.preprocessing.temporal_features.market_close_seconds,
            start_offset_minutes=self.config.preprocessing.temporal_features.start_offset_minutes,
            end_offset_minutes=self.config.preprocessing.temporal_features.end_offset_minutes,
        ) 
        self.labeler = TargetLabelPipeline(self.config.preprocessing.labels)
        self.message_processor = MessageFeatureProcessor(
            time_column=self.config.data.time_column,
            message_config=self.config.preprocessing.message,
        )
        self.snapshot_processor = SnapshotBatchProcessor(self.config.data, self.config.preprocessing)
        self.sequence_builder = DailySequenceBuilder(self.config.data)

    def _resolve_path(self, path: Path) -> Path:
        return path if path.is_absolute() else (self.config_dir / path).resolve()

    def discover_pairs(self) -> list[LobFilePair]:
        """gathers message/orderbook data by day"""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"LOBSTER directory not found: {self.data_dir}")

        files = list(self.data_dir.iterdir())
        grouped: dict[str, dict[str, Path]] = {}

        for file_path in files:
            parts = file_path.name.split("_")
            if len(parts) < 2:
                continue
            date = parts[1]
            grouped.setdefault(date, {})
            if "message" in file_path.name:
                grouped[date]["message"] = file_path
            elif "orderbook" in file_path.name:
                grouped[date]["orderbook"] = file_path

        pairs = [
            LobFilePair(date=date, message_path=paths["message"], orderbook_path=paths["orderbook"])
            for date, paths in grouped.items()
            if {"message", "orderbook"} <= set(paths)
        ]
        return sorted(pairs, key=lambda pair: pair.date)

    def split_pairs(self, pairs: list[LobFilePair]) -> dict[str, list[LobFilePair]]:
        split_dates = {
            "train": set(self.config.dataset_splits.train_dates),
            "validation": set(self.config.dataset_splits.validation_dates),
            "test": set(self.config.dataset_splits.test_dates),
        }

        overlaps = (
            split_dates["train"] & split_dates["validation"]
            or split_dates["train"] & split_dates["test"]
            or split_dates["validation"] & split_dates["test"]
        )
        if overlaps:
            raise ValueError(f"Some dates are assigned to multiple splits: {sorted(overlaps)}")

        split_pairs = {name: [] for name in SPLIT_NAMES}
        for pair in pairs:
            assigned = [split for split, dates in split_dates.items() if pair.date in dates]
            if len(assigned) > 1:
                raise ValueError(f"Date {pair.date} is assigned to multiple splits: {assigned}")
            if assigned:
                split_pairs[assigned[0]].append(pair)

        missing = [
            pair.date
            for pair in pairs
            if pair.date not in split_dates["train"]
            and pair.date not in split_dates["validation"]
            and pair.date not in split_dates["test"]
        ]
        if missing:
            raise ValueError(
                "Some discovered dates are not assigned to any split in the config: "
                f"{missing}"
            )

        if not split_pairs["train"]:
            raise ValueError("At least one training date must be provided in dataset_splits.train_dates.")

        return split_pairs

    def load_and_trim_pair(self, pair: LobFilePair) -> pd.DataFrame:
        """this function: delete 'ghost' level if placeholder values are recognized;
        joins orderbook/message data for a particular day;
        filters out the first/last 15 minutes."""
        message_df = pd.read_csv(pair.message_path)
        orderbook_df = pd.read_csv(pair.orderbook_path)
        handle_abnormal_prices([message_df, orderbook_df])
        joined = self.joiner.transform(message_df, orderbook_df)
        return self.session_filter.transform(joined) #session_filter.transform gets rid of first/last 15 mins

    def preprocess_pair(self, pair: LobFilePair, split: str) -> ProcessedDay:
        """"""
        trimmed = self.load_and_trim_pair(pair)
        labeled = self.labeler.transform(trimmed)
        message_features = self.message_processor.transform(labeled)
        processed = self.snapshot_processor.transform(message_features)
        return ProcessedDay(
            split=split,
            pair=pair,
            raw=trimmed,
            joined=trimmed,
            labeled=labeled,
            message_features=message_features,
            processed=processed,
        )

    def preprocess_splits(self, split_pairs: dict[str, list[LobFilePair]]) -> dict[str, list[ProcessedDay]]:
        processed = {name: [] for name in SPLIT_NAMES}
        for split, pairs in split_pairs.items():
            for pair in pairs:
                processed[split].append(self.preprocess_pair(pair, split))
        return processed

    def fit_train_normalizer(self, train_days: list[ProcessedDay]) -> DerivativeNormalizer:
        normalizer = DerivativeNormalizer(self.derivatives_stats_path)
        normalizer.fit([day.processed for day in train_days])
        return normalizer

    def apply_normalization(self, processed_splits: dict[str, list[ProcessedDay]], normalizer: DerivativeNormalizer) -> None:
        for split_days in processed_splits.values():
            for day in split_days:
                day.normalized = normalizer.transform(day.processed)

    def save_split_outputs(self, processed_splits: dict[str, list[ProcessedDay]]) -> None:
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.sequence_dir.mkdir(parents=True, exist_ok=True)

        for split, days in processed_splits.items():
            processed_split_dir = self.processed_dir / split
            sequence_split_dir = self.sequence_dir / split
            processed_split_dir.mkdir(parents=True, exist_ok=True)
            sequence_split_dir.mkdir(parents=True, exist_ok=True)

            for day in days:
                if day.normalized is None:
                    raise ValueError(f"Day {day.pair.date} in split {split} has not been normalized yet.")

                csv_path = processed_split_dir / f"{day.pair.date}_processed.csv"
                day.normalized.to_csv(csv_path, index=False)
                day.processed_csv_path = csv_path

                prefix = sequence_split_dir / day.pair.date
                day.sequence_paths = self.sequence_builder.save(day.normalized, prefix)

    def run(self) -> dict[str, dict[str, tuple[int, int]]]:
        pairs = self.discover_pairs()
        split_pairs = self.split_pairs(pairs)
        processed_splits = self.preprocess_splits(split_pairs)
        normalizer = self.fit_train_normalizer(processed_splits["train"])
        self.apply_normalization(processed_splits, normalizer)
        self.save_split_outputs(processed_splits)

        return {
            split: {
                day.pair.date: day.normalized.shape if day.normalized is not None else day.processed.shape
                for day in days
            }
            for split, days in processed_splits.items()
        }