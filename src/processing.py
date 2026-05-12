from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from configuration import ExperimentConfig, load_config
    from datasets import DailySequenceBuilder
    from fast_kinematic_preprocessing import optimize_smoothing_lambda_gcv
    from horizon import TargetLabelPipeline
    from kinematic_preprocessing import (
        DerivativeNormalizer,
        MessageFeatureProcessor,
        MessageOrderbookJoiner,
        SnapshotBatchProcessor,
        TradingSessionFilter,
        handle_abnormal_prices,
    )
    from run_logging import save_preprocessing_metadata
except ImportError:  # pragma: no cover
    from .configuration import ExperimentConfig, load_config
    from .datasets import DailySequenceBuilder
    from .fast_kinematic_preprocessing import optimize_smoothing_lambda_gcv
    from .horizon import TargetLabelPipeline
    from .kinematic_preprocessing import (
        DerivativeNormalizer,
        MessageFeatureProcessor,
        MessageOrderbookJoiner,
        SnapshotBatchProcessor,
        TradingSessionFilter,
        handle_abnormal_prices,
    )
    from .run_logging import save_preprocessing_metadata


SPLIT_NAMES = ("train", "validation", "test")


@dataclass(slots=True)
class LobFilePair:
    symbol: str
    date: str
    message_path: Path
    orderbook_path: Path

    @property
    def output_stem(self) -> str:
        return f"{self.symbol}_{self.date}"

    @property
    def label(self) -> str:
        return f"{self.symbol} {self.date}"


@dataclass(slots=True)
class ProcessedDay:
    split: str
    pair: LobFilePair
    raw: pd.DataFrame
    joined: pd.DataFrame
    labeled: pd.DataFrame
    message_features: pd.DataFrame
    processed: pd.DataFrame | None = None
    normalized: pd.DataFrame | None = None
    processed_csv_path: Path | None = None
    sequence_paths: tuple[Path, Path, Path] | None = None


class LobProcessingPipeline:
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
        self.fast_smoothing_lambda_results: dict[str, dict[str, float]] = {}

    def _resolve_path(self, path: Path) -> Path:
        return path if path.is_absolute() else (self.config_dir / path).resolve()

    def discover_pairs(self) -> list[LobFilePair]:
        """gathers message/orderbook data by asset and day"""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"LOBSTER directory not found: {self.data_dir}")

        files = list(self.data_dir.iterdir())
        grouped: dict[tuple[str, str], dict[str, Path]] = {}

        for file_path in files:
            parts = file_path.name.split("_")
            if len(parts) < 2:
                continue
            symbol = parts[0]
            date = parts[1]
            key = (symbol, date)
            grouped.setdefault(key, {})
            if "message" in file_path.name:
                grouped[key]["message"] = file_path
            elif "orderbook" in file_path.name:
                grouped[key]["orderbook"] = file_path

        pairs = [
            LobFilePair(symbol=symbol, date=date, message_path=paths["message"], orderbook_path=paths["orderbook"])
            for (symbol, date), paths in grouped.items()
            if {"message", "orderbook"} <= set(paths)
        ]
        return sorted(pairs, key=lambda pair: (pair.symbol, pair.date))

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
        print(f"Loading raw data for {pair.label}.")
        message_df = pd.read_csv(pair.message_path)
        orderbook_df = pd.read_csv(pair.orderbook_path)
        handle_abnormal_prices([message_df, orderbook_df])
        joined = self.joiner.transform(message_df, orderbook_df)
        trimmed = self.session_filter.transform(joined) #session_filter.transform gets rid of first/last 15 mins
        print(f"Loaded and trimmed {pair.label}: {len(trimmed)} rows.")
        return trimmed

    def prepare_pair(self, pair: LobFilePair, split: str) -> ProcessedDay:
        print(f"Starting raw feature preparation for {pair.label} ({split}).")
        trimmed = self.load_and_trim_pair(pair)
        print(f"Starting label generation for {pair.label}.")
        labeled = self.labeler.transform(trimmed)
        print(f"Starting message feature processing for {pair.label}.")
        message_features = self.message_processor.transform(labeled)
        print(f"Finished raw feature preparation for {pair.label}: {message_features.shape[0]} rows.")
        return ProcessedDay(
            split=split,
            pair=pair,
            raw=trimmed,
            joined=trimmed,
            labeled=labeled,
            message_features=message_features,
        )

    def preprocess_pair(self, pair: LobFilePair, split: str) -> ProcessedDay:
        day = self.prepare_pair(pair, split)
        day.processed = self.snapshot_processor.transform(day.message_features, source_label=pair.label)
        print(
            f"Finished preprocessing for {pair.label}: "
            f"{day.processed.shape[0]} rows, {day.processed.shape[1]} columns."
        )
        return day

    def preprocess_splits(self, split_pairs: dict[str, list[LobFilePair]]) -> dict[str, list[ProcessedDay]]:
        processed = {name: [] for name in SPLIT_NAMES}
        for split, pairs in split_pairs.items():
            print(f"Starting preprocessing split '{split}' with {len(pairs)} day(s).")
            for pair in pairs:
                processed[split].append(self.prepare_pair(pair, split))
            print(f"Finished preprocessing split '{split}'.")
        return processed

    def _stream_values_for_lambda_optimization(
        self,
        train_days: list[ProcessedDay],
        *,
        kind: str,
    ) -> list[np.ndarray]:
        if kind not in {"price", "volume"}:
            raise ValueError("kind must be 'price' or 'volume'.")

        stream_config = (
            self.config.preprocessing.price_kinematic
            if kind == "price"
            else self.config.preprocessing.volume_kinematic
        )
        values_by_day: list[np.ndarray] = []
        for day in train_days:
            columns = self.snapshot_processor.window_processor._resolve_stream_columns(
                day.message_features,
                stream_config,
                kind,
            )
            if not columns:
                continue
            values = day.message_features[columns].to_numpy(dtype=float)
            if kind == "volume":
                values = np.log1p(values)
            values_by_day.append(values)
        return values_by_day

    def _optimize_fast_stream_lambda(
        self,
        train_days: list[ProcessedDay],
        *,
        kind: str,
    ) -> None:
        preprocessing = self.config.preprocessing
        stream_config = (
            preprocessing.price_kinematic
            if kind == "price"
            else preprocessing.volume_kinematic
        )
        if not stream_config.enabled:
            return

        window = preprocessing.snapshot_window
        values_by_day = [
            values
            for values in self._stream_values_for_lambda_optimization(train_days, kind=kind)
            if len(values) >= window
        ]
        if not values_by_day:
            return

        fast_config = stream_config.fast
        gcv_chunk_size = min(preprocessing.kinematic_tokenization.chunk_size, 4096)
        print(
            f"Starting fast {kind} smoothing-lambda optimization "
            f"on train windows with max df={fast_config.df}."
        )
        result = optimize_smoothing_lambda_gcv(
            values_by_day=values_by_day,
            window=window,
            n_basis=fast_config.n_basis,
            max_df=fast_config.df,
            chunk_size=gcv_chunk_size,
        )
        fast_config.selected_smoothing_lambda = result.smoothing_lambda
        self.fast_smoothing_lambda_results[kind] = {
            "selected_smoothing_lambda": float(result.smoothing_lambda),
            "effective_df": float(result.effective_df),
            "mean_gcv": float(result.gcv_score),
        }
        print(
            f"Selected fast {kind} smoothing_lambda={result.smoothing_lambda:.8g} "
            f"(effective_df={result.effective_df:.4f}, mean_gcv={result.gcv_score:.8g})."
        )

    def optimize_fast_smoothing_lambdas(self, train_days: list[ProcessedDay]) -> None:
        if self.config.preprocessing.kinematic_tokenization.method != "fast":
            return
        self._optimize_fast_stream_lambda(train_days, kind="price")
        self._optimize_fast_stream_lambda(train_days, kind="volume")
        self.snapshot_processor = SnapshotBatchProcessor(self.config.data, self.config.preprocessing)

    def build_snapshot_features(self, processed_splits: dict[str, list[ProcessedDay]]) -> None:
        print("Starting snapshot feature construction for all splits.")
        for split_days in processed_splits.values():
            for day in split_days:
                day.processed = self.snapshot_processor.transform(day.message_features, source_label=day.pair.label)
                print(
                    f"Finished preprocessing for {day.pair.label}: "
                    f"{day.processed.shape[0]} rows, {day.processed.shape[1]} columns."
                )
        print("Finished snapshot feature construction for all splits.")

    def fit_train_normalizer(self, train_days: list[ProcessedDay]) -> DerivativeNormalizer:
        print(f"Fitting derivative normalizer on {len(train_days)} training day(s).")
        if self.derivatives_stats_path.exists():
            self.derivatives_stats_path.unlink()
            print(f"Removed previous derivative statistics: {self.derivatives_stats_path}")
        processed_train_days = []
        for day in train_days:
            if day.processed is None:
                raise ValueError(f"{day.pair.label} has not been snapshot-processed yet.")
            processed_train_days.append(day.processed)
        normalizer = DerivativeNormalizer(self.derivatives_stats_path)
        normalizer.fit(processed_train_days)
        print("Derivative normalizer fitted.")
        return normalizer

    def apply_normalization(self, processed_splits: dict[str, list[ProcessedDay]], normalizer: DerivativeNormalizer) -> None:
        print("Starting normalization for all splits.")
        for split_days in processed_splits.values():
            for day in split_days:
                if day.processed is None:
                    raise ValueError(f"{day.pair.label} has not been snapshot-processed yet.")
                day.normalized = normalizer.transform(day.processed)
        print("Finished normalization for all splits.")

    def save_split_outputs(self, processed_splits: dict[str, list[ProcessedDay]]) -> None:
        print("Saving processed CSV and sequence outputs.")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.sequence_dir.mkdir(parents=True, exist_ok=True)

        for split, days in processed_splits.items():
            processed_split_dir = self.processed_dir / split
            sequence_split_dir = self.sequence_dir / split
            processed_split_dir.mkdir(parents=True, exist_ok=True)
            sequence_split_dir.mkdir(parents=True, exist_ok=True)

            for day in days:
                if day.normalized is None:
                    raise ValueError(f"{day.pair.label} in split {split} has not been normalized yet.")

                csv_path = processed_split_dir / f"{day.pair.output_stem}_processed.csv"
                day.normalized.to_csv(csv_path, index=False)
                day.processed_csv_path = csv_path

                prefix = sequence_split_dir / day.pair.output_stem
                day.sequence_paths = self.sequence_builder.save(day.normalized, prefix)
                print(f"Saved outputs for {day.pair.label} ({split}).")

    def run(self) -> dict[str, dict[str, tuple[int, int]]]:
        print("Starting LOB processing pipeline.")
        pairs = self.discover_pairs()
        print(f"Discovered {len(pairs)} message/orderbook file pair(s).")
        split_pairs = self.split_pairs(pairs)
        processed_splits = self.preprocess_splits(split_pairs)
        self.optimize_fast_smoothing_lambdas(processed_splits["train"])
        self.build_snapshot_features(processed_splits)
        normalizer = self.fit_train_normalizer(processed_splits["train"])
        self.apply_normalization(processed_splits, normalizer)
        self.save_split_outputs(processed_splits)
        metadata_path = save_preprocessing_metadata(
            self.config,
            self.sequence_dir,
            lambda_results=self.fast_smoothing_lambda_results,
        )
        print(f"Saved preprocessing metadata to {metadata_path}.")
        print("LOB processing pipeline finished.")

        return {
            split: {
                day.pair.output_stem: day.normalized.shape if day.normalized is not None else day.processed.shape
                for day in days
            }
            for split, days in processed_splits.items()
        }
