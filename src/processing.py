from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from configuration import ExperimentConfig, FoldConfig, load_config
    from datasets import DailySequenceBuilder
    from fast_kinematic_preprocessing import optimize_smoothing_lambda_gcv
    from horizon import (
        SmoothingMethodC,
        TargetLabelPipeline,
        calculate_adaptive_method_c_threshold_components,
        calculate_midprice,
    )
    from kinematic_preprocessing import (
        DerivativeNormalizer,
        MessageFeatureProcessor,
        MessageOrderbookJoiner,
        SnapshotBatchProcessor,
        TradingSessionFilter,
        fit_exp_scaling_parameters,
        fit_plgs_parameters,
        handle_abnormal_prices,
        price_static_distance_frame,
    )
    from run_logging import save_preprocessing_metadata
except ImportError:  # pragma: no cover
    from .configuration import ExperimentConfig, FoldConfig, load_config
    from .datasets import DailySequenceBuilder
    from .fast_kinematic_preprocessing import optimize_smoothing_lambda_gcv
    from .horizon import (
        SmoothingMethodC,
        TargetLabelPipeline,
        calculate_adaptive_method_c_threshold_components,
        calculate_midprice,
    )
    from .kinematic_preprocessing import (
        DerivativeNormalizer,
        MessageFeatureProcessor,
        MessageOrderbookJoiner,
        SnapshotBatchProcessor,
        TradingSessionFilter,
        fit_exp_scaling_parameters,
        fit_plgs_parameters,
        handle_abnormal_prices,
        price_static_distance_frame,
    )
    from .run_logging import save_preprocessing_metadata


SPLIT_NAMES = ("train", "validation", "test")
DERIVATIVES_STATS_FILENAME = "derivatives_stats.yaml"


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
        self.derivatives_stats_dir = self._resolve_path(
            Path(self.config.preprocessing.normalization.derivatives_stats_dir)
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
        self.price_static_plgs_results: dict[str, float] = {}
        self.volume_static_exp_results: dict[str, float] = {}

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

    def split_pairs_for_fold(self, pairs: list[LobFilePair], fold: FoldConfig) -> dict[str, list[LobFilePair]]:
        split_dates = {
            "train": set(fold.train_dates),
            "validation": set(fold.validation_dates),
            "test": set(fold.test_dates),
        }
        available_dates = {pair.date for pair in pairs}
        requested_dates = set().union(*split_dates.values())
        missing_dates = sorted(requested_dates - available_dates)
        if missing_dates:
            raise ValueError(f"Fold {fold.id} references dates with no discovered LOBSTER pair: {missing_dates}")

        split_pairs = {name: [] for name in SPLIT_NAMES}
        for pair in pairs:
            for split, dates in split_dates.items():
                if pair.date in dates:
                    split_pairs[split].append(pair)

        if not split_pairs["train"]:
            raise ValueError(f"Fold {fold.id} has no training file pairs.")
        return split_pairs

    def _required_fold_dates(self) -> set[str]:
        requested_dates: set[str] = set()
        for fold in self.config.folds:
            requested_dates.update(fold.train_dates)
            requested_dates.update(fold.validation_dates)
            requested_dates.update(fold.test_dates)
        return requested_dates

    def prepare_required_days(self, pairs: list[LobFilePair]) -> dict[str, ProcessedDay]:
        required_dates = self._required_fold_dates()
        prepared: dict[str, ProcessedDay] = {}
        for pair in pairs:
            if pair.date not in required_dates:
                continue
            prepared[pair.output_stem] = self.prepare_pair(pair, split="prepared")
        return prepared

    @staticmethod
    def _clone_prepared_day(prepared_day: ProcessedDay, split: str) -> ProcessedDay:
        return ProcessedDay(
            split=split,
            pair=prepared_day.pair,
            raw=prepared_day.raw,
            joined=prepared_day.joined,
            labeled=prepared_day.labeled,
            message_features=prepared_day.message_features,
        )

    def prepared_splits_for_fold(
        self,
        fold: FoldConfig,
        split_pairs: dict[str, list[LobFilePair]],
        prepared_days: dict[str, ProcessedDay],
    ) -> dict[str, list[ProcessedDay]]:
        processed = {name: [] for name in SPLIT_NAMES}
        for split, pairs in split_pairs.items():
            for pair in pairs:
                processed[split].append(self._clone_prepared_day(prepared_days[pair.output_stem], split))
        return processed

    def load_and_trim_pair(self, pair: LobFilePair) -> pd.DataFrame:
        """this function: delete 'ghost' levels if placeholder values are recognized;
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
    ) -> tuple[list[np.ndarray], list[np.ndarray] | None, float]:
        if kind not in {"price", "volume"}:
            raise ValueError("kind must be 'price' or 'volume'.")

        stream_config = (
            self.config.preprocessing.price_kinematic
            if kind == "price"
            else self.config.preprocessing.volume_kinematic
        )
        values_by_day: list[np.ndarray] = []
        centers_by_day: list[np.ndarray] | None = [] if kind == "price" else None
        scale = (
            self.config.preprocessing.price_kinematic.tick_size
            if kind == "price"
            else 1.0
        )
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
            if centers_by_day is not None:
                centers_by_day.append(calculate_midprice(day.message_features).to_numpy(dtype=float))
        return values_by_day, centers_by_day, scale

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
        raw_values_by_day, raw_centers_by_day, scale = self._stream_values_for_lambda_optimization(
            train_days,
            kind=kind,
        )
        values_by_day: list[np.ndarray] = []
        centers_by_day: list[np.ndarray] | None = [] if raw_centers_by_day is not None else None
        for day_index, values in enumerate(raw_values_by_day):
            if len(values) < window:
                continue
            values_by_day.append(values)
            if centers_by_day is not None and raw_centers_by_day is not None:
                centers_by_day.append(raw_centers_by_day[day_index])
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
            centers_by_day=centers_by_day,
            scale=scale,
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

    def reset_fold_state(self) -> None:
        self.config.preprocessing.price_kinematic.fast.selected_smoothing_lambda = None
        self.config.preprocessing.volume_kinematic.fast.selected_smoothing_lambda = None
        self.fast_smoothing_lambda_results = {}
        self.config.preprocessing.price_static.tau_clip = None
        self.config.preprocessing.price_static.tau_max = None
        self.price_static_plgs_results = {}
        self.config.preprocessing.volume_static.k = None
        self.volume_static_exp_results = {}
        self.snapshot_processor = SnapshotBatchProcessor(self.config.data, self.config.preprocessing)

    def fit_price_static_plgs_parameters(self, train_days: list[ProcessedDay]) -> dict[str, float] | None:
        price_static_config = self.config.preprocessing.price_static
        if not price_static_config.enabled:
            return None

        window = self.config.preprocessing.snapshot_window
        train_values: list[pd.Series] = []
        for day in train_days:
            df = day.message_features
            if len(df) < window:
                continue
            columns = self.snapshot_processor.window_processor._resolve_stream_columns(
                df,
                price_static_config,
                "price",
            )
            if not columns:
                continue
            end_positions = np.arange(window - 1, len(df))
            end_rows = df.iloc[end_positions].reset_index(drop=True)
            distances = price_static_distance_frame(
                end_rows,
                columns,
                price_static_config.tick_size,
            )
            if distances.empty:
                continue
            train_values.append(pd.Series(distances.to_numpy(dtype=float).ravel()))

        if not train_values:
            raise ValueError("Cannot fit price static PLGS parameters: no train price static values found.")

        all_values = pd.concat(train_values, ignore_index=True)
        result = fit_plgs_parameters(all_values, tau_start=price_static_config.tau_start)
        price_static_config.tau_clip = float(result["tau_clip"])
        price_static_config.tau_max = float(result["tau_max"])
        self.price_static_plgs_results = result
        self.snapshot_processor = SnapshotBatchProcessor(self.config.data, self.config.preprocessing)

        print(
            "Selected price static PLGS parameters from train: "
            f"tau_start={result['tau_start']:.6g}, "
            f"tau_clip(q99)={result['tau_clip']:.6g}, "
            f"tau_max={result['tau_max']:.6g}, "
            f"x95={result['x95']:.6g}, n_values={int(result['n_values'])}."
        )
        return result

    def fit_volume_static_exp_parameters(self, train_days: list[ProcessedDay]) -> dict[str, float] | None:
        volume_static_config = self.config.preprocessing.volume_static
        if not volume_static_config.enabled:
            return None

        window = self.config.preprocessing.snapshot_window
        train_values: list[pd.Series] = []
        for day in train_days:
            df = day.message_features
            if len(df) < window:
                continue
            columns = self.snapshot_processor.window_processor._resolve_stream_columns(
                df,
                volume_static_config,
                "volume",
            )
            if not columns:
                continue
            end_positions = np.arange(window - 1, len(df))
            end_rows = df.iloc[end_positions].reset_index(drop=True)
            train_values.append(pd.Series(end_rows[columns].to_numpy(dtype=float).ravel()))

        if not train_values:
            raise ValueError("Cannot fit volume static exponential scaling k: no train volume values found.")

        all_values = pd.concat(train_values, ignore_index=True)
        result = fit_exp_scaling_parameters(
            all_values,
            quantile=volume_static_config.quantile,
            target=volume_static_config.target,
        )
        volume_static_config.k = float(result["k"])
        self.volume_static_exp_results = result
        self.snapshot_processor = SnapshotBatchProcessor(self.config.data, self.config.preprocessing)

        print(
            "Selected volume static exponential scaling from train: "
            f"quantile={result['quantile']:.6g}, "
            f"target={result['target']:.6g}, "
            f"quantile_value={result['quantile_value']:.6g}, "
            f"k={result['k']:.6g}, n_values={int(result['n_values'])}."
        )
        return result

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

    def uses_adaptive_method_c_labels(self) -> bool:
        label_config = self.config.preprocessing.labels
        smoothing_config = label_config.smoothing
        return (
            label_config.strategy.lower() == "smoothing"
            and smoothing_config.method.upper() == "C"
            and smoothing_config.adaptive_threshold is not None
            and smoothing_config.adaptive_threshold.enabled
        )

    def train_label_distribution(self, train_days: list[ProcessedDay]) -> dict[str, object] | None:
        if not self.uses_adaptive_method_c_labels():
            return None

        label_column = self.config.data.label_column
        label_values = [
            day.labeled[label_column]
            for day in train_days
            if label_column in day.labeled.columns
        ]
        labels = pd.concat(label_values, ignore_index=True) if label_values else pd.Series(dtype=int)
        total = int(len(labels))
        distribution: dict[str, object] = {
            "method": "smoothing_C_adaptive",
            "train": {
                "total": total,
            },
        }
        train_distribution = distribution["train"]
        if not isinstance(train_distribution, dict):
            raise RuntimeError("Invalid label distribution payload.")

        for class_value in (-1, 0, 1):
            count = int((labels == class_value).sum())
            train_distribution[str(class_value)] = {
                "count": count,
                "percentage": 0.0 if total == 0 else float(100.0 * count / total),
            }
        floor_comparison = self.adaptive_threshold_floor_comparison(train_days)
        if floor_comparison is not None:
            distribution["adaptive_threshold_floor_comparison"] = floor_comparison
        return distribution

    def adaptive_threshold_floor_comparison(self, train_days: list[ProcessedDay]) -> dict[str, object] | None:
        smoothing_config = self.config.preprocessing.labels.smoothing
        adaptive_config = smoothing_config.adaptive_threshold
        if adaptive_config is None or not adaptive_config.enabled:
            return None

        total_valid = 0
        cost_greater = 0
        volatility_greater = 0
        equal = 0
        for day in train_days:
            midprices = calculate_midprice(
                day.joined,
                bid_col=smoothing_config.bid_column,
                ask_col=smoothing_config.ask_column,
            )
            pct_changes = SmoothingMethodC(k=smoothing_config.k, h=smoothing_config.h)(midprices)
            components = calculate_adaptive_method_c_threshold_components(
                day.joined,
                midprices,
                k=smoothing_config.k,
                h=smoothing_config.h,
                bid_col=smoothing_config.bid_column,
                ask_col=smoothing_config.ask_column,
                config=adaptive_config,
            )
            valid_mask = (
                pct_changes.notna()
                & np.isfinite(pct_changes)
                & components["cost_floor"].notna()
                & np.isfinite(components["cost_floor"])
                & components["volatility_floor"].notna()
                & np.isfinite(components["volatility_floor"])
            )
            cost_floor = components.loc[valid_mask, "cost_floor"]
            volatility_floor = components.loc[valid_mask, "volatility_floor"]
            total_valid += int(valid_mask.sum())
            cost_greater += int((cost_floor > volatility_floor).sum())
            volatility_greater += int((volatility_floor > cost_floor).sum())
            equal += int((cost_floor == volatility_floor).sum())

        def entry(count: int) -> dict[str, float | int]:
            return {
                "count": count,
                "percentage": 0.0 if total_valid == 0 else float(100.0 * count / total_valid),
            }

        return {
            "valid_rows": total_valid,
            "cost_floor_gt_volatility_floor": entry(cost_greater),
            "volatility_floor_gt_cost_floor": entry(volatility_greater),
            "equal": entry(equal),
        }

    @staticmethod
    def print_label_distribution(fold_id: str, distribution: dict[str, object] | None) -> None:
        if distribution is None:
            return
        train_distribution = distribution.get("train")
        if not isinstance(train_distribution, dict):
            return

        print(f"{fold_id} adaptive method C train label distribution:")
        for class_value in ("-1", "0", "1"):
            class_distribution = train_distribution.get(class_value)
            if not isinstance(class_distribution, dict):
                continue
            count = int(class_distribution.get("count", 0))
            percentage = float(class_distribution.get("percentage", 0.0))
            print(f"- class {int(class_value):>2}: {count} ({percentage:.2f}%)")

        floor_comparison = distribution.get("adaptive_threshold_floor_comparison")
        if not isinstance(floor_comparison, dict):
            return
        valid_rows = int(floor_comparison.get("valid_rows", 0))
        print(f"{fold_id} adaptive method C threshold floor comparison over {valid_rows} valid row(s):")
        for key, label in (
            ("cost_floor_gt_volatility_floor", "cost_floor > volatility_floor"),
            ("volatility_floor_gt_cost_floor", "volatility_floor > cost_floor"),
        ):
            values = floor_comparison.get(key)
            if not isinstance(values, dict):
                continue
            count = int(values.get("count", 0))
            percentage = float(values.get("percentage", 0.0))
            print(f"- {label}: {percentage:.2f}% ({count}/{valid_rows})")

    def fold_derivatives_stats_path(self, fold_id: str) -> Path:
        return self.derivatives_stats_dir / fold_id / DERIVATIVES_STATS_FILENAME

    def fit_train_normalizer(self, train_days: list[ProcessedDay], stats_path: Path | None = None) -> DerivativeNormalizer:
        print(f"Fitting derivative normalizer on {len(train_days)} training day(s).")
        target_stats_path = stats_path or (self.derivatives_stats_dir / DERIVATIVES_STATS_FILENAME)
        if target_stats_path.exists():
            target_stats_path.unlink()
            print(f"Removed previous derivative statistics: {target_stats_path}")
        processed_train_days = []
        for day in train_days:
            if day.processed is None:
                raise ValueError(f"{day.pair.label} has not been snapshot-processed yet.")
            processed_train_days.append(day.processed)
        normalizer = DerivativeNormalizer(target_stats_path)
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

    def save_split_outputs(
        self,
        processed_splits: dict[str, list[ProcessedDay]],
        *,
        processed_dir: Path | None = None,
        sequence_dir: Path | None = None,
    ) -> None:
        print("Saving processed CSV and sequence outputs.")
        target_processed_dir = processed_dir or self.processed_dir
        target_sequence_dir = sequence_dir or self.sequence_dir
        target_processed_dir.mkdir(parents=True, exist_ok=True)
        target_sequence_dir.mkdir(parents=True, exist_ok=True)

        for split, days in processed_splits.items():
            processed_split_dir = target_processed_dir / split
            sequence_split_dir = target_sequence_dir / split
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

    def run_fold(
        self,
        fold: FoldConfig,
        split_pairs: dict[str, list[LobFilePair]],
        prepared_days: dict[str, ProcessedDay],
    ) -> dict[str, dict[str, tuple[int, int]]]:
        print(f"Starting fold {fold.id}.")
        self.reset_fold_state()
        processed_splits = self.prepared_splits_for_fold(fold, split_pairs, prepared_days)
        label_distribution = self.train_label_distribution(processed_splits["train"])
        self.print_label_distribution(fold.id, label_distribution)
        plgs_parameters = self.fit_price_static_plgs_parameters(processed_splits["train"])
        volume_static_exp = self.fit_volume_static_exp_parameters(processed_splits["train"])
        self.optimize_fast_smoothing_lambdas(processed_splits["train"])
        self.build_snapshot_features(processed_splits)
        stats_path = self.fold_derivatives_stats_path(fold.id)
        normalizer = self.fit_train_normalizer(processed_splits["train"], stats_path=stats_path)
        self.apply_normalization(processed_splits, normalizer)

        fold_processed_dir = self.processed_dir / fold.id
        fold_sequence_dir = self.sequence_dir / fold.id
        self.save_split_outputs(
            processed_splits,
            processed_dir=fold_processed_dir,
            sequence_dir=fold_sequence_dir,
        )
        metadata_path = save_preprocessing_metadata(
            self.config,
            fold_sequence_dir,
            lambda_results=self.fast_smoothing_lambda_results,
            label_distribution=label_distribution,
            price_static_plgs=plgs_parameters,
            volume_static_exp=volume_static_exp,
        )
        print(f"Saved preprocessing metadata for fold {fold.id} to {metadata_path}.")
        print(f"Finished fold {fold.id}.")
        return {
            split: {
                day.pair.output_stem: day.normalized.shape if day.normalized is not None else day.processed.shape
                for day in days
            }
            for split, days in processed_splits.items()
        }

    def run(self) -> dict[str, dict[str, dict[str, tuple[int, int]]]]:
        print("Starting LOB processing pipeline.")
        pairs = self.discover_pairs()
        print(f"Discovered {len(pairs)} message/orderbook file pair(s).")
        prepared_days = self.prepare_required_days(pairs)
        summary: dict[str, dict[str, dict[str, tuple[int, int]]]] = {}
        for fold in self.config.folds:
            split_pairs = self.split_pairs_for_fold(pairs, fold)
            summary[fold.id] = self.run_fold(fold, split_pairs, prepared_days)
        print("LOB processing pipeline finished.")
        return summary
