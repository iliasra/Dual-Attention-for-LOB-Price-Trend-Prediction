from __future__ import annotations

import gc
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import yaml

try:
    from configuration import ExperimentConfig, FoldConfig, load_config
    from datasets import DailySequenceBuilder
    from fast_kinematic_preprocessing import optimize_smoothing_lambda_gcv
    from horizon import (
        SmoothingMethodC,
        TargetLabelPipeline,
        calculate_adaptive_method_c_threshold_components,
        calculate_midprice,
        fit_smoothing_threshold,
        fit_train_smoothing_threshold,
        is_fitted_smoothing_threshold,
    )
    from gcv_lambda_cache import (
        aggregate_daily_gcv_caches,
        daily_lambda_gcv_cache_path,
        lambda_gcv_cache_key,
        load_daily_gcv_cache,
    )
    from kinematic_preprocessing import (
        DerivativeNormalizer,
        MessageFeatureProcessor,
        MessageOrderbookJoiner,
        SnapshotBatchProcessor,
        TradingSessionFilter,
        derivative_feature_columns,
        fit_exp_scaling_parameters,
        fit_plgs_parameters,
        handle_abnormal_prices,
        normalizable_feature_columns,
        price_kinematic_values,
        price_static_distance_frame,
    )
    from lobster_io import read_lobster_message_csv, read_lobster_orderbook_csv
    from run_logging import format_duration, save_preprocessing_metadata
    from volume_clock import VolumeBarFeatureProcessor, VolumeClockSampler
except ImportError:  # pragma: no cover
    from .configuration import ExperimentConfig, FoldConfig, load_config
    from .datasets import DailySequenceBuilder
    from .fast_kinematic_preprocessing import optimize_smoothing_lambda_gcv
    from .horizon import (
        SmoothingMethodC,
        TargetLabelPipeline,
        calculate_adaptive_method_c_threshold_components,
        calculate_midprice,
        fit_smoothing_threshold,
        fit_train_smoothing_threshold,
        is_fitted_smoothing_threshold,
    )
    from .gcv_lambda_cache import (
        aggregate_daily_gcv_caches,
        daily_lambda_gcv_cache_path,
        lambda_gcv_cache_key,
        load_daily_gcv_cache,
    )
    from .kinematic_preprocessing import (
        DerivativeNormalizer,
        MessageFeatureProcessor,
        MessageOrderbookJoiner,
        SnapshotBatchProcessor,
        TradingSessionFilter,
        derivative_feature_columns,
        fit_exp_scaling_parameters,
        fit_plgs_parameters,
        handle_abnormal_prices,
        normalizable_feature_columns,
        price_kinematic_values,
        price_static_distance_frame,
    )
    from .lobster_io import read_lobster_message_csv, read_lobster_orderbook_csv
    from .run_logging import format_duration, save_preprocessing_metadata
    from .volume_clock import VolumeBarFeatureProcessor, VolumeClockSampler


SPLIT_NAMES = ("train", "validation", "test")
DERIVATIVES_STATS_FILENAME = "derivatives_stats.yaml"
FEATURE_SCHEMA_FILENAME = "feature_schema.yaml"


@dataclass(slots=True)
class LobFileSegment:
    message_path: Path
    orderbook_path: Path

    def __post_init__(self) -> None:
        self.message_path = Path(self.message_path)
        self.orderbook_path = Path(self.orderbook_path)


@dataclass(slots=True)
class LobFilePair:
    symbol: str
    date: str
    segments: tuple[LobFileSegment, ...]

    def __post_init__(self) -> None:
        self.segments = tuple(self.segments)
        if not self.segments:
            raise ValueError("LobFilePair must contain at least one segment.")

    @property
    def output_stem(self) -> str:
        return f"{self.symbol}_{self.date}"

    @property
    def label(self) -> str:
        suffix = "" if self.segment_count == 1 else f" ({self.segment_count} segments)"
        return f"{self.symbol} {self.date}{suffix}"

    @property
    def segment_count(self) -> int:
        return len(self.segments)


@dataclass(slots=True)
class ProcessedDay:
    """Temporary per-day payload; dataframe fields may be released to reduce RAM."""

    split: str
    pair: LobFilePair
    raw: pd.DataFrame | None
    joined: pd.DataFrame | None
    labeled: pd.DataFrame | None
    message_features: pd.DataFrame | None
    processed: pd.DataFrame | None = None
    normalized: pd.DataFrame | None = None
    processed_csv_path: Path | None = None
    sequence_paths: tuple[Path, Path, Path] | None = None

    def release_raw_frames(self) -> None:
        """Release raw/joined/labeled frames after train-only diagnostics."""
        self.raw = None
        self.joined = None
        self.labeled = None

    def release_processed_frames(self) -> None:
        """Release feature, processed, and normalized frames after saving."""
        self.message_features = None
        self.processed = None
        self.normalized = None

    def release_all_frames(self) -> None:
        """Release all dataframe payloads held by this day."""
        self.release_raw_frames()
        self.release_processed_frames()


class LobProcessingPipeline:
    def __init__(
        self,
        config: ExperimentConfig | None = None,
        data_dir: str | Path | None = None,
        lambda_cache_dir: str | Path | None = None,
        require_lambda_cache: bool = False,
    ) -> None:
        self.config = config or load_config()
        self.root_dir = Path(__file__).resolve().parent.parent
        self.config_dir = self.config.path.parent
        self.downstream_data_config = self._build_downstream_data_config()
        configured_data_dir = Path(self.config.data.raw_data_dir)
        self.data_dir = Path(data_dir) if data_dir is not None else self._resolve_path(configured_data_dir)
        self.processed_dir = self._resolve_path(Path(self.config.data.processed_data_dir))
        self.sequence_dir = self._resolve_path(Path(self.config.data.sequence_data_dir))
        self.derivatives_stats_dir = self._resolve_path(
            Path(self.config.preprocessing.normalization.derivatives_stats_dir)
        )
        self.lambda_cache_dir = None if lambda_cache_dir is None else Path(lambda_cache_dir).resolve()
        self.require_lambda_cache = bool(require_lambda_cache)

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
        self.volume_clock_sampler = VolumeClockSampler(
            sample_clock_config=self.config.preprocessing.sample_clock,
            message_config=self.config.preprocessing.message,
            time_column=self.config.data.time_column,
        )
        self.volume_bar_processor = VolumeBarFeatureProcessor(
            message_config=self.config.preprocessing.message,
        )
        self.snapshot_processor = SnapshotBatchProcessor(self.downstream_data_config, self.config.preprocessing)
        self.sequence_builder = DailySequenceBuilder(self.downstream_data_config)
        self.fast_smoothing_lambda_results: dict[str, dict[str, float]] = {}
        self.price_static_plgs_results: dict[str, float] = {}
        self.volume_static_exp_results: dict[str, float] = {}
        self.volume_bar_scaling_results: dict[str, dict[str, float | int]] = {}
        self.smoothing_threshold_result: dict[str, object] | None = None
        self.smoothing_threshold_results: dict[str, dict[str, object]] = {}

    def _build_downstream_data_config(self):
        """Return the data config used after optional sample-clock conversion."""
        if not self.config.preprocessing.sample_clock.enabled:
            return self.config.data
        excluded = list(dict.fromkeys([*self.config.data.feature_exclude_columns, "volume_wall_time"]))
        return replace(
            self.config.data,
            time_column="volume_time",
            feature_exclude_columns=excluded,
        )

    def uses_volume_clock(self) -> bool:
        """Return whether preprocessing samples events into volume bars."""
        return self.config.preprocessing.sample_clock.enabled

    def _sample_clock_metadata(self) -> dict[str, object]:
        """Return serializable sample-clock settings for preprocessing artifacts."""
        sample_clock = self.config.preprocessing.sample_clock
        return {
            "mode": sample_clock.mode,
            "volume_step_shares": sample_clock.volume_step_shares,
            "volume_source": sample_clock.volume_source,
            "trade_type_values": list(sample_clock.trade_type_values),
            "downstream_time_column": self.downstream_data_config.time_column,
            "wall_time_column": "volume_wall_time" if sample_clock.enabled else self.config.data.time_column,
        }

    def _resolve_path(self, path: Path) -> Path:
        return path if path.is_absolute() else (self.config_dir / path).resolve()

    @staticmethod
    def _require_frame(day: ProcessedDay, attr: str, context: str) -> pd.DataFrame:
        """Return a day dataframe or fail clearly if it was released."""
        frame = getattr(day, attr)
        if frame is None:
            raise ValueError(f"{day.pair.label} has no '{attr}' dataframe available for {context}.")
        return frame

    @staticmethod
    def _lobster_file_key(file_path: Path) -> tuple[str, str] | None:
        parts = file_path.name.split("_")
        if len(parts) < 4:
            return None
        symbol = parts[0]
        date = parts[1]
        return symbol, date

    @staticmethod
    def _lobster_segment_sort_key(file_path: Path) -> tuple[int, str]:
        parts = file_path.name.split("_")
        try:
            return int(parts[2]), file_path.name
        except (IndexError, ValueError):
            return 0, file_path.name

    def discover_pairs(self) -> list[LobFilePair]:
        """gathers message/orderbook data by asset and day"""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"LOBSTER directory not found: {self.data_dir}")

        grouped_segments: dict[tuple[str, str], list[tuple[Path, Path]]] = {}
        unmatched: list[Path] = []
        orderbook_paths = {path.name: path for path in self.data_dir.glob("*_orderbook_*.csv")}
        matched_orderbooks: set[Path] = set()

        for message_path in sorted(self.data_dir.glob("*_message_*.csv")):
            key = self._lobster_file_key(message_path)
            if key is None:
                continue
            orderbook_path = orderbook_paths.get(message_path.name.replace("_message_", "_orderbook_"))
            if orderbook_path is None:
                unmatched.append(message_path)
                continue
            grouped_segments.setdefault(key, []).append((message_path, orderbook_path))
            matched_orderbooks.add(orderbook_path)

        unmatched.extend(
            orderbook_path
            for orderbook_path in sorted(orderbook_paths.values())
            if orderbook_path not in matched_orderbooks
        )
        if unmatched:
            raise ValueError(
                "Unmatched LOBSTER files without message/orderbook counterpart: "
                + ", ".join(path.name for path in unmatched)
            )

        pairs: list[LobFilePair] = []
        for (symbol, date), segments in grouped_segments.items():
            sorted_segments = sorted(segments, key=lambda segment: self._lobster_segment_sort_key(segment[0]))
            pairs.append(
                LobFilePair(
                    symbol=symbol,
                    date=date,
                    segments=tuple(
                        LobFileSegment(message_path=message_path, orderbook_path=orderbook_path)
                        for message_path, orderbook_path in sorted_segments
                    ),
                )
            )
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

    def _required_fold_dates(self, folds: list[FoldConfig] | None = None) -> set[str]:
        requested_dates: set[str] = set()
        selected_folds = self.config.folds if folds is None else folds
        for fold in selected_folds:
            requested_dates.update(fold.train_dates)
            requested_dates.update(fold.validation_dates)
            requested_dates.update(fold.test_dates)
        return requested_dates

    def prepare_required_days(
        self,
        pairs: list[LobFilePair],
        folds: list[FoldConfig] | None = None,
    ) -> dict[str, ProcessedDay]:
        """Legacy in-memory preparation path; not used by RAM-safe HPC preprocessing."""
        required_dates = self._required_fold_dates(folds)
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
        """Legacy helper that clones preloaded days into fold splits."""
        processed = {name: [] for name in SPLIT_NAMES}
        for split, pairs in split_pairs.items():
            for pair in pairs:
                processed[split].append(self._clone_prepared_day(prepared_days[pair.output_stem], split))
        return processed

    def _load_joined_segment(self, segment: LobFileSegment) -> pd.DataFrame:
        message_result = read_lobster_message_csv(
            segment.message_path,
            time_column=self.config.data.time_column,
            size_column=self.config.preprocessing.message.size_column,
            price_column=self.config.preprocessing.message.price_column,
            order_id_column=self.config.preprocessing.message.order_id_column,
            categorical_value_map=self.config.preprocessing.message.categorical_value_map,
        )
        message_df = message_result.dataframe
        orderbook_df = read_lobster_orderbook_csv(segment.orderbook_path).dataframe
        trading_session_mask = message_df[self.config.data.time_column].between(
            self.config.preprocessing.temporal_features.market_open_seconds,
            self.config.preprocessing.temporal_features.market_close_seconds,
            inclusive="both",
        )
        if bool(trading_session_mask.any()):
            handle_abnormal_prices([message_df, orderbook_df], row_mask=trading_session_mask)
        else:
            print(
                f"Segment {segment.message_path.name} has no rows inside the configured trading session; "
                "it will be trimmed."
            )
        return self.joiner.transform(message_df, orderbook_df)

    def load_and_trim_pair(self, pair: LobFilePair) -> pd.DataFrame:
        """this function: delete 'ghost' levels if placeholder values are recognized;
        joins orderbook/message data for a particular day;
        filters out the first/last 15 minutes."""
        print(f"Loading raw data for {pair.label}.")
        joined_segments = [
            self._load_joined_segment(segment)
            for segment in pair.segments
        ]
        joined = pd.concat(joined_segments, ignore_index=True) if len(joined_segments) > 1 else joined_segments[0]
        if len(joined_segments) > 1:
            before_dedup = len(joined)
            dedupe_columns = [column for column in joined.columns if column != "delta_t"]
            joined = joined.drop_duplicates(subset=dedupe_columns, ignore_index=True)
            duplicate_count = before_dedup - len(joined)
            if duplicate_count:
                print(f"Removed {duplicate_count} duplicated rows across segments for {pair.label}.")
        joined = joined.sort_values(
            self.config.data.time_column,
            kind="mergesort",
            ignore_index=True,
        )
        joined["delta_t"] = joined[self.config.data.time_column] - joined[self.config.data.time_column].shift(1)
        trimmed = self.session_filter.transform(joined) #session_filter.transform gets rid of first/last 15 mins
        print(f"Loaded and trimmed {pair.label}: {len(trimmed)} rows.")
        return trimmed

    def _sample_clock_frame(self, df: pd.DataFrame, pair: LobFilePair) -> pd.DataFrame:
        """Apply the configured sample clock after raw event trimming."""
        if not self.uses_volume_clock():
            return df
        sampled = self.volume_clock_sampler.transform(df)
        print(
            f"Volume-clock sampled {pair.label}: {len(df)} events -> {len(sampled)} complete bars "
            f"(step={self.config.preprocessing.sample_clock.volume_step_shares:g} shares)."
        )
        return sampled

    def _message_features_from_labeled(self, labeled: pd.DataFrame) -> pd.DataFrame:
        """Build model-facing pre-snapshot features for the active sample clock."""
        if not self.uses_volume_clock():
            return self.message_processor.transform(labeled)
        if not self.volume_bar_processor.fitted:
            return labeled.copy()
        return self.volume_bar_processor.transform(labeled)

    def _fit_volume_bar_features(self, train_days: list[ProcessedDay]) -> dict[str, dict[str, float | int]] | None:
        """Fit train-only scalers for volume-bar aggregate features."""
        if not self.uses_volume_clock():
            return None
        labeled_frames = [
            self._require_frame(day, "labeled", "volume-bar feature scaling")
            for day in train_days
        ]
        self.volume_bar_scaling_results = self.volume_bar_processor.fit(labeled_frames)
        print(
            "Fitted volume-bar feature scaling on train: "
            + ", ".join(
                f"{column}: k={float(stats['k']):.6g}"
                for column, stats in sorted(self.volume_bar_scaling_results.items())
            )
        )
        for day in train_days:
            labeled = self._require_frame(day, "labeled", "volume-bar feature transformation")
            day.message_features = self.volume_bar_processor.transform(labeled)
        return self.volume_bar_scaling_results

    def _sample_clock_counts_for_day(self, day: ProcessedDay) -> dict[str, int]:
        """Return row counts before and after optional sample-clock conversion."""
        raw = self._require_frame(day, "raw", "sample-clock count metadata")
        joined = self._require_frame(day, "joined", "sample-clock count metadata")
        labeled = self._require_frame(day, "labeled", "sample-clock count metadata")
        message_features = self._require_frame(day, "message_features", "sample-clock count metadata")
        return {
            "raw_event_rows": int(len(raw)),
            "sampled_rows": int(len(joined)),
            "labeled_rows": int(len(labeled)),
            "feature_rows": int(len(message_features)),
        }

    def uses_train_fitted_smoothing_threshold(self) -> bool:
        """Return whether smoothing labels need a scalar train-fitted threshold."""
        label_config = self.config.preprocessing.labels
        smoothing_config = label_config.smoothing
        return (
            label_config.strategy.lower() == "smoothing"
            and is_fitted_smoothing_threshold(smoothing_config.threshold)
            and smoothing_config.resolved_fit_scope() == "train"
        )

    def uses_split_fitted_smoothing_threshold(self) -> bool:
        """Return whether smoothing labels need a scalar split-fitted threshold."""
        label_config = self.config.preprocessing.labels
        smoothing_config = label_config.smoothing
        return (
            label_config.strategy.lower() == "smoothing"
            and is_fitted_smoothing_threshold(smoothing_config.threshold)
            and smoothing_config.resolved_fit_scope() == "per_split"
        )

    def uses_fitted_smoothing_threshold(self) -> bool:
        """Return whether smoothing labels need any fitted scalar threshold."""
        return self.uses_train_fitted_smoothing_threshold() or self.uses_split_fitted_smoothing_threshold()

    def _resolved_smoothing_threshold(self, split: str) -> float | None:
        """Return the fitted smoothing threshold required for current split labels."""
        if not self.uses_fitted_smoothing_threshold():
            return None
        if self.uses_split_fitted_smoothing_threshold():
            result = self.smoothing_threshold_results.get(split)
            if result is None:
                raise ValueError(
                    f"Smoothing threshold must be fitted on {split} before labeling that split."
                )
            return float(result["value"])

        if self.smoothing_threshold_result is None:
            raise ValueError(
                "Smoothing threshold must be fitted on train before labeling validation/test. "
                "Use run_fold or call _fit_train_smoothing_threshold first."
            )
        return float(self.smoothing_threshold_result["value"])

    def _fit_train_smoothing_threshold(self, train_days: list[ProcessedDay]) -> dict[str, object] | None:
        """Fit optional train-derived smoothing threshold from sampled train frames."""
        if not self.uses_train_fitted_smoothing_threshold():
            return None
        sampled_frames = [
            self._require_frame(day, "joined", "train smoothing threshold fitting")
            for day in train_days
        ]
        self.smoothing_threshold_result = fit_train_smoothing_threshold(
            sampled_frames,
            self.config.preprocessing.labels.smoothing,
        )
        self.smoothing_threshold_result["fit_scope"] = "train"
        self.smoothing_threshold_results["train"] = self.smoothing_threshold_result
        print(
            "Fitted smoothing label threshold on train: "
            f"mode={self.smoothing_threshold_result['mode']}, "
            f"value={float(self.smoothing_threshold_result['value']):.10g}, "
            f"n_values={int(self.smoothing_threshold_result['n_values'])}."
        )
        return self.smoothing_threshold_result

    def _fit_split_smoothing_threshold(
        self,
        split_days: list[ProcessedDay],
        split: str,
    ) -> dict[str, object] | None:
        """Fit optional split-derived smoothing threshold from sampled frames."""
        if not self.uses_split_fitted_smoothing_threshold():
            return None
        sampled_frames = [
            self._require_frame(day, "joined", f"{split} smoothing threshold fitting")
            for day in split_days
        ]
        result = fit_smoothing_threshold(
            sampled_frames,
            self.config.preprocessing.labels.smoothing,
            fit_split=split,
        )
        result["fit_scope"] = "per_split"
        self.smoothing_threshold_results[split] = result
        print(
            f"Fitted smoothing label threshold on {split}: "
            f"mode={result['mode']}, "
            f"value={float(result['value']):.10g}, "
            f"n_values={int(result['n_values'])}."
        )
        return result

    def _smoothing_threshold_metadata(self) -> dict[str, object] | None:
        """Return fitted smoothing-threshold metadata for artifacts."""
        if self.uses_split_fitted_smoothing_threshold():
            return {
                "enabled": True,
                "mode": str(self.config.preprocessing.labels.smoothing.threshold).lower(),
                "fit_scope": "per_split",
                "splits": dict(self.smoothing_threshold_results),
            }
        return self.smoothing_threshold_result

    def prepare_unlabeled_pair(self, pair: LobFilePair, split: str) -> ProcessedDay:
        """Load, trim, and sample one day before target labeling."""
        print(f"Starting raw feature preparation for {pair.label} ({split}).")
        trimmed = self.load_and_trim_pair(pair)
        sampled = self._sample_clock_frame(trimmed, pair)
        return ProcessedDay(
            split=split,
            pair=pair,
            raw=trimmed,
            joined=sampled,
            labeled=None,
            message_features=None,
        )

    def _label_and_build_message_features(self, day: ProcessedDay) -> ProcessedDay:
        """Add target labels and pre-snapshot model features to one sampled day."""
        sampled = self._require_frame(day, "joined", "label generation")
        print(f"Starting label generation for {day.pair.label}.")
        labeled = self.labeler.transform(
            sampled,
            smoothing_threshold_override=self._resolved_smoothing_threshold(day.split),
        )
        print(f"Starting message feature processing for {day.pair.label}.")
        day.labeled = labeled
        day.message_features = self._message_features_from_labeled(labeled)
        print(f"Finished raw feature preparation for {day.pair.label}: {day.message_features.shape[0]} rows.")
        return day

    def prepare_pair(self, pair: LobFilePair, split: str) -> ProcessedDay:
        day = self.prepare_unlabeled_pair(pair, split)
        return self._label_and_build_message_features(day)

    def prepare_pair_for_lambda_cache(self, pair: LobFilePair) -> ProcessedDay:
        """Prepare one sampled day for GCV cache building without target labels."""
        day = self.prepare_unlabeled_pair(pair, split="lambda_cache")
        sampled = self._require_frame(day, "joined", "lambda cache stream extraction")
        day.message_features = sampled.copy()
        print(f"Finished lambda-cache feature preparation for {pair.label}: {sampled.shape[0]} rows.")
        return day

    def preprocess_pair(self, pair: LobFilePair, split: str) -> ProcessedDay:
        """Legacy eager preprocessing helper retained for notebooks/tests."""
        day = self.prepare_pair(pair, split)
        message_features = self._require_frame(day, "message_features", "snapshot preprocessing")
        day.processed = self.snapshot_processor.transform(message_features, source_label=pair.label)
        print(
            f"Finished preprocessing for {pair.label}: "
            f"{day.processed.shape[0]} rows, {day.processed.shape[1]} columns."
        )
        return day

    def preprocess_splits(self, split_pairs: dict[str, list[LobFilePair]]) -> dict[str, list[ProcessedDay]]:
        """Legacy in-memory split preparation; run_fold now streams days."""
        processed = {name: [] for name in SPLIT_NAMES}
        if self.uses_fitted_smoothing_threshold():
            for pair in split_pairs.get("train", []):
                processed["train"].append(self.prepare_unlabeled_pair(pair, "train"))
            if self.uses_split_fitted_smoothing_threshold():
                self._fit_split_smoothing_threshold(processed["train"], "train")
            else:
                self._fit_train_smoothing_threshold(processed["train"])
            for day in processed["train"]:
                self._label_and_build_message_features(day)
            split_iterable = [split for split in SPLIT_NAMES if split != "train"]
        else:
            split_iterable = list(SPLIT_NAMES)
        for split, pairs in split_pairs.items():
            if split not in split_iterable:
                continue
            print(f"Starting preprocessing split '{split}' with {len(pairs)} day(s).")
            if self.uses_split_fitted_smoothing_threshold():
                for pair in pairs:
                    processed[split].append(self.prepare_unlabeled_pair(pair, split))
                if processed[split]:
                    self._fit_split_smoothing_threshold(processed[split], split)
                for day in processed[split]:
                    self._label_and_build_message_features(day)
            else:
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
            message_features = self._require_frame(day, "message_features", f"{kind} lambda optimization")
            columns = self.snapshot_processor.window_processor._resolve_kinematic_columns(
                message_features,
                stream_config,
                kind,
            )
            if kind == "price":
                values, _ = price_kinematic_values(
                    message_features,
                    columns,
                    microprice_levels=self.snapshot_processor.window_processor._microprice_levels_for_price_kinematic(),
                )
            else:
                if not columns:
                    continue
                values = message_features[columns].to_numpy(dtype=float)
                values = np.log1p(values)
            if values.shape[1] == 0:
                continue
            values_by_day.append(values)
            if centers_by_day is not None:
                centers_by_day.append(calculate_midprice(message_features).to_numpy(dtype=float))
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

        if self._try_optimize_fast_stream_lambda_from_cache(train_days, kind=kind):
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
            n_df_candidates=preprocessing.kinematic_tokenization.n_df_candidates,
            chunk_size=gcv_chunk_size,
            centers_by_day=centers_by_day,
            scale=scale,
        )
        fast_config.selected_smoothing_lambda = result.smoothing_lambda
        self.fast_smoothing_lambda_results[kind] = {
            "selected_smoothing_lambda": float(result.smoothing_lambda),
            "effective_df": float(result.effective_df),
            "mean_gcv": float(result.gcv_score),
            "stream_signature": self._lambda_cache_stream_signature(kind=kind),
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
        self.snapshot_processor = SnapshotBatchProcessor(self.downstream_data_config, self.config.preprocessing)

    def _stream_config_for_kind(self, kind: str):
        if kind == "price":
            return self.config.preprocessing.price_kinematic
        if kind == "volume":
            return self.config.preprocessing.volume_kinematic
        raise ValueError("kind must be 'price' or 'volume'.")

    def _lambda_cache_key_for_stream(self, *, kind: str, scale: float) -> str:
        preprocessing = self.config.preprocessing
        fast_config = self._stream_config_for_kind(kind).fast
        return lambda_gcv_cache_key(
            window=preprocessing.snapshot_window,
            n_basis=fast_config.n_basis,
            max_df=fast_config.df,
            scale=scale,
            n_df_candidates=preprocessing.kinematic_tokenization.n_df_candidates,
            stream_signature=self._lambda_cache_stream_signature(kind=kind),
        )

    def _lambda_cache_stream_signature(self, *, kind: str) -> str:
        """Return the stream feature signature used by daily GCV caches."""
        preprocessing = self.config.preprocessing
        top_k = preprocessing.kinematic_tokenization.orderbook_top_k_levels
        top_token = "all" if top_k is None else str(int(top_k))
        sample_clock = preprocessing.sample_clock
        if sample_clock.enabled:
            trade_types = "-".join(str(value) for value in sample_clock.trade_type_values)
            clock_token = (
                f"clockvolume_src{sample_clock.volume_source}"
                f"_step{float(sample_clock.volume_step_shares):g}"
                f"_types{trade_types}"
            )
        else:
            clock_token = "clockevent"
        if kind == "price" and preprocessing.microprice.enabled:
            microprice_token = f"mp{int(preprocessing.microprice.levels)}"
        else:
            microprice_token = "mpoff"
        return f"{clock_token}_top{top_token}_{microprice_token}"

    def _try_optimize_fast_stream_lambda_from_cache(
        self,
        train_days: list[ProcessedDay],
        *,
        kind: str,
    ) -> bool:
        """Use daily GCV cache when --lambda-cache-dir is configured."""
        if self.lambda_cache_dir is None:
            return False

        preprocessing = self.config.preprocessing
        stream_config = self._stream_config_for_kind(kind)
        if not stream_config.enabled:
            return True

        scale = preprocessing.price_kinematic.tick_size if kind == "price" else 1.0
        cache_key = self._lambda_cache_key_for_stream(kind=kind, scale=scale)

        caches = []
        missing: list[Path] = []

        for day in train_days:
            message_features = self._require_frame(day, "message_features", f"{kind} GCV cache lookup")
            if len(message_features) < preprocessing.snapshot_window:
                continue

            cache_path = daily_lambda_gcv_cache_path(
                self.lambda_cache_dir,
                cache_key=cache_key,
                kind=kind,
                output_stem=day.pair.output_stem,
            )
            if not cache_path.exists():
                missing.append(cache_path)
                continue

            caches.append(load_daily_gcv_cache(cache_path))

        if missing:
            message = (
                f"Missing {kind} lambda GCV cache file(s): "
                + ", ".join(str(path) for path in missing[:5])
                + (" ..." if len(missing) > 5 else "")
            )
            if self.require_lambda_cache:
                raise FileNotFoundError(message)
            print(message)
            print(f"Falling back to in-fold {kind} lambda optimization.")
            return False

        if not caches:
            if self.require_lambda_cache:
                raise ValueError(f"No {kind} lambda GCV caches available for this fold.")
            return False

        result, total_count = aggregate_daily_gcv_caches(caches)
        stream_config.fast.selected_smoothing_lambda = result.smoothing_lambda

        self.fast_smoothing_lambda_results[kind] = {
            "selected_smoothing_lambda": float(result.smoothing_lambda),
            "effective_df": float(result.effective_df),
            "mean_gcv": float(result.gcv_score),
            "stream_signature": self._lambda_cache_stream_signature(kind=kind),
            "gcv_cache_total_count": int(total_count),
            "gcv_cache_days": int(len(caches)),
        }

        print(
            f"Selected fast {kind} smoothing_lambda={result.smoothing_lambda:.8g} "
            f"from daily GCV cache "
            f"(effective_df={result.effective_df:.4f}, "
            f"mean_gcv={result.gcv_score:.8g}, "
            f"count={total_count})."
        )
        return True

    def reset_fold_state(self) -> None:
        self.config.preprocessing.price_kinematic.fast.selected_smoothing_lambda = None
        self.config.preprocessing.volume_kinematic.fast.selected_smoothing_lambda = None
        self.fast_smoothing_lambda_results = {}
        self.smoothing_threshold_result = None
        self.smoothing_threshold_results = {}
        self.volume_bar_scaling_results = {}
        self.volume_bar_processor = VolumeBarFeatureProcessor(
            message_config=self.config.preprocessing.message,
        )
        self.config.preprocessing.price_static.tau_clip = None
        self.config.preprocessing.price_static.tau_max = None
        self.price_static_plgs_results = {}
        self.config.preprocessing.volume_static.k = None
        self.volume_static_exp_results = {}
        self.snapshot_processor = SnapshotBatchProcessor(self.downstream_data_config, self.config.preprocessing)

    def fit_price_static_plgs_parameters(self, train_days: list[ProcessedDay]) -> dict[str, float] | None:
        price_static_config = self.config.preprocessing.price_static
        if not price_static_config.enabled:
            return None

        window = self.config.preprocessing.snapshot_window
        train_values: list[pd.Series] = []
        for day in train_days:
            df = self._require_frame(day, "message_features", "price static PLGS fitting")
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
        self.snapshot_processor = SnapshotBatchProcessor(self.downstream_data_config, self.config.preprocessing)

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
            df = self._require_frame(day, "message_features", "volume static scaling fitting")
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
        self.snapshot_processor = SnapshotBatchProcessor(self.downstream_data_config, self.config.preprocessing)

        print(
            "Selected volume static exponential scaling from train: "
            f"quantile={result['quantile']:.6g}, "
            f"target={result['target']:.6g}, "
            f"quantile_value={result['quantile_value']:.6g}, "
            f"k={result['k']:.6g}, n_values={int(result['n_values'])}."
        )
        return result

    def build_snapshot_features(self, processed_splits: dict[str, list[ProcessedDay]]) -> None:
        """Legacy eager snapshot construction for in-memory split payloads."""
        print("Starting snapshot feature construction for all splits.")
        for split_days in processed_splits.values():
            for day in split_days:
                message_features = self._require_frame(day, "message_features", "snapshot feature construction")
                day.processed = self.snapshot_processor.transform(message_features, source_label=day.pair.label)
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

    def _tracks_label_distribution(self) -> bool:
        """Return whether preprocessing metadata should include label counts."""
        return self.uses_adaptive_method_c_labels() or self.uses_fitted_smoothing_threshold()

    def _label_distribution_method(self) -> str:
        """Return the label-distribution method name for metadata."""
        if self.uses_adaptive_method_c_labels():
            return "smoothing_C_adaptive"
        if self.uses_split_fitted_smoothing_threshold():
            mode = str(self.config.preprocessing.labels.smoothing.threshold).lower()
            return f"smoothing_{mode}_per_split_fitted"
        if self.uses_train_fitted_smoothing_threshold():
            mode = str(self.config.preprocessing.labels.smoothing.threshold).lower()
            return f"smoothing_{mode}_train_fitted"
        return self.config.preprocessing.labels.strategy.lower()

    def _label_distribution_for_days(self, days: list[ProcessedDay]) -> dict[str, object]:
        label_column = self.config.data.label_column
        label_values = []
        for day in days:
            labeled = self._require_frame(day, "labeled", "label distribution")
            if label_column in labeled.columns:
                label_values.append(labeled[label_column])
        labels = pd.concat(label_values, ignore_index=True) if label_values else pd.Series(dtype=int)
        total = int(len(labels))
        distribution: dict[str, object] = {"total": total}

        for class_value in (-1, 0, 1):
            count = int((labels == class_value).sum())
            distribution[str(class_value)] = {
                "count": count,
                "percentage": 0.0 if total == 0 else float(100.0 * count / total),
            }
        return distribution

    def label_distribution(self, processed_splits: dict[str, list[ProcessedDay]]) -> dict[str, object] | None:
        if not self._tracks_label_distribution():
            return None

        distribution: dict[str, object] = {
            "method": self._label_distribution_method(),
        }
        for split in SPLIT_NAMES:
            distribution[split] = self._label_distribution_for_days(processed_splits.get(split, []))

        train_days = processed_splits.get("train", [])
        floor_comparison = self.adaptive_threshold_floor_comparison(train_days)
        if floor_comparison is not None:
            distribution["adaptive_threshold_floor_comparison"] = floor_comparison
        return distribution

    def train_label_distribution(self, train_days: list[ProcessedDay]) -> dict[str, object] | None:
        if not self._tracks_label_distribution():
            return None

        distribution: dict[str, object] = {
            "method": self._label_distribution_method(),
            "train": self._label_distribution_for_days(train_days),
        }
        floor_comparison = self.adaptive_threshold_floor_comparison(train_days)
        if floor_comparison is not None:
            distribution["adaptive_threshold_floor_comparison"] = floor_comparison
        return distribution

    def _new_label_distribution_accumulator(self) -> dict[str, dict[str, int]]:
        """Create split label counters for adaptive method-C metadata."""
        return {split: {"total": 0, "-1": 0, "0": 0, "1": 0} for split in SPLIT_NAMES}

    def _accumulate_label_distribution(
        self,
        accumulator: dict[str, dict[str, int]],
        split: str,
        day: ProcessedDay,
    ) -> None:
        """Accumulate one labeled day without keeping labels in memory."""
        labeled = self._require_frame(day, "labeled", "streaming label distribution")
        label_column = self.config.data.label_column
        if label_column not in labeled.columns:
            return

        labels = labeled[label_column]
        split_counts = accumulator[split]
        split_counts["total"] += int(len(labels))
        for class_value in (-1, 0, 1):
            split_counts[str(class_value)] += int((labels == class_value).sum())

    @staticmethod
    def _finalize_split_label_distribution(counts: dict[str, int]) -> dict[str, object]:
        """Convert accumulated label counts to metadata format."""
        total = int(counts["total"])
        distribution: dict[str, object] = {"total": total}
        for class_value in ("-1", "0", "1"):
            count = int(counts[class_value])
            distribution[class_value] = {
                "count": count,
                "percentage": 0.0 if total == 0 else float(100.0 * count / total),
            }
        return distribution

    def _finalize_label_distribution(
        self,
        accumulator: dict[str, dict[str, int]] | None,
        floor_comparison: dict[str, object] | None,
    ) -> dict[str, object] | None:
        """Build final label metadata after streaming all splits."""
        if accumulator is None:
            return None

        distribution: dict[str, object] = {"method": self._label_distribution_method()}
        for split in SPLIT_NAMES:
            distribution[split] = self._finalize_split_label_distribution(accumulator[split])
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
            joined = self._require_frame(day, "joined", "adaptive threshold floor comparison")
            midprices = calculate_midprice(
                joined,
                bid_col=smoothing_config.bid_column,
                ask_col=smoothing_config.ask_column,
            )
            pct_changes = SmoothingMethodC(k=smoothing_config.k, h=smoothing_config.h)(midprices)
            components = calculate_adaptive_method_c_threshold_components(
                joined,
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
        method = str(distribution.get("method", "labels"))
        label = "adaptive method C" if method == "smoothing_C_adaptive" else method
        for split in SPLIT_NAMES:
            split_distribution = distribution.get(split)
            if not isinstance(split_distribution, dict):
                continue

            print(f"{fold_id} {label} {split} label distribution:")
            for class_value in ("-1", "0", "1"):
                class_distribution = split_distribution.get(class_value)
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

    def fold_feature_schema_path(self, fold_id: str) -> Path:
        return self.sequence_dir / fold_id / FEATURE_SCHEMA_FILENAME

    def fit_train_normalizer(self, train_days: list[ProcessedDay], stats_path: Path | None = None) -> DerivativeNormalizer:
        """Legacy in-memory normalizer fit; RAM-safe run_fold uses streaming fit."""
        print(f"Fitting derivative normalizer on {len(train_days)} training day(s).")
        target_stats_path = stats_path or (self.derivatives_stats_dir / DERIVATIVES_STATS_FILENAME)
        if target_stats_path.exists():
            target_stats_path.unlink()
            print(f"Removed previous derivative statistics: {target_stats_path}")
        processed_train_days = []
        for day in train_days:
            processed_train_days.append(self._require_frame(day, "processed", "normalizer fitting"))
        normalizer = DerivativeNormalizer(
            target_stats_path,
            method=self.config.preprocessing.normalization.derivative_scaling_method,
            position_method=self.config.preprocessing.normalization.position_scaling_method,
            size_log1p_method=self.config.preprocessing.normalization.size_log1p_scaling_method,
            price_static_method=self.config.preprocessing.normalization.price_static_scaling_method,
            adaptive_label_feature_method=(
                self.config.preprocessing.normalization.adaptive_label_feature_scaling_method
            ),
            delta_t_method=self.config.preprocessing.normalization.delta_t_scaling_method,
            delta_t_transform=self.config.preprocessing.normalization.delta_t_transform,
        )
        normalizer.fit(processed_train_days)
        print("Derivative normalizer fitted.")
        return normalizer

    def apply_normalization(self, processed_splits: dict[str, list[ProcessedDay]], normalizer: DerivativeNormalizer) -> None:
        """Legacy in-memory normalization; RAM-safe run_fold normalizes one day at a time."""
        print("Starting normalization for all splits.")
        for split_days in processed_splits.values():
            for day in split_days:
                processed = self._require_frame(day, "processed", "normalization")
                day.normalized = normalizer.transform(processed)
        print("Finished normalization for all splits.")

    def _save_feature_schema(self, ordered_feature_columns: list[str], schema_path: Path) -> None:
        """Persist the fold feature order used by all saved arrays."""
        schema_path.parent.mkdir(parents=True, exist_ok=True)
        schema_path.write_text(
            yaml.safe_dump(
                {"ordered_feature_columns": ordered_feature_columns},
                sort_keys=False,
                allow_unicode=False,
            ),
            encoding="utf-8",
        )
        print(f"Saved feature schema with {len(ordered_feature_columns)} columns to {schema_path}.")

    def _build_feature_schema_from_day(self, day: ProcessedDay, schema_path: Path) -> list[str]:
        """Build the fold feature schema from the first normalized train day."""
        normalized = self._require_frame(day, "normalized", "feature schema construction")
        ordered_feature_columns = self.sequence_builder.feature_columns(normalized)
        if not ordered_feature_columns:
            raise ValueError("Cannot build feature schema: the first training day has no feature columns.")
        self._save_feature_schema(ordered_feature_columns, schema_path)
        return ordered_feature_columns

    def _align_day_to_feature_schema(self, day: ProcessedDay, ordered_feature_columns: list[str]) -> None:
        """Validate and reorder one normalized day to the fold feature schema."""
        normalized = self._require_frame(day, "normalized", "feature schema alignment")
        expected_features = set(ordered_feature_columns)
        current_feature_columns = self.sequence_builder.feature_columns(normalized)
        current_features = set(current_feature_columns)
        missing = sorted(expected_features - current_features)
        extra = sorted(current_features - expected_features)
        if missing or extra:
            raise ValueError(
                f"Feature schema mismatch for {day.pair.label} ({day.split}): "
                f"missing columns={missing}, extra columns={extra}."
            )

        data_config = getattr(self, "downstream_data_config", self.config.data)
        excluded_columns = {
            data_config.time_column,
            data_config.label_column,
            *data_config.feature_exclude_columns,
        }
        non_feature_columns = [column for column in normalized.columns if column in excluded_columns]
        day.normalized = normalized.loc[:, non_feature_columns + ordered_feature_columns]

    def apply_feature_schema(self, processed_splits: dict[str, list[ProcessedDay]], schema_path: Path) -> list[str]:
        """Legacy in-memory schema validation; RAM-safe run_fold validates per day."""
        train_days = processed_splits.get("train", [])
        if not train_days:
            raise ValueError("Cannot build feature schema: no training day is available.")
        ordered_feature_columns = self._build_feature_schema_from_day(train_days[0], schema_path)
        for split in SPLIT_NAMES:
            for day in processed_splits.get(split, []):
                day.split = split
                self._align_day_to_feature_schema(day, ordered_feature_columns)
        return ordered_feature_columns

    def _derivative_stats_metadata(
        self,
        message_features: pd.DataFrame,
        derivative_columns: list[str],
    ) -> dict[str, object]:
        """Return audit metadata saved next to derivative normalization stats."""
        window_processor = self.snapshot_processor.window_processor
        price_columns = (
            window_processor._resolve_kinematic_columns(
                message_features,
                self.config.preprocessing.price_kinematic,
                "price",
            )
            if self.config.preprocessing.price_kinematic.enabled
            else []
        )
        volume_columns = (
            window_processor._resolve_kinematic_columns(
                message_features,
                self.config.preprocessing.volume_kinematic,
                "volume",
            )
            if self.config.preprocessing.volume_kinematic.enabled
            else []
        )
        price_labels = (
            window_processor._price_kinematic_labels(price_columns)
            if self.config.preprocessing.price_kinematic.enabled
            else []
        )
        return {
            "kinematic_tokenization": {
                "orderbook_top_k_levels": self.config.preprocessing.kinematic_tokenization.orderbook_top_k_levels,
            },
            "microprice": {
                "enabled": bool(self.config.preprocessing.microprice.enabled),
                "levels": int(self.config.preprocessing.microprice.levels),
            },
            "sample_clock": self._sample_clock_metadata(),
            "kinematic_columns": {
                "price": price_labels,
                "volume": volume_columns,
            },
            "derivative_columns": derivative_columns,
        }

    def _fit_train_normalizer_streaming(
        self,
        train_days: list[ProcessedDay],
        *,
        stats_path: Path,
        schema_path: Path,
    ) -> tuple[DerivativeNormalizer, list[str]]:
        """Fit derivative stats from train snapshots while retaining only one full day."""
        print(f"Fitting derivative normalizer on {len(train_days)} training day(s).")
        if stats_path.exists():
            stats_path.unlink()
            print(f"Removed previous derivative statistics: {stats_path}")

        normalizer = DerivativeNormalizer(
            stats_path,
            method=self.config.preprocessing.normalization.derivative_scaling_method,
            position_method=self.config.preprocessing.normalization.position_scaling_method,
            size_log1p_method=self.config.preprocessing.normalization.size_log1p_scaling_method,
            price_static_method=self.config.preprocessing.normalization.price_static_scaling_method,
            adaptive_label_feature_method=(
                self.config.preprocessing.normalization.adaptive_label_feature_scaling_method
            ),
            delta_t_method=self.config.preprocessing.normalization.delta_t_scaling_method,
            delta_t_transform=self.config.preprocessing.normalization.delta_t_transform,
        )
        normalizer_frames: list[pd.DataFrame] = []
        first_schema_day: ProcessedDay | None = None
        stats_metadata: dict[str, object] | None = None

        for day in train_days:
            message_features = self._require_frame(day, "message_features", "train snapshot fitting")
            day.processed = self.snapshot_processor.transform(message_features, source_label=day.pair.label)
            processed = self._require_frame(day, "processed", "train normalizer fitting")
            print(
                f"Finished preprocessing for {day.pair.label}: "
                f"{processed.shape[0]} rows, {processed.shape[1]} columns."
            )

            derivative_columns = derivative_feature_columns(processed)
            normalization_columns = normalizable_feature_columns(processed)
            if stats_metadata is None:
                stats_metadata = self._derivative_stats_metadata(message_features, derivative_columns)
                stats_metadata["normalization_columns"] = normalization_columns
                stats_metadata["normalization_methods"] = {
                    "derivatives": self.config.preprocessing.normalization.derivative_scaling_method,
                    "kinematic_positions": self.config.preprocessing.normalization.position_scaling_method,
                    "size_log1p": self.config.preprocessing.normalization.size_log1p_scaling_method,
                    "price_static": self.config.preprocessing.normalization.price_static_scaling_method,
                    "adaptive_label_features": (
                        self.config.preprocessing.normalization.adaptive_label_feature_scaling_method
                    ),
                    "delta_t": self.config.preprocessing.normalization.delta_t_scaling_method,
                    "delta_t_transform": self.config.preprocessing.normalization.delta_t_transform,
                }
            normalizer_frames.append(processed.loc[:, normalization_columns].copy())
            if first_schema_day is None:
                first_schema_day = day
            else:
                day.processed = None
            gc.collect()

        if first_schema_day is None:
            raise ValueError("Cannot fit derivative normalizer: no training day is available.")

        normalizer.fit(normalizer_frames, metadata=stats_metadata)
        first_processed = self._require_frame(first_schema_day, "processed", "feature schema bootstrap")
        first_schema_day.normalized = normalizer.transform(first_processed)
        ordered_feature_columns = self._build_feature_schema_from_day(first_schema_day, schema_path)
        first_schema_day.processed = None
        first_schema_day.normalized = None
        normalizer_frames.clear()
        gc.collect()

        print("Derivative normalizer fitted.")
        return normalizer, ordered_feature_columns

    def save_split_outputs(
        self,
        processed_splits: dict[str, list[ProcessedDay]],
        *,
        processed_dir: Path | None = None,
        sequence_dir: Path | None = None,
    ) -> None:
        """Legacy in-memory output writer; RAM-safe run_fold writes per day."""
        save_processed_csv = self.config.preprocessing.save_processed_dataframes
        print(
            "Saving sequence outputs"
            + (" and processed CSV outputs." if save_processed_csv else " without processed CSV outputs.")
        )
        target_processed_dir = processed_dir or self.processed_dir
        target_sequence_dir = sequence_dir or self.sequence_dir
        if save_processed_csv:
            target_processed_dir.mkdir(parents=True, exist_ok=True)
        target_sequence_dir.mkdir(parents=True, exist_ok=True)

        for split, days in processed_splits.items():
            processed_split_dir = target_processed_dir / split
            sequence_split_dir = target_sequence_dir / split
            if save_processed_csv:
                processed_split_dir.mkdir(parents=True, exist_ok=True)
            sequence_split_dir.mkdir(parents=True, exist_ok=True)

            for day in days:
                normalized = self._require_frame(day, "normalized", "output saving")

                if save_processed_csv:
                    csv_path = processed_split_dir / f"{day.pair.output_stem}_processed.csv"
                    normalized.to_csv(csv_path, index=False)
                    day.processed_csv_path = csv_path
                else:
                    day.processed_csv_path = None

                prefix = sequence_split_dir / day.pair.output_stem
                day.sequence_paths = self.sequence_builder.save(normalized, prefix)
                print(f"Saved outputs for {day.pair.label} ({split}).")

    def _build_and_save_one_day(
        self,
        day: ProcessedDay,
        *,
        split: str,
        normalizer: DerivativeNormalizer,
        ordered_feature_columns: list[str],
        processed_dir: Path,
        sequence_dir: Path,
    ) -> tuple[int, int]:
        """Build snapshots, normalize, align, and save one day."""
        day.split = split
        message_features = self._require_frame(day, "message_features", "single-day snapshot construction")
        day.processed = self.snapshot_processor.transform(message_features, source_label=day.pair.label)
        processed = self._require_frame(day, "processed", "single-day normalization")
        print(
            f"Finished preprocessing for {day.pair.label}: "
            f"{processed.shape[0]} rows, {processed.shape[1]} columns."
        )
        day.normalized = normalizer.transform(processed)
        self._align_day_to_feature_schema(day, ordered_feature_columns)
        normalized = self._require_frame(day, "normalized", "single-day output saving")

        processed_split_dir = processed_dir / split
        sequence_split_dir = sequence_dir / split
        save_processed_csv = self.config.preprocessing.save_processed_dataframes
        if save_processed_csv:
            processed_split_dir.mkdir(parents=True, exist_ok=True)
        sequence_split_dir.mkdir(parents=True, exist_ok=True)

        if save_processed_csv:
            csv_path = processed_split_dir / f"{day.pair.output_stem}_processed.csv"
            normalized.to_csv(csv_path, index=False)
            day.processed_csv_path = csv_path
        else:
            day.processed_csv_path = None

        prefix = sequence_split_dir / day.pair.output_stem
        day.sequence_paths = self.sequence_builder.save(normalized, prefix)
        print(f"Saved outputs for {day.pair.label} ({split}).")
        return normalized.shape

    def run_fold(
        self,
        fold: FoldConfig,
        split_pairs: dict[str, list[LobFilePair]],
    ) -> dict[str, dict[str, tuple[int, int]]]:
        fold_start = perf_counter()
        print(f"Starting fold {fold.id}.")
        self.reset_fold_state()

        train_days: list[ProcessedDay] = []
        train_pairs = split_pairs.get("train", [])
        if self.uses_fitted_smoothing_threshold():
            for pair in train_pairs:
                train_days.append(self.prepare_unlabeled_pair(pair, "train"))
            if self.uses_split_fitted_smoothing_threshold():
                self._fit_split_smoothing_threshold(train_days, "train")
            else:
                self._fit_train_smoothing_threshold(train_days)
            for day in train_days:
                self._label_and_build_message_features(day)
        else:
            for pair in train_pairs:
                train_days.append(self.prepare_pair(pair, "train"))
        volume_bar_scaling = self._fit_volume_bar_features(train_days)
        sample_clock_counts: dict[str, dict[str, dict[str, int]]] = {split: {} for split in SPLIT_NAMES}
        for day in train_days:
            sample_clock_counts["train"][day.pair.output_stem] = self._sample_clock_counts_for_day(day)

        label_accumulator = self._new_label_distribution_accumulator() if self._tracks_label_distribution() else None
        floor_comparison: dict[str, object] | None = None
        stats_path = self.fold_derivatives_stats_path(fold.id)
        fold_processed_dir = self.processed_dir / fold.id
        fold_sequence_dir = self.sequence_dir / fold.id
        try:
            floor_comparison = self.adaptive_threshold_floor_comparison(train_days)
            for day in train_days:
                day.release_raw_frames()
            gc.collect()

            plgs_parameters = self.fit_price_static_plgs_parameters(train_days)
            volume_static_exp = self.fit_volume_static_exp_parameters(train_days)
            self.optimize_fast_smoothing_lambdas(train_days)
            normalizer, ordered_feature_columns = self._fit_train_normalizer_streaming(
                train_days,
                stats_path=stats_path,
                schema_path=self.fold_feature_schema_path(fold.id),
            )
        finally:
            for day in train_days:
                day.release_all_frames()
            train_days.clear()
            gc.collect()

        summary: dict[str, dict[str, tuple[int, int]]] = {split: {} for split in SPLIT_NAMES}

        def process_streamed_day(day: ProcessedDay, split: str) -> None:
            try:
                sample_clock_counts[split][day.pair.output_stem] = self._sample_clock_counts_for_day(day)
                if label_accumulator is not None:
                    self._accumulate_label_distribution(label_accumulator, split, day)
                summary[split][day.pair.output_stem] = self._build_and_save_one_day(
                    day,
                    split=split,
                    normalizer=normalizer,
                    ordered_feature_columns=ordered_feature_columns,
                    processed_dir=fold_processed_dir,
                    sequence_dir=fold_sequence_dir,
                )
            finally:
                day.release_all_frames()
                gc.collect()

        for split in SPLIT_NAMES:
            pairs = split_pairs.get(split, [])
            print(f"Starting streaming preprocessing split '{split}' with {len(pairs)} day(s).")
            if (
                self.uses_split_fitted_smoothing_threshold()
                and split not in self.smoothing_threshold_results
                and pairs
            ):
                split_days = [self.prepare_unlabeled_pair(pair, split) for pair in pairs]
                self._fit_split_smoothing_threshold(split_days, split)
                for day in split_days:
                    self._label_and_build_message_features(day)
                    process_streamed_day(day, split)
            else:
                for pair in pairs:
                    day = self.prepare_pair(pair, split)
                    process_streamed_day(day, split)
            print(f"Finished streaming preprocessing split '{split}'.")

        label_distribution = self._finalize_label_distribution(label_accumulator, floor_comparison)
        self.print_label_distribution(fold.id, label_distribution)
        fold_duration_seconds = perf_counter() - fold_start
        metadata_path = save_preprocessing_metadata(
            self.config,
            fold_sequence_dir,
            lambda_results=self.fast_smoothing_lambda_results,
            label_distribution=label_distribution,
            price_static_plgs=plgs_parameters,
            volume_static_exp=volume_static_exp,
            volume_bar_scaling=volume_bar_scaling,
            smoothing_threshold=self._smoothing_threshold_metadata(),
            sample_clock=self._sample_clock_metadata(),
            sample_clock_counts=sample_clock_counts,
            timing={
                "fold_preprocessing_seconds": round(fold_duration_seconds, 6),
                "fold_preprocessing_duration": format_duration(fold_duration_seconds),
            },
        )
        print(f"Saved preprocessing metadata for fold {fold.id} to {metadata_path}.")
        print(f"Finished fold {fold.id} ({format_duration(fold_duration_seconds)}).")
        return summary

    def run(
        self,
        selected_fold_ids: set[str] | None = None,
    ) -> dict[str, dict[str, dict[str, tuple[int, int]]]]:
        pipeline_start = perf_counter()
        print("Starting LOB processing pipeline.")
        pairs = self.discover_pairs()
        print(f"Discovered {len(pairs)} message/orderbook file pair(s).")

        selected_folds = [
            fold
            for fold in self.config.folds
            if selected_fold_ids is None or fold.id in selected_fold_ids
        ]
        if selected_fold_ids is not None:
            found = {fold.id for fold in selected_folds}
            missing = sorted(selected_fold_ids - found)
            if missing:
                raise ValueError(f"Unknown fold id(s): {missing}")
            print(f"Selected {len(selected_folds)} fold(s): {', '.join(fold.id for fold in selected_folds)}.")

        summary: dict[str, dict[str, dict[str, tuple[int, int]]]] = {}
        for fold in selected_folds:
            split_pairs = self.split_pairs_for_fold(pairs, fold)
            summary[fold.id] = self.run_fold(fold, split_pairs)
        print(f"LOB processing pipeline finished ({format_duration(perf_counter() - pipeline_start)}).")
        return summary
