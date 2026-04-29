from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, NamedTuple, Union
import re

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from tqdm import tqdm

try:
    from configuration import DataConfig, PreprocessingConfig, StreamConfig, load_config
    from utils import append_to_yaml
except ImportError:  # pragma: no cover
    from .configuration import DataConfig, PreprocessingConfig, StreamConfig, load_config
    from .utils import append_to_yaml

ArrayLike = Union[float, int, np.ndarray, pd.Series]


def sanity_scheck(dataframes: list[pd.DataFrame]) -> None:
    for dataframe in dataframes:
        if 9999999999 in dataframe.max(numeric_only=True).tolist():
            raise ValueError("One dataframe contains abnormal value 9999999999.")
        if -9999999999 in dataframe.min(numeric_only=True).tolist():
            raise ValueError("One dataframe contains abnormal value -9999999999.")


def _price_level_metadata(column_name: str) -> tuple[str, str, int] | None:
    match = re.fullmatch(r"(ask|bid)_(price|size)_(\d+)", column_name.lower())
    if match is None:
        return None
    side, kind, level = match.groups()
    return side, kind, int(level)


def _ghost_level_columns(
    df: pd.DataFrame,
    dummy_values: list[float | int],
) -> list[str]:
    columns_to_drop: set[str] = set()
    grouped_columns: dict[tuple[str, int], dict[str, str]] = {}

    for column in df.columns:
        metadata = _price_level_metadata(column)
        if metadata is None:
            continue
        side, kind, level = metadata
        grouped_columns.setdefault((side, level), {})[kind] = column

    for (_, _), columns in grouped_columns.items():
        price_column = columns.get("price")
        if price_column is None:
            continue
        is_ghost_level = df[price_column].isin(dummy_values).all()
        if not is_ghost_level:
            continue
        columns_to_drop.add(price_column)
        size_column = columns.get("size")
        if size_column is not None:
            columns_to_drop.add(size_column)

    return sorted(columns_to_drop)


def handle_abnormal_prices(dataframes: list[pd.DataFrame]) -> None:
    dummy_values = [9999999999, -9999999999, 9999999999.0, -9999999999.0]
    for dataframe in dataframes:
        columns_to_drop = _ghost_level_columns(dataframe, dummy_values)
        if columns_to_drop:
            dataframe.drop(columns=columns_to_drop, inplace=True)
    sanity_scheck(dataframes)

class ZScoreResult(NamedTuple):
    normalized: pd.Series
    mean: float
    std: float

class DerivativeStats(NamedTuple):
    mean: float
    std: float
    q001: float
    q999: float
    n_nan: int
    n_inf: int

def _zscore(series: pd.Series) -> ZScoreResult:
    mean = float(series.mean())
    std = float(series.std(ddof=0))
    return ZScoreResult((series - mean) / (std + 1e-8), mean, std)

def _derivative_stats(series: pd.Series) -> DerivativeStats:
    values = pd.to_numeric(series, errors="coerce")
    finite_mask = np.isfinite(values)
    finite_values = values[finite_mask]
    zscore_result = _zscore(series)
    return DerivativeStats(
        mean=zscore_result.mean,
        std=zscore_result.std,
        q001=float(finite_values.quantile(0.001)),
        q999=float(finite_values.quantile(0.999)),
        n_nan=int(values.isna().sum()),
        n_inf=int(np.isinf(values).sum()),
    )

def _log1p(x: ArrayLike) -> pd.Series:
    return np.log1p(x)

def _exp_scaling(x: ArrayLike, k: float) -> pd.Series:
    return 1 - np.exp(-x / k)

def _static_centering(price: ArrayLike, center: ArrayLike, tick: float) -> pd.Series:
    return np.round(price - center) / tick - 1

def _kine_centering(x: ArrayLike, mid_price: float, tick: float) -> pd.Series:
    return (x - mid_price) / tick

def PLGS(x: ArrayLike, tau_start: float, tau_clip: float | None, tau_max: float) -> pd.Series:
    if tau_max <= tau_start:
        raise ValueError("tau_max must be > tau_start")

    mu = 1 - (1 / (tau_max - tau_start))
    clipped = x.clip(upper=tau_clip) if tau_clip is not None else x.copy()

    scaled = pd.Series(0.0, index=clipped.index)
    linear_mask = clipped <= tau_start
    scaled[linear_mask] = clipped[linear_mask]

    geom_mask = ~linear_mask
    x_geom = clipped[geom_mask] - tau_start
    scaled[geom_mask] = tau_start + (tau_max - tau_start) * (1.0 - (mu**x_geom))

    return scaled / tau_max

def time_to_sincos(timestamp: np.ndarray, freq: int = 86400) -> np.ndarray:
    angle = 2 * np.pi * (timestamp % freq) / freq
    return np.stack((np.sin(angle), np.cos(angle)), axis=-1)

def min_max_norm(series: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(series, dtype=float)
    span = values.max() - values.min()
    if span == 0:
        return np.zeros_like(values, dtype=float)
    return (values - values.min()) / span

def _detect_price_volume_columns(
    df: pd.DataFrame,
    price_cols: list[str] | None,
    volume_cols: list[str] | None,
) -> tuple[list[str], list[str]]:
    resolved_price = price_cols or [column for column in df.columns if "_price_" in column.lower()]
    resolved_volume = volume_cols or [column for column in df.columns if "_size_" in column.lower()]
    return resolved_price, resolved_volume


def _best_prices(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    return df["bid_price_1"], df["ask_price_1"]


@dataclass(slots=True)
class ColumnResolver:
    data_config: DataConfig

    def price_columns(self, df: pd.DataFrame, override: list[str] | None = None) -> list[str]:
        detected, _ = _detect_price_volume_columns(df, override or self.data_config.price_columns, None)
        return [column for column in detected if column in df.columns]

    def volume_columns(self, df: pd.DataFrame, override: list[str] | None = None) -> list[str]:
        _, detected = _detect_price_volume_columns(df, None, override or self.data_config.volume_columns)
        return [column for column in detected if column in df.columns]


@dataclass(slots=True)
class KinematicTokenExtractor:
    alpha: float = 5.0
    reference: Literal["tick", "time"] = "tick"

    def extract(self, series: pd.Series, time: pd.Series, label: str) -> dict[str, float]:
        if len(series) < 2:
            raise ValueError("At least two samples are required to extract kinematic tokens.")

        if self.reference == "time":
            tau = min_max_norm(time)
            dt = max(float(time.max() - time.min()), 1e-8)
        else:
            tau = min_max_norm(np.arange(len(series), dtype=float))
            dt = max(float(len(series) - 1), 1e-8)

        x_values = series.to_numpy(dtype=float)
        spline_order = min(3, len(x_values) - 1)
        variance = float(np.var(x_values))
        smoothing = len(x_values) * variance * (self.alpha**2) if variance > 0 else 1.0

        spline = UnivariateSpline(tau, x_values, k=spline_order, s=smoothing)
        return {
            f"{label}_pos": float(spline(1.0)),
            f"{label}_vel": float(spline.derivative(1)(1.0) / dt),
            f"{label}_acc": float(spline.derivative(min(2, spline_order))(1.0) / (dt**2))
            if spline_order >= 2
            else 0.0,
            f"{label}_jrk": float(spline.derivative(3)(1.0) / (dt**3)) if spline_order >= 3 else 0.0,
        }


@dataclass(slots=True)
class PriceKinematicProcessor:
    time_column: str
    tick_size: float
    extractor: KinematicTokenExtractor

    def transform(self, df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
        best_bid, best_ask = _best_prices(df)
        mid_price = float((best_bid.iloc[0] + best_ask.iloc[0]) * 0.5)
        tokens: dict[str, float] = {}

        for column in columns:
            centered = _kine_centering(df[column], mid_price, self.tick_size)
            tokens.update(self.extractor.extract(centered, df[self.time_column], f"{column}_kin"))

        return tokens


@dataclass(slots=True)
class PriceStaticProcessor:
    tick_size: float
    tau_start: float
    tau_clip: float | None
    tau_max: float

    def transform(self, df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
        best_bid, best_ask = _best_prices(df)
        last_row = df.iloc[[-1]]
        features: dict[str, float] = {}

        for column in columns:
            if "ask" in column.lower():
                center = best_bid.iloc[[-1]]
            elif "bid" in column.lower():
                center = best_ask.iloc[[-1]]
            else:
                continue

            centered = _static_centering(last_row[column], center, self.tick_size).abs()
            transformed = PLGS(
                centered,
                tau_start=self.tau_start,
                tau_clip=self.tau_clip,
                tau_max=self.tau_max,
            )
            features[f"{column}_static_plgs"] = float(transformed.iloc[-1])

        return features


@dataclass(slots=True)
class VolumeKinematicProcessor:
    time_column: str
    extractor: KinematicTokenExtractor

    def transform(self, df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
        tokens: dict[str, float] = {}
        for column in columns:
            transformed = pd.Series(_log1p(df[column]), index=df.index)
            tokens.update(self.extractor.extract(transformed, df[self.time_column], f"{column}_kin"))
        return tokens


@dataclass(slots=True)
class VolumeStaticProcessor:
    k: float

    def transform(self, df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
        last_row = df.iloc[[-1]]
        features: dict[str, float] = {}
        for column in columns:
            transformed = _exp_scaling(last_row[column], self.k)
            features[f"{column}_static_exp"] = float(transformed.iloc[-1])
        return features


@dataclass(slots=True)
class MessageOrderbookJoiner:
    time_column: str = "time"
    method: str = "ffill"

    def transform(self, message_df: pd.DataFrame, orderbook_df: pd.DataFrame) -> pd.DataFrame:
        if self.time_column not in message_df.columns:
            raise ValueError(f"Column '{self.time_column}' not found in message dataframe.")

        msg = message_df.reset_index(drop=True)
        ob = orderbook_df.reset_index(drop=True)

        if len(ob) != len(msg):
            raise ValueError(
                "Message and orderbook dataframes must have the same number of rows, "
                f"got {len(msg)} and {len(ob)}."
            )

        ob[self.time_column] = msg[self.time_column]
        ob["delta_t"] = msg[self.time_column] - msg[self.time_column].shift(1)
        if ob[self.time_column].isna().any():
            raise ValueError(f"Column '{self.time_column}' contains missing timestamps.")

        return pd.concat([ob, msg.drop(columns=[self.time_column], errors="ignore")], axis=1)


@dataclass(slots=True)
class MessageFeatureProcessor:
    time_column: str
    message_config: object

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.time_column not in df.columns:
            raise ValueError(f"Column '{self.time_column}' not found.")

        result = df.copy()
        size_column = self.message_config.size_column
        price_column = self.message_config.price_column

        if size_column not in result.columns or price_column not in result.columns:
            raise ValueError("Message dataframe must contain size and price columns.")
        if "bid_price_1" not in result.columns or "ask_price_1" not in result.columns:
            raise ValueError("bid_price_1 and ask_price_1 are required for message feature processing.")

        result["size_log1p"] = _log1p(result[size_column])

        best_bid = result["bid_price_1"]
        best_ask = result["ask_price_1"]
        opposite_best = pd.Series(index=result.index, dtype=float)

        if "direction" in result.columns:
            buy_mask = result["direction"] == 1
            sell_mask = result["direction"] == -1
            opposite_best.loc[buy_mask] = best_ask.loc[buy_mask]
            opposite_best.loc[sell_mask] = best_bid.loc[sell_mask]

        unknown_mask = opposite_best.isna()
        if unknown_mask.any():
            dist_to_bid = (result.loc[unknown_mask, price_column] - best_bid.loc[unknown_mask]).abs()
            dist_to_ask = (result.loc[unknown_mask, price_column] - best_ask.loc[unknown_mask]).abs()
            opposite_best.loc[unknown_mask] = np.where(
                dist_to_bid <= dist_to_ask,
                best_bid.loc[unknown_mask],
                best_ask.loc[unknown_mask],
            )

        result["price_static"] = _static_centering(
            result[price_column],
            opposite_best,
            self.message_config.tick_size,
        )

        for column, allowed_values in self.message_config.categorical_value_map.items():
            if column not in result.columns:
                continue
            for value in allowed_values:
                result[f"{column}_{value}"] = (result[column] == value).astype(int)

        return result.drop(columns=self.message_config.drop_columns, errors="ignore")


@dataclass(slots=True)
class SnapshotWindowProcessor:
    data_config: DataConfig
    preprocessing_config: PreprocessingConfig
    column_resolver: ColumnResolver = field(init=False)

    def __post_init__(self) -> None:
        self.column_resolver = ColumnResolver(self.data_config)

    def _resolve_stream_columns(
        self,
        df: pd.DataFrame,
        stream_config: StreamConfig,
        kind: Literal["price", "volume"],
    ) -> list[str]:
        if kind == "price":
            return self.column_resolver.price_columns(df, override=stream_config.columns)
        return self.column_resolver.volume_columns(df, override=stream_config.columns)

    def _passthrough_columns(self, df: pd.DataFrame, price_cols: list[str], volume_cols: list[str]) -> list[str]:
        excluded = set(price_cols) | set(volume_cols)
        return [column for column in df.columns if column not in excluded]

    def transform_window(self, df: pd.DataFrame) -> dict[str, float]:
        time_column = self.data_config.time_column
        if time_column not in df.columns:
            raise ValueError(f"Unable to find timestamp column '{time_column}'.")

        window = df.tail(self.preprocessing_config.snapshot_window).reset_index(drop=True)
        price_cols = self.column_resolver.price_columns(window)
        volume_cols = self.column_resolver.volume_columns(window)
        result: dict[str, float] = {}

        if self.preprocessing_config.price_static.enabled:
            processor = PriceStaticProcessor(
                tick_size=self.preprocessing_config.price_static.tick_size,
                tau_start=self.preprocessing_config.price_static.tau_start,
                tau_clip=self.preprocessing_config.price_static.tau_clip,
                tau_max=self.preprocessing_config.price_static.tau_max,
            )
            result.update(
                processor.transform(
                    window,
                    self._resolve_stream_columns(window, self.preprocessing_config.price_static, "price"),
                )
            )

        if self.preprocessing_config.volume_static.enabled:
            processor = VolumeStaticProcessor(k=self.preprocessing_config.volume_static.k)
            result.update(
                processor.transform(
                    window,
                    self._resolve_stream_columns(window, self.preprocessing_config.volume_static, "volume"),
                )
            )

        if self.preprocessing_config.price_kinematic.enabled:
            processor = PriceKinematicProcessor(
                time_column=time_column,
                tick_size=self.preprocessing_config.price_kinematic.tick_size,
                extractor=KinematicTokenExtractor(
                    alpha=self.preprocessing_config.price_kinematic.alpha,
                    reference=self.preprocessing_config.price_kinematic.reference,
                ),
            )
            result.update(
                processor.transform(
                    window,
                    self._resolve_stream_columns(window, self.preprocessing_config.price_kinematic, "price"),
                )
            )

        if self.preprocessing_config.volume_kinematic.enabled:
            processor = VolumeKinematicProcessor(
                time_column=time_column,
                extractor=KinematicTokenExtractor(
                    alpha=self.preprocessing_config.volume_kinematic.alpha,
                    reference=self.preprocessing_config.volume_kinematic.reference,
                ),
            )
            result.update(
                processor.transform(
                    window,
                    self._resolve_stream_columns(window, self.preprocessing_config.volume_kinematic, "volume"),
                )
            )

        passthrough = self._passthrough_columns(window, price_cols, volume_cols)
        last_row = window.iloc[-1]
        for column in passthrough:
            if column == time_column and not self.preprocessing_config.temporal_features.keep_timestamp:
                continue
            result[column] = last_row[column]

        final_time = float(window[time_column].iloc[-1])
        if self.preprocessing_config.temporal_features.keep_timestamp:
            result[time_column] = final_time
        else:
            result["time_rel"] = final_time - float(window[time_column].iloc[0])

        if self.preprocessing_config.temporal_features.add_day_sincos:
            sincos_day = time_to_sincos(
                np.array([final_time]),
                freq=self.preprocessing_config.temporal_features.day_frequency,
            )[0]
            result["time_day_sin"] = float(sincos_day[0])
            result["time_day_cos"] = float(sincos_day[1])

        return result


@dataclass(slots=True)
class SnapshotBatchProcessor:
    data_config: DataConfig
    preprocessing_config: PreprocessingConfig
    window_processor: SnapshotWindowProcessor = field(init=False)

    def __post_init__(self) -> None:
        self.window_processor = SnapshotWindowProcessor(self.data_config, self.preprocessing_config)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        window_size = self.preprocessing_config.snapshot_window
        if len(df) < window_size:
            raise ValueError(f"Dataframe length ({len(df)}) < window size ({window_size}).")

        results: list[dict[str, float]] = []
        for index in tqdm(
            range(window_size - 1, len(df)),
            desc="Processing snapshot windows",
            total=len(df) - window_size + 1,
            mininterval=2,
        ):
            window_data = df.iloc[index - window_size + 1 : index + 1].reset_index(drop=True)
            results.append(self.window_processor.transform_window(window_data))

        return pd.DataFrame(results)


def derivative_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if any(token in column.lower() for token in ("vel", "acc", "jrk"))
    ]


@dataclass(slots=True)
class TradingSessionFilter:
    time_column: str
    market_open_seconds: float
    market_close_seconds: float
    start_offset_minutes: int = 15
    end_offset_minutes: int = 15

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out the first/last 15 minutes of each trading day, to avoid any
        edge effect."""
        if self.time_column not in df.columns:
            raise ValueError(f"Column '{self.time_column}' not found for trading session filtering.")

        start_time = self.market_open_seconds + self.start_offset_minutes * 60
        end_time = self.market_close_seconds - self.end_offset_minutes * 60
        filtered = df.loc[(df[self.time_column] >= start_time) & (df[self.time_column] <= end_time)].copy()

        if filtered.empty:
            raise ValueError(
                f"No rows left after trimming to trading session window [{start_time}, {end_time}] seconds."
            )

        return filtered.reset_index(drop=True)


@dataclass(slots=True)
class DerivativeNormalizer:
    output_path: str | Path
    stats_: dict[str, dict[str, float | int]] = field(default_factory=dict)

    def fit(self, dataframes: list[pd.DataFrame]) -> "DerivativeNormalizer":
        if not dataframes:
            raise ValueError("Cannot fit derivative normalizer on an empty list of dataframes.")

        concatenated = pd.concat(dataframes, axis=0, ignore_index=True)
        self.stats_ = {}
        for column in derivative_feature_columns(concatenated):
            stats = _derivative_stats(concatenated[column])
            self.stats_[column] = {
                "mean": stats.mean,
                "std": stats.std,
                "q001": stats.q001,
                "q999": stats.q999,
                "n_nan": stats.n_nan,
                "n_inf": stats.n_inf,
            }

        append_to_yaml(self.output_path, self.stats_)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.stats_:
            self.stats_ = self.load_stats(self.output_path)
        if not self.stats_:
            raise ValueError("Derivative normalizer has no fitted statistics to apply.")

        normalized = df.copy()
        for column, stats in self.stats_.items():
            if column not in normalized.columns:
                continue
            mean = stats["mean"]
            std = stats["std"]
            normalized[column] = (normalized[column] - mean) / (std + 1e-8)
        return normalized

    @staticmethod
    def load_stats(path: str | Path) -> dict[str, dict[str, float | int]]:
        stats_path = Path(path)
        if not stats_path.exists():
            return {}

        import yaml

        with stats_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        loaded: dict[str, dict[str, float | int]] = {}
        for column, values in payload.items():
            if not isinstance(values, dict) or not {"mean", "std"} <= set(values):
                continue
            loaded[str(column)] = {
                "mean": float(values["mean"]),
                "std": float(values["std"]),
            }
            for optional_float in ("q001", "q999"):
                if optional_float in values:
                    loaded[str(column)][optional_float] = float(values[optional_float])
            for optional_count in ("n_nan", "n_inf"):
                if optional_count in values:
                    loaded[str(column)][optional_count] = int(values[optional_count])
        return loaded


@dataclass(slots=True)
class FittedDerivativeNormalizer:
    stats: dict[str, dict[str, float | int]]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        normalized = df.copy()
        for column, values in self.stats.items():
            if column not in normalized.columns:
                continue
            normalized[column] = (normalized[column] - values["mean"]) / (values["std"] + 1e-8)
        return normalized

# ---------------------------------------------------------------------------
# Legacy functional API
# ---------------------------------------------------------------------------
# These wrappers are kept for compatibility with notebooks or older
# scripts. The production pipeline uses the class-based processors above.

def price_kinematic_stream(
    df: pd.DataFrame,
    price_cols: list[str],
    time_col: str,
    tick: float,
    alpha: float,
    ref: Literal["tick", "time"],
) -> pd.DataFrame:
    """Legacy wrapper"""
    processor = PriceKinematicProcessor(
        time_column=time_col,
        tick_size=tick,
        extractor=KinematicTokenExtractor(alpha=alpha, reference=ref),
    )
    return pd.DataFrame([processor.transform(df, price_cols)], index=[df.index[-1]])


def price_static_stream(
    df: pd.DataFrame,
    price_cols: list[str],
    tick: float,
    tau_start: float,
    tau_clip: float | None,
    tau_max: float,
) -> pd.DataFrame:
    """Legacy wrapper"""
    processor = PriceStaticProcessor(tick_size=tick, tau_start=tau_start, tau_clip=tau_clip, tau_max=tau_max)
    return pd.DataFrame([processor.transform(df, price_cols)], index=[df.index[-1]])


def volume_kinematic_stream(
    df: pd.DataFrame,
    volume_cols: list[str],
    timezone_col: str,
    alpha: float,
    ref: Literal["tick", "time"],
) -> pd.DataFrame:
    """Legacy wrapper"""
    processor = VolumeKinematicProcessor(
        time_column=timezone_col,
        extractor=KinematicTokenExtractor(alpha=alpha, reference=ref),
    )
    return pd.DataFrame([processor.transform(df, volume_cols)], index=[df.index[-1]])


def volume_static_stream(df: pd.DataFrame, volume_cols: list[str], k: float) -> pd.DataFrame:
    """Legacy wrapper"""
    processor = VolumeStaticProcessor(k=k)
    return pd.DataFrame([processor.transform(df, volume_cols)], index=[df.index[-1]])


def join_message_orderbook(
    message_df: pd.DataFrame,
    orderbook_df: pd.DataFrame,
    time_col: str = "time",
    method: str = "ffill",
) -> pd.DataFrame:
    """Legacy wrapper"""
    return MessageOrderbookJoiner(time_column=time_col, method=method).transform(message_df, orderbook_df)


def extract_kinematic_tokens(
    series: pd.Series,
    time: pd.Series,
    label: str,
    ref: Literal["tick", "time"],
    alpha: float = 5.0,
) -> dict[str, float]:
    """Legacy wrapper"""
    return KinematicTokenExtractor(alpha=alpha, reference=ref).extract(series, time, label)


def process_orderbook_snapshot_df_window(
    df: pd.DataFrame,
    window: int = 100,
    timestamp_col: str = "time",
    price_cols: list[str] | None = None,
    volume_cols: list[str] | None = None,
    tick: float = 1.0,
    alpha: float = 5.0,
    tau_start: float = 1.0,
    tau_clip: float | None = 50.0,
    tau_max: float = 100.0,
    keep_timestamp: bool = True,
    ref: Literal["tick", "time"] = "tick",
    k: float = 2000.0,
) -> dict[str, float]:
    """Legacy wrapper"""
    config = load_config()
    data_config = DataConfig(
        time_column=timestamp_col,
        label_column=config.data.label_column,
        price_columns=price_cols,
        volume_columns=volume_cols,
        feature_exclude_columns=config.data.feature_exclude_columns,
        sequence_window=config.data.sequence_window,
    )
    preprocessing_config = config.preprocessing
    preprocessing_config.snapshot_window = window
    preprocessing_config.temporal_features.keep_timestamp = keep_timestamp
    preprocessing_config.price_kinematic.alpha = alpha
    preprocessing_config.price_kinematic.reference = ref
    preprocessing_config.price_kinematic.tick_size = tick
    preprocessing_config.volume_kinematic.alpha = alpha
    preprocessing_config.volume_kinematic.reference = ref
    preprocessing_config.price_static.tick_size = tick
    preprocessing_config.price_static.tau_start = tau_start
    preprocessing_config.price_static.tau_clip = tau_clip
    preprocessing_config.price_static.tau_max = tau_max
    preprocessing_config.volume_static.k = k
    return SnapshotWindowProcessor(data_config, preprocessing_config).transform_window(df)


def process_all_snapshot_windows(
    df: pd.DataFrame,
    window: int = 100,
    timestamp_col: str = "time",
    price_cols: list[str] | None = None,
    volume_cols: list[str] | None = None,
    tick: float = 1.0,
    alpha: float = 5.0,
    tau_start: float = 1.0,
    tau_clip: float | None = 50.0,
    tau_max: float = 100.0,
    keep_timestamp: bool = True,
    ref: Literal["tick", "time"] = "tick",
    k: float = 2000.0,
) -> pd.DataFrame:
    """Legacy wrapper"""
    config = load_config()
    data_config = DataConfig(
        time_column=timestamp_col,
        label_column=config.data.label_column,
        price_columns=price_cols,
        volume_columns=volume_cols,
        feature_exclude_columns=config.data.feature_exclude_columns,
        sequence_window=config.data.sequence_window,
    )
    preprocessing_config = config.preprocessing
    preprocessing_config.snapshot_window = window
    preprocessing_config.temporal_features.keep_timestamp = keep_timestamp
    preprocessing_config.price_kinematic.alpha = alpha
    preprocessing_config.price_kinematic.reference = ref
    preprocessing_config.price_kinematic.tick_size = tick
    preprocessing_config.volume_kinematic.alpha = alpha
    preprocessing_config.volume_kinematic.reference = ref
    preprocessing_config.price_static.tick_size = tick
    preprocessing_config.price_static.tau_start = tau_start
    preprocessing_config.price_static.tau_clip = tau_clip
    preprocessing_config.price_static.tau_max = tau_max
    preprocessing_config.volume_static.k = k
    return SnapshotBatchProcessor(data_config, preprocessing_config).transform(df)


def process_message_col(message_df: pd.DataFrame, tick: float) -> pd.DataFrame:
    """Legacy wrapper"""
    config = load_config()
    config.preprocessing.message.tick_size = tick
    return MessageFeatureProcessor(config.data.time_column, config.preprocessing.message).transform(message_df)


def z_score_derivatives(df: pd.DataFrame, filepath: str) -> pd.DataFrame:
    """Legacy wrapper"""
    normalizer = DerivativeNormalizer(filepath).fit([df])
    return normalizer.transform(df)
