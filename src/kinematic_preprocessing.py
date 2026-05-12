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
    from configuration import DataConfig, FastKinematicConfig, PreprocessingConfig, load_config
    from fast_kinematic_preprocessing import (
        KINEMATIC_SUFFIXES,
        PenalizedBSplineKinematicTokenizer,
        lambda_for_effective_degrees_of_freedom,
    )
    from utils import save_yaml
except ImportError:  # pragma: no cover
    from .configuration import DataConfig, FastKinematicConfig, PreprocessingConfig, load_config
    from .fast_kinematic_preprocessing import (
        KINEMATIC_SUFFIXES,
        PenalizedBSplineKinematicTokenizer,
        lambda_for_effective_degrees_of_freedom,
    )
    from .utils import save_yaml

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
    if k <= 0:
        raise ValueError("Exponential scaling k must be > 0.")
    return 1 - np.exp(-x / k)


def choose_exp_scaling_k(quantile_value: float, target: float) -> float:
    if quantile_value <= 0:
        raise ValueError("Cannot fit exponential scaling k from a non-positive quantile value.")
    if not 0.0 < target < 1.0:
        raise ValueError("Exponential scaling target must be in (0, 1).")
    return float(-quantile_value / np.log(1.0 - target))


def fit_exp_scaling_parameters(
    values: pd.Series | np.ndarray,
    *,
    quantile: float,
    target: float,
) -> dict[str, float]:
    if not 0.0 <= quantile <= 100.0:
        raise ValueError("Exponential scaling quantile must be in [0, 100].")

    numeric = pd.to_numeric(pd.Series(values).astype(float), errors="coerce")
    finite_values = numeric[np.isfinite(numeric)]
    if finite_values.empty:
        raise ValueError("Cannot fit exponential scaling parameters without finite training values.")

    quantile_value = float(finite_values.quantile(quantile / 100.0))
    k = choose_exp_scaling_k(quantile_value, target)
    return {
        "quantile": float(quantile),
        "target": float(target),
        "quantile_value": quantile_value,
        "k": k,
        "n_values": int(len(finite_values)),
    }

def _static_centering(
    price: ArrayLike,
    center: ArrayLike,
    tick: float,
    *,
    absolute: bool = False,
    remove_touch_tick: bool = False,
    side: ArrayLike | None = None,
) -> pd.Series:
    if tick <= 0:
        raise ValueError("tick must be > 0 for static price centering.")

    distance = np.round((price - center) / tick)
    if not isinstance(distance, pd.Series):
        distance = pd.Series(distance)

    if side is not None:
        side_values = side if isinstance(side, pd.Series) else pd.Series(side, index=distance.index)
        side_values = side_values.reindex(distance.index)
        known_side = side_values.notna()
        distance.loc[known_side] = distance.loc[known_side] * side_values.loc[known_side]

    if absolute:
        distance = distance.abs()

    if remove_touch_tick:
        distance = (distance - 1).clip(lower=0)

    return distance

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


def plgs_value(x: float, tau_start: float, tau_max: float) -> float:
    if tau_max <= tau_start:
        raise ValueError("tau_max must be > tau_start")

    mu = 1 - (1 / (tau_max - tau_start))
    if x <= tau_start:
        scaled = x
    else:
        scaled = tau_start + (tau_max - tau_start) * (1 - mu ** (x - tau_start))
    return float(scaled / tau_max)


def choose_plgs_tau_max(x95: float, tau_start: float, target: float = 0.50) -> float:
    if target <= 0.0 or target >= 1.0:
        raise ValueError("PLGS target must be in (0, 1).")

    lo = tau_start + 1.0001
    hi = max(10.0, 100.0 * float(x95))
    if hi <= lo:
        hi = lo * 2.0

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        y_value = plgs_value(float(x95), tau_start, mid)
        if y_value > target:
            lo = mid
        else:
            hi = mid
    return float(hi)


def price_static_distance_frame(df: pd.DataFrame, columns: list[str], tick_size: float) -> pd.DataFrame:
    best_bid, best_ask = _best_prices(df)
    features: dict[str, np.ndarray] = {}

    for column in columns:
        if "ask" in column.lower():
            center = best_bid
        elif "bid" in column.lower():
            center = best_ask
        else:
            continue

        centered = _static_centering(
            df[column],
            center,
            tick_size,
            absolute=True,
            remove_touch_tick=True,
        )
        features[column] = centered.to_numpy(dtype=float)

    return pd.DataFrame(features, index=df.index)


def fit_plgs_parameters(
    values: pd.Series | np.ndarray,
    *,
    tau_start: float,
    tau_clip_quantile: float = 0.99,
    tau_max_quantile: float = 0.95,
    tau_max_target: float = 0.50,
) -> dict[str, float]:
    numeric = pd.to_numeric(pd.Series(values).astype(float), errors="coerce")
    finite_values = numeric[np.isfinite(numeric)]
    if finite_values.empty:
        raise ValueError("Cannot fit PLGS parameters without finite training values.")

    tau_clip = float(finite_values.quantile(tau_clip_quantile))
    x95 = float(finite_values.quantile(tau_max_quantile))
    tau_max = choose_plgs_tau_max(x95, tau_start=tau_start, target=tau_max_target)
    return {
        "tau_start": float(tau_start),
        "tau_clip": tau_clip,
        "tau_max": tau_max,
        "x95": x95,
        "x99": tau_clip,
        "tau_max_target": float(tau_max_target),
        "n_values": int(len(finite_values)),
    }


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
    alpha: float
    reference: Literal["tick", "time"]

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
    tau_max: float | None

    def transform_rows(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        if self.tau_max is None:
            raise ValueError("Price static PLGS tau_max has not been fitted.")

        distances = price_static_distance_frame(df, columns, self.tick_size)
        features: dict[str, np.ndarray] = {}

        for column in distances.columns:
            transformed = PLGS(
                distances[column],
                tau_start=self.tau_start,
                tau_clip=self.tau_clip,
                tau_max=self.tau_max,
            )
            features[f"{column}_static_plgs"] = transformed.to_numpy(dtype=float)

        return pd.DataFrame(features, index=df.index)

    def transform(self, df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
        rows = self.transform_rows(df, columns)
        return {column: float(rows[column].iloc[-1]) for column in rows.columns}


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


def _make_fast_tokenizer(
    window: int,
    fast_config: FastKinematicConfig,
    chunk_size: int,
) -> PenalizedBSplineKinematicTokenizer:
    smoothing_lambda = fast_config.selected_smoothing_lambda
    if smoothing_lambda is None:
        smoothing_lambda = lambda_for_effective_degrees_of_freedom(
            target_df=fast_config.df,
            window=window,
            n_basis=fast_config.n_basis,
        )
    return PenalizedBSplineKinematicTokenizer(
        window=window,
        n_basis=fast_config.n_basis,
        smoothing_lambda=smoothing_lambda,
        eval_at=fast_config.eval_at,
        chunk_size=chunk_size,
        dtype=np.float64,
    )


def _empty_kinematic_tokens(n_windows: int) -> np.ndarray:
    return np.empty((n_windows, 0, len(KINEMATIC_SUFFIXES)), dtype=np.float64)


def _fast_price_tokens(
    df: pd.DataFrame,
    columns: list[str],
    *,
    window: int,
    tick_size: float,
    fast_config: FastKinematicConfig,
    chunk_size: int,
    progress_desc: str | None = None,
) -> np.ndarray:
    n_windows = len(df) - window + 1
    if not columns:
        return _empty_kinematic_tokens(n_windows)

    tokenizer = _make_fast_tokenizer(window, fast_config, chunk_size)
    values = df[columns].to_numpy(dtype=np.float64)
    tokens = tokenizer.transform_values(values, progress_desc=progress_desc)

    best_bid, best_ask = _best_prices(df)
    centers = ((best_bid.to_numpy(dtype=np.float64) + best_ask.to_numpy(dtype=np.float64)) * 0.5)[:n_windows]
    constant_response = tokenizer.H @ np.ones(window, dtype=np.float64)

    return (tokens - centers[:, None, None] * constant_response[None, None, :]) / tick_size


def _fast_volume_tokens(
    df: pd.DataFrame,
    columns: list[str],
    *,
    window: int,
    fast_config: FastKinematicConfig,
    chunk_size: int,
    progress_desc: str | None = None,
) -> np.ndarray:
    n_windows = len(df) - window + 1
    if not columns:
        return _empty_kinematic_tokens(n_windows)

    tokenizer = _make_fast_tokenizer(window, fast_config, chunk_size)
    values = np.log1p(df[columns].to_numpy(dtype=np.float64))
    return tokenizer.transform_values(values, progress_desc=progress_desc)


def _kinematic_tokens_to_frame(tokens: np.ndarray, columns: list[str]) -> pd.DataFrame:
    data: dict[str, np.ndarray] = {}
    for feature_index, column in enumerate(columns):
        for suffix_index, suffix in enumerate(KINEMATIC_SUFFIXES):
            data[f"{column}_kin_{suffix}"] = tokens[:, feature_index, suffix_index]
    return pd.DataFrame(data)


def _kinematic_tokens_to_dict(tokens: np.ndarray, columns: list[str]) -> dict[str, float]:
    if len(tokens) != 1:
        raise ValueError("Expected exactly one fast kinematic token window.")
    return {
        column_name: float(values.iloc[0])
        for column_name, values in _kinematic_tokens_to_frame(tokens, columns).items()
    }


@dataclass(slots=True)
class FastPriceKinematicProcessor:
    tick_size: float
    fast_config: FastKinematicConfig
    chunk_size: int

    def transform(self, df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
        tokens = _fast_price_tokens(
            df,
            columns,
            window=len(df),
            tick_size=self.tick_size,
            fast_config=self.fast_config,
            chunk_size=self.chunk_size,
        )
        return _kinematic_tokens_to_dict(tokens, columns)


@dataclass(slots=True)
class FastVolumeKinematicProcessor:
    fast_config: FastKinematicConfig
    chunk_size: int

    def transform(self, df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
        tokens = _fast_volume_tokens(
            df,
            columns,
            window=len(df),
            fast_config=self.fast_config,
            chunk_size=self.chunk_size,
        )
        return _kinematic_tokens_to_dict(tokens, columns)


@dataclass(slots=True)
class VolumeStaticProcessor:
    k: float | None

    def transform_rows(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        if self.k is None:
            raise ValueError("Volume static exponential scaling k has not been fitted.")

        features: dict[str, np.ndarray] = {}
        for column in columns:
            transformed = _exp_scaling(df[column], self.k)
            features[f"{column}_static_exp"] = np.asarray(transformed, dtype=float)
        return pd.DataFrame(features, index=df.index)

    def transform(self, df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
        rows = self.transform_rows(df, columns)
        return {column: float(rows[column].iloc[-1]) for column in rows.columns}


@dataclass(slots=True)
class MessageOrderbookJoiner:
    time_column: str

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
        directional_side: pd.Series | None = None

        if "direction" in result.columns:
            buy_mask = result["direction"] == 1
            sell_mask = result["direction"] == -1
            opposite_best.loc[buy_mask] = best_ask.loc[buy_mask]
            opposite_best.loc[sell_mask] = best_bid.loc[sell_mask]
            directional_side = pd.Series(index=result.index, dtype=float)
            directional_side.loc[buy_mask] = 1.0
            directional_side.loc[sell_mask] = -1.0

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
            side=directional_side,
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
        stream_config: object,
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

        tokenization_method = self.preprocessing_config.kinematic_tokenization.method

        if self.preprocessing_config.price_kinematic.enabled:
            if tokenization_method == "fast":
                processor = FastPriceKinematicProcessor(
                    tick_size=self.preprocessing_config.price_kinematic.tick_size,
                    fast_config=self.preprocessing_config.price_kinematic.fast,
                    chunk_size=self.preprocessing_config.kinematic_tokenization.chunk_size,
                )
            else:
                processor = PriceKinematicProcessor(
                    time_column=time_column,
                    tick_size=self.preprocessing_config.price_kinematic.tick_size,
                    extractor=KinematicTokenExtractor(
                        alpha=self.preprocessing_config.price_kinematic.basis.alpha,
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
            if tokenization_method == "fast":
                processor = FastVolumeKinematicProcessor(
                    fast_config=self.preprocessing_config.volume_kinematic.fast,
                    chunk_size=self.preprocessing_config.kinematic_tokenization.chunk_size,
                )
            else:
                processor = VolumeKinematicProcessor(
                    time_column=time_column,
                    extractor=KinematicTokenExtractor(
                        alpha=self.preprocessing_config.volume_kinematic.basis.alpha,
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

    @staticmethod
    def _label(source_label: str | None) -> str:
        return source_label or "dataset"

    def transform(self, df: pd.DataFrame, source_label: str | None = None) -> pd.DataFrame:
        label = self._label(source_label)
        print(f"Starting snapshot processing for {label}: {len(df)} input rows.")
        if self.preprocessing_config.kinematic_tokenization.method == "fast":
            return self._transform_fast(df, source_label=label)
        return self._transform_basis(df, source_label=label)

    def _window_end_positions(self, df: pd.DataFrame) -> np.ndarray:
        window_size = self.preprocessing_config.snapshot_window
        if len(df) < window_size:
            raise ValueError(f"Dataframe length ({len(df)}) < window size ({window_size}).")
        return np.arange(window_size - 1, len(df))

    @staticmethod
    def _empty_component(n_rows: int) -> pd.DataFrame:
        return pd.DataFrame(index=pd.RangeIndex(n_rows))

    @staticmethod
    def _slice_window_ends(frame: pd.DataFrame, end_positions: np.ndarray) -> pd.DataFrame:
        return frame.iloc[end_positions].reset_index(drop=True)

    def _build_static_frame(
        self,
        df: pd.DataFrame,
        end_positions: np.ndarray,
        source_label: str | None = None,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        end_rows = df.iloc[end_positions].reset_index(drop=True)

        if self.preprocessing_config.price_static.enabled:
            processor = PriceStaticProcessor(
                tick_size=self.preprocessing_config.price_static.tick_size,
                tau_start=self.preprocessing_config.price_static.tau_start,
                tau_clip=self.preprocessing_config.price_static.tau_clip,
                tau_max=self.preprocessing_config.price_static.tau_max,
            )
            price_columns = self.window_processor._resolve_stream_columns(
                df,
                self.preprocessing_config.price_static,
                "price",
            )
            frames.append(processor.transform_rows(end_rows, price_columns).reset_index(drop=True))

        if self.preprocessing_config.volume_static.enabled:
            processor = VolumeStaticProcessor(k=self.preprocessing_config.volume_static.k)
            volume_columns = self.window_processor._resolve_stream_columns(
                df,
                self.preprocessing_config.volume_static,
                "volume",
            )
            frames.append(processor.transform_rows(end_rows, volume_columns).reset_index(drop=True))

        if not frames:
            static_frame = self._empty_component(len(end_positions))
        else:
            static_frame = pd.concat(frames, axis=1)

        print(f"Static stream calculated for {self._label(source_label)}: {len(end_positions)} window-end rows.")
        return static_frame

    def _build_passthrough_temporal_frame(self, df: pd.DataFrame, end_positions: np.ndarray) -> pd.DataFrame:
        time_column = self.data_config.time_column
        if time_column not in df.columns:
            raise ValueError(f"Unable to find timestamp column '{time_column}'.")

        price_cols = self.window_processor.column_resolver.price_columns(df)
        volume_cols = self.window_processor.column_resolver.volume_columns(df)
        passthrough = self.window_processor._passthrough_columns(df, price_cols, volume_cols)
        end_rows = df.iloc[end_positions].reset_index(drop=True)
        result = self._empty_component(len(end_positions))

        for column in passthrough:
            if column == time_column and not self.preprocessing_config.temporal_features.keep_timestamp:
                continue
            result[column] = end_rows[column].to_numpy()

        final_time = end_rows[time_column].to_numpy(dtype=float)
        if self.preprocessing_config.temporal_features.keep_timestamp:
            result[time_column] = final_time
        else:
            window_size = self.preprocessing_config.snapshot_window
            start_positions = end_positions - window_size + 1
            start_time = df.iloc[start_positions][time_column].to_numpy(dtype=float)
            result["time_rel"] = final_time - start_time

        if self.preprocessing_config.temporal_features.add_day_sincos:
            sincos_day = time_to_sincos(
                final_time,
                freq=self.preprocessing_config.temporal_features.day_frequency,
            )
            result["time_day_sin"] = sincos_day[:, 0]
            result["time_day_cos"] = sincos_day[:, 1]

        return result

    def _build_basis_kinematic_frame(
        self,
        df: pd.DataFrame,
        end_positions: np.ndarray,
        source_label: str | None = None,
    ) -> pd.DataFrame:
        results: list[dict[str, float]] = []
        time_column = self.data_config.time_column
        window_size = self.preprocessing_config.snapshot_window
        print(f"Starting basis kinematic stream for {self._label(source_label)}: {len(end_positions)} rows.")

        price_processor: PriceKinematicProcessor | None = None
        price_columns: list[str] = []
        if self.preprocessing_config.price_kinematic.enabled:
            price_processor = PriceKinematicProcessor(
                time_column=time_column,
                tick_size=self.preprocessing_config.price_kinematic.tick_size,
                extractor=KinematicTokenExtractor(
                    alpha=self.preprocessing_config.price_kinematic.basis.alpha,
                    reference=self.preprocessing_config.price_kinematic.reference,
                ),
            )
            price_columns = self.window_processor._resolve_stream_columns(
                df,
                self.preprocessing_config.price_kinematic,
                "price",
            )

        volume_processor: VolumeKinematicProcessor | None = None
        volume_columns: list[str] = []
        if self.preprocessing_config.volume_kinematic.enabled:
            volume_processor = VolumeKinematicProcessor(
                time_column=time_column,
                extractor=KinematicTokenExtractor(
                    alpha=self.preprocessing_config.volume_kinematic.basis.alpha,
                    reference=self.preprocessing_config.volume_kinematic.reference,
                ),
            )
            volume_columns = self.window_processor._resolve_stream_columns(
                df,
                self.preprocessing_config.volume_kinematic,
                "volume",
            )

        for end_position in tqdm(
            end_positions,
            desc="Processing snapshot windows",
            total=len(end_positions),
            mininterval=5,
        ):
            window_data = df.iloc[end_position - window_size + 1 : end_position + 1].reset_index(drop=True)
            tokens: dict[str, float] = {}
            if price_processor is not None:
                tokens.update(price_processor.transform(window_data, price_columns))
            if volume_processor is not None:
                tokens.update(volume_processor.transform(window_data, volume_columns))
            results.append(tokens)

        if not results:
            frame = self._empty_component(len(end_positions))
        else:
            frame = pd.DataFrame(results)
        print(f"Basis kinematic stream calculated for {self._label(source_label)}: {len(end_positions)} rows.")
        return frame

    def _build_fast_kinematic_frame(self, df: pd.DataFrame, source_label: str | None = None) -> pd.DataFrame:
        window_size = self.preprocessing_config.snapshot_window
        token_frames: list[pd.DataFrame] = []
        chunk_size = self.preprocessing_config.kinematic_tokenization.chunk_size
        n_windows = len(df) - window_size + 1
        label = self._label(source_label)
        print(f"Starting fast kinematic stream for {label}: {n_windows} rows.")

        if self.preprocessing_config.price_kinematic.enabled:
            price_columns = self.window_processor._resolve_stream_columns(
                df,
                self.preprocessing_config.price_kinematic,
                "price",
            )
            price_tokens = _fast_price_tokens(
                df,
                price_columns,
                window=window_size,
                tick_size=self.preprocessing_config.price_kinematic.tick_size,
                fast_config=self.preprocessing_config.price_kinematic.fast,
                chunk_size=chunk_size,
                progress_desc=f"Fast price kinematic [{label}]",
            )
            token_frames.append(_kinematic_tokens_to_frame(price_tokens, price_columns))

        if self.preprocessing_config.volume_kinematic.enabled:
            volume_columns = self.window_processor._resolve_stream_columns(
                df,
                self.preprocessing_config.volume_kinematic,
                "volume",
            )
            volume_tokens = _fast_volume_tokens(
                df,
                volume_columns,
                window=window_size,
                fast_config=self.preprocessing_config.volume_kinematic.fast,
                chunk_size=chunk_size,
                progress_desc=f"Fast volume kinematic [{label}]",
            )
            token_frames.append(_kinematic_tokens_to_frame(volume_tokens, volume_columns))

        if not token_frames:
            frame = self._empty_component(n_windows)
        else:
            frame = pd.concat(token_frames, axis=1)
        print(f"Fast kinematic stream calculated for {label}: {n_windows} rows.")
        return frame

    def _transform_basis(self, df: pd.DataFrame, source_label: str | None = None) -> pd.DataFrame:
        end_positions = self._window_end_positions(df)
        static_frame = self._build_static_frame(df, end_positions, source_label=source_label)
        kinematic_frame = self._build_basis_kinematic_frame(df, end_positions, source_label=source_label)
        passthrough_frame = self._build_passthrough_temporal_frame(df, end_positions)
        result = pd.concat([static_frame, kinematic_frame, passthrough_frame], axis=1)
        print(f"Snapshot processing finished for {self._label(source_label)}: {len(result)} output rows.")
        return result

    def _transform_fast(self, df: pd.DataFrame, source_label: str | None = None) -> pd.DataFrame:
        end_positions = self._window_end_positions(df)
        static_frame = self._build_static_frame(df, end_positions, source_label=source_label)
        passthrough_frame = self._build_passthrough_temporal_frame(df, end_positions)
        kinematic_frame = self._build_fast_kinematic_frame(df, source_label=source_label)
        result = pd.concat([static_frame, passthrough_frame, kinematic_frame], axis=1)
        print(f"Snapshot processing finished for {self._label(source_label)}: {len(result)} output rows.")
        return result


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

        save_yaml(self.output_path, self.stats_)
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
) -> pd.DataFrame:
    """Legacy wrapper"""
    return MessageOrderbookJoiner(time_column=time_col).transform(message_df, orderbook_df)


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
        raw_data_dir=config.data.raw_data_dir,
        processed_data_dir=config.data.processed_data_dir,
        sequence_data_dir=config.data.sequence_data_dir,
        logs_dir=config.data.logs_dir,
        tick_size=tick,
        time_column=timestamp_col,
        label_column=config.data.label_column,
        label_mapping=config.data.label_mapping,
        price_columns=price_cols,
        volume_columns=volume_cols,
        feature_exclude_columns=config.data.feature_exclude_columns,
        sequence_window=config.data.sequence_window,
    )
    preprocessing_config = config.preprocessing
    preprocessing_config.snapshot_window = window
    preprocessing_config.temporal_features.keep_timestamp = keep_timestamp
    preprocessing_config.price_kinematic.basis.alpha = alpha
    preprocessing_config.price_kinematic.reference = ref
    preprocessing_config.price_kinematic.tick_size = tick
    preprocessing_config.volume_kinematic.basis.alpha = alpha
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
        raw_data_dir=config.data.raw_data_dir,
        processed_data_dir=config.data.processed_data_dir,
        sequence_data_dir=config.data.sequence_data_dir,
        logs_dir=config.data.logs_dir,
        tick_size=tick,
        time_column=timestamp_col,
        label_column=config.data.label_column,
        label_mapping=config.data.label_mapping,
        price_columns=price_cols,
        volume_columns=volume_cols,
        feature_exclude_columns=config.data.feature_exclude_columns,
        sequence_window=config.data.sequence_window,
    )
    preprocessing_config = config.preprocessing
    preprocessing_config.snapshot_window = window
    preprocessing_config.temporal_features.keep_timestamp = keep_timestamp
    preprocessing_config.price_kinematic.basis.alpha = alpha
    preprocessing_config.price_kinematic.reference = ref
    preprocessing_config.price_kinematic.tick_size = tick
    preprocessing_config.volume_kinematic.basis.alpha = alpha
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
