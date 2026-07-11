from __future__ import annotations

from dataclasses import dataclass, field
from math import log
from typing import Any

import numpy as np
import pandas as pd

try:
    from configuration import MessageConfig, SampleClockConfig
except ImportError:  # pragma: no cover
    from .configuration import MessageConfig, SampleClockConfig


POSITIVE_VOLUME_COLUMNS = (
    "bar_buy_trade_volume",
    "bar_sell_trade_volume",
    "bar_add_size",
    "bar_cancel_size",
    "bar_delete_size",
    "bar_execution_size",
)
SIGNED_VOLUME_COLUMNS = ("bar_signed_trade_volume",)
COUNT_COLUMNS = ("bar_trade_count", "bar_message_count")
TIME_COLUMNS = ("bar_duration_seconds",)
RAW_BAR_COLUMNS = {
    *POSITIVE_VOLUME_COLUMNS,
    *SIGNED_VOLUME_COLUMNS,
    *COUNT_COLUMNS,
    *TIME_COLUMNS,
    "bar_total_trade_volume",
}
SCALING_QUANTILE = 95.0
SCALING_TARGET = 0.5
EPS = 1e-12


def _value_token(value: Any) -> str:
    """Return a stable column-name token for a categorical value."""
    if isinstance(value, (int, np.integer)):
        return f"minus_{abs(int(value))}" if int(value) < 0 else str(int(value))
    text = str(value).replace("-", "minus_").replace(".", "p")
    return text.strip() or "unknown"


def bar_type_count_column(value: int) -> str:
    """Return the raw count column for one LOBSTER message type."""
    return f"bar_type_{_value_token(value)}_count"


def bar_direction_count_column(value: int) -> str:
    """Return the raw count column for one LOBSTER direction."""
    return f"bar_direction_{_value_token(value)}_count"


def _log1p_nonnegative(values: pd.Series | np.ndarray) -> np.ndarray:
    """Apply log1p after clipping tiny numerical negatives to zero."""
    array = np.asarray(values, dtype=np.float64)
    return np.log1p(np.clip(array, 0.0, None))


def _signed_log1p_exp(values: pd.Series | np.ndarray, k: float) -> np.ndarray:
    """Scale a signed volume by sign-preserving log1p exponential compression."""
    array = np.asarray(values, dtype=np.float64)
    magnitude = np.log1p(np.abs(array))
    return np.sign(array) * (1.0 - np.exp(-magnitude / max(float(k), EPS)))


def _positive_log1p_exp(values: pd.Series | np.ndarray, k: float) -> np.ndarray:
    """Scale a positive volume by log1p followed by exponential compression."""
    transformed = _log1p_nonnegative(values)
    return 1.0 - np.exp(-transformed / max(float(k), EPS))


def _fit_exp_k(values: pd.Series) -> dict[str, float | int]:
    """Fit the exponential scale k on finite log1p-transformed train values."""
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)].clip(lower=0.0)
    if finite.empty:
        return {"k": 1.0, "quantile": SCALING_QUANTILE, "target": SCALING_TARGET, "quantile_value": 0.0, "n_values": 0}
    transformed = np.log1p(finite.to_numpy(dtype=np.float64, copy=False))
    quantile_value = float(np.quantile(transformed, SCALING_QUANTILE / 100.0))
    if quantile_value <= 0.0:
        k = 1.0
    else:
        k = float(-quantile_value / log(1.0 - SCALING_TARGET))
    return {
        "k": k,
        "quantile": SCALING_QUANTILE,
        "target": SCALING_TARGET,
        "quantile_value": quantile_value,
        "n_values": int(len(transformed)),
    }


@dataclass(slots=True)
class VolumeClockSampler:
    sample_clock_config: SampleClockConfig
    message_config: MessageConfig
    time_column: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert an event stream into exact-grid volume-clock bars."""
        if not self.sample_clock_config.enabled:
            return df.copy()
        self._validate_input(df)

        type_values = [int(value) for value in self.message_config.categorical_value_map.get("type", [1, 2, 3, 4, 5])]
        direction_values = [
            int(value)
            for value in self.message_config.categorical_value_map.get("direction", [-1, 1])
        ]
        snapshot_columns = self._snapshot_columns(df)
        accumulator = self._new_accumulator(type_values, direction_values)
        bars: list[dict[str, float]] = []

        step = float(self.sample_clock_config.volume_step_shares)
        current_volume = 0.0
        volume_time = 0
        first_wall_time = float(df[self.time_column].iloc[0])
        previous_bar_wall_time: float | None = None

        for _, row in df.iterrows():
            clock_volume = self._clock_volume(row)
            if clock_volume <= 0.0:
                self._add_contribution(accumulator, row, 1.0, type_values, direction_values)
                continue

            remaining = clock_volume
            while current_volume + remaining >= step - EPS:
                portion = max(step - current_volume, 0.0)
                fraction = portion / clock_volume if clock_volume > 0.0 else 0.0
                if fraction > 0.0:
                    self._add_contribution(accumulator, row, fraction, type_values, direction_values)

                volume_time += 1
                close_time = float(row[self.time_column])
                if previous_bar_wall_time is None:
                    duration = max(close_time - first_wall_time, 0.0)
                else:
                    duration = max(close_time - previous_bar_wall_time, 0.0)
                accumulator["bar_duration_seconds"] = duration
                bars.append(self._emit_bar(row, snapshot_columns, accumulator, volume_time, close_time))
                previous_bar_wall_time = close_time
                accumulator = self._new_accumulator(type_values, direction_values)
                current_volume = 0.0
                remaining -= portion

                if portion <= EPS and remaining <= EPS:
                    break

            if remaining > EPS:
                fraction = remaining / clock_volume
                self._add_contribution(accumulator, row, fraction, type_values, direction_values)
                current_volume += remaining

        if not bars:
            raise ValueError(
                "Volume-clock sampling emitted no complete bars; "
                "decrease preprocessing.sample_clock.volume_step_shares or check traded volume."
            )
        return pd.DataFrame(bars).reset_index(drop=True)

    def _validate_input(self, df: pd.DataFrame) -> None:
        required = {
            self.time_column,
            self.message_config.size_column,
            "type",
            "direction",
        }
        missing = sorted(column for column in required if column not in df.columns)
        if missing:
            raise ValueError(f"Cannot build volume-clock bars; missing column(s): {missing}.")

    def _snapshot_columns(self, df: pd.DataFrame) -> list[str]:
        excluded = {
            self.time_column,
            self.message_config.size_column,
            self.message_config.price_column,
            self.message_config.order_id_column,
            "type",
            "direction",
            "delta_t",
        }
        return [column for column in df.columns if column not in excluded]

    def _clock_volume(self, row: pd.Series) -> float:
        size = max(float(row[self.message_config.size_column]), 0.0)
        if self.sample_clock_config.volume_source == "message_size":
            return size
        return size if int(row["type"]) in set(self.sample_clock_config.trade_type_values) else 0.0

    def _new_accumulator(self, type_values: list[int], direction_values: list[int]) -> dict[str, float]:
        accumulator = {column: 0.0 for column in RAW_BAR_COLUMNS}
        for value in type_values:
            accumulator[bar_type_count_column(value)] = 0.0
        for value in direction_values:
            accumulator[bar_direction_count_column(value)] = 0.0
        return accumulator

    def _add_contribution(
        self,
        accumulator: dict[str, float],
        row: pd.Series,
        fraction: float,
        type_values: list[int],
        direction_values: list[int],
    ) -> None:
        size = max(float(row[self.message_config.size_column]), 0.0) * float(fraction)
        message_type = int(row["type"])
        direction = int(row["direction"])
        trade_types = set(self.sample_clock_config.trade_type_values)

        accumulator["bar_message_count"] += float(fraction)
        if message_type in type_values:
            accumulator[bar_type_count_column(message_type)] += float(fraction)
        if direction in direction_values:
            accumulator[bar_direction_count_column(direction)] += float(fraction)

        if message_type == 1:
            accumulator["bar_add_size"] += size
        elif message_type == 2:
            accumulator["bar_cancel_size"] += size
        elif message_type == 3:
            accumulator["bar_delete_size"] += size

        if message_type in trade_types:
            accumulator["bar_trade_count"] += float(fraction)
            accumulator["bar_execution_size"] += size
            accumulator["bar_total_trade_volume"] += size
            # LOBSTER direction is the passive limit-order side. Executions
            # against a sell limit order (direction=-1) are buyer-initiated;
            # executions against a buy limit order are seller-initiated.
            if direction == -1:
                accumulator["bar_buy_trade_volume"] += size
                accumulator["bar_signed_trade_volume"] += size
            elif direction == 1:
                accumulator["bar_sell_trade_volume"] += size
                accumulator["bar_signed_trade_volume"] -= size

    def _emit_bar(
        self,
        row: pd.Series,
        snapshot_columns: list[str],
        accumulator: dict[str, float],
        volume_time: int,
        close_time: float,
    ) -> dict[str, float]:
        output = {column: row[column] for column in snapshot_columns}
        output["volume_time"] = float(volume_time)
        output["volume_wall_time"] = close_time
        output.update(accumulator)
        return output


@dataclass(slots=True)
class VolumeBarFeatureProcessor:
    message_config: MessageConfig
    scaling_: dict[str, dict[str, float | int]] = field(default_factory=dict)

    def fit(self, dataframes: list[pd.DataFrame]) -> dict[str, dict[str, float | int]]:
        """Fit train-only exponential scales for raw volume-bar aggregates."""
        if not dataframes:
            raise ValueError("Cannot fit volume-bar feature scaling on an empty dataframe list.")

        concatenated = pd.concat(dataframes, axis=0, ignore_index=True)
        scaling: dict[str, dict[str, float | int]] = {}
        for raw_column in (*POSITIVE_VOLUME_COLUMNS, *SIGNED_VOLUME_COLUMNS):
            if raw_column not in concatenated.columns:
                continue
            values = concatenated[raw_column].abs() if raw_column in SIGNED_VOLUME_COLUMNS else concatenated[raw_column]
            scaling[raw_column] = _fit_exp_k(pd.Series(values))
        self.scaling_ = scaling
        return self.scaling_

    @property
    def fitted(self) -> bool:
        """Return whether train-only volume scales have been fitted."""
        return bool(self.scaling_)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create model-facing transformed volume-bar features."""
        if not self.fitted:
            raise ValueError("VolumeBarFeatureProcessor must be fitted before transform().")

        raw_columns = self.raw_bar_columns(df)
        result = df.drop(columns=raw_columns, errors="ignore").copy()

        for column in COUNT_COLUMNS:
            if column in df.columns:
                result[f"{column}_log1p"] = _log1p_nonnegative(df[column])
        for column in self._dynamic_count_columns(df):
            result[f"{column}_log1p"] = _log1p_nonnegative(df[column])
        for column in POSITIVE_VOLUME_COLUMNS:
            if column in df.columns:
                result[f"{column}_log1p_exp"] = _positive_log1p_exp(df[column], float(self.scaling_[column]["k"]))
        for column in SIGNED_VOLUME_COLUMNS:
            if column in df.columns:
                result[f"{column}_signed_log1p_exp"] = _signed_log1p_exp(df[column], float(self.scaling_[column]["k"]))
        if "bar_total_trade_volume" in df.columns and "bar_signed_trade_volume" in df.columns:
            total = df["bar_total_trade_volume"].to_numpy(dtype=np.float64)
            signed = df["bar_signed_trade_volume"].to_numpy(dtype=np.float64)
            imbalance = np.divide(signed, total, out=np.zeros_like(signed), where=total > 0.0)
            result["bar_trade_imbalance"] = np.clip(imbalance, -1.0, 1.0)
        if "bar_duration_seconds" in df.columns:
            result["bar_duration_seconds_log1p"] = _log1p_nonnegative(df["bar_duration_seconds"])
        return result

    def raw_bar_columns(self, df: pd.DataFrame) -> list[str]:
        """Return raw volume-bar aggregate columns present in a dataframe."""
        raw = set(RAW_BAR_COLUMNS)
        raw.update(self._dynamic_count_columns(df))
        return [column for column in df.columns if column in raw]

    @staticmethod
    def _dynamic_count_columns(df: pd.DataFrame) -> list[str]:
        return [
            column
            for column in df.columns
            if (
                column.startswith("bar_type_")
                or column.startswith("bar_direction_")
            )
            and column.endswith("_count")
        ]
