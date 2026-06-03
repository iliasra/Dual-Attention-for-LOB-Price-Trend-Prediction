from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

try:
    from configuration import AdaptiveThresholdConfig, LabelConfig, SmoothingLabelConfig, TripleBarrierLabelConfig
except ImportError:  # pragma: no cover
    from .configuration import AdaptiveThresholdConfig, LabelConfig, SmoothingLabelConfig, TripleBarrierLabelConfig


TRAIN_FITTED_SMOOTHING_THRESHOLDS = {"mean_spread", "mean_pct"}


def calculate_midprice(
    df: pd.DataFrame,
    bid_col: str = "bid_price_1",
    ask_col: str = "ask_price_1",
) -> pd.Series:
    return (df[bid_col] + df[ask_col]) / 2.0


def calculate_spread(
    df: pd.DataFrame,
    bid_col: str = "bid_price_1",
    ask_col: str = "ask_price_1",
) -> pd.Series:
    return df[ask_col] - df[bid_col]


class SmoothingStrategy(Protocol):
    def __call__(self, midprices: pd.Series) -> pd.Series:
        ...


@dataclass(slots=True)
class SmoothingMethodA:
    k: int = 10

    def __post_init__(self) -> None:
        if self.k < 0:
            raise ValueError("SmoothingMethodA requires k >= 0.")

    def __call__(self, midprices: pd.Series) -> pd.Series:
        m_plus = midprices.rolling(window=self.k + 1).mean().shift(-self.k)
        return (m_plus - midprices) / midprices


@dataclass(slots=True)
class SmoothingMethodB:
    k: int = 10

    def __post_init__(self) -> None:
        if self.k < 0:
            raise ValueError("SmoothingMethodB requires k >= 0.")

    def __call__(self, midprices: pd.Series) -> pd.Series:
        m_minus = midprices.rolling(window=self.k + 1).mean()
        m_plus = midprices.rolling(window=self.k + 1).mean().shift(-self.k)
        return (m_plus - m_minus) / m_minus


@dataclass(slots=True)
class SmoothingMethodC:
    k: int = 5
    h: int = 10

    def __post_init__(self) -> None:
        _validate_method_c_windows(self.k, self.h)

    def __call__(self, midprices: pd.Series) -> pd.Series:
        w_minus = midprices.rolling(window=self.k + 1).mean()
        w_plus = midprices.rolling(window=self.k + 1).mean().shift(-self.h)
        return (w_plus - w_minus) / w_minus


def _validate_method_c_windows(k: int, h: int) -> None:
    if k < 0:
        raise ValueError("Smoothing method C requires k >= 0.")
    if h <= 0:
        raise ValueError("Smoothing method C requires h > 0.")
    if k >= h:
        raise ValueError("Smoothing method C requires k < h.")


def smoothing_method_A(midprices: pd.Series, k: int = 10) -> pd.Series:
    return SmoothingMethodA(k=k)(midprices)


def smoothing_method_B(midprices: pd.Series, k: int = 10) -> pd.Series:
    return SmoothingMethodB(k=k)(midprices)


def smoothing_method_C(midprices: pd.Series, k: int = 5, h: int = 10) -> pd.Series:
    return SmoothingMethodC(k=k, h=h)(midprices)


def calculate_adaptive_method_c_threshold(
    df: pd.DataFrame,
    midprices: pd.Series,
    *,
    k: int,
    h: int,
    bid_col: str,
    ask_col: str,
    config: AdaptiveThresholdConfig,
) -> pd.Series:
    return calculate_adaptive_method_c_threshold_components(
        df,
        midprices,
        k=k,
        h=h,
        bid_col=bid_col,
        ask_col=ask_col,
        config=config,
    )["threshold"]


def calculate_adaptive_method_c_threshold_components(
    df: pd.DataFrame,
    midprices: pd.Series,
    *,
    k: int,
    h: int,
    bid_col: str,
    ask_col: str,
    config: AdaptiveThresholdConfig,
) -> pd.DataFrame:
    _validate_method_c_windows(k, h)
    w_minus = midprices.rolling(window=k + 1).mean()
    spread = calculate_spread(df, bid_col=bid_col, ask_col=ask_col)
    exit_spread_min_periods = min(config.exit_spread_window, max(10, config.exit_spread_window // 10))
    exit_spread = spread.rolling(
        window=config.exit_spread_window,
        min_periods=exit_spread_min_periods,
    ).median()
    fee_price = midprices * config.round_trip_fees_bps / 10000.0
    cost_floor = (((spread + exit_spread) / 2.0) + fee_price) / w_minus

    if config.volatility_lambda == 0:
        volatility_floor = pd.Series(0.0, index=midprices.index)
    else:
        realized_c_returns = w_minus.pct_change(periods=h)
        volatility_min_periods = min(config.volatility_window, max(32, config.volatility_window // 10))
        local_sigma = realized_c_returns.rolling(
            window=config.volatility_window,
            min_periods=volatility_min_periods,
        ).std(ddof=0)
        volatility_floor = config.volatility_lambda * local_sigma

    threshold_values = np.maximum(cost_floor.to_numpy(dtype=float), volatility_floor.to_numpy(dtype=float))
    return pd.DataFrame(
        {
            "cost_floor": cost_floor,
            "volatility_floor": volatility_floor,
            "threshold": threshold_values,
        },
        index=midprices.index,
    )


@dataclass(slots=True)
class TrendThresholdClassifier:
    threshold: float | pd.Series

    def __call__(self, l_values: pd.Series) -> pd.Series:
        threshold = self.threshold
        if isinstance(threshold, pd.Series):
            threshold = threshold.reindex(l_values.index)
        labels = pd.Series(0, index=l_values.index, dtype=int)
        labels[l_values > threshold] = 1
        labels[l_values < -threshold] = -1
        return labels


def classify_trend(l_values: pd.Series, threshold: float | pd.Series) -> pd.Series:
    return TrendThresholdClassifier(threshold=threshold)(l_values)


def is_train_fitted_smoothing_threshold(value: object) -> bool:
    """Return whether a smoothing threshold must be fitted on train."""
    return isinstance(value, str) and value.strip().lower() in TRAIN_FITTED_SMOOTHING_THRESHOLDS


def _finite_spread_midprice_arrays(
    df: pd.DataFrame,
    *,
    bid_col: str,
    ask_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite spread and midprice arrays from one orderbook frame."""
    spread = calculate_spread(df, bid_col=bid_col, ask_col=ask_col).to_numpy(dtype=float)
    midprice = calculate_midprice(df, bid_col=bid_col, ask_col=ask_col).to_numpy(dtype=float)
    valid_mask = np.isfinite(spread) & np.isfinite(midprice) & (midprice != 0.0)
    return spread[valid_mask], midprice[valid_mask]


def fit_train_smoothing_threshold(
    dataframes: list[pd.DataFrame],
    config: SmoothingLabelConfig,
) -> dict[str, object]:
    """Fit a scalar smoothing threshold from train orderbook frames."""
    if not is_train_fitted_smoothing_threshold(config.threshold):
        raise ValueError("Smoothing threshold fitting requires threshold='mean_spread' or 'mean_pct'.")

    mode = str(config.threshold).lower()
    n_values = 0
    spread_sum = 0.0
    midprice_sum = 0.0
    pct_sum = 0.0

    for df in dataframes:
        spread, midprice = _finite_spread_midprice_arrays(
            df,
            bid_col=config.bid_column,
            ask_col=config.ask_column,
        )
        if spread.size == 0:
            continue
        n_values += int(spread.size)
        spread_sum += float(spread.sum())
        midprice_sum += float(midprice.sum())
        pct_sum += float((spread / midprice).sum())

    if n_values == 0:
        raise ValueError("Cannot fit smoothing threshold: no finite train spread/midprice values found.")
    if midprice_sum == 0.0:
        raise ValueError("Cannot fit smoothing threshold: train midprice sum is zero.")

    mean_spread = spread_sum / n_values
    mean_midprice = midprice_sum / n_values
    if mode == "mean_pct":
        threshold = pct_sum / n_values
        formula = "mean((ask_price - bid_price) / midprice)"
    else:
        threshold = mean_spread / mean_midprice
        formula = "mean(ask_price - bid_price) / mean(midprice)"

    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError(f"Fitted smoothing threshold must be finite and non-negative, got {threshold}.")

    return {
        "enabled": True,
        "mode": mode,
        "fit_split": "train",
        "value": float(threshold),
        "formula": formula,
        "n_values": int(n_values),
        "bid_column": config.bid_column,
        "ask_column": config.ask_column,
        "mean_spread": float(mean_spread),
        "mean_midprice": float(mean_midprice),
        "mean_pct": float(pct_sum / n_values),
    }


@dataclass(slots=True)
class TripleBarrierLabeler:
    horizon: int = 10
    upper_barrier_ticks: float = 2.0
    lower_barrier_ticks: float = 3.0
    price_col: str = "midprice"

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        prices = df[self.price_col]
        labels = pd.Series(0, index=prices.index, dtype=int)

        for index in range(len(prices) - self.horizon):
            current_price = prices.iloc[index]
            upper_barrier = current_price + self.upper_barrier_ticks
            lower_barrier = current_price - self.lower_barrier_ticks
            future_prices = prices.iloc[index + 1 : index + self.horizon + 1]

            upper_candidates = future_prices[future_prices >= upper_barrier]
            lower_candidates = future_prices[future_prices <= lower_barrier]
            upper_hit_idx = upper_candidates.index[0] if not upper_candidates.empty else None
            lower_hit_idx = lower_candidates.index[0] if not lower_candidates.empty else None

            if upper_hit_idx is not None and lower_hit_idx is not None:
                labels.iloc[index] = 1 if upper_hit_idx < lower_hit_idx else -1
            elif upper_hit_idx is not None:
                labels.iloc[index] = 1
            elif lower_hit_idx is not None:
                labels.iloc[index] = -1

        return labels


def triple_barrier_method(
    df: pd.DataFrame,
    horizon: int = 10,
    upper_barrier_ticks: int = 2,
    lower_barrier_ticks: int = 3,
    price_col: str = "midprice",
) -> pd.Series:
    labeler = TripleBarrierLabeler(
        horizon=horizon,
        upper_barrier_ticks=upper_barrier_ticks,
        lower_barrier_ticks=lower_barrier_ticks,
        price_col=price_col,
    )
    return labeler(df)


class TargetLabelPipeline:
    def __init__(self, config: LabelConfig):
        self.config = config

    def transform(
        self,
        df: pd.DataFrame,
        *,
        smoothing_threshold_override: float | None = None,
    ) -> pd.DataFrame:
        strategy = self.config.strategy.lower()
        if strategy == "smoothing":
            return self._add_smoothing_labels(
                df,
                self.config.smoothing,
                threshold_override=smoothing_threshold_override,
            )
        if strategy == "triple_barrier":
            return self._add_triple_barrier_labels(df, self.config.triple_barrier)
        raise ValueError(f"Unsupported labeling strategy: {self.config.strategy}")

    def _add_smoothing_labels(
        self,
        df: pd.DataFrame,
        config: SmoothingLabelConfig,
        *,
        threshold_override: float | None = None,
    ) -> pd.DataFrame:
        result = df.copy()
        midprices = calculate_midprice(result, bid_col=config.bid_column, ask_col=config.ask_column)

        method_name = config.method.upper()
        if method_name == "A":
            pct_changes = SmoothingMethodA(k=config.k)(midprices)
        elif method_name == "B":
            pct_changes = SmoothingMethodB(k=config.k)(midprices)
        elif method_name == "C":
            pct_changes = SmoothingMethodC(k=config.k, h=config.h)(midprices)
        else:
            raise ValueError(f"Unknown smoothing method: {config.method}")

        threshold = config.threshold if threshold_override is None else float(threshold_override)
        if threshold_override is not None and (
            config.adaptive_threshold is not None and config.adaptive_threshold.enabled
        ):
            raise ValueError("A fitted smoothing threshold override cannot be combined with adaptive labels.")
        if (
            method_name == "C"
            and config.adaptive_threshold is not None
            and config.adaptive_threshold.enabled
        ):
            threshold = calculate_adaptive_method_c_threshold(
                result,
                midprices,
                k=config.k,
                h=config.h,
                bid_col=config.bid_column,
                ask_col=config.ask_column,
                config=config.adaptive_threshold,
            )
        elif threshold is None:
            raise ValueError(
                "Smoothing label threshold cannot be null unless adaptive method-C thresholding is enabled. "
                "Set a numeric threshold, use mean_spread/mean_pct through the preprocessing pipeline, "
                "or enable adaptive_threshold."
            )
        elif is_train_fitted_smoothing_threshold(threshold):
            raise ValueError(
                "preprocessing.labels.smoothing.threshold is train-fitted; "
                "fit it on train and pass smoothing_threshold_override before labeling."
            )

        valid_mask = pct_changes.notna() & np.isfinite(pct_changes)
        if isinstance(threshold, pd.Series):
            valid_mask = valid_mask & threshold.notna() & np.isfinite(threshold)
            threshold = threshold.loc[valid_mask].reset_index(drop=True)
        result = result.loc[valid_mask].copy().reset_index(drop=True)
        pct_changes = pct_changes.loc[valid_mask].reset_index(drop=True)
        result["trend_label"] = TrendThresholdClassifier(threshold)(pct_changes)
        return result

    def _add_triple_barrier_labels(
        self,
        df: pd.DataFrame,
        config: TripleBarrierLabelConfig,
    ) -> pd.DataFrame:
        result = df.copy()
        price_column = config.price_column or "midprice"

        if config.price_column is None:
            result["midprice"] = calculate_midprice(
                result,
                bid_col=config.bid_column,
                ask_col=config.ask_column,
            )

        result["trend_label"] = TripleBarrierLabeler(
            horizon=config.horizon,
            upper_barrier_ticks=config.upper_barrier_ticks,
            lower_barrier_ticks=config.lower_barrier_ticks,
            price_col=price_column,
        )(result)
        return result


def add_target_labels_smoothing(
    df: pd.DataFrame,
    threshold: float | None = None,
    method: str = "A",
    k: int = 10,
    h: int = 10,
    bid_col: str = "bid_price_1",
    ask_col: str = "ask_price_1",
) -> pd.DataFrame:
    config = LabelConfig(
        strategy="smoothing",
        smoothing=SmoothingLabelConfig(
            threshold=threshold,
            method=method,
            k=k,
            h=h,
            bid_column=bid_col,
            ask_column=ask_col,
            adaptive_threshold=None,
        ),
        triple_barrier=TripleBarrierLabelConfig(
            horizon=10,
            upper_barrier_ticks=2.0,
            lower_barrier_ticks=3.0,
            bid_column=bid_col,
            ask_column=ask_col,
            price_column=None,
        ),
    )
    return TargetLabelPipeline(config).transform(df)


def add_target_labels_triple_barrier(
    df: pd.DataFrame,
    horizon: int = 10,
    upper_barrier_ticks: float = 0.01,
    lower_barrier_ticks: float = -0.01,
    bid_col: str = "bid_price_1",
    ask_col: str = "ask_price_1",
    price_col: str | None = None,
) -> pd.DataFrame:
    config = LabelConfig(
        strategy="triple_barrier",
        smoothing=SmoothingLabelConfig(
            threshold=None,
            method="C",
            k=5,
            h=10,
            bid_column=bid_col,
            ask_column=ask_col,
            adaptive_threshold=None,
        ),
        triple_barrier=TripleBarrierLabelConfig(
            horizon=horizon,
            upper_barrier_ticks=upper_barrier_ticks,
            lower_barrier_ticks=lower_barrier_ticks,
            bid_column=bid_col,
            ask_column=ask_col,
            price_column=price_col,
        ),
    )
    return TargetLabelPipeline(config).transform(df)
