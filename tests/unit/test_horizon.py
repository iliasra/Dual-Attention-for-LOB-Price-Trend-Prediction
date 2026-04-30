from __future__ import annotations

import numpy as np
import pandas as pd

from configuration import LabelConfig, SmoothingLabelConfig, TripleBarrierLabelConfig
from horizon import (
    SmoothingMethodA,
    TargetLabelPipeline,
    TrendThresholdClassifier,
    TripleBarrierLabeler,
    calculate_midprice,
    calculate_spread,
)


def make_smoothing_config(
    method: str = "C",
    threshold: float | None = None,
    k: int = 5,
    h: int = 10,
) -> LabelConfig:
    return LabelConfig(
        strategy="smoothing",
        smoothing=SmoothingLabelConfig(
            method=method,
            threshold=threshold,
            k=k,
            h=h,
            bid_column="bid_price_1",
            ask_column="ask_price_1",
        ),
        triple_barrier=TripleBarrierLabelConfig(
            horizon=10,
            upper_barrier_ticks=2.0,
            lower_barrier_ticks=3.0,
            bid_column="bid_price_1",
            ask_column="ask_price_1",
            price_column=None,
        ),
    )


def make_triple_barrier_config(
    horizon: int = 10,
    upper_barrier_ticks: float = 2.0,
    lower_barrier_ticks: float = 3.0,
) -> LabelConfig:
    return LabelConfig(
        strategy="triple_barrier",
        smoothing=SmoothingLabelConfig(
            method="C",
            threshold=None,
            k=5,
            h=10,
            bid_column="bid_price_1",
            ask_column="ask_price_1",
        ),
        triple_barrier=TripleBarrierLabelConfig(
            horizon=horizon,
            upper_barrier_ticks=upper_barrier_ticks,
            lower_barrier_ticks=lower_barrier_ticks,
            bid_column="bid_price_1",
            ask_column="ask_price_1",
            price_column=None,
        ),
    )


def test_calculate_midprice_and_spread() -> None:
    df = pd.DataFrame(
        {
            "bid_price_1": [99.0, 101.0],
            "ask_price_1": [101.0, 104.0],
        }
    )

    np.testing.assert_allclose(calculate_midprice(df), [100.0, 102.5])
    np.testing.assert_allclose(calculate_spread(df), [2.0, 3.0])


def test_smoothing_method_a_uses_future_rolling_mean() -> None:
    midprices = pd.Series([10.0, 12.0, 14.0, 16.0])

    result = SmoothingMethodA(k=1)(midprices)

    expected = pd.Series([0.1, 1.0 / 12.0, 1.0 / 14.0, np.nan])
    np.testing.assert_allclose(result.dropna(), expected.dropna())
    assert result.isna().iloc[-1]


def test_trend_threshold_classifier_maps_three_classes() -> None:
    values = pd.Series([-0.20, -0.05, 0.0, 0.05, 0.20])

    labels = TrendThresholdClassifier(threshold=0.1)(values)

    assert labels.tolist() == [-1, 0, 0, 0, 1]


def test_triple_barrier_labeler_uses_first_hit() -> None:
    df = pd.DataFrame({"midprice": [100.0, 103.0, 97.0, 100.0]})

    labels = TripleBarrierLabeler(
        horizon=2,
        upper_barrier_ticks=2.0,
        lower_barrier_ticks=2.0,
        price_col="midprice",
    )(df)

    assert labels.tolist() == [1, -1, 0, 0]


def test_target_label_pipeline_adds_smoothing_labels() -> None:
    df = pd.DataFrame(
        {
            "bid_price_1": [99.0, 100.0, 101.0, 102.0, 103.0],
            "ask_price_1": [101.0, 102.0, 103.0, 104.0, 105.0],
        }
    )
    config = make_smoothing_config(method="A", threshold=0.001, k=1, h=10)

    result = TargetLabelPipeline(config).transform(df)

    assert "trend_label" in result.columns
    assert result["trend_label"].tolist() == [1, 1, 1, 1]


def test_target_label_pipeline_adds_triple_barrier_labels() -> None:
    df = pd.DataFrame(
        {
            "bid_price_1": [99.0, 102.0, 96.0, 99.0],
            "ask_price_1": [101.0, 104.0, 98.0, 101.0],
        }
    )
    config = make_triple_barrier_config(horizon=2, upper_barrier_ticks=2.0, lower_barrier_ticks=2.0)

    result = TargetLabelPipeline(config).transform(df)

    assert result["trend_label"].tolist() == [1, -1, 0, 0]
