from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from configuration import AdaptiveThresholdConfig, LabelConfig, SmoothingLabelConfig, TripleBarrierLabelConfig
from horizon import (
    SmoothingMethodC,
    SmoothingMethodA,
    TargetLabelPipeline,
    TrendThresholdClassifier,
    TripleBarrierLabeler,
    calculate_adaptive_method_c_threshold,
    calculate_adaptive_method_c_threshold_components,
    calculate_midprice,
    calculate_spread,
    fit_train_smoothing_threshold,
)


def make_smoothing_config(
    method: str = "C",
    threshold: float | str | None = None,
    k: int = 5,
    h: int = 10,
    adaptive_threshold: AdaptiveThresholdConfig | None = None,
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
            adaptive_threshold=adaptive_threshold,
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
            adaptive_threshold=None,
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


def test_fit_train_smoothing_threshold_mean_pct() -> None:
    frame = pd.DataFrame(
        {
            "bid_price_1": [99.0, 198.0],
            "ask_price_1": [101.0, 202.0],
        }
    )
    config = make_smoothing_config(method="C", threshold="mean_pct").smoothing

    result = fit_train_smoothing_threshold([frame], config)

    expected = np.mean([2.0 / 100.0, 4.0 / 200.0])
    assert result["mode"] == "mean_pct"
    assert result["value"] == pytest.approx(expected)
    assert result["fit_split"] == "train"


def test_fit_train_smoothing_threshold_mean_spread() -> None:
    frame = pd.DataFrame(
        {
            "bid_price_1": [99.0, 196.0],
            "ask_price_1": [101.0, 204.0],
        }
    )
    config = make_smoothing_config(method="C", threshold="mean_spread").smoothing

    result = fit_train_smoothing_threshold([frame], config)

    expected = np.mean([2.0, 8.0]) / np.mean([100.0, 200.0])
    assert result["mode"] == "mean_spread"
    assert result["value"] == pytest.approx(expected)


def test_smoothing_method_a_uses_future_rolling_mean() -> None:
    midprices = pd.Series([10.0, 12.0, 14.0, 16.0])

    result = SmoothingMethodA(k=1)(midprices)

    expected = pd.Series([0.1, 1.0 / 12.0, 1.0 / 14.0, np.nan])
    np.testing.assert_allclose(result.dropna(), expected.dropna())
    assert result.isna().iloc[-1]


def test_smoothing_method_c_validates_windows() -> None:
    for k, h in [(-1, 2), (1, 0), (2, 2)]:
        try:
            SmoothingMethodC(k=k, h=h)
        except ValueError:
            continue
        raise AssertionError(f"Expected SmoothingMethodC to reject k={k}, h={h}.")


def test_trend_threshold_classifier_maps_three_classes() -> None:
    values = pd.Series([-0.20, -0.05, 0.0, 0.05, 0.20])

    labels = TrendThresholdClassifier(threshold=0.1)(values)

    assert labels.tolist() == [-1, 0, 0, 0, 1]


def test_trend_threshold_classifier_accepts_series_threshold() -> None:
    values = pd.Series([-0.3, -0.15, 0.1, 0.3])
    thresholds = pd.Series([0.2, 0.1, 0.2, 0.2])

    labels = TrendThresholdClassifier(threshold=thresholds)(values)

    assert labels.tolist() == [-1, -1, 0, 1]


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


def test_target_label_pipeline_keeps_constant_threshold_when_adaptive_threshold_is_disabled() -> None:
    df = pd.DataFrame(
        {
            "bid_price_1": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0],
            "ask_price_1": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
        }
    )
    disabled_adaptive = AdaptiveThresholdConfig(
        enabled=False,
        exit_spread_window=2,
        volatility_window=2,
        round_trip_fees_bps=0.0,
        volatility_lambda=1.0,
    )

    constant_result = TargetLabelPipeline(
        make_smoothing_config(method="C", threshold=0.001, k=1, h=2)
    ).transform(df)
    disabled_result = TargetLabelPipeline(
        make_smoothing_config(
            method="C",
            threshold=0.001,
            k=1,
            h=2,
            adaptive_threshold=disabled_adaptive,
        )
    ).transform(df)

    pd.testing.assert_frame_equal(disabled_result, constant_result)


def test_calculate_adaptive_method_c_threshold_uses_cost_floor() -> None:
    midprices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
    df = pd.DataFrame(
        {
            "bid_price_1": midprices - 1.0,
            "ask_price_1": midprices + 1.0,
        }
    )
    config = AdaptiveThresholdConfig(
        enabled=True,
        exit_spread_window=2,
        volatility_window=2,
        round_trip_fees_bps=10.0,
        volatility_lambda=0.0,
    )

    threshold = calculate_adaptive_method_c_threshold(
        df,
        midprices,
        k=1,
        h=2,
        bid_col="bid_price_1",
        ask_col="ask_price_1",
        config=config,
    )

    expected_index = 2
    w_minus = (midprices.iloc[1] + midprices.iloc[2]) / 2.0
    expected_fee = midprices.iloc[expected_index] * 10.0 / 10000.0
    expected = (2.0 + expected_fee) / w_minus
    np.testing.assert_allclose(threshold.iloc[expected_index], expected)


def test_calculate_adaptive_method_c_threshold_components_expose_floor_dominance() -> None:
    midprices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
    df = pd.DataFrame(
        {
            "bid_price_1": midprices - 1.0,
            "ask_price_1": midprices + 1.0,
        }
    )
    config = AdaptiveThresholdConfig(
        enabled=True,
        exit_spread_window=2,
        volatility_window=2,
        round_trip_fees_bps=0.0,
        volatility_lambda=0.0,
    )

    components = calculate_adaptive_method_c_threshold_components(
        df,
        midprices,
        k=1,
        h=2,
        bid_col="bid_price_1",
        ask_col="ask_price_1",
        config=config,
    )

    valid = components.dropna()
    assert {"cost_floor", "volatility_floor", "threshold"} <= set(components.columns)
    assert (valid["cost_floor"] > valid["volatility_floor"]).all()
    pd.testing.assert_series_equal(
        valid["threshold"],
        valid["cost_floor"],
        check_names=False,
    )


def test_adaptive_method_c_threshold_does_not_use_future_values() -> None:
    midprices = pd.Series(np.linspace(100.0, 109.0, 10))
    df = pd.DataFrame(
        {
            "bid_price_1": midprices - 0.5,
            "ask_price_1": midprices + 0.5,
        }
    )
    modified_midprices = midprices.copy()
    modified_midprices.iloc[8:] += 1000.0
    modified_df = pd.DataFrame(
        {
            "bid_price_1": modified_midprices - 0.5,
            "ask_price_1": modified_midprices + 0.5,
        }
    )
    config = AdaptiveThresholdConfig(
        enabled=True,
        exit_spread_window=2,
        volatility_window=2,
        round_trip_fees_bps=5.0,
        volatility_lambda=1.0,
    )

    threshold = calculate_adaptive_method_c_threshold(
        df,
        midprices,
        k=1,
        h=2,
        bid_col="bid_price_1",
        ask_col="ask_price_1",
        config=config,
    )
    modified_threshold = calculate_adaptive_method_c_threshold(
        modified_df,
        modified_midprices,
        k=1,
        h=2,
        bid_col="bid_price_1",
        ask_col="ask_price_1",
        config=config,
    )

    np.testing.assert_allclose(threshold.iloc[5], modified_threshold.iloc[5])


def test_target_label_pipeline_uses_adaptive_threshold_for_method_c() -> None:
    midprices = pd.Series(np.arange(100.0, 108.0))
    df = pd.DataFrame(
        {
            "bid_price_1": midprices,
            "ask_price_1": midprices,
        }
    )
    config = make_smoothing_config(
        method="C",
        threshold=1.0,
        k=1,
        h=2,
        adaptive_threshold=AdaptiveThresholdConfig(
            enabled=True,
            exit_spread_window=2,
            volatility_window=2,
            round_trip_fees_bps=0.0,
            volatility_lambda=0.0,
        ),
    )

    result = TargetLabelPipeline(config).transform(df)

    assert not result.empty
    assert set(result["trend_label"]) == {1}


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
