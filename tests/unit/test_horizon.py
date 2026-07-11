from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from configuration import (
    AdaptiveThresholdConfig,
    ExecutableReturnLabelConfig,
    LabelConfig,
    SmoothingLabelConfig,
    TripleBarrierLabelConfig,
)
from horizon import (
    ADAPTIVE_LABEL_FEATURE_COLUMNS,
    SmoothingMethodC,
    SmoothingMethodA,
    TargetLabelPipeline,
    TrendThresholdClassifier,
    TripleBarrierLabeler,
    calculate_adaptive_method_c_threshold,
    calculate_adaptive_method_c_threshold_components,
    calculate_midprice,
    calculate_spread,
    fit_smoothing_threshold,
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
            "bid_price_1": [99.0, 101.0, 103.0, 105.0, 107.0],
            "ask_price_1": [101.0, 103.0, 105.0, 107.0, 109.0],
        }
    )
    config = make_smoothing_config(method="C", threshold="mean_pct", k=1, h=2).smoothing

    result = fit_train_smoothing_threshold([frame], config)

    expected = np.mean([4.0 / 101.0, 4.0 / 103.0])
    assert result["mode"] == "mean_pct"
    assert result["value"] == pytest.approx(expected)
    assert result["mean_pct"] == pytest.approx(expected)
    assert result["formula"] == "mean(abs(l_t)), where l_t is the configured smoothing percentage change"
    assert result["fit_split"] == "train"


def test_fit_smoothing_threshold_mean_pct_can_use_validation_split() -> None:
    frame = pd.DataFrame(
        {
            "bid_price_1": [99.0, 101.0, 103.0, 105.0, 107.0],
            "ask_price_1": [101.0, 103.0, 105.0, 107.0, 109.0],
        }
    )
    config = make_smoothing_config(method="C", threshold="mean_pct", k=1, h=2).smoothing

    result = fit_smoothing_threshold([frame], config, fit_split="validation")

    expected = np.mean([4.0 / 101.0, 4.0 / 103.0])
    assert result["mode"] == "mean_pct"
    assert result["value"] == pytest.approx(expected)
    assert result["mean_pct"] == pytest.approx(expected)
    assert result["fit_split"] == "validation"


def test_fit_smoothing_threshold_mean_pct_2_halves_mean_abs_change() -> None:
    frame = pd.DataFrame(
        {
            "bid_price_1": [99.0, 101.0, 103.0, 105.0, 107.0],
            "ask_price_1": [101.0, 103.0, 105.0, 107.0, 109.0],
        }
    )
    config = make_smoothing_config(method="C", threshold="mean_pct_2", k=1, h=2).smoothing

    result = fit_smoothing_threshold([frame], config, fit_split="validation")

    mean_abs_change = np.mean([4.0 / 101.0, 4.0 / 103.0])
    assert result["mode"] == "mean_pct_2"
    assert result["value"] == pytest.approx(0.5 * mean_abs_change)
    assert result["mean_pct"] == pytest.approx(mean_abs_change)
    assert result["multiplier"] == pytest.approx(0.5)
    assert result["formula"] == "0.5 * mean(abs(l_t)), where l_t is the configured smoothing percentage change"


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

    assert labels.iloc[:2].tolist() == [1, -1]
    assert labels.iloc[2:].isna().all()


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


def test_target_label_pipeline_rejects_null_smoothing_threshold_without_adaptive() -> None:
    df = pd.DataFrame(
        {
            "bid_price_1": [99.0, 100.0, 101.0, 102.0],
            "ask_price_1": [101.0, 102.0, 103.0, 104.0],
        }
    )
    smoothing = SmoothingLabelConfig(
        method="A",
        threshold=None,
        k=1,
        h=10,
        bid_column="bid_price_1",
        ask_column="ask_price_1",
        adaptive_threshold=None,
    )

    with pytest.raises(ValueError, match="threshold cannot be null"):
        TargetLabelPipeline(make_triple_barrier_config())._add_smoothing_labels(df, smoothing)


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
    assert {
        "exit_spread_median",
        "local_volatility",
        "cost_floor",
        "volatility_floor",
        "threshold",
        *ADAPTIVE_LABEL_FEATURE_COLUMNS,
    } <= set(components.columns)
    assert (valid["cost_floor"] > valid["volatility_floor"]).all()
    pd.testing.assert_series_equal(
        valid["threshold"],
        valid["cost_floor"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        components["adaptive_threshold"],
        components["threshold"],
        check_names=False,
    )


def test_adaptive_method_c_components_include_exante_local_volatility() -> None:
    midprices = pd.Series([100.0, 101.0, 103.0, 102.0, 104.0, 107.0, 109.0])
    df = pd.DataFrame(
        {
            "bid_price_1": midprices - 0.5,
            "ask_price_1": midprices + 0.5,
        }
    )
    config = AdaptiveThresholdConfig(
        enabled=True,
        exit_spread_window=2,
        volatility_window=2,
        round_trip_fees_bps=0.0,
        volatility_lambda=2.0,
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

    expected_local_vol = (
        midprices.pct_change().rolling(window=2, min_periods=2).std(ddof=0) * np.sqrt(2.0)
    )
    pd.testing.assert_series_equal(
        components["adaptive_local_volatility"],
        expected_local_vol,
        check_names=False,
    )
    np.testing.assert_allclose(
        components["adaptive_volatility_floor"].dropna(),
        (2.0 * expected_local_vol).dropna(),
    )
    valid = components[["adaptive_threshold", "adaptive_cost_floor", "adaptive_volatility_floor"]].dropna()
    np.testing.assert_allclose(
        valid["adaptive_threshold"],
        np.maximum(valid["adaptive_cost_floor"].to_numpy(), valid["adaptive_volatility_floor"].to_numpy()),
    )


def test_adaptive_method_c_can_use_postex_threshold_for_labels() -> None:
    midprices = pd.Series([100.0, 100.0, 100.0, 106.0, 106.0, 106.0, 106.0])
    spreads = pd.Series([1.0, 1.0, 1.0, 1.0, 20.0, 1.0, 1.0])
    df = pd.DataFrame(
        {
            "row_id": np.arange(len(midprices)),
            "bid_price_1": midprices - spreads / 2.0,
            "ask_price_1": midprices + spreads / 2.0,
        }
    )
    exante_adaptive = AdaptiveThresholdConfig(
        enabled=True,
        exit_spread_window=2,
        volatility_window=2,
        round_trip_fees_bps=0.0,
        volatility_lambda=0.0,
        label_timing="ex_ante",
        include_exante_features=True,
    )
    postex_adaptive = AdaptiveThresholdConfig(
        enabled=True,
        exit_spread_window=2,
        volatility_window=2,
        round_trip_fees_bps=0.0,
        volatility_lambda=0.0,
        label_timing="ex_post",
        include_exante_features=True,
    )

    exante_components = calculate_adaptive_method_c_threshold_components(
        df,
        midprices,
        k=1,
        h=2,
        bid_col="bid_price_1",
        ask_col="ask_price_1",
        config=exante_adaptive,
    )
    postex_components = calculate_adaptive_method_c_threshold_components(
        df,
        midprices,
        k=1,
        h=2,
        bid_col="bid_price_1",
        ask_col="ask_price_1",
        config=postex_adaptive,
    )
    idx = 2
    w_minus = midprices.rolling(window=2).mean()
    forward_return = (midprices.rolling(window=2).mean().shift(-2) - w_minus) / w_minus

    assert exante_components.loc[idx, "threshold"] == pytest.approx(
        exante_components.loc[idx, "adaptive_threshold"],
    )
    assert postex_components.loc[idx, "threshold"] == pytest.approx(
        postex_components.loc[idx, "postex_threshold"],
    )
    assert postex_components.loc[idx, "adaptive_threshold"] == pytest.approx(
        exante_components.loc[idx, "adaptive_threshold"],
    )
    assert exante_components.loc[idx, "threshold"] < forward_return.loc[idx]
    assert postex_components.loc[idx, "threshold"] > forward_return.loc[idx]

    exante_result = TargetLabelPipeline(
        make_smoothing_config(k=1, h=2, adaptive_threshold=exante_adaptive),
    ).transform(df)
    postex_result = TargetLabelPipeline(
        make_smoothing_config(k=1, h=2, adaptive_threshold=postex_adaptive),
    ).transform(df)

    assert exante_result.loc[exante_result["row_id"] == idx, "trend_label"].item() == 1
    assert postex_result.loc[postex_result["row_id"] == idx, "trend_label"].item() == 0
    assert set(ADAPTIVE_LABEL_FEATURE_COLUMNS) <= set(postex_result.columns)
    assert not any(column.startswith("postex_") for column in postex_result.columns)


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
            include_exante_features=True,
        ),
    )

    result = TargetLabelPipeline(config).transform(df)

    assert not result.empty
    assert set(ADAPTIVE_LABEL_FEATURE_COLUMNS) <= set(result.columns)
    assert not any(column.startswith("realized_") for column in result.columns)
    assert set(result["trend_label"]) == {1}


def test_target_label_pipeline_can_omit_adaptive_exante_features() -> None:
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
            include_exante_features=False,
        ),
    )

    result = TargetLabelPipeline(config).transform(df)

    assert not result.empty
    assert set(ADAPTIVE_LABEL_FEATURE_COLUMNS).isdisjoint(result.columns)
    assert set(result["trend_label"]) == {1}


def test_postex_labels_do_not_require_exante_features_when_omitted() -> None:
    midprices = pd.Series(np.arange(100.0, 108.0))
    df = pd.DataFrame(
        {
            "bid_price_1": midprices - 0.5,
            "ask_price_1": midprices + 0.5,
        }
    )
    omitted_config = make_smoothing_config(
        method="C",
        threshold=None,
        k=1,
        h=2,
        adaptive_threshold=AdaptiveThresholdConfig(
            enabled=True,
            exit_spread_window=100,
            volatility_window=2,
            round_trip_fees_bps=0.0,
            volatility_lambda=0.0,
            label_timing="ex_post",
            include_exante_features=False,
        ),
    )
    included_config = make_smoothing_config(
        method="C",
        threshold=None,
        k=1,
        h=2,
        adaptive_threshold=AdaptiveThresholdConfig(
            enabled=True,
            exit_spread_window=100,
            volatility_window=2,
            round_trip_fees_bps=0.0,
            volatility_lambda=0.0,
            label_timing="ex_post",
            include_exante_features=True,
        ),
    )

    omitted_result = TargetLabelPipeline(omitted_config).transform(df)
    included_result = TargetLabelPipeline(included_config).transform(df)

    assert not omitted_result.empty
    assert set(ADAPTIVE_LABEL_FEATURE_COLUMNS).isdisjoint(omitted_result.columns)
    assert included_result.empty


def test_target_label_pipeline_adds_triple_barrier_labels() -> None:
    df = pd.DataFrame(
        {
            "bid_price_1": [99.0, 102.0, 96.0, 99.0],
            "ask_price_1": [101.0, 104.0, 98.0, 101.0],
        }
    )
    config = make_triple_barrier_config(horizon=2, upper_barrier_ticks=2.0, lower_barrier_ticks=2.0)

    result = TargetLabelPipeline(config).transform(df)

    assert result["trend_label"].tolist() == [1, -1]


def test_executable_return_targets_match_crossing_pnl_and_censor_tail() -> None:
    frame = pd.DataFrame(
        {
            "bid_price_1": [100.0, 101.0, 103.0, 102.0, 105.0],
            "ask_price_1": [101.0, 102.0, 104.0, 103.0, 106.0],
        }
    )
    config = make_smoothing_config(threshold=0.0, k=0, h=2)
    config.strategy = "executable_return"
    config.executable_return = ExecutableReturnLabelConfig(
        horizon_events=2,
        minimum_edge_ticks=0.0,
        tick_size=1.0,
    )

    result = TargetLabelPipeline(config).transform(frame)

    assert len(result) == 3
    np.testing.assert_allclose(result["long_net_return_ticks"], [2.0, 0.0, 1.0])
    np.testing.assert_allclose(result["short_net_return_ticks"], [-4.0, -2.0, -3.0])
    assert result["trend_label"].tolist() == [1, 0, 1]


def test_executable_return_targets_include_fees_latency_and_slippage() -> None:
    frame = pd.DataFrame(
        {
            "bid_price_1": [10_000.0, 10_010.0, 10_030.0, 10_040.0],
            "ask_price_1": [10_010.0, 10_020.0, 10_040.0, 10_050.0],
        }
    )
    config = make_smoothing_config(threshold=0.0, k=0, h=2)
    config.strategy = "executable_return"
    config.executable_return = ExecutableReturnLabelConfig(
        horizon_events=3,
        entry_lag_events=1,
        round_trip_fees_bps=10.0,
        slippage_ticks_per_side=0.5,
        tick_size=10.0,
    )

    result = TargetLabelPipeline(config).transform(frame)

    expected_fee_ticks = ((10_010.0 + 10_020.0) / 2.0) * 10.0 / 10_000.0 / 10.0
    expected_long = (10_040.0 - 10_020.0) / 10.0 - expected_fee_ticks - 1.0
    expected_short = (10_010.0 - 10_050.0) / 10.0 - expected_fee_ticks - 1.0
    assert len(result) == 1
    assert result.loc[0, "long_net_return_ticks"] == pytest.approx(expected_long)
    assert result.loc[0, "short_net_return_ticks"] == pytest.approx(expected_short)
