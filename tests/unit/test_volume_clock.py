from __future__ import annotations

import numpy as np
import pandas as pd

from configuration import DataConfig, MessageConfig, SampleClockConfig
from kinematic_preprocessing import ColumnResolver
from volume_clock import VolumeBarFeatureProcessor, VolumeClockSampler


def _message_config() -> MessageConfig:
    return MessageConfig(
        tick_size=1.0,
        size_column="size",
        price_column="price",
        order_id_column="order_id",
        categorical_value_map={"type": [1, 2, 3, 4, 5], "direction": [-1, 1]},
        drop_columns=["price", "size", "type", "direction", "order_id"],
    )


def test_volume_clock_sampler_splits_large_trade_on_exact_grid() -> None:
    df = pd.DataFrame(
        {
            "time": [0.0, 1.0],
            "type": [1, 4],
            "order_id": [10, 11],
            "size": [20.0, 300.0],
            "price": [101.0, 101.0],
            "direction": [1, 1],
            "bid_price_1": [100.0, 100.5],
            "ask_price_1": [101.0, 101.5],
            "bid_size_1": [50.0, 60.0],
            "ask_size_1": [55.0, 65.0],
        }
    )
    sampler = VolumeClockSampler(
        sample_clock_config=SampleClockConfig(
            mode="volume",
            volume_step_shares=100.0,
            volume_source="traded",
            trade_type_values=[4, 5],
        ),
        message_config=_message_config(),
        time_column="time",
    )

    bars = sampler.transform(df)

    assert bars["volume_time"].tolist() == [1.0, 2.0, 3.0]
    assert bars["volume_wall_time"].tolist() == [1.0, 1.0, 1.0]
    assert bars["bar_duration_seconds"].tolist() == [1.0, 0.0, 0.0]
    np.testing.assert_allclose(bars["bar_buy_trade_volume"], [100.0, 100.0, 100.0])
    np.testing.assert_allclose(bars["bar_trade_count"], [1 / 3, 1 / 3, 1 / 3])
    np.testing.assert_allclose(bars["bar_type_1_count"], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(bars["bar_type_4_count"], [1 / 3, 1 / 3, 1 / 3])
    assert bars["ask_price_1"].tolist() == [101.5, 101.5, 101.5]


def test_volume_clock_sampler_drops_incomplete_final_bar() -> None:
    df = pd.DataFrame(
        {
            "time": [0.0, 1.0],
            "type": [4, 4],
            "order_id": [10, 11],
            "size": [100.0, 50.0],
            "price": [101.0, 101.0],
            "direction": [1, -1],
            "bid_price_1": [100.0, 100.5],
            "ask_price_1": [101.0, 101.5],
            "bid_size_1": [50.0, 60.0],
            "ask_size_1": [55.0, 65.0],
        }
    )
    sampler = VolumeClockSampler(
        sample_clock_config=SampleClockConfig(
            mode="volume",
            volume_step_shares=100.0,
            volume_source="traded",
            trade_type_values=[4, 5],
        ),
        message_config=_message_config(),
        time_column="time",
    )

    bars = sampler.transform(df)

    assert len(bars) == 1
    assert bars["volume_time"].tolist() == [1.0]
    np.testing.assert_allclose(bars["bar_total_trade_volume"], [100.0])


def test_volume_bar_feature_processor_creates_suffix_features_and_drops_raw_columns() -> None:
    raw_bars = pd.DataFrame(
        {
            "volume_time": [1.0, 2.0],
            "volume_wall_time": [10.0, 11.0],
            "trend_label": [1, -1],
            "bid_price_1": [100.0, 100.5],
            "ask_price_1": [101.0, 101.5],
            "bar_trade_count": [1.0, 2.0],
            "bar_message_count": [3.0, 4.0],
            "bar_type_1_count": [1.0, 0.0],
            "bar_type_4_count": [2.0, 4.0],
            "bar_direction_1_count": [2.0, 1.0],
            "bar_direction_minus_1_count": [1.0, 3.0],
            "bar_buy_trade_volume": [100.0, 0.0],
            "bar_sell_trade_volume": [0.0, 50.0],
            "bar_signed_trade_volume": [100.0, -50.0],
            "bar_total_trade_volume": [100.0, 50.0],
            "bar_add_size": [10.0, 0.0],
            "bar_cancel_size": [0.0, 5.0],
            "bar_delete_size": [0.0, 7.0],
            "bar_execution_size": [100.0, 50.0],
            "bar_duration_seconds": [1.5, 0.0],
        }
    )
    processor = VolumeBarFeatureProcessor(_message_config())

    processor.fit([raw_bars])
    features = processor.transform(raw_bars)

    assert "bar_trade_count" not in features.columns
    assert "bar_trade_count_log1p" in features.columns
    assert "bar_type_4_count_log1p" in features.columns
    assert "bar_direction_minus_1_count_log1p" in features.columns
    assert "bar_buy_trade_volume_log1p_exp" in features.columns
    assert "bar_signed_trade_volume_signed_log1p_exp" in features.columns
    assert "bar_duration_seconds_log1p" in features.columns
    assert "bar_trade_imbalance" in features.columns
    assert features["volume_wall_time"].tolist() == [10.0, 11.0]
    np.testing.assert_allclose(features["bar_trade_imbalance"], [1.0, -1.0])
    assert np.isfinite(features.select_dtypes(include=[np.number]).to_numpy()).all()


def test_orderbook_auto_detection_ignores_volume_bar_size_features() -> None:
    data_config = DataConfig(
        raw_data_dir="",
        processed_data_dir="",
        sequence_data_dir="",
        logs_dir="",
        tick_size=1.0,
        time_column="volume_time",
        label_column="trend_label",
        label_mapping={-1: 0, 0: 1, 1: 2},
        price_columns=None,
        volume_columns=None,
        feature_exclude_columns=["volume_wall_time"],
        sequence_window=2,
    )
    frame = pd.DataFrame(
        {
            "ask_price_1": [101.0],
            "bid_price_1": [100.0],
            "ask_size_1": [10.0],
            "bid_size_1": [11.0],
            "bar_add_size_log1p_exp": [0.2],
            "bar_execution_size_log1p_exp": [0.3],
        }
    )

    resolver = ColumnResolver(data_config)

    assert resolver.price_columns(frame) == ["ask_price_1", "bid_price_1"]
    assert resolver.volume_columns(frame) == ["ask_size_1", "bid_size_1"]

