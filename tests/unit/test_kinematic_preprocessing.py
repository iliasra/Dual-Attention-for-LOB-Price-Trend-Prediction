from __future__ import annotations

import numpy as np
import pandas as pd

from configuration import MessageConfig
from kinematic_preprocessing import (
    MessageFeatureProcessor,
    MessageOrderbookJoiner,
    handle_abnormal_prices,
    min_max_norm,
    time_to_sincos,
)


def test_handle_abnormal_prices_drops_complete_ghost_levels() -> None:
    df = pd.DataFrame(
        {
            "ask_price_1": [101.0, 102.0],
            "ask_size_1": [20, 21],
            "bid_price_2": [9999999999.0, 9999999999.0],
            "bid_size_2": [0, 0],
        }
    )

    handle_abnormal_prices([df])

    assert "bid_price_2" not in df.columns
    assert "bid_size_2" not in df.columns
    assert {"ask_price_1", "ask_size_1"} <= set(df.columns)


def test_time_to_sincos_maps_day_quarters() -> None:
    result = time_to_sincos(np.array([0, 21600]), freq=86400)

    np.testing.assert_allclose(result[0], [0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(result[1], [1.0, 0.0], atol=1e-12)


def test_min_max_norm_returns_zeros_for_constant_values() -> None:
    result = min_max_norm(pd.Series([5.0, 5.0, 5.0]))

    np.testing.assert_allclose(result, [0.0, 0.0, 0.0])


def test_message_orderbook_joiner_copies_time_and_delta_t() -> None:
    message_df = pd.DataFrame(
        {
            "time": [10.0, 10.5, 11.0],
            "type": [1, 2, 3],
        }
    )
    orderbook_df = pd.DataFrame({"bid_price_1": [100.0, 100.5, 101.0]})

    joined = MessageOrderbookJoiner(time_column="time").transform(message_df, orderbook_df)

    assert joined["time"].tolist() == [10.0, 10.5, 11.0]
    assert np.isnan(joined["delta_t"].iloc[0])
    assert joined["delta_t"].iloc[1:].tolist() == [0.5, 0.5]
    assert "type" in joined.columns


def test_message_feature_processor_adds_log_static_and_one_hot_features() -> None:
    df = pd.DataFrame(
        {
            "time": [1.0, 2.0],
            "bid_price_1": [99.0, 100.0],
            "ask_price_1": [101.0, 102.0],
            "size": [9.0, 99.0],
            "price": [101.0, 100.0],
            "type": [1, 5],
            "direction": [1, -1],
            "order_id": [10, 11],
        }
    )

    config = MessageConfig(
        tick_size=1.0,
        size_column="size",
        price_column="price",
        order_id_column="order_id",
        categorical_value_map={"type": [1, 2, 3, 4, 5], "direction": [-1, 1]},
        drop_columns=["price", "size", "type", "direction", "order_id"],
    )

    result = MessageFeatureProcessor("time", config).transform(df)

    assert {"size_log1p", "price_static", "type_1", "type_5", "direction_1", "direction_-1"} <= set(result.columns)
    assert {"size", "price", "type", "direction", "order_id"}.isdisjoint(result.columns)
    np.testing.assert_allclose(result["size_log1p"], np.log1p([9.0, 99.0]))
