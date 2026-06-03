from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import yaml

from configuration import (
    BasisKinematicConfig,
    DataConfig,
    FastKinematicConfig,
    KinematicTokenizationConfig,
    LabelConfig,
    MessageConfig,
    MicropriceConfig,
    NormalizationConfig,
    PriceKinematicConfig,
    PriceStaticConfig,
    PreprocessingConfig,
    SmoothingLabelConfig,
    TemporalFeaturesConfig,
    TripleBarrierLabelConfig,
    VolumeKinematicConfig,
    VolumeStaticConfig,
)
from kinematic_preprocessing import (
    DerivativeNormalizer,
    MessageFeatureProcessor,
    MessageOrderbookJoiner,
    PriceStaticProcessor,
    SnapshotBatchProcessor,
    VolumeStaticProcessor,
    _fast_price_tokens,
    _static_centering,
    calculate_microprice,
    derivative_feature_columns,
    fit_exp_scaling_parameters,
    fit_plgs_parameters,
    handle_abnormal_prices,
    min_max_norm,
    plgs_value,
    time_to_sincos,
)
from fast_kinematic_preprocessing import PenalizedBSplineKinematicTokenizer
from lobster_io import read_lobster_message_csv


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


def test_handle_abnormal_prices_ignores_dummy_values_outside_row_mask() -> None:
    df = pd.DataFrame(
        {
            "ask_price_1": [9999999999.0, 101.0, 102.0],
            "ask_size_1": [0, 20, 21],
            "bid_price_1": [99.0, 100.0, 101.0],
            "bid_size_1": [19, 20, 21],
        }
    )

    handle_abnormal_prices([df], row_mask=[False, True, True])

    assert {"ask_price_1", "ask_size_1", "bid_price_1", "bid_size_1"} <= set(df.columns)


def test_handle_abnormal_prices_drops_ghost_levels_inside_row_mask() -> None:
    df = pd.DataFrame(
        {
            "ask_price_1": [9999999999.0, 101.0, 102.0],
            "ask_size_1": [0, 20, 21],
            "bid_price_2": [98.0, -9999999999.0, -9999999999.0],
            "bid_size_2": [19, 0, 0],
        }
    )

    handle_abnormal_prices([df], row_mask=[False, True, True])

    assert {"bid_price_2", "bid_size_2"} & set(df.columns) == set()
    assert {"ask_price_1", "ask_size_1"} <= set(df.columns)


def test_read_lobster_message_csv_drops_trailing_extra_column() -> None:
    artifact_dir = Path(__file__).resolve().parent / ".test_artifacts" / "read_lobster_message_csv"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True)
    message_path = artifact_dir / "sample_message.csv"
    message_path.write_text(
        "1.0,1,100,10,451200,1,UBSS\n"
        "2.0,3,100,10,451200,-1,\n",
        encoding="utf-8",
    )

    result = read_lobster_message_csv(
        message_path,
        time_column="time",
        size_column="size",
        price_column="price",
        order_id_column="order_id",
        categorical_value_map={"type": [1, 2, 3, 4, 5], "direction": [-1, 1]},
    )

    assert not result.had_header
    assert result.dropped_trailing_extra_column
    assert result.dataframe.columns.tolist() == ["time", "type", "order_id", "size", "price", "direction"]
    assert result.dataframe.shape == (2, 6)


def test_time_to_sincos_maps_day_quarters() -> None:
    result = time_to_sincos(np.array([0, 21600]), freq=86400)

    np.testing.assert_allclose(result[0], [0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(result[1], [1.0, 0.0], atol=1e-12)


def test_min_max_norm_returns_zeros_for_constant_values() -> None:
    result = min_max_norm(pd.Series([5.0, 5.0, 5.0]))

    np.testing.assert_allclose(result, [0.0, 0.0, 0.0])


def test_derivative_normalizer_fit_overwrites_stale_stats() -> None:
    artifact_dir = Path(__file__).resolve().parent / ".test_artifacts" / "derivative_normalizer_overwrite"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True)
    stats_path = artifact_dir / "derivatives_stats.yaml"
    first_train = pd.DataFrame(
        {
            "price_kin_vel": [1.0, 2.0, 3.0],
            "price_kin_acc": [0.1, 0.2, 0.3],
        }
    )
    second_train = pd.DataFrame({"price_kin_vel": [10.0, 20.0, 30.0]})

    DerivativeNormalizer(stats_path).fit([first_train])
    DerivativeNormalizer(stats_path).fit([second_train])

    loaded_stats = DerivativeNormalizer.load_stats(stats_path)

    assert set(loaded_stats) == {"price_kin_vel"}
    assert loaded_stats["price_kin_vel"]["mean"] == 20.0


def test_derivative_normalizer_fit_matches_full_concat_stats() -> None:
    artifact_dir = Path(__file__).resolve().parent / ".test_artifacts" / "derivative_normalizer_exact"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True)
    stats_path = artifact_dir / "derivatives_stats.yaml"
    first_train = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0],
            "price_kin_vel": [1.0, np.nan, 3.0],
            "price_kin_acc": [0.1, 0.2, 0.3],
            "static_feature": [10.0, 20.0, 30.0],
        }
    )
    second_train = pd.DataFrame(
        {
            "time": [4.0, 5.0],
            "price_kin_vel": [np.inf, 5.0],
            "price_kin_jrk": [7.0, 8.0],
            "static_feature": [40.0, 50.0],
        }
    )

    DerivativeNormalizer(stats_path).fit([first_train, second_train])
    loaded_stats = DerivativeNormalizer.load_stats(stats_path)
    concatenated = pd.concat([first_train, second_train], ignore_index=True)

    assert set(loaded_stats) == set(derivative_feature_columns(concatenated))
    for column in derivative_feature_columns(concatenated):
        values = pd.to_numeric(concatenated[column], errors="coerce")
        finite_values = values[np.isfinite(values)]
        np.testing.assert_allclose(loaded_stats[column]["mean"], float(concatenated[column].mean()), equal_nan=True)
        np.testing.assert_allclose(loaded_stats[column]["std"], float(concatenated[column].std(ddof=0)), equal_nan=True)
        np.testing.assert_allclose(loaded_stats[column]["q001"], float(finite_values.quantile(0.001)), equal_nan=True)
        np.testing.assert_allclose(loaded_stats[column]["q999"], float(finite_values.quantile(0.999)), equal_nan=True)
        assert loaded_stats[column]["n_nan"] == int(values.isna().sum())
        assert loaded_stats[column]["n_inf"] == int(np.isinf(values).sum())


def test_derivative_normalizer_saves_metadata_without_polluting_loaded_stats() -> None:
    artifact_dir = Path(__file__).resolve().parent / ".test_artifacts" / "derivative_normalizer_metadata"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True)
    stats_path = artifact_dir / "derivatives_stats.yaml"
    train = pd.DataFrame({"microprice_2_kin_vel": [1.0, 2.0, 3.0]})
    metadata = {
        "kinematic_tokenization": {"orderbook_top_k_levels": 2},
        "microprice": {"enabled": True, "levels": 2},
    }

    DerivativeNormalizer(stats_path).fit([train], metadata=metadata)
    raw_payload = yaml.safe_load(stats_path.read_text(encoding="utf-8"))
    loaded_stats = DerivativeNormalizer.load_stats(stats_path)

    assert raw_payload["__metadata__"] == metadata
    assert set(loaded_stats) == {"microprice_2_kin_vel"}


def test_derivative_normalizer_zscore_transform_matches_existing_formula() -> None:
    artifact_dir = Path(__file__).resolve().parent / ".test_artifacts" / "derivative_normalizer_zscore"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True)
    stats_path = artifact_dir / "derivatives_stats.yaml"
    train = pd.DataFrame({"price_kin_vel": [1.0, 2.0, 3.0]})

    normalizer = DerivativeNormalizer(stats_path, method="zscore").fit([train])
    normalized = normalizer.transform(train)
    expected = (train["price_kin_vel"] - train["price_kin_vel"].mean()) / (
        train["price_kin_vel"].std(ddof=0) + 1e-8
    )

    np.testing.assert_allclose(normalized["price_kin_vel"], expected)


def test_derivative_normalizer_robust_mad_uses_scaled_mad() -> None:
    artifact_dir = Path(__file__).resolve().parent / ".test_artifacts" / "derivative_normalizer_robust_mad"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True)
    stats_path = artifact_dir / "derivatives_stats.yaml"
    train = pd.DataFrame({"price_kin_vel": [1.0, 2.0, 3.0]})

    normalizer = DerivativeNormalizer(stats_path, method="robust_mad").fit([train])
    loaded_stats = DerivativeNormalizer.load_stats(stats_path)
    stats = loaded_stats["price_kin_vel"]
    normalized = normalizer.transform(train)

    assert stats["median"] == 2.0
    assert stats["mad"] == 1.0
    assert stats["scale_source"] == "mad"
    np.testing.assert_allclose(float(stats["scale"]), 1.4826)
    np.testing.assert_allclose(normalized["price_kin_vel"], [-1.0 / 1.4826, 0.0, 1.0 / 1.4826])


def test_derivative_normalizer_robust_mad_falls_back_to_std_when_mad_is_zero() -> None:
    artifact_dir = Path(__file__).resolve().parent / ".test_artifacts" / "derivative_normalizer_robust_std"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True)
    stats_path = artifact_dir / "derivatives_stats.yaml"
    train = pd.DataFrame({"price_kin_vel": [1.0, 1.0, 1.0, 10.0]})

    normalizer = DerivativeNormalizer(stats_path, method="robust_mad").fit([train])
    stats = DerivativeNormalizer.load_stats(stats_path)["price_kin_vel"]
    scale = float(np.std(train["price_kin_vel"].to_numpy(dtype=float)))
    normalized = normalizer.transform(train)

    assert stats["median"] == 1.0
    assert stats["mad"] == 0.0
    assert stats["scale_source"] == "std"
    np.testing.assert_allclose(float(stats["scale"]), scale)
    np.testing.assert_allclose(normalized["price_kin_vel"], (train["price_kin_vel"] - 1.0) / scale)


def test_derivative_normalizer_robust_mad_constant_column_uses_unit_scale() -> None:
    artifact_dir = Path(__file__).resolve().parent / ".test_artifacts" / "derivative_normalizer_robust_unit"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True)
    stats_path = artifact_dir / "derivatives_stats.yaml"
    train = pd.DataFrame({"price_kin_vel": [5.0, 5.0, 5.0]})

    normalizer = DerivativeNormalizer(stats_path, method="robust_mad").fit([train])
    stats = DerivativeNormalizer.load_stats(stats_path)["price_kin_vel"]
    normalized = normalizer.transform(train)

    assert stats["median"] == 5.0
    assert stats["mad"] == 0.0
    assert stats["scale"] == 1.0
    assert stats["scale_source"] == "unit"
    np.testing.assert_allclose(normalized["price_kin_vel"], [0.0, 0.0, 0.0])


def test_derivative_normalizer_robust_mad_without_finite_values_uses_empty_scale() -> None:
    artifact_dir = Path(__file__).resolve().parent / ".test_artifacts" / "derivative_normalizer_robust_empty"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True)
    stats_path = artifact_dir / "derivatives_stats.yaml"
    train = pd.DataFrame({"price_kin_vel": [np.nan, np.inf, -np.inf]})

    DerivativeNormalizer(stats_path, method="robust_mad").fit([train])
    stats = DerivativeNormalizer.load_stats(stats_path)["price_kin_vel"]

    assert stats["median"] == 0.0
    assert stats["mad"] == 0.0
    assert stats["scale"] == 1.0
    assert stats["scale_source"] == "empty"


def test_derivative_normalizer_quantile_scaling_uses_std_floor_and_clips() -> None:
    artifact_dir = Path(__file__).resolve().parent / ".test_artifacts" / "derivative_normalizer_quantile"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True)
    stats_path = artifact_dir / "derivatives_stats.yaml"
    train = pd.DataFrame({"price_kin_vel": np.linspace(-1.0, 1.0, 1001)})
    test = pd.DataFrame({"price_kin_vel": [-100.0, 0.0, 100.0]})

    normalizer = DerivativeNormalizer(stats_path, method="quantile_scaling").fit([train])
    stats = DerivativeNormalizer.load_stats(stats_path)["price_kin_vel"]
    quantile_scale = (float(stats["q999"]) - float(stats["q001"])) / (2 * 3.090232306)
    expected_scale = max(float(np.std(train["price_kin_vel"].to_numpy(dtype=float))), quantile_scale)
    normalized = normalizer.transform(test)

    assert stats["median"] == 0.0
    assert stats["scale_source"] == "std"
    np.testing.assert_allclose(float(stats["scale"]), expected_scale)
    np.testing.assert_allclose(normalized["price_kin_vel"], [-10.0, 0.0, 10.0])


def test_static_centering_removes_touch_tick_symmetrically() -> None:
    price = pd.Series([101.0, 102.0, 100.0, 99.0])
    opposite_best = pd.Series([100.0, 100.0, 101.0, 101.0])

    result = _static_centering(price, opposite_best, tick=1.0, absolute=True, remove_touch_tick=True)

    np.testing.assert_allclose(result, [0.0, 1.0, 0.0, 1.0])


def test_static_centering_can_encode_directional_message_distance() -> None:
    price = pd.Series([102.0, 100.0, 98.0, 102.0])
    opposite_best = pd.Series([101.0, 101.0, 100.0, 100.0])
    direction = pd.Series([1.0, 1.0, -1.0, -1.0])

    result = _static_centering(price, opposite_best, tick=1.0, side=direction)

    np.testing.assert_allclose(result, [1.0, -1.0, 2.0, -2.0])


def test_fit_plgs_parameters_uses_train_quantiles_and_targets_q95() -> None:
    values = pd.Series(np.arange(1.0, 101.0))

    result = fit_plgs_parameters(values, tau_start=2.0)

    np.testing.assert_allclose(result["tau_clip"], values.quantile(0.99))
    np.testing.assert_allclose(result["x95"], values.quantile(0.95))
    np.testing.assert_allclose(
        plgs_value(result["x95"], tau_start=2.0, tau_max=result["tau_max"]),
        0.5,
        atol=1e-8,
    )
    assert result["n_values"] == len(values)


def test_fit_exp_scaling_parameters_uses_quantile_and_target() -> None:
    values = pd.Series(np.arange(1.0, 101.0))

    result = fit_exp_scaling_parameters(values, quantile=95.0, target=0.5)

    expected_quantile = float(values.quantile(0.95))
    expected_k = -expected_quantile / np.log(1.0 - 0.5)
    np.testing.assert_allclose(result["quantile_value"], expected_quantile)
    np.testing.assert_allclose(result["k"], expected_k)
    np.testing.assert_allclose(1.0 - np.exp(-expected_quantile / result["k"]), 0.5)
    assert result["n_values"] == len(values)


def test_fast_price_tokens_center_windows_before_spline_filtering() -> None:
    df = pd.DataFrame(
        {
            "bid_price_1": [99.0, 100.0, 101.0, 102.0],
            "ask_price_1": [101.0, 102.0, 103.0, 104.0],
            "ask_price_2": [102.0, 103.0, 104.0, 105.0],
        }
    )
    fast_config = FastKinematicConfig(
        n_basis=4,
        df=4.0,
        eval_at=1.0,
        selected_smoothing_lambda=0.0,
    )
    window = 3
    tick_size = 2.0

    tokens, labels = _fast_price_tokens(
        df,
        ["ask_price_1", "ask_price_2"],
        window=window,
        tick_size=tick_size,
        fast_config=fast_config,
        chunk_size=2,
    )
    tokenizer = PenalizedBSplineKinematicTokenizer(
        window=window,
        n_basis=fast_config.n_basis,
        smoothing_lambda=0.0,
        eval_at=fast_config.eval_at,
        chunk_size=2,
        dtype=np.float64,
    )
    centers = ((df["bid_price_1"] + df["ask_price_1"]) * 0.5).to_numpy(dtype=float)
    values = df[["ask_price_1", "ask_price_2"]].to_numpy(dtype=float)
    expected = []
    for start in range(len(df) - window + 1):
        centered_window = (values[start : start + window] - centers[start]) / tick_size
        expected.append(np.einsum("dw,wf->fd", tokenizer.H, centered_window, optimize=True))

    assert labels == ["ask_price_1", "ask_price_2"]
    np.testing.assert_allclose(tokens, np.asarray(expected))


def test_microprice_uses_opposite_side_liquidity() -> None:
    df = pd.DataFrame(
        {
            "ask_price_1": [101.0],
            "bid_price_1": [99.0],
            "ask_size_1": [20.0],
            "bid_size_1": [10.0],
            "ask_price_2": [102.0],
            "bid_price_2": [98.0],
            "ask_size_2": [30.0],
            "bid_size_2": [40.0],
        }
    )

    result = calculate_microprice(df, levels=2)

    expected = ((101.0 * 10.0) + (99.0 * 20.0) + (102.0 * 40.0) + (98.0 * 30.0)) / (
        10.0 + 20.0 + 40.0 + 30.0
    )
    np.testing.assert_allclose(result.to_numpy(), [expected])


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


def test_message_price_static_uses_directional_distance_to_opposite_best() -> None:
    df = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "bid_price_1": [100.0, 100.0, 100.0, 100.0],
            "ask_price_1": [101.0, 101.0, 101.0, 101.0],
            "size": [10.0, 10.0, 10.0, 10.0],
            "price": [102.0, 100.0, 98.0, 102.0],
            "type": [1, 1, 1, 1],
            "direction": [1, 1, -1, -1],
            "order_id": [10, 11, 12, 13],
        }
    )
    config = MessageConfig(
        tick_size=1.0,
        size_column="size",
        price_column="price",
        order_id_column="order_id",
        categorical_value_map={"type": [1], "direction": [-1, 1]},
        drop_columns=["price", "size", "type", "direction", "order_id"],
    )

    result = MessageFeatureProcessor("time", config).transform(df)

    np.testing.assert_allclose(result["price_static"], [1.0, -1.0, 2.0, -2.0])


def _make_static_refactor_frame(rows: int = 12) -> pd.DataFrame:
    times = np.arange(rows, dtype=float)
    bid_price_1 = 100.0 + 0.1 * times + 0.02 * np.sin(times)
    ask_price_1 = bid_price_1 + 1.0 + 0.01 * np.cos(times)
    bid_price_2 = bid_price_1 - 1.0
    ask_price_2 = ask_price_1 + 1.0
    bid_price_3 = bid_price_1 - 2.0
    ask_price_3 = ask_price_1 + 2.0
    return pd.DataFrame(
        {
            "time": times,
            "ask_price_1": ask_price_1,
            "ask_size_1": 120 + np.arange(rows) * 2,
            "bid_price_1": bid_price_1,
            "bid_size_1": 100 + np.arange(rows) * 3,
            "ask_price_2": ask_price_2,
            "ask_size_2": 130 + np.arange(rows) * 2,
            "bid_price_2": bid_price_2,
            "bid_size_2": 90 + np.arange(rows) * 3,
            "ask_price_3": ask_price_3,
            "ask_size_3": 140 + np.arange(rows) * 2,
            "bid_price_3": bid_price_3,
            "bid_size_3": 80 + np.arange(rows) * 3,
            "delta_t": np.r_[np.nan, np.diff(times)],
            "trend_label": np.where(np.arange(rows) % 3 == 0, 1, 0),
            "size_log1p": np.log1p(10 + np.arange(rows)),
            "price_static": np.arange(rows, dtype=float),
        }
    )


def _make_static_refactor_config(method: str) -> tuple[DataConfig, PreprocessingConfig]:
    data_config = DataConfig(
        raw_data_dir="",
        processed_data_dir="",
        sequence_data_dir="",
        logs_dir="",
        tick_size=1.0,
        time_column="time",
        label_column="trend_label",
        label_mapping={-1: 0, 0: 1, 1: 2},
        price_columns=None,
        volume_columns=None,
        feature_exclude_columns=[],
        sequence_window=3,
    )
    preprocessing_config = PreprocessingConfig(
        snapshot_window=4,
        labels=LabelConfig(
            strategy="smoothing",
            smoothing=SmoothingLabelConfig(
                method="C",
                threshold=0.0,
                k=1,
                h=2,
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
        ),
        message=MessageConfig(
            tick_size=1.0,
            size_column="size",
            price_column="price",
            order_id_column="order_id",
            categorical_value_map={"type": [1], "direction": [-1, 1]},
            drop_columns=[],
        ),
        temporal_features=TemporalFeaturesConfig(
            add_day_sincos=True,
            day_frequency=86400,
            keep_timestamp=True,
            market_open_seconds=34200.0,
            market_close_seconds=57600.0,
            start_offset_minutes=15,
            end_offset_minutes=15,
        ),
        normalization=NormalizationConfig(
            derivatives_stats_dir=".",
            scope="train_only",
        ),
        kinematic_tokenization=KinematicTokenizationConfig(method=method, chunk_size=3, n_df_candidates=4),
        price_kinematic=PriceKinematicConfig(
            enabled=True,
            columns=None,
            tick_size=1.0,
            reference="tick",
            basis=BasisKinematicConfig(alpha=2.0),
            fast=FastKinematicConfig(n_basis=6, df=5.0, eval_at=1.0),
        ),
        price_static=PriceStaticConfig(
            enabled=True,
            columns=None,
            tick_size=1.0,
            tau_start=1.0,
            tau_clip=50.0,
            tau_max=100.0,
        ),
        volume_kinematic=VolumeKinematicConfig(
            enabled=True,
            columns=None,
            reference="tick",
            basis=BasisKinematicConfig(alpha=2.0),
            fast=FastKinematicConfig(n_basis=6, df=5.0, eval_at=1.0),
        ),
        volume_static=VolumeStaticConfig(
            enabled=True,
            columns=None,
            quantile=95.0,
            target=0.5,
            k=2000.0,
        ),
    )
    return data_config, preprocessing_config


def test_snapshot_batch_static_streams_are_aligned_to_window_end_rows() -> None:
    df = _make_static_refactor_frame()

    for method in ("basis", "fast"):
        data_config, preprocessing_config = _make_static_refactor_config(method)
        processed = SnapshotBatchProcessor(data_config, preprocessing_config).transform(df)
        end_positions = np.arange(preprocessing_config.snapshot_window - 1, len(df))

        assert len(processed) == len(df) - preprocessing_config.snapshot_window + 1
        assert processed.columns[0] == "ask_price_1_static_plgs"
        np.testing.assert_allclose(processed["time"], df.iloc[end_positions]["time"])

        price_processor = PriceStaticProcessor(
            tick_size=preprocessing_config.price_static.tick_size,
            tau_start=preprocessing_config.price_static.tau_start,
            tau_clip=preprocessing_config.price_static.tau_clip,
            tau_max=preprocessing_config.price_static.tau_max,
        )
        volume_processor = VolumeStaticProcessor(k=preprocessing_config.volume_static.k)
        expected_static_rows: list[dict[str, float]] = []
        for end_position in end_positions:
            window = df.iloc[end_position - preprocessing_config.snapshot_window + 1 : end_position + 1]
            expected_static_rows.append(
                {
                    **price_processor.transform(window.reset_index(drop=True), ["ask_price_1", "bid_price_1"]),
                    **volume_processor.transform(window.reset_index(drop=True), ["ask_size_1", "bid_size_1"]),
                }
            )
        expected_static = pd.DataFrame(expected_static_rows)

        pd.testing.assert_frame_equal(
            processed[expected_static.columns],
            expected_static,
            check_dtype=False,
            rtol=1e-12,
            atol=1e-12,
        )
        assert np.isfinite(processed.select_dtypes(include=[np.number]).to_numpy()).all()


def test_snapshot_batch_top_k_filters_only_kinematic_streams_and_adds_microprice() -> None:
    df = _make_static_refactor_frame()

    for method in ("basis", "fast"):
        data_config, preprocessing_config = _make_static_refactor_config(method)
        preprocessing_config.kinematic_tokenization.orderbook_top_k_levels = 2
        preprocessing_config.microprice = MicropriceConfig(enabled=True, levels=2)

        processed = SnapshotBatchProcessor(data_config, preprocessing_config).transform(df)
        columns = set(processed.columns)

        assert "ask_price_3_static_plgs" in columns
        assert "bid_size_3_static_exp" in columns
        assert "microprice_2_kin_pos" in columns
        assert "microprice_2_kin_vel" in columns
        assert "ask_price_3_kin_vel" not in columns
        assert "bid_price_3_kin_vel" not in columns
        assert "ask_size_3_kin_vel" not in columns
        assert "bid_size_3_kin_vel" not in columns
        assert np.isfinite(processed.select_dtypes(include=[np.number]).to_numpy()).all()


def test_snapshot_batch_microprice_can_be_disabled() -> None:
    df = _make_static_refactor_frame()
    data_config, preprocessing_config = _make_static_refactor_config("fast")
    preprocessing_config.kinematic_tokenization.orderbook_top_k_levels = 2
    preprocessing_config.microprice = MicropriceConfig(enabled=False, levels=2)

    processed = SnapshotBatchProcessor(data_config, preprocessing_config).transform(df)

    assert not any(column.startswith("microprice_") for column in processed.columns)
