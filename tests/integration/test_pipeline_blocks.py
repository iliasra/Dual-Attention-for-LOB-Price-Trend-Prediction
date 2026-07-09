from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from configuration import (
    AuxiliaryHeadsConfig,
    BasisKinematicConfig,
    ClassifierPoolingConfig,
    DataConfig,
    FastKinematicConfig,
    KinematicTokenizationConfig,
    LabelConfig,
    MessageConfig,
    ModelConfig,
    NormalizationConfig,
    PriceKinematicConfig,
    PriceStaticConfig,
    PreprocessingConfig,
    SmoothingLabelConfig,
    TemporalFeaturesConfig,
    TripleBarrierLabelConfig,
    VolumeKinematicConfig,
    VolumeStaticConfig,
    ExperimentConfig,
    load_config,
)
from datasets import DailySequenceBuilder, LOBDataset
from horizon import ADAPTIVE_LABEL_FEATURE_COLUMNS, TargetLabelPipeline
from kinematic_preprocessing import (
    DerivativeNormalizer,
    MessageFeatureProcessor,
    MessageOrderbookJoiner,
    SnapshotBatchProcessor,
    TradingSessionFilter,
)
from model import build_model
from processing import LobFilePair, LobFileSegment, LobProcessingPipeline, ProcessedDay


@pytest.fixture()
def artifact_dir(request: pytest.FixtureRequest) -> Path:
    path = Path(__file__).resolve().parent / ".test_artifacts" / request.node.name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def make_synthetic_lob_frames(rows: int = 14) -> tuple[pd.DataFrame, pd.DataFrame]:
    times = np.arange(rows, dtype=float)
    bid_price_1 = 100.0 + 0.1 * times
    ask_price_1 = bid_price_1 + 1.0
    direction = np.where(np.arange(rows) % 2 == 0, 1, -1)

    message_df = pd.DataFrame(
        {
            "time": times,
            "type": (np.arange(rows) % 5) + 1,
            "order_id": np.arange(1000, 1000 + rows),
            "size": 10 + np.arange(rows),
            "price": np.where(direction == 1, ask_price_1, bid_price_1),
            "direction": direction,
        }
    )
    orderbook_df = pd.DataFrame(
        {
            "ask_price_1": ask_price_1,
            "ask_size_1": 120 + np.arange(rows),
            "bid_price_1": bid_price_1,
            "bid_size_1": 100 + np.arange(rows),
            "ask_price_2": ask_price_1 + 0.5,
            "ask_size_2": 140 + np.arange(rows),
            "bid_price_2": bid_price_1 - 0.5,
            "bid_size_2": 80 + np.arange(rows),
        }
    )
    return message_df, orderbook_df


def write_lobster_day(raw_dir: Path, symbol: str, date: str, rows: int = 14) -> None:
    message_df, orderbook_df = make_synthetic_lob_frames(rows=rows)
    message_df["time"] = 34200.0 + message_df["time"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    message_df.to_csv(raw_dir / f"{symbol}_{date}_34200000_57600000_message_10.csv", index=False)
    orderbook_df.to_csv(raw_dir / f"{symbol}_{date}_34200000_57600000_orderbook_10.csv", index=False)


def make_test_configs(tokenization_method: str = "basis") -> tuple[DataConfig, PreprocessingConfig]:
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
            categorical_value_map={"type": [1, 2, 3, 4, 5], "direction": [-1, 1]},
            drop_columns=["price", "size", "type", "direction", "order_id"],
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
        kinematic_tokenization=KinematicTokenizationConfig(
            method=tokenization_method,
            chunk_size=3,
            n_df_candidates=4,
        ),
        price_kinematic=PriceKinematicConfig(
            enabled=True,
            columns=None,
            tick_size=1.0,
            reference="tick",
            basis=BasisKinematicConfig(
                alpha=2.0,
            ),
            fast=FastKinematicConfig(
                n_basis=6,
                df=5.0,
                eval_at=1.0,
            ),
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
            basis=BasisKinematicConfig(
                alpha=2.0,
            ),
            fast=FastKinematicConfig(
                n_basis=6,
                df=5.0,
                eval_at=1.0,
            ),
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


def run_preprocessing_pipeline(artifact_dir: Path, tokenization_method: str = "basis") -> pd.DataFrame:
    data_config, preprocessing_config = make_test_configs(tokenization_method=tokenization_method)
    message_df, orderbook_df = make_synthetic_lob_frames()

    joined = MessageOrderbookJoiner(time_column=data_config.time_column).transform(message_df, orderbook_df)
    labeled = TargetLabelPipeline(preprocessing_config.labels).transform(joined)
    enriched = MessageFeatureProcessor(data_config.time_column, preprocessing_config.message).transform(labeled)
    processed = SnapshotBatchProcessor(data_config, preprocessing_config).transform(enriched)

    normalizer = DerivativeNormalizer(artifact_dir / "derivatives_stats.yaml").fit([processed])
    return normalizer.transform(processed)


def test_preprocessing_blocks_produce_expected_columns(artifact_dir: Path) -> None:
    normalized = run_preprocessing_pipeline(artifact_dir)

    expected_columns = {
        "time",
        "trend_label",
        "delta_t",
        "time_day_sin",
        "time_day_cos",
        "size_log1p",
        "price_static",
        "type_1",
        "type_5",
        "direction_1",
        "direction_-1",
        "bid_price_1_kin_vel",
        "ask_price_1_static_plgs",
        "bid_size_1_kin_acc",
        "ask_size_1_static_exp",
    }

    assert expected_columns <= set(normalized.columns)
    assert not normalized.empty
    assert normalized["trend_label"].isin([-1, 0, 1]).all()
    assert np.isfinite(normalized.select_dtypes(include=[np.number]).to_numpy()).all()


def test_fast_kinematic_tokenization_produces_expected_columns(artifact_dir: Path) -> None:
    normalized = run_preprocessing_pipeline(artifact_dir, tokenization_method="fast")

    expected_columns = {
        "bid_price_1_kin_pos",
        "bid_price_1_kin_vel",
        "ask_price_1_kin_acc",
        "bid_size_1_kin_jrk",
        "ask_size_1_kin_pos",
    }

    assert expected_columns <= set(normalized.columns)
    assert not normalized.empty
    assert np.isfinite(normalized.select_dtypes(include=[np.number]).to_numpy()).all()


def test_sequence_dataset_and_model_forward_use_matching_tensor_shapes(artifact_dir: Path) -> None:
    torch.manual_seed(0)
    data_config, _ = make_test_configs()
    normalized = run_preprocessing_pipeline(artifact_dir)

    x_path, t_path, y_path = DailySequenceBuilder(data_config).save(normalized, artifact_dir / "synthetic_lob")
    dataset = LOBDataset([str(x_path)], [str(t_path)], [str(y_path)], sequence_window=data_config.sequence_window)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    x_batch, t_batch, y_batch = next(iter(loader))

    assert x_batch.ndim == 3
    assert t_batch.shape == x_batch.shape[:2]
    assert y_batch.shape == (x_batch.shape[0],)
    assert x_batch.shape[1] == data_config.sequence_window

    model_config = ModelConfig(
        d_input=x_batch.shape[-1],
        d_model=16,
        feature_embed_dim=4,
        feature_num_frequencies=3,
        feature_sigma=1.0,
        num_heads=2,
        max_dt=3.0,
        num_experts=2,
        top_k=1,
        num_classes=3,
        rope_type="hybrid_crope",
        rope_base=10000,
        attention_dropout=0.0,
        moe_dropout=0.0,
        moe_expansion_factor=2,
        moe_router_noise=0.0,
        moe_load_balancing_weight=0.0,
        classifier_dropout=0.0,
    )
    model = build_model(model_config)

    logits = model(x_batch, t_batch)

    assert logits.shape == (x_batch.shape[0], model_config.num_classes)
    assert torch.isfinite(logits).all()
    assert model.classifier.trunk[2].in_features == model_config.d_model
    assert model.encoder.layers[0].moe is not None
    assert model.moe_routing is not None
    assert model.auxiliary_outputs == {}


def test_model_forward_tokenwise_outputs_one_logit_per_token() -> None:
    torch.manual_seed(0)
    model_config = ModelConfig(
        d_input=4,
        d_model=16,
        feature_embed_dim=4,
        feature_num_frequencies=3,
        feature_sigma=1.0,
        num_heads=2,
        max_dt=10.0,
        num_experts=2,
        top_k=1,
        num_classes=3,
        rope_type="hybrid_crope",
        rope_base=10000,
        attention_dropout=0.0,
        moe_dropout=0.0,
        moe_expansion_factor=2,
        moe_router_noise=0.0,
        moe_load_balancing_weight=0.0,
        classifier_dropout=0.0,
        local_attention_context_tokens=4,
    )
    model = build_model(model_config)
    x = torch.randn(2, 8, 4)
    t = torch.arange(8, dtype=torch.float32).repeat(2, 1)

    logits = model(x, t, tokenwise=True)

    assert logits.shape == (2, 8, 3)
    assert torch.isfinite(logits).all()


def test_multi_layer_model_uses_dense_intermediate_blocks_and_final_moe() -> None:
    torch.manual_seed(0)
    model_config = ModelConfig(
        d_input=6,
        d_model=16,
        feature_embed_dim=4,
        feature_num_frequencies=3,
        feature_sigma=1.0,
        num_heads=2,
        max_dt=3.0,
        num_experts=2,
        top_k=1,
        num_classes=3,
        rope_type="hybrid_crope",
        rope_base=10000,
        attention_dropout=0.0,
        moe_dropout=0.0,
        moe_expansion_factor=2,
        moe_router_noise=0.0,
        moe_load_balancing_weight=0.0,
        classifier_dropout=0.0,
        num_layers=3,
        latent_spatial_embed_dim=4,
    )
    model = build_model(model_config)
    x = torch.randn(2, 5, model_config.d_input)
    t = torch.arange(5, dtype=torch.float32).repeat(2, 1)

    logits = model(x, t)

    assert logits.shape == (2, model_config.num_classes)
    assert model.encoder.layers[0].moe is None
    assert model.encoder.layers[0].dense_fnn is not None
    assert model.encoder.layers[1].moe is None
    assert model.encoder.layers[1].dense_fnn is not None
    assert model.encoder.layers[2].moe is not None
    assert model.encoder.layers[2].dense_fnn is None
    assert model.moe_load_balancing_loss is not None
    assert model.moe_routing is not None


def test_single_layer_model_can_replace_moe_with_dense_fnn() -> None:
    torch.manual_seed(0)
    model_config = ModelConfig(
        d_input=6,
        d_model=16,
        feature_embed_dim=4,
        feature_num_frequencies=3,
        feature_sigma=1.0,
        num_heads=2,
        max_dt=3.0,
        num_experts=2,
        top_k=1,
        num_classes=3,
        rope_type="hybrid_crope",
        rope_base=10000,
        attention_dropout=0.0,
        moe_dropout=0.0,
        moe_expansion_factor=2,
        moe_router_noise=0.0,
        moe_load_balancing_weight=0.0,
        classifier_dropout=0.0,
        use_moe=False,
    )
    model = build_model(model_config)
    x = torch.randn(2, 5, model_config.d_input)
    t = torch.arange(5, dtype=torch.float32).repeat(2, 1)

    logits = model(x, t)

    assert logits.shape == (2, model_config.num_classes)
    assert model.encoder.layers[0].moe is None
    assert model.encoder.layers[0].dense_fnn is not None
    assert model.encoder.moe is None
    assert model.moe_load_balancing_loss is None
    assert model.moe_routing is None


def test_model_accepts_discrete_rope_type() -> None:
    torch.manual_seed(0)
    model_config = ModelConfig(
        d_input=6,
        d_model=16,
        feature_embed_dim=4,
        feature_num_frequencies=3,
        feature_sigma=1.0,
        num_heads=2,
        max_dt=3.0,
        num_experts=2,
        top_k=1,
        num_classes=3,
        rope_type="rope",
        rope_base=10000,
        attention_dropout=0.0,
        moe_dropout=0.0,
        moe_expansion_factor=2,
        moe_router_noise=0.0,
        moe_load_balancing_weight=0.0,
        classifier_dropout=0.0,
        use_moe=False,
    )
    model = build_model(model_config)
    x = torch.randn(2, 5, model_config.d_input)
    t = torch.tensor([[0.0, 0.1, 0.4, 1.2, 3.0], [0.0, 0.0, 0.2, 0.2, 2.0]])

    logits = model(x, t)

    assert logits.shape == (2, model_config.num_classes)
    assert torch.isfinite(logits).all()


def test_classifier_pooling_can_concatenate_last_mean_max() -> None:
    torch.manual_seed(0)
    model_config = ModelConfig(
        d_input=6,
        d_model=16,
        feature_embed_dim=4,
        feature_num_frequencies=3,
        feature_sigma=1.0,
        num_heads=2,
        max_dt=3.0,
        num_experts=2,
        top_k=1,
        num_classes=3,
        rope_type="hybrid_crope",
        rope_base=10000,
        attention_dropout=0.0,
        moe_dropout=0.0,
        moe_expansion_factor=2,
        moe_router_noise=0.0,
        moe_load_balancing_weight=0.0,
        classifier_dropout=0.0,
        use_moe=False,
        classifier_pooling=ClassifierPoolingConfig(methods=("last", "mean", "max"), last_k=3),
    )
    model = build_model(model_config)
    x = torch.randn(2, 5, model_config.d_input)
    t = torch.arange(5, dtype=torch.float32).repeat(2, 1)

    logits = model(x, t)

    assert logits.shape == (2, model_config.num_classes)
    assert torch.isfinite(logits).all()
    assert model.classifier.trunk[0].normalized_shape == (3 * model_config.d_model,)
    assert model.classifier.trunk[2].in_features == 3 * model_config.d_model


def test_auxiliary_heads_expose_movement_and_direction_outputs() -> None:
    torch.manual_seed(0)
    model_config = ModelConfig(
        d_input=6,
        d_model=16,
        feature_embed_dim=4,
        feature_num_frequencies=3,
        feature_sigma=1.0,
        num_heads=2,
        max_dt=3.0,
        num_experts=2,
        top_k=1,
        num_classes=3,
        rope_type="hybrid_crope",
        rope_base=10000,
        attention_dropout=0.0,
        moe_dropout=0.0,
        moe_expansion_factor=2,
        moe_router_noise=0.0,
        moe_load_balancing_weight=0.0,
        classifier_dropout=0.0,
        use_moe=False,
        classifier_pooling=ClassifierPoolingConfig(methods=("last", "mean", "max"), last_k=3),
        auxiliary_heads=AuxiliaryHeadsConfig(enabled=True, movement=True, direction=True, hidden_dim=8),
    )
    model = build_model(model_config)
    x = torch.randn(2, 5, model_config.d_input)
    t = torch.arange(5, dtype=torch.float32).repeat(2, 1)

    logits = model(x, t)

    assert logits.shape == (2, model_config.num_classes)
    assert model.classifier.trunk[2].out_features == 8
    assert model.auxiliary_outputs["movement_logit"].shape == (2,)
    assert model.auxiliary_outputs["direction_logits"].shape == (2, 2)


def test_classifier_loads_legacy_sequential_head_state_dict_keys() -> None:
    torch.manual_seed(0)
    model_config = ModelConfig(
        d_input=6,
        d_model=16,
        feature_embed_dim=4,
        feature_num_frequencies=3,
        feature_sigma=1.0,
        num_heads=2,
        max_dt=3.0,
        num_experts=2,
        top_k=1,
        num_classes=3,
        rope_type="hybrid_crope",
        rope_base=10000,
        attention_dropout=0.0,
        moe_dropout=0.0,
        moe_expansion_factor=2,
        moe_router_noise=0.0,
        moe_load_balancing_weight=0.0,
        classifier_dropout=0.0,
        use_moe=False,
        classifier_pooling=ClassifierPoolingConfig(methods=("last", "mean", "max"), last_k=3),
    )
    current_model = build_model(model_config)
    legacy_state_dict = {}
    replacements = {
        "classifier.trunk.0.": "classifier.head.0.",
        "classifier.trunk.2.": "classifier.head.2.",
        "classifier.class_head.": "classifier.head.5.",
    }
    for key, value in current_model.state_dict().items():
        legacy_key = key
        for current_prefix, legacy_prefix in replacements.items():
            if key.startswith(current_prefix):
                legacy_key = key.replace(current_prefix, legacy_prefix, 1)
                break
        legacy_state_dict[legacy_key] = value.clone()

    reloaded_model = build_model(model_config)
    reloaded_model.load_state_dict(legacy_state_dict, strict=True)

    x = torch.randn(2, 5, model_config.d_input)
    t = torch.arange(5, dtype=torch.float32).repeat(2, 1)
    assert torch.isfinite(reloaded_model(x, t)).all()


def test_classifier_pooling_last_k_can_exceed_sequence_length() -> None:
    torch.manual_seed(0)
    model_config = ModelConfig(
        d_input=6,
        d_model=16,
        feature_embed_dim=4,
        feature_num_frequencies=3,
        feature_sigma=1.0,
        num_heads=2,
        max_dt=3.0,
        num_experts=2,
        top_k=1,
        num_classes=3,
        rope_type="hybrid_crope",
        rope_base=10000,
        attention_dropout=0.0,
        moe_dropout=0.0,
        moe_expansion_factor=2,
        moe_router_noise=0.0,
        moe_load_balancing_weight=0.0,
        classifier_dropout=0.0,
        use_moe=False,
        classifier_pooling=ClassifierPoolingConfig(methods=("mean", "max"), last_k=16),
    )
    model = build_model(model_config)
    x = torch.randn(2, 5, model_config.d_input)
    t = torch.arange(5, dtype=torch.float32).repeat(2, 1)

    logits = model(x, t)

    assert logits.shape == (2, model_config.num_classes)
    assert torch.isfinite(logits).all()
    assert model.classifier.trunk[0].normalized_shape == (2 * model_config.d_model,)


def test_multi_layer_model_can_disable_moe_for_all_blocks() -> None:
    torch.manual_seed(0)
    model_config = ModelConfig(
        d_input=6,
        d_model=16,
        feature_embed_dim=4,
        feature_num_frequencies=3,
        feature_sigma=1.0,
        num_heads=2,
        max_dt=3.0,
        num_experts=2,
        top_k=1,
        num_classes=3,
        rope_type="hybrid_crope",
        rope_base=10000,
        attention_dropout=0.0,
        moe_dropout=0.0,
        moe_expansion_factor=2,
        moe_router_noise=0.0,
        moe_load_balancing_weight=0.0,
        classifier_dropout=0.0,
        num_layers=3,
        latent_spatial_embed_dim=4,
        use_moe=False,
    )
    model = build_model(model_config)
    x = torch.randn(2, 5, model_config.d_input)
    t = torch.arange(5, dtype=torch.float32).repeat(2, 1)

    logits = model(x, t)

    assert logits.shape == (2, model_config.num_classes)
    assert all(layer.moe is None for layer in model.encoder.layers)
    assert all(layer.dense_fnn is not None for layer in model.encoder.layers)
    assert model.encoder.moe is None
    assert model.moe_load_balancing_loss is None
    assert model.moe_routing is None


def test_processing_pipeline_writes_fold_scoped_outputs(artifact_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    raw_dir = artifact_dir / "raw"
    write_lobster_day(raw_dir, "TEST", "2020-01-01")
    write_lobster_day(raw_dir, "TEST", "2020-01-02")
    write_lobster_day(raw_dir, "TEST", "2020-01-03")

    base_config = load_config()
    payload = yaml.safe_load(base_config.path.read_text(encoding="utf-8"))
    payload["data"]["raw_data_dir"] = "raw"
    payload["data"]["processed_data_dir"] = "processed"
    payload["data"]["sequence_data_dir"] = "sequences"
    payload["data"]["logs_dir"] = "logs"
    payload["data"]["tick_size"] = 1.0
    payload["data"]["sequence_window"] = 3
    payload["dataset_splits"] = {
        "train_dates": ["2020-01-01"],
        "validation_dates": ["2020-01-02"],
        "test_dates": ["2020-01-03"],
    }
    payload["folds"] = [
        {
            "id": "fold_001",
            "train_dates": ["2020-01-01"],
            "validation_dates": ["2020-01-02"],
            "test_dates": ["2020-01-03"],
        }
    ]
    payload["preprocessing"]["snapshot_window"] = 4
    payload["preprocessing"]["labels"]["smoothing"]["k"] = 1
    payload["preprocessing"]["labels"]["smoothing"]["h"] = 2
    payload["preprocessing"]["labels"]["smoothing"]["adaptive_threshold"] = {
        "enabled": True,
        "exit_spread_window": 2,
        "volatility_window": 2,
        "round_trip_fees_bps": 0.0,
        "volatility_lambda": 0.0,
    }
    payload["preprocessing"]["temporal_features"]["market_open_seconds"] = 0
    payload["preprocessing"]["temporal_features"]["market_close_seconds"] = 100000
    payload["preprocessing"]["temporal_features"]["start_offset_minutes"] = 0
    payload["preprocessing"]["temporal_features"]["end_offset_minutes"] = 0
    payload["preprocessing"]["normalization"]["derivatives_stats_dir"] = "derivatives"
    payload["preprocessing"]["kinematic_tokenization"]["method"] = "basis"
    payload["preprocessing"]["microprice"]["enabled"] = False

    config_path = artifact_dir / "fold_pipeline.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    summary = LobProcessingPipeline(ExperimentConfig.from_yaml(config_path)).run()
    output = capsys.readouterr().out

    assert "fold_001" in summary
    assert "fold_001 adaptive method C train label distribution:" in output
    assert "fold_001 adaptive method C validation label distribution:" in output
    assert "fold_001 adaptive method C test label distribution:" in output
    assert "cost_floor > volatility_floor" in output
    assert "volatility_floor > cost_floor" in output
    assert "Selected price static PLGS parameters from train:" in output
    assert "Selected volume static exponential scaling from train:" in output
    assert not (artifact_dir / "processed" / "fold_001" / "train" / "TEST_2020-01-01_processed.csv").exists()
    train_features_path = artifact_dir / "sequences" / "fold_001" / "train" / "TEST_2020-01-01_features.npy"
    assert train_features_path.exists()
    assert (artifact_dir / "sequences" / "fold_001" / "validation" / "TEST_2020-01-02_features.npy").exists()
    assert (artifact_dir / "sequences" / "fold_001" / "test" / "TEST_2020-01-03_features.npy").exists()
    metadata_path = artifact_dir / "sequences" / "fold_001" / "preprocessing_metadata.yaml"
    feature_schema_path = artifact_dir / "sequences" / "fold_001" / "feature_schema.yaml"
    assert metadata_path.exists()
    assert feature_schema_path.exists()
    assert (artifact_dir / "derivatives" / "fold_001" / "derivatives_stats.yaml").exists()

    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    feature_schema = yaml.safe_load(feature_schema_path.read_text(encoding="utf-8"))
    assert metadata["save_processed_dataframes"] is False
    assert metadata["adaptive_label_features"] == {"enabled": False, "method": None}
    assert len(feature_schema["ordered_feature_columns"]) == np.load(train_features_path).shape[1]
    assert set(ADAPTIVE_LABEL_FEATURE_COLUMNS).isdisjoint(feature_schema["ordered_feature_columns"])
    label_distribution = metadata["label_distribution"]
    assert label_distribution["method"] == "smoothing_C_adaptive"
    for split in ("train", "validation", "test"):
        assert label_distribution[split]["total"] > 0
        assert set(label_distribution[split]) >= {"-1", "0", "1"}
    floor_comparison = label_distribution["adaptive_threshold_floor_comparison"]
    assert floor_comparison["valid_rows"] > 0
    assert floor_comparison["cost_floor_gt_volatility_floor"]["percentage"] > 0.0
    assert floor_comparison["volatility_floor_gt_cost_floor"]["percentage"] == 0.0
    plgs_metadata = metadata["price_static_plgs"]
    assert plgs_metadata["tau_start"] == payload["preprocessing"]["price_static"]["tau_start"]
    assert plgs_metadata["tau_clip"] == plgs_metadata["x99"]
    assert plgs_metadata["tau_max"] > plgs_metadata["tau_start"]
    assert plgs_metadata["n_values"] > 0
    volume_metadata = metadata["volume_static_exp"]
    assert volume_metadata["quantile"] == payload["preprocessing"]["volume_static"]["quantile"]
    assert volume_metadata["target"] == payload["preprocessing"]["volume_static"]["target"]
    assert volume_metadata["k"] > 0.0
    assert volume_metadata["n_values"] > 0


def test_processing_pipeline_uses_split_fitted_mean_pct_smoothing_threshold(artifact_dir: Path) -> None:
    raw_dir = artifact_dir / "raw"
    write_lobster_day(raw_dir, "TEST", "2020-01-01", rows=18)
    write_lobster_day(raw_dir, "TEST", "2020-01-02", rows=18)
    write_lobster_day(raw_dir, "TEST", "2020-01-03", rows=18)

    base_config = load_config()
    payload = yaml.safe_load(base_config.path.read_text(encoding="utf-8"))
    payload["data"]["raw_data_dir"] = "raw"
    payload["data"]["processed_data_dir"] = "processed"
    payload["data"]["sequence_data_dir"] = "sequences"
    payload["data"]["logs_dir"] = "logs"
    payload["data"]["tick_size"] = 1.0
    payload["data"]["sequence_window"] = 3
    payload["dataset_splits"] = {
        "train_dates": ["2020-01-01"],
        "validation_dates": ["2020-01-02"],
        "test_dates": ["2020-01-03"],
    }
    payload["folds"] = [
        {
            "id": "fold_001",
            "train_dates": ["2020-01-01"],
            "validation_dates": ["2020-01-02"],
            "test_dates": ["2020-01-03"],
        }
    ]
    payload["preprocessing"]["snapshot_window"] = 4
    payload["preprocessing"]["labels"]["smoothing"]["threshold"] = "mean_pct"
    payload["preprocessing"]["labels"]["smoothing"]["fit_scope"] = "per_split"
    payload["preprocessing"]["labels"]["smoothing"]["k"] = 1
    payload["preprocessing"]["labels"]["smoothing"]["h"] = 2
    payload["preprocessing"]["labels"]["smoothing"]["adaptive_threshold"]["enabled"] = False
    payload["preprocessing"]["temporal_features"]["market_open_seconds"] = 0
    payload["preprocessing"]["temporal_features"]["market_close_seconds"] = 100000
    payload["preprocessing"]["temporal_features"]["start_offset_minutes"] = 0
    payload["preprocessing"]["temporal_features"]["end_offset_minutes"] = 0
    payload["preprocessing"]["normalization"]["derivatives_stats_dir"] = "derivatives"
    payload["preprocessing"]["kinematic_tokenization"]["method"] = "basis"
    payload["preprocessing"]["microprice"]["enabled"] = False

    config_path = artifact_dir / "split_fitted_threshold_pipeline.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    LobProcessingPipeline(ExperimentConfig.from_yaml(config_path)).run()

    metadata_path = artifact_dir / "sequences" / "fold_001" / "preprocessing_metadata.yaml"
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))

    threshold = metadata["smoothing_threshold"]
    assert threshold["mode"] == "mean_pct"
    assert threshold["fit_scope"] == "per_split"
    assert set(threshold["splits"]) == {"train", "validation", "test"}
    for split, split_threshold in threshold["splits"].items():
        assert split_threshold["mode"] == "mean_pct"
        assert split_threshold["fit_split"] == split
        assert split_threshold["value"] > 0.0
        assert split_threshold["n_values"] > 0
    assert metadata["label_distribution"]["method"] == "smoothing_mean_pct_per_split_fitted"
    for split in ("train", "validation", "test"):
        assert metadata["label_distribution"][split]["total"] > 0


def test_processing_pipeline_volume_clock_writes_bar_features(artifact_dir: Path) -> None:
    raw_dir = artifact_dir / "raw"
    write_lobster_day(raw_dir, "TEST", "2020-01-01", rows=30)
    write_lobster_day(raw_dir, "TEST", "2020-01-02", rows=30)
    write_lobster_day(raw_dir, "TEST", "2020-01-03", rows=30)

    base_config = load_config()
    payload = yaml.safe_load(base_config.path.read_text(encoding="utf-8"))
    payload["data"]["raw_data_dir"] = "raw"
    payload["data"]["processed_data_dir"] = "processed"
    payload["data"]["sequence_data_dir"] = "sequences"
    payload["data"]["logs_dir"] = "logs"
    payload["data"]["tick_size"] = 1.0
    payload["data"]["sequence_window"] = 2
    payload["dataset_splits"] = {
        "train_dates": ["2020-01-01"],
        "validation_dates": ["2020-01-02"],
        "test_dates": ["2020-01-03"],
    }
    payload["folds"] = [
        {
            "id": "fold_001",
            "train_dates": ["2020-01-01"],
            "validation_dates": ["2020-01-02"],
            "test_dates": ["2020-01-03"],
        }
    ]
    payload["preprocessing"]["snapshot_window"] = 2
    payload["preprocessing"]["labels"]["smoothing"]["threshold"] = 0.0
    payload["preprocessing"]["labels"]["smoothing"]["k"] = 1
    payload["preprocessing"]["labels"]["smoothing"]["h"] = 2
    payload["preprocessing"]["labels"]["smoothing"]["adaptive_threshold"]["enabled"] = False
    payload["preprocessing"]["temporal_features"]["market_open_seconds"] = 0
    payload["preprocessing"]["temporal_features"]["market_close_seconds"] = 100000
    payload["preprocessing"]["temporal_features"]["start_offset_minutes"] = 0
    payload["preprocessing"]["temporal_features"]["end_offset_minutes"] = 0
    payload["preprocessing"]["normalization"]["derivatives_stats_dir"] = "derivatives"
    payload["preprocessing"]["kinematic_tokenization"]["method"] = "basis"
    payload["preprocessing"]["microprice"]["enabled"] = False
    payload["preprocessing"]["sample_clock"] = {
        "mode": "volume",
        "volume_step_shares": 20.0,
        "volume_source": "traded",
        "trade_type_values": [4, 5],
    }

    config_path = artifact_dir / "volume_clock_pipeline.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    LobProcessingPipeline(ExperimentConfig.from_yaml(config_path)).run()

    fold_sequence_dir = artifact_dir / "sequences" / "fold_001"
    schema = yaml.safe_load((fold_sequence_dir / "feature_schema.yaml").read_text(encoding="utf-8"))
    features = schema["ordered_feature_columns"]
    times = np.load(fold_sequence_dir / "train" / "TEST_2020-01-01_times.npy")
    metadata = yaml.safe_load((fold_sequence_dir / "preprocessing_metadata.yaml").read_text(encoding="utf-8"))
    derivative_stats = yaml.safe_load(
        (artifact_dir / "derivatives" / "fold_001" / "derivatives_stats.yaml").read_text(encoding="utf-8")
    )

    assert "bar_trade_count_log1p" in features
    assert "bar_buy_trade_volume_log1p_exp" in features
    assert "bar_signed_trade_volume_signed_log1p_exp" in features
    assert "volume_wall_time" not in features
    assert np.all(np.diff(times) > 0)
    assert metadata["sample_clock"]["mode"] == "volume"
    assert metadata["sample_clock_counts"]["train"]["TEST_2020-01-01"]["sampled_rows"] > 0
    assert "volume_bar_scaling" in metadata
    assert derivative_stats["__metadata__"]["sample_clock"]["mode"] == "volume"


def test_feature_schema_rejects_missing_or_extra_split_columns(artifact_dir: Path) -> None:
    data_config, _ = make_test_configs()
    pipeline = object.__new__(LobProcessingPipeline)
    pipeline.config = type("Config", (), {"data": data_config})()
    pipeline.sequence_builder = DailySequenceBuilder(data_config)

    def make_processed_day(split: str, date: str, df: pd.DataFrame) -> ProcessedDay:
        pair = LobFilePair(
            symbol="TEST",
            date=date,
            segments=(
                LobFileSegment(
                    message_path=artifact_dir / f"{date}_message.csv",
                    orderbook_path=artifact_dir / f"{date}_orderbook.csv",
                ),
            ),
        )
        return ProcessedDay(
            split=split,
            pair=pair,
            raw=df,
            joined=df,
            labeled=df,
            message_features=df,
            processed=df,
            normalized=df.copy(),
        )

    train_df = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0],
            "trend_label": [0, 0, 1],
            "feature_a": [1.0, 2.0, 3.0],
            "feature_b": [4.0, 5.0, 6.0],
        }
    )
    validation_df = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0],
            "trend_label": [0, 1, 1],
            "feature_a": [1.0, 2.0, 3.0],
            "feature_c": [7.0, 8.0, 9.0],
        }
    )
    processed_splits = {
        "train": [make_processed_day("train", "2020-01-01", train_df)],
        "validation": [make_processed_day("validation", "2020-01-02", validation_df)],
        "test": [],
    }

    with pytest.raises(ValueError, match="Feature schema mismatch.*missing columns=.*feature_b.*extra columns=.*feature_c"):
        pipeline.apply_feature_schema(processed_splits, artifact_dir / "feature_schema.yaml")


def test_pipeline_loads_multiple_lobster_segments_for_one_trading_day(artifact_dir: Path) -> None:
    data_config, preprocessing_config = make_test_configs()
    preprocessing_config.temporal_features.market_open_seconds = 34200.0
    preprocessing_config.temporal_features.market_close_seconds = 34204.0
    preprocessing_config.temporal_features.start_offset_minutes = 0
    preprocessing_config.temporal_features.end_offset_minutes = 0

    message_columns = ["time", "type", "order_id", "size", "price", "direction"]
    orderbook_columns = ["ask_price_1", "ask_size_1", "bid_price_1", "bid_size_1"]
    first_messages = pd.DataFrame(
        [
            [34200.0, 1, 1, 10, 101.0, 1],
            [34201.0, 1, 2, 11, 102.0, -1],
            [34202.0, 3, 2, 11, 102.0, -1],
        ],
        columns=message_columns,
    )
    second_messages = pd.DataFrame(
        [
            [34202.0, 3, 2, 11, 102.0, -1],
            [34203.0, 1, 3, 12, 103.0, 1],
            [34204.0, 1, 4, 13, 104.0, -1],
        ],
        columns=message_columns,
    )
    first_orderbook = pd.DataFrame(
        [
            [101.0, 20, 100.0, 19],
            [102.0, 21, 101.0, 20],
            [102.0, 21, 101.0, 20],
        ],
        columns=orderbook_columns,
    )
    second_orderbook = pd.DataFrame(
        [
            [102.0, 21, 101.0, 20],
            [103.0, 22, 102.0, 21],
            [104.0, 23, 103.0, 22],
        ],
        columns=orderbook_columns,
    )

    for suffix, message_df, orderbook_df in (
        ("34200000_34202000", first_messages, first_orderbook),
        ("34202000_34204000", second_messages, second_orderbook),
    ):
        message_df.to_csv(artifact_dir / f"TEST_2020-01-01_{suffix}_message_1.csv", index=False)
        orderbook_df.to_csv(artifact_dir / f"TEST_2020-01-01_{suffix}_orderbook_1.csv", index=False)

    pipeline = object.__new__(LobProcessingPipeline)
    pipeline.config = type(
        "Config",
        (),
        {
            "data": data_config,
            "preprocessing": preprocessing_config,
        },
    )()
    pipeline.data_dir = artifact_dir
    pipeline.joiner = MessageOrderbookJoiner(time_column=data_config.time_column)
    pipeline.session_filter = TradingSessionFilter(
        time_column=data_config.time_column,
        market_open_seconds=preprocessing_config.temporal_features.market_open_seconds,
        market_close_seconds=preprocessing_config.temporal_features.market_close_seconds,
        start_offset_minutes=0,
        end_offset_minutes=0,
    )

    pairs = pipeline.discover_pairs()
    assert len(pairs) == 1
    assert pairs[0].segment_count == 2

    trimmed = pipeline.load_and_trim_pair(pairs[0])

    assert trimmed[data_config.time_column].tolist() == [34200.0, 34201.0, 34202.0, 34203.0, 34204.0]
    assert len(trimmed) == 5
