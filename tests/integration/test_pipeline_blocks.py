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
    BasisKinematicConfig,
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
from horizon import TargetLabelPipeline
from kinematic_preprocessing import (
    DerivativeNormalizer,
    MessageFeatureProcessor,
    MessageOrderbookJoiner,
    SnapshotBatchProcessor,
)
from model import build_model
from processing import LobProcessingPipeline


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
    message_df.to_csv(raw_dir / f"{symbol}_{date}_message.csv", index=False)
    orderbook_df.to_csv(raw_dir / f"{symbol}_{date}_orderbook.csv", index=False)


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
                h=1,
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
            derivatives_stats_path="derivatives_stats.yaml",
            scope="train_only",
        ),
        kinematic_tokenization=KinematicTokenizationConfig(
            method=tokenization_method,
            chunk_size=3,
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


def test_processing_pipeline_writes_fold_scoped_outputs(artifact_dir: Path) -> None:
    raw_dir = artifact_dir / "raw"
    write_lobster_day(raw_dir, "TEST", "2020-01-01")
    write_lobster_day(raw_dir, "TEST", "2020-01-02")

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
        "test_dates": [],
    }
    payload["folds"] = [
        {
            "id": "fold_001",
            "train_dates": ["2020-01-01"],
            "validation_dates": ["2020-01-02"],
            "test_dates": [],
        }
    ]
    payload["preprocessing"]["snapshot_window"] = 4
    payload["preprocessing"]["labels"]["smoothing"]["k"] = 1
    payload["preprocessing"]["labels"]["smoothing"]["h"] = 1
    payload["preprocessing"]["temporal_features"]["market_open_seconds"] = 0
    payload["preprocessing"]["temporal_features"]["market_close_seconds"] = 100000
    payload["preprocessing"]["temporal_features"]["start_offset_minutes"] = 0
    payload["preprocessing"]["temporal_features"]["end_offset_minutes"] = 0
    payload["preprocessing"]["normalization"]["derivatives_stats_path"] = "derivatives/derivatives_stats.yaml"
    payload["preprocessing"]["kinematic_tokenization"]["method"] = "basis"

    config_path = artifact_dir / "fold_pipeline.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    summary = LobProcessingPipeline(ExperimentConfig.from_yaml(config_path)).run()

    assert "fold_001" in summary
    assert (artifact_dir / "processed" / "fold_001" / "train" / "TEST_2020-01-01_processed.csv").exists()
    assert (artifact_dir / "sequences" / "fold_001" / "train" / "TEST_2020-01-01_features.npy").exists()
    assert (artifact_dir / "sequences" / "fold_001" / "validation" / "TEST_2020-01-02_features.npy").exists()
    assert (artifact_dir / "sequences" / "fold_001" / "preprocessing_metadata.yaml").exists()
    assert (artifact_dir / "derivatives" / "fold_001" / "derivatives_stats.yaml").exists()
