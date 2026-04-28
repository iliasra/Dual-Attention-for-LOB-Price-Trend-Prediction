from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _ensure_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "auto", "none"}:
            return None
        return [value]
    return list(value)


@dataclass(slots=True)
class DataConfig:
    raw_data_dir: str = "../data/LOBSTER"
    processed_data_dir: str = "../data/processed_dataframes"
    sequence_data_dir: str = "../data/sequences"
    time_column: str = "time"
    label_column: str = "trend_label"
    label_mapping: dict[int, int] = field(default_factory=lambda: {-1: 0, 0: 1, 1: 2})
    price_columns: list[str] | None = None
    volume_columns: list[str] | None = None
    feature_exclude_columns: list[str] = field(default_factory=list)
    sequence_window: int = 64

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DataConfig":
        return cls(
            raw_data_dir=str(payload.get("raw_data_dir", "../data/LOBSTER")),
            processed_data_dir=str(payload.get("processed_data_dir", "../data/processed_dataframes")),
            sequence_data_dir=str(payload.get("sequence_data_dir", "../data/sequences")),
            time_column=payload.get("time_column", "time"),
            label_column=payload.get("label_column", "trend_label"),
            label_mapping={int(key): int(value) for key, value in payload.get("label_mapping", {-1: 0, 0: 1, 1: 2}).items()},
            price_columns=_ensure_list(payload.get("price_columns")),
            volume_columns=_ensure_list(payload.get("volume_columns")),
            feature_exclude_columns=list(payload.get("feature_exclude_columns", [])),
            sequence_window=int(payload.get("sequence_window", 64)),
        )


@dataclass(slots=True)
class StreamConfig:
    enabled: bool = True
    columns: list[str] | None = None
    alpha: float = 5.0
    tick_size: float = 1.0
    tau_start: float = 1.0
    tau_clip: float | None = 50.0
    tau_max: float = 100.0
    k: float = 2000.0
    reference: str = "tick"

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None, **defaults: Any) -> "StreamConfig":
        payload = payload or {}
        merged = {**defaults, **payload}
        return cls(
            enabled=bool(merged.get("enabled", True)),
            columns=_ensure_list(merged.get("columns")),
            alpha=float(merged.get("alpha", 5.0)),
            tick_size=float(merged.get("tick_size", 1.0)),
            tau_start=float(merged.get("tau_start", 1.0)),
            tau_clip=None if merged.get("tau_clip") is None else float(merged.get("tau_clip")),
            tau_max=float(merged.get("tau_max", 100.0)),
            k=float(merged.get("k", 2000.0)),
            reference=str(merged.get("reference", "tick")),
        )


@dataclass(slots=True)
class JoinConfig:
    method: str = "ffill" #to avoid look-ahead bias

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "JoinConfig":
        return cls(method=str(payload.get("method", "ffill")))


@dataclass(slots=True)
class SmoothingLabelConfig:
    method: str = "C"
    threshold: float | None = None
    k: int = 5
    h: int = 10
    bid_column: str = "bid_price_1"
    ask_column: str = "ask_price_1"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SmoothingLabelConfig":
        return cls(
            method=str(payload.get("method", "C")),
            threshold=None if payload.get("threshold") is None else float(payload.get("threshold")),
            k=int(payload.get("k", 5)),
            h=int(payload.get("h", 10)),
            bid_column=str(payload.get("bid_column", "bid_price_1")),
            ask_column=str(payload.get("ask_column", "ask_price_1")),
        )


@dataclass(slots=True)
class TripleBarrierLabelConfig:
    horizon: int = 10
    upper_barrier_ticks: float = 2.0
    lower_barrier_ticks: float = 3.0
    bid_column: str = "bid_price_1"
    ask_column: str = "ask_price_1"
    price_column: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TripleBarrierLabelConfig":
        return cls(
            horizon=int(payload.get("horizon", 10)),
            upper_barrier_ticks=float(payload.get("upper_barrier_ticks", 2.0)),
            lower_barrier_ticks=float(payload.get("lower_barrier_ticks", 3.0)),
            bid_column=str(payload.get("bid_column", "bid_price_1")),
            ask_column=str(payload.get("ask_column", "ask_price_1")),
            price_column=payload.get("price_column"),
        )


@dataclass(slots=True)
class LabelConfig:
    strategy: str = "smoothing"
    smoothing: SmoothingLabelConfig = field(default_factory=SmoothingLabelConfig)
    triple_barrier: TripleBarrierLabelConfig = field(default_factory=TripleBarrierLabelConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LabelConfig":
        return cls(
            strategy=str(payload.get("strategy", "smoothing")),
            smoothing=SmoothingLabelConfig.from_dict(payload.get("smoothing", {})),
            triple_barrier=TripleBarrierLabelConfig.from_dict(payload.get("triple_barrier", {})),
        )


@dataclass(slots=True)
class MessageConfig:
    tick_size: float = 1.0
    size_column: str = "size"
    price_column: str = "price"
    order_id_column: str = "order_id"
    categorical_value_map: dict[str, list[int]] = field(
        default_factory=lambda: {
            "type": [1, 2, 3, 4, 5],
            "direction": [-1, 1],
        }
    )
    drop_columns: list[str] = field(default_factory=lambda: ["price", "size", "type", "direction", "order_id"])

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MessageConfig":
        return cls(
            tick_size=float(payload.get("tick_size", 1.0)),
            size_column=str(payload.get("size_column", "size")),
            price_column=str(payload.get("price_column", "price")),
            order_id_column=str(payload.get("order_id_column", "order_id")),
            categorical_value_map={
                str(column): [int(value) for value in values]
                for column, values in payload.get(
                    "categorical_value_map",
                    {
                        "type": [1, 2, 3, 4, 5],
                        "direction": [-1, 1],
                    },
                ).items()
            },
            drop_columns=list(payload.get("drop_columns", ["price", "size", "type", "direction", "order_id"])),
        )


@dataclass(slots=True)
class TemporalFeaturesConfig:
    add_day_sincos: bool = True
    day_frequency: int = 86400
    keep_timestamp: bool = True
    market_open_seconds: float = 34200.0
    market_close_seconds: float = 57600.0
    start_offset_minutes: int = 15
    end_offset_minutes: int = 15

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TemporalFeaturesConfig":
        return cls(
            add_day_sincos=bool(payload.get("add_day_sincos", True)),
            day_frequency=int(payload.get("day_frequency", 86400)),
            keep_timestamp=bool(payload.get("keep_timestamp", True)),
            market_open_seconds=float(payload.get("market_open_seconds", 34200.0)),
            market_close_seconds=float(payload.get("market_close_seconds", 57600.0)),
            start_offset_minutes=int(payload.get("start_offset_minutes", 15)),
            end_offset_minutes=int(payload.get("end_offset_minutes", 15)),
        )


@dataclass(slots=True)
class NormalizationConfig:
    derivatives_stats_path: str = "../data/derivatives_z_scores/derivatives_stats.yaml"
    scope: str = "train_only"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NormalizationConfig":
        return cls(
            derivatives_stats_path=str(
                payload.get(
                    "derivatives_stats_path",
                    "../data/derivatives_z_scores/derivatives_stats.yaml",
                )
            ),
            scope=str(payload.get("scope", "train_only")),
        )


@dataclass(slots=True)
class DatasetSplitConfig:
    train_dates: list[str] = field(default_factory=list)
    validation_dates: list[str] = field(default_factory=list)
    test_dates: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetSplitConfig":
        return cls(
            train_dates=[str(value) for value in payload.get("train_dates", [])],
            validation_dates=[str(value) for value in payload.get("validation_dates", [])],
            test_dates=[str(value) for value in payload.get("test_dates", [])],
        )


@dataclass(slots=True)
class PreprocessingConfig:
    snapshot_window: int = 100
    join: JoinConfig = field(default_factory=JoinConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    message: MessageConfig = field(default_factory=MessageConfig)
    temporal_features: TemporalFeaturesConfig = field(default_factory=TemporalFeaturesConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    price_kinematic: StreamConfig = field(default_factory=StreamConfig)
    price_static: StreamConfig = field(default_factory=StreamConfig)
    volume_kinematic: StreamConfig = field(default_factory=StreamConfig)
    volume_static: StreamConfig = field(default_factory=StreamConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PreprocessingConfig":
        return cls(
            snapshot_window=int(payload.get("snapshot_window", 100)),
            join=JoinConfig.from_dict(payload.get("join", {})),
            labels=LabelConfig.from_dict(payload.get("labels", {})),
            message=MessageConfig.from_dict(payload.get("message", {})),
            temporal_features=TemporalFeaturesConfig.from_dict(payload.get("temporal_features", {})),
            normalization=NormalizationConfig.from_dict(payload.get("normalization", {})),
            price_kinematic=StreamConfig.from_dict(payload.get("price_kinematic", {})),
            price_static=StreamConfig.from_dict(payload.get("price_static", {})),
            volume_kinematic=StreamConfig.from_dict(payload.get("volume_kinematic", {})),
            volume_static=StreamConfig.from_dict(payload.get("volume_static", {})),
        )


@dataclass(slots=True)
class ModelConfig:
    d_input: int | None = None
    d_model: int = 128
    feature_embed_dim: int = 16
    feature_num_frequencies: int = 8
    feature_sigma: float = 1.0
    num_heads: int = 8
    max_dt: float = 3.0
    num_experts: int = 4
    top_k: int = 2
    num_classes: int = 3
    rope_type: str = "crope"
    rope_base: int = 10000
    attention_dropout: float = 0.1
    moe_dropout: float = 0.1
    moe_expansion_factor: int = 4
    moe_router_noise: float = 1e-2  # routing noise for MoE training
    moe_load_balancing_weight: float = 1e-2  # load-balancing coefficient for MoE training
    classifier_dropout: float = 0.1

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelConfig":
        return cls(
            d_input=None if payload.get("d_input") is None else int(payload["d_input"]),
            d_model=int(payload.get("d_model", 128)),
            feature_embed_dim=int(payload.get("feature_embed_dim", 16)),
            feature_num_frequencies=int(payload.get("feature_num_frequencies", 8)),
            feature_sigma=float(payload.get("feature_sigma", 1.0)),
            num_heads=int(payload.get("num_heads", 8)),
            max_dt=float(payload.get("max_dt", 3.0)),
            num_experts=int(payload.get("num_experts", 4)),
            top_k=int(payload.get("top_k", 2)),
            num_classes=int(payload.get("num_classes", 3)),
            rope_type=str(payload.get("rope_type", "crope")),
            rope_base=int(payload.get("rope_base", 10000)),
            attention_dropout=float(payload.get("attention_dropout", 0.1)),
            moe_dropout=float(payload.get("moe_dropout", 0.1)),
            moe_expansion_factor=int(payload.get("moe_expansion_factor", 4)),
            moe_router_noise=float(payload.get("moe_router_noise", 1e-2)),  # routing noise for MoE training
            moe_load_balancing_weight=float(payload.get("moe_load_balancing_weight", 1e-2)),  # load-balancing coefficient for MoE training
            classifier_dropout=float(payload.get("classifier_dropout", 0.1)),
        )

    def resolved_d_input(self, inferred_feature_count: int | None = None) -> int:
        if self.d_input is not None:
            return self.d_input
        if inferred_feature_count is None:
            raise ValueError("d_input is missing from config and could not be inferred from the data.")
        return inferred_feature_count


@dataclass(slots=True)
class TrainingConfig:
    device: str = "cuda"
    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    focal_gamma: float = 2.0
    class_weights: list[float] | None = field(default_factory=lambda: [2.0, 10.0, 1.0])
    grad_clip_norm: float = 1.0
    best_model_path: str = "best_lob_transformer.pth"
    use_amp: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingConfig":
        raw_weights = payload.get("class_weights", [2.0, 10.0, 1.0])
        return cls(
            device=str(payload.get("device", "cuda")),
            epochs=int(payload.get("epochs", 5)),
            batch_size=int(payload.get("batch_size", 16)),
            learning_rate=float(payload.get("learning_rate", 1e-4)),
            weight_decay=float(payload.get("weight_decay", 1e-4)),
            focal_gamma=float(payload.get("focal_gamma", 2.0)),
            class_weights=None if raw_weights is None else [float(weight) for weight in raw_weights],
            grad_clip_norm=float(payload.get("grad_clip_norm", 1.0)),
            best_model_path=str(payload.get("best_model_path", "best_lob_transformer.pth")),
            use_amp=bool(payload.get("use_amp", True)),
        )


@dataclass(slots=True)
class ExperimentConfig:
    path: Path
    data: DataConfig = field(default_factory=DataConfig)
    dataset_splits: DatasetSplitConfig = field(default_factory=DatasetSplitConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}

        return cls(
            path=config_path.resolve(),
            data=DataConfig.from_dict(payload.get("data", {})),
            dataset_splits=DatasetSplitConfig.from_dict(payload.get("dataset_splits", {})),
            preprocessing=PreprocessingConfig.from_dict(payload.get("preprocessing", {})),
            model=ModelConfig.from_dict(payload.get("model", {})),
            training=TrainingConfig.from_dict(payload.get("training", {})),
        )


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    default_path = Path(__file__).with_name("pipeline_config.yaml")
    return ExperimentConfig.from_yaml(path or default_path)
