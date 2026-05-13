from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


BEST_MODEL_FILENAME = "best_lob_transformer.pth"


REQUIRED_CONFIG_SCHEMA: dict[str, Any] = {
    "seed": None,
    "data": {
        "raw_data_dir": None,
        "processed_data_dir": None,
        "sequence_data_dir": None,
        "logs_dir": None,
        "tick_size": None,
        "time_column": None,
        "label_column": None,
        "label_mapping": None,
        "price_columns": None,
        "volume_columns": None,
        "feature_exclude_columns": None,
        "sequence_window": None,
    },
    "dataset_splits": {
        "train_dates": None,
        "validation_dates": None,
        "test_dates": None,
    },
    "preprocessing": {
        "snapshot_window": None,
        "labels": {
            "strategy": None,
            "smoothing": {
                "method": None,
                "threshold": None,
                "k": None,
                "h": None,
                "bid_column": None,
                "ask_column": None,
                "adaptive_threshold": {
                    "enabled": None,
                    "exit_spread_window": None,
                    "volatility_window": None,
                    "round_trip_fees_bps": None,
                    "volatility_lambda": None,
                },
            },
            "triple_barrier": {
                "horizon": None,
                "upper_barrier_ticks": None,
                "lower_barrier_ticks": None,
                "bid_column": None,
                "ask_column": None,
                "price_column": None,
            },
        },
        "message": {
            "size_column": None,
            "price_column": None,
            "order_id_column": None,
            "categorical_value_map": None,
            "drop_columns": None,
        },
        "temporal_features": {
            "add_day_sincos": None,
            "day_frequency": None,
            "keep_timestamp": None,
            "market_open_seconds": None,
            "market_close_seconds": None,
            "start_offset_minutes": None,
            "end_offset_minutes": None,
        },
        "normalization": {
            "derivatives_stats_dir": None,
            "scope": None,
        },
        "kinematic_tokenization": {
            "method": None,
            "chunk_size": None,
        },
        "price_kinematic": {
            "enabled": None,
            "columns": None,
            "reference": None,
            "basis": {
                "alpha": None,
            },
            "fast": {
                "n_basis": None,
                "df": None,
                "eval_at": None,
            },
        },
        "price_static": {
            "enabled": None,
            "columns": None,
            "tau_start": None,
            "tau_clip": None,
            "tau_max": None,
        },
        "volume_kinematic": {
            "enabled": None,
            "columns": None,
            "reference": None,
            "basis": {
                "alpha": None,
            },
            "fast": {
                "n_basis": None,
                "df": None,
                "eval_at": None,
            },
        },
        "volume_static": {
            "enabled": None,
            "columns": None,
            "quantile": None,
            "target": None,
        },
    },
    "model": {
        "d_input": None,
        "d_model": None,
        "feature_embed_dim": None,
        "feature_num_frequencies": None,
        "feature_sigma": None,
        "num_heads": None,
        "max_dt_quantile": None,
        "num_experts": None,
        "top_k": None,
        "num_classes": None,
        "rope_type": None,
        "rope_base": None,
        "attention_dropout": None,
        "moe_dropout": None,
        "moe_expansion_factor": None,
        "moe_router_noise": None,
        "moe_load_balancing_weight": None,
        "classifier_dropout": None,
    },
    "training": {
        "device": None,
        "epochs": None,
        "batch_size": None,
        "num_workers": None,
        "early_stopping_patience": None,
        "persistent_workers": None,
        "learning_rate": None,
        "weight_decay": None,
        "focal_gamma": None,
        "grad_clip_norm": None,
        "model_dir": None,
        "use_amp": None,
    },
}

OPTIONAL_TOP_LEVEL_KEYS = {"folds"}
OPTIONAL_CONFIG_KEYS = {
    "preprocessing.labels.smoothing.adaptive_threshold",
    "preprocessing.price_static.tau_clip",
    "preprocessing.price_static.tau_max",
}


ALLOWED_CONFIG_VALUES: dict[str, set[Any]] = {
    "preprocessing.labels.strategy": {"smoothing", "triple_barrier"},
    "preprocessing.labels.smoothing.method": {"A", "B", "C"},
    "preprocessing.normalization.scope": {"train_only"},
    "preprocessing.kinematic_tokenization.method": {"basis", "fast"},
    "preprocessing.price_kinematic.reference": {"tick", "time"},
    "preprocessing.volume_kinematic.reference": {"tick", "time"},
    "model.rope_type": {"crope", "hybrid_crope", "hybrid-crope", "hybrid"},
    "training.device": {"cpu", "cuda"},
}


def _ensure_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "auto", "none"}:
            return None
        return [value]
    return list(value)


def _validate_required_config(payload: Any, schema: dict[str, Any]) -> None:
    missing: list[str] = []
    invalid_mappings: list[str] = []
    unexpected: list[str] = []

    def walk(node: Any, subtree: dict[str, Any], prefix: str = "") -> None:
        if not isinstance(node, dict):
            invalid_mappings.append(prefix or "<root>")
            return

        for key, child_schema in subtree.items():
            key_path = f"{prefix}.{key}" if prefix else key
            if key not in node:
                if key_path in OPTIONAL_CONFIG_KEYS:
                    continue
                missing.append(key_path)
                continue
            if isinstance(child_schema, dict):
                walk(node[key], child_schema, key_path)

        allowed_keys = set(subtree)
        if prefix == "":
            allowed_keys |= OPTIONAL_TOP_LEVEL_KEYS
        for key in node:
            if key not in allowed_keys:
                unexpected.append(f"{prefix}.{key}" if prefix else str(key))

    walk(payload, schema)
    if missing or invalid_mappings or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing keys: " + ", ".join(missing))
        if invalid_mappings:
            details.append("expected mappings at: " + ", ".join(invalid_mappings))
        if unexpected:
            details.append("unexpected keys: " + ", ".join(unexpected))
        raise ValueError("Invalid experiment config; " + "; ".join(details))


def _get_nested(payload: dict[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for part in dotted_path.split("."):
        current = current[part]
    return current


def _require_explicit_value(value: Any, dotted_path: str) -> Any:
    if value is None:
        raise ValueError(f"Invalid experiment config; {dotted_path} must be set explicitly.")
    return value


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return float(value)


def _validate_allowed_values(payload: dict[str, Any]) -> None:
    invalid: list[str] = []

    for path, allowed_values in ALLOWED_CONFIG_VALUES.items():
        value = _get_nested(payload, path)
        if isinstance(value, str):
            comparable = value.lower()
            allowed = {str(item).lower() for item in allowed_values}
        else:
            comparable = value
            allowed = allowed_values

        if comparable not in allowed:
            invalid.append(f"{path}={value!r}; allowed values are {sorted(allowed_values)}")

    if invalid:
        raise ValueError("Invalid experiment config; invalid values: " + "; ".join(invalid))


@dataclass(slots=True)
class DataConfig:
    raw_data_dir: str
    processed_data_dir: str
    sequence_data_dir: str
    logs_dir: str
    tick_size: float
    time_column: str
    label_column: str
    label_mapping: dict[int, int]
    price_columns: list[str] | None
    volume_columns: list[str] | None
    feature_exclude_columns: list[str]
    sequence_window: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DataConfig":
        return cls(
            raw_data_dir=str(payload["raw_data_dir"]),
            processed_data_dir=str(payload["processed_data_dir"]),
            sequence_data_dir=str(payload["sequence_data_dir"]),
            logs_dir=str(payload["logs_dir"]),
            tick_size=float(payload["tick_size"]),
            time_column=str(payload["time_column"]),
            label_column=str(payload["label_column"]),
            label_mapping={int(key): int(value) for key, value in payload["label_mapping"].items()},
            price_columns=_ensure_list(payload["price_columns"]),
            volume_columns=_ensure_list(payload["volume_columns"]),
            feature_exclude_columns=list(payload["feature_exclude_columns"]),
            sequence_window=int(payload["sequence_window"]),
        )


@dataclass(slots=True)
class KinematicTokenizationConfig:
    method: str
    chunk_size: int

    def __post_init__(self) -> None:
        self.method = self.method.lower()
        if self.method not in {"basis", "fast"}:
            raise ValueError("preprocessing.kinematic_tokenization.method must be 'basis' or 'fast'.")
        if self.chunk_size <= 0:
            raise ValueError("preprocessing.kinematic_tokenization.chunk_size must be > 0.")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "KinematicTokenizationConfig":
        return cls(
            method=str(_require_explicit_value(payload["method"], "preprocessing.kinematic_tokenization.method")),
            chunk_size=int(
                _require_explicit_value(payload["chunk_size"], "preprocessing.kinematic_tokenization.chunk_size")
            ),
        )


@dataclass(slots=True)
class FastKinematicConfig:
    n_basis: int
    df: float
    eval_at: float
    selected_smoothing_lambda: float | None = None

    def __post_init__(self) -> None:
        if self.n_basis <= 3:
            raise ValueError("Fast kinematic n_basis must be > 3 for cubic B-splines.")
        if not 0.0 < self.df <= self.n_basis:
            raise ValueError("Fast kinematic df must be in (0, n_basis].")
        if not 0.0 <= self.eval_at <= 1.0:
            raise ValueError("Fast kinematic eval_at must be in [0, 1].")
        if self.selected_smoothing_lambda is not None and self.selected_smoothing_lambda < 0:
            raise ValueError("Fast kinematic selected_smoothing_lambda must be >= 0.")

    @classmethod
    def from_dict(cls, payload: dict[str, Any], prefix: str) -> "FastKinematicConfig":
        return cls(
            n_basis=int(_require_explicit_value(payload["n_basis"], f"{prefix}.n_basis")),
            df=float(_require_explicit_value(payload["df"], f"{prefix}.df")),
            eval_at=float(_require_explicit_value(payload["eval_at"], f"{prefix}.eval_at")),
        )


@dataclass(slots=True)
class BasisKinematicConfig:
    alpha: float

    @classmethod
    def from_dict(cls, payload: dict[str, Any], prefix: str) -> "BasisKinematicConfig":
        return cls(
            alpha=float(_require_explicit_value(payload["alpha"], f"{prefix}.alpha")),
        )


@dataclass(slots=True)
class PriceKinematicConfig:
    enabled: bool
    columns: list[str] | None
    tick_size: float
    reference: str
    basis: BasisKinematicConfig
    fast: FastKinematicConfig

    @classmethod
    def from_dict(cls, payload: dict[str, Any], tick_size: float) -> "PriceKinematicConfig":
        return cls(
            enabled=bool(payload["enabled"]),
            columns=_ensure_list(payload["columns"]),
            tick_size=float(tick_size),
            reference=str(payload["reference"]),
            basis=BasisKinematicConfig.from_dict(
                payload["basis"],
                "preprocessing.price_kinematic.basis",
            ),
            fast=FastKinematicConfig.from_dict(
                payload["fast"],
                "preprocessing.price_kinematic.fast",
            ),
        )


@dataclass(slots=True)
class PriceStaticConfig:
    enabled: bool
    columns: list[str] | None
    tick_size: float
    tau_start: float
    tau_clip: float | None
    tau_max: float | None

    @classmethod
    def from_dict(cls, payload: dict[str, Any], tick_size: float) -> "PriceStaticConfig":
        return cls(
            enabled=bool(payload["enabled"]),
            columns=_ensure_list(payload["columns"]),
            tick_size=float(tick_size),
            tau_start=float(payload["tau_start"]),
            tau_clip=_optional_float(payload.get("tau_clip")),
            tau_max=_optional_float(payload.get("tau_max")),
        )


@dataclass(slots=True)
class VolumeKinematicConfig:
    enabled: bool
    columns: list[str] | None
    reference: str
    basis: BasisKinematicConfig
    fast: FastKinematicConfig

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VolumeKinematicConfig":
        return cls(
            enabled=bool(payload["enabled"]),
            columns=_ensure_list(payload["columns"]),
            reference=str(payload["reference"]),
            basis=BasisKinematicConfig.from_dict(
                payload["basis"],
                "preprocessing.volume_kinematic.basis",
            ),
            fast=FastKinematicConfig.from_dict(
                payload["fast"],
                "preprocessing.volume_kinematic.fast",
            ),
        )


@dataclass(slots=True)
class VolumeStaticConfig:
    enabled: bool
    columns: list[str] | None
    quantile: float
    target: float
    k: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.quantile <= 100.0:
            raise ValueError("preprocessing.volume_static.quantile must be in [0, 100].")
        if not 0.0 < self.target < 1.0:
            raise ValueError("preprocessing.volume_static.target must be in (0, 1).")
        if self.k is not None and self.k <= 0:
            raise ValueError("preprocessing.volume_static.k must be > 0 when fitted.")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VolumeStaticConfig":
        return cls(
            enabled=bool(payload["enabled"]),
            columns=_ensure_list(payload["columns"]),
            quantile=float(payload["quantile"]),
            target=float(payload["target"]),
        )

@dataclass(slots=True)
class AdaptiveThresholdConfig:
    enabled: bool
    exit_spread_window: int
    volatility_window: int
    round_trip_fees_bps: float
    volatility_lambda: float

    def __post_init__(self) -> None:
        if self.exit_spread_window <= 0:
            raise ValueError("preprocessing.labels.smoothing.adaptive_threshold.exit_spread_window must be > 0.")
        if self.volatility_window <= 0:
            raise ValueError("preprocessing.labels.smoothing.adaptive_threshold.volatility_window must be > 0.")
        if self.round_trip_fees_bps < 0:
            raise ValueError("preprocessing.labels.smoothing.adaptive_threshold.round_trip_fees_bps must be >= 0.")
        if self.volatility_lambda < 0:
            raise ValueError("preprocessing.labels.smoothing.adaptive_threshold.volatility_lambda must be >= 0.")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AdaptiveThresholdConfig":
        return cls(
            enabled=bool(payload["enabled"]),
            exit_spread_window=int(payload["exit_spread_window"]),
            volatility_window=int(payload["volatility_window"]),
            round_trip_fees_bps=float(payload["round_trip_fees_bps"]),
            volatility_lambda=float(payload["volatility_lambda"]),
        )


@dataclass(slots=True)
class SmoothingLabelConfig:
    method: str
    threshold: float | None
    k: int
    h: int
    bid_column: str
    ask_column: str
    adaptive_threshold: AdaptiveThresholdConfig | None = None

    def __post_init__(self) -> None:
        if self.k < 0:
            raise ValueError("preprocessing.labels.smoothing.k must be >= 0.")
        if self.h <= 0:
            raise ValueError("preprocessing.labels.smoothing.h must be > 0.")
        if self.method.upper() == "C" and self.k >= self.h:
            raise ValueError("preprocessing.labels.smoothing method C requires k < h.")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SmoothingLabelConfig":
        adaptive_payload = payload.get("adaptive_threshold")
        return cls(
            method=str(payload["method"]),
            threshold=_optional_float(payload["threshold"]),
            k=int(payload["k"]),
            h=int(payload["h"]),
            bid_column=str(payload["bid_column"]),
            ask_column=str(payload["ask_column"]),
            adaptive_threshold=None
            if adaptive_payload is None
            else AdaptiveThresholdConfig.from_dict(adaptive_payload),
        )


@dataclass(slots=True)
class TripleBarrierLabelConfig:
    horizon: int
    upper_barrier_ticks: float
    lower_barrier_ticks: float
    bid_column: str
    ask_column: str
    price_column: str | None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TripleBarrierLabelConfig":
        return cls(
            horizon=int(payload["horizon"]),
            upper_barrier_ticks=float(payload["upper_barrier_ticks"]),
            lower_barrier_ticks=float(payload["lower_barrier_ticks"]),
            bid_column=str(payload["bid_column"]),
            ask_column=str(payload["ask_column"]),
            price_column=None if payload["price_column"] is None else str(payload["price_column"]),
        )


@dataclass(slots=True)
class LabelConfig:
    strategy: str
    smoothing: SmoothingLabelConfig
    triple_barrier: TripleBarrierLabelConfig

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LabelConfig":
        return cls(
            strategy=str(payload["strategy"]),
            smoothing=SmoothingLabelConfig.from_dict(payload["smoothing"]),
            triple_barrier=TripleBarrierLabelConfig.from_dict(payload["triple_barrier"]),
        )


@dataclass(slots=True)
class MessageConfig:
    tick_size: float
    size_column: str
    price_column: str
    order_id_column: str
    categorical_value_map: dict[str, list[int]]
    drop_columns: list[str]

    @classmethod
    def from_dict(cls, payload: dict[str, Any], tick_size: float) -> "MessageConfig":
        return cls(
            tick_size=float(tick_size),
            size_column=str(payload["size_column"]),
            price_column=str(payload["price_column"]),
            order_id_column=str(payload["order_id_column"]),
            categorical_value_map={
                str(column): [int(value) for value in values]
                for column, values in payload["categorical_value_map"].items()
            },
            drop_columns=list(payload["drop_columns"]),
        )


@dataclass(slots=True)
class TemporalFeaturesConfig:
    add_day_sincos: bool
    day_frequency: int
    keep_timestamp: bool
    market_open_seconds: float
    market_close_seconds: float
    start_offset_minutes: int
    end_offset_minutes: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TemporalFeaturesConfig":
        return cls(
            add_day_sincos=bool(payload["add_day_sincos"]),
            day_frequency=int(payload["day_frequency"]),
            keep_timestamp=bool(payload["keep_timestamp"]),
            market_open_seconds=float(payload["market_open_seconds"]),
            market_close_seconds=float(payload["market_close_seconds"]),
            start_offset_minutes=int(payload["start_offset_minutes"]),
            end_offset_minutes=int(payload["end_offset_minutes"]),
        )


@dataclass(slots=True)
class NormalizationConfig:
    derivatives_stats_dir: str
    scope: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NormalizationConfig":
        return cls(
            derivatives_stats_dir=str(payload["derivatives_stats_dir"]),
            scope=str(payload["scope"]),
        )


def _validate_split_dates(
    context: str,
    *,
    train_dates: list[str],
    validation_dates: list[str],
    test_dates: list[str],
) -> None:
    missing = [
        split_name
        for split_name, dates in (
            ("train_dates", train_dates),
            ("validation_dates", validation_dates),
            ("test_dates", test_dates),
        )
        if not dates
    ]
    if missing:
        raise ValueError(f"{context} must provide non-empty {', '.join(missing)}.")

    split_dates = {
        "train": set(train_dates),
        "validation": set(validation_dates),
        "test": set(test_dates),
    }
    overlap_messages: list[str] = []
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        overlap = split_dates[left] & split_dates[right]
        if overlap:
            overlap_messages.append(f"{left}/{right}: {sorted(overlap)}")
    if overlap_messages:
        raise ValueError(f"{context} assigns dates to multiple splits: " + "; ".join(overlap_messages))

    if max(train_dates) >= min(validation_dates):
        raise ValueError(f"{context} must have train dates strictly before validation dates.")
    if max(validation_dates) >= min(test_dates):
        raise ValueError(f"{context} must have validation dates strictly before test dates.")


@dataclass(slots=True)
class DatasetSplitConfig:
    train_dates: list[str]
    validation_dates: list[str]
    test_dates: list[str]

    def __post_init__(self) -> None:
        _validate_split_dates(
            "dataset_splits",
            train_dates=self.train_dates,
            validation_dates=self.validation_dates,
            test_dates=self.test_dates,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetSplitConfig":
        return cls(
            train_dates=[str(value) for value in payload["train_dates"]],
            validation_dates=[str(value) for value in payload["validation_dates"]],
            test_dates=[str(value) for value in payload["test_dates"]],
        )


@dataclass(slots=True)
class FoldConfig:
    id: str
    train_dates: list[str]
    validation_dates: list[str]
    test_dates: list[str]

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("folds[].id must be a non-empty string.")
        _validate_split_dates(
            f"Fold {self.id}",
            train_dates=self.train_dates,
            validation_dates=self.validation_dates,
            test_dates=self.test_dates,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FoldConfig":
        return cls(
            id=str(_require_explicit_value(payload.get("id"), "folds[].id")),
            train_dates=[str(value) for value in payload.get("train_dates", [])],
            validation_dates=[str(value) for value in payload.get("validation_dates", [])],
            test_dates=[str(value) for value in payload.get("test_dates", [])],
        )

    @classmethod
    def from_dataset_splits(cls, payload: DatasetSplitConfig) -> "FoldConfig":
        return cls(
            id="single",
            train_dates=payload.train_dates,
            validation_dates=payload.validation_dates,
            test_dates=payload.test_dates,
        )


def _folds_from_payload(payload: dict[str, Any], dataset_splits: DatasetSplitConfig) -> list[FoldConfig]:
    raw_folds = payload.get("folds")
    if raw_folds in (None, []):
        return [FoldConfig.from_dataset_splits(dataset_splits)]
    if not isinstance(raw_folds, list):
        raise ValueError("Invalid experiment config; folds must be a list when provided.")

    folds = [FoldConfig.from_dict(raw_fold) for raw_fold in raw_folds]
    seen_ids: set[str] = set()
    duplicates: list[str] = []
    for fold in folds:
        if fold.id in seen_ids:
            duplicates.append(fold.id)
        seen_ids.add(fold.id)
    if duplicates:
        raise ValueError(f"Invalid experiment config; duplicate fold ids: {sorted(set(duplicates))}")
    return folds


@dataclass(slots=True)
class PreprocessingConfig:
    snapshot_window: int
    labels: LabelConfig
    message: MessageConfig
    temporal_features: TemporalFeaturesConfig
    normalization: NormalizationConfig
    kinematic_tokenization: KinematicTokenizationConfig
    price_kinematic: PriceKinematicConfig
    price_static: PriceStaticConfig
    volume_kinematic: VolumeKinematicConfig
    volume_static: VolumeStaticConfig

    def __post_init__(self) -> None:
        if self.kinematic_tokenization.method != "fast":
            return

        invalid_references: list[str] = []
        if self.price_kinematic.enabled and self.price_kinematic.reference != "tick":
            invalid_references.append("preprocessing.price_kinematic.reference")
        if self.volume_kinematic.enabled and self.volume_kinematic.reference != "tick":
            invalid_references.append("preprocessing.volume_kinematic.reference")
        if invalid_references:
            raise ValueError(
                "Fast kinematic tokenization only supports tick reference; set "
                + ", ".join(invalid_references)
                + " to 'tick'."
            )

    @classmethod
    def from_dict(cls, payload: dict[str, Any], tick_size: float) -> "PreprocessingConfig":
        return cls(
            snapshot_window=int(payload["snapshot_window"]),
            labels=LabelConfig.from_dict(payload["labels"]),
            message=MessageConfig.from_dict(payload["message"], tick_size=tick_size),
            temporal_features=TemporalFeaturesConfig.from_dict(payload["temporal_features"]),
            normalization=NormalizationConfig.from_dict(payload["normalization"]),
            kinematic_tokenization=KinematicTokenizationConfig.from_dict(payload["kinematic_tokenization"]),
            price_kinematic=PriceKinematicConfig.from_dict(payload["price_kinematic"], tick_size=tick_size),
            price_static=PriceStaticConfig.from_dict(payload["price_static"], tick_size=tick_size),
            volume_kinematic=VolumeKinematicConfig.from_dict(payload["volume_kinematic"]),
            volume_static=VolumeStaticConfig.from_dict(payload["volume_static"]),
        )


@dataclass(slots=True)
class ModelConfig:
    d_input: int | None
    d_model: int
    feature_embed_dim: int
    feature_num_frequencies: int
    feature_sigma: float
    num_heads: int
    num_experts: int
    top_k: int
    num_classes: int
    rope_type: str
    rope_base: int
    attention_dropout: float
    moe_dropout: float
    moe_expansion_factor: int
    moe_router_noise: float
    moe_load_balancing_weight: float
    classifier_dropout: float
    max_dt_quantile: float = 95.0
    max_dt: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_dt_quantile <= 100.0:
            raise ValueError("model.max_dt_quantile must be in [0, 100].")
        if self.max_dt is not None and self.max_dt < 0.0:
            raise ValueError("model.max_dt must be >= 0 when resolved.")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelConfig":
        return cls(
            d_input=None if payload["d_input"] is None else int(payload["d_input"]),
            d_model=int(payload["d_model"]),
            feature_embed_dim=int(payload["feature_embed_dim"]),
            feature_num_frequencies=int(payload["feature_num_frequencies"]),
            feature_sigma=float(payload["feature_sigma"]),
            num_heads=int(payload["num_heads"]),
            num_experts=int(payload["num_experts"]),
            top_k=int(payload["top_k"]),
            num_classes=int(payload["num_classes"]),
            rope_type=str(payload["rope_type"]),
            rope_base=int(payload["rope_base"]),
            attention_dropout=float(payload["attention_dropout"]),
            moe_dropout=float(payload["moe_dropout"]),
            moe_expansion_factor=int(payload["moe_expansion_factor"]),
            moe_router_noise=float(payload["moe_router_noise"]),
            moe_load_balancing_weight=float(payload["moe_load_balancing_weight"]),
            classifier_dropout=float(payload["classifier_dropout"]),
            max_dt_quantile=float(_require_explicit_value(payload["max_dt_quantile"], "model.max_dt_quantile")),
        )

    def resolved_d_input(self, inferred_feature_count: int | None = None) -> int:
        if self.d_input is not None:
            return self.d_input
        if inferred_feature_count is None:
            raise ValueError("d_input is missing from config and could not be inferred from the data.")
        return inferred_feature_count


@dataclass(slots=True)
class TrainingConfig:
    device: str
    epochs: int
    batch_size: int
    num_workers: int
    early_stopping_patience: int
    persistent_workers: bool
    learning_rate: float
    weight_decay: float
    focal_gamma: float
    grad_clip_norm: float
    model_dir: str
    use_amp: bool
    class_weights: list[float] | None = None

    def __post_init__(self) -> None:
        if self.num_workers < 0:
            raise ValueError("training.num_workers must be >= 0.")
        if self.early_stopping_patience < 0:
            raise ValueError("training.early_stopping_patience must be >= 0.")
        if self.persistent_workers and self.num_workers == 0:
            raise ValueError("training.persistent_workers requires training.num_workers > 0.")

    @property
    def pin_memory(self) -> bool:
        return self.device.lower() == "cuda"

    @property
    def best_model_path(self) -> Path:
        return Path(self.model_dir) / BEST_MODEL_FILENAME

    def data_loader_kwargs(self) -> dict[str, bool | int]:
        return {
            "num_workers": self.num_workers,
            "persistent_workers": self.persistent_workers,
            "pin_memory": self.pin_memory,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingConfig":
        return cls(
            device=str(payload["device"]).lower(),
            epochs=int(payload["epochs"]),
            batch_size=int(payload["batch_size"]),
            num_workers=int(payload["num_workers"]),
            early_stopping_patience=int(payload["early_stopping_patience"]),
            persistent_workers=bool(payload["persistent_workers"]),
            learning_rate=float(payload["learning_rate"]),
            weight_decay=float(payload["weight_decay"]),
            focal_gamma=float(payload["focal_gamma"]),
            grad_clip_norm=float(payload["grad_clip_norm"]),
            model_dir=str(payload["model_dir"]),
            use_amp=bool(payload["use_amp"]),
        )


@dataclass(slots=True)
class ExperimentConfig:
    path: Path
    seed: int
    data: DataConfig
    dataset_splits: DatasetSplitConfig
    folds: list[FoldConfig]
    preprocessing: PreprocessingConfig
    model: ModelConfig
    training: TrainingConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}

        _validate_required_config(payload, REQUIRED_CONFIG_SCHEMA)
        _validate_allowed_values(payload)

        data_config = DataConfig.from_dict(payload["data"])
        dataset_splits = DatasetSplitConfig.from_dict(payload["dataset_splits"])
        seed = int(_require_explicit_value(payload["seed"], "seed"))
        if seed < 0:
            raise ValueError("Invalid experiment config; seed must be >= 0.")

        return cls(
            path=config_path.resolve(),
            seed=seed,
            data=data_config,
            dataset_splits=dataset_splits,
            folds=_folds_from_payload(payload, dataset_splits),
            preprocessing=PreprocessingConfig.from_dict(payload["preprocessing"], tick_size=data_config.tick_size),
            model=ModelConfig.from_dict(payload["model"]),
            training=TrainingConfig.from_dict(payload["training"]),
        )


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    default_path = Path(__file__).resolve().parent.parent / "configs" / "pipeline_config.yaml"
    return ExperimentConfig.from_yaml(path or default_path)
