from __future__ import annotations

from dataclasses import dataclass, field
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
        "save_processed_dataframes": None,
        "labels": {
            "strategy": None,
            "smoothing": {
                "method": None,
                "threshold": None,
                "fit_scope": None,
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
            "derivative_scaling_method": None,
        },
        "kinematic_tokenization": {
            "method": None,
            "chunk_size": None,
            "n_df_candidates": None,
            "orderbook_top_k_levels": None,
        },
        "sample_clock": {
            "mode": None,
            "volume_step_shares": None,
            "volume_source": None,
            "trade_type_values": None,
        },
        "microprice": {
            "enabled": None,
            "levels": None,
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
        "num_layers": None,
        "latent_spatial_embed_dim": None,
        "use_moe": None,
        "num_heads": None,
        "max_dt_quantile": None,
        "max_dt": None,
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
        "classifier_pooling": {
            "methods": None,
            "last_k": None,
        },
    },
    "training": {
        "device": None,
        "epochs": None,
        "batch_size": None,
        "eval_batch_size": None,
        "num_workers": None,
        "early_stopping_patience": None,
        "early_stopping_warmup": None,
        "early_stopping_min_delta": None,
        "monitor": None,
        "monitor_mode": None,
        "top_k_checkpoints": None,
        "monitor_params": {
            "base_metric": None,
            "lambda_ece": None,
            "lambda_rate": None,
        },
        "persistent_workers": None,
        "optimizer": None,
        "learning_rate": None,
        "weight_decay": None,
        "focal_gamma": None,
        "class_weight_beta": None,
        "class_weight_min": None,
        "class_weight_max": None,
        "grad_clip_norm": None,
        "model_dir": None,
        "use_amp": None,
        "deterministic_torch": None,
        "temperature_scaling": {
            "enabled": None,
            "class_bias_calibration": None,
        },
        "directional_thresholds": {
            "enabled": None,
            "method": None,
            "score": None,
            "min": None,
            "max": None,
            "step": None,
            "delta": None,
            "up_precision_floor": None,
            "down_precision_floor": None,
            "up_quantile": None,
            "down_quantile": None,
        },
        "sampling": {
            "neutral_to_directional_ratio": None,
        },
    },
}

OPTIONAL_TOP_LEVEL_KEYS = {"experiment", "folds", "run_metadata"}
OPTIONAL_CONFIG_KEYS = {
    "preprocessing.labels.smoothing.adaptive_threshold",
    "preprocessing.labels.smoothing.fit_scope",
    "preprocessing.price_static.tau_clip",
    "preprocessing.price_static.tau_max",
    "model.max_dt",
    "model.num_layers",
    "model.latent_spatial_embed_dim",
    "model.use_moe",
    "model.classifier_pooling",
    "training.monitor_params",
    "training.monitor_params.base_metric",
    "training.top_k_checkpoints",
    "preprocessing.save_processed_dataframes",
    "preprocessing.kinematic_tokenization.orderbook_top_k_levels",
    "preprocessing.sample_clock",
    "preprocessing.sample_clock.mode",
    "preprocessing.sample_clock.volume_step_shares",
    "preprocessing.sample_clock.volume_source",
    "preprocessing.sample_clock.trade_type_values",
    "preprocessing.microprice",
    "preprocessing.microprice.enabled",
    "preprocessing.microprice.levels",
    "training.early_stopping_warmup",
    "training.early_stopping_min_delta",
    "training.optimizer",
    "training.temperature_scaling",
    "training.temperature_scaling.enabled",
    "training.temperature_scaling.class_bias_calibration",
    "training.directional_thresholds",
    "training.directional_thresholds.enabled",
    "training.directional_thresholds.method",
    "training.directional_thresholds.score",
    "training.directional_thresholds.min",
    "training.directional_thresholds.max",
    "training.directional_thresholds.step",
    "training.directional_thresholds.delta",
    "training.directional_thresholds.up_precision_floor",
    "training.directional_thresholds.down_precision_floor",
    "training.directional_thresholds.up_quantile",
    "training.directional_thresholds.down_quantile",
    "training.sampling",
}


ALLOWED_CONFIG_VALUES: dict[str, set[Any]] = {
    "preprocessing.labels.strategy": {"smoothing", "triple_barrier"},
    "preprocessing.labels.smoothing.method": {"A", "B", "C"},
    "preprocessing.normalization.scope": {"train_only"},
    "preprocessing.normalization.derivative_scaling_method": {"zscore", "robust_mad", "quantile_scaling"},
    "preprocessing.kinematic_tokenization.method": {"basis", "fast"},
    "preprocessing.price_kinematic.reference": {"tick", "time"},
    "preprocessing.volume_kinematic.reference": {"tick", "time"},
    "model.rope_type": {"rope", "crope", "hybrid_crope", "hybrid-crope", "hybrid"},
    "training.monitor": {"val_loss", "val_macro_f1", "val_directional_macro_f1", "tailored_score"},
    "training.monitor_mode": {"min", "max"},
}


def _ensure_list(value: Any) -> list[str] | None:
    """
    Force the input value (sourced from the YAML config file) 
    to be converted as a list. Treats None, "auto", "none", "" as None.
    """
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "auto", "none"}:
            return None
        return [value]
    return list(value)


def _validate_required_config(payload: Any, schema: dict[str, Any]) -> None:
    """
    Validate the YAML config file structure. Raises ValueError if the configuration 
    YAML implies a missing/unexpected/invalid argument. 

    Args:
      payload (Any) : the loaded yaml config, typically a Python dict.
      schema  (dict) : The required config schema to compare the payload with.
    """
    missing: list[str] = []
    invalid_mappings: list[str] = []
    unexpected: list[str] = []

    def walk(node: Any, subtree: dict[str, Any], prefix: str = "") -> None:
        """Recursive walk into a dict schema. The function compares each node 
        against a schema subtree.
        
        Args: 
          node (Any) : typically a dict (the node) we currently explore.
          subtree (dict) : the subset of parameters to explore/check.
          prefix (str) : the prefix used to navigate inside the tree. 
        """
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
    """Return a dotted path from a nested mapping. Typically used for logging
    or to generate human-readable outputs. 

    Args:
        payload (dict) : Nested configuration mapping to traverse.
        dotted_path (str) : Dot-separated path, for example "training.device".

    Returns:
        The value stored at dotted_path.

    Raises:
        KeyError: If any path component is missing.
    """
    current: Any = payload
    for part in dotted_path.split("."):
        current = current[part]
    return current


def _require_explicit_value(value: Any, dotted_path: str) -> Any:
    """Require a configuration value to be explicitly set.

    Args:
        value: Value read from the configuration payload.
        dotted_path: Human-readable dotted path used in the error message.

    Returns:
        The original value when it is not None.

    Raises:
        ValueError: If value is None.
    """
    if value is None:
        raise ValueError(f"Invalid experiment config; {dotted_path} must be set explicitly.")
    return value


def _optional_float(value: Any) -> float | None:
    """Convert an optional configuration value to float.

    Args:
        value: Raw value read from the configuration payload.

    Returns:
        None for null-like inputs, otherwise value converted to float.

    Raises:
        TypeError: If value cannot be converted to float.
        ValueError: If value cannot be parsed as a valid float.
    """
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return float(value)


TRAIN_FITTED_SMOOTHING_THRESHOLDS = {"mean_spread"}
SPLIT_FITTED_SMOOTHING_THRESHOLDS = {"mean_pct", "mean_pct_2"}
FITTED_SMOOTHING_THRESHOLDS = TRAIN_FITTED_SMOOTHING_THRESHOLDS | SPLIT_FITTED_SMOOTHING_THRESHOLDS
SMOOTHING_THRESHOLD_FIT_SCOPES = {"per_split", "train"}


def _optional_smoothing_threshold(value: Any) -> float | str | None:
    """Parse a smoothing threshold as null, float, or train-fitted mode."""
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "none", "null"}:
            return None
        if lowered in FITTED_SMOOTHING_THRESHOLDS:
            return lowered
        try:
            return float(value)
        except ValueError as exc:
            allowed = sorted(FITTED_SMOOTHING_THRESHOLDS)
            raise ValueError(
                "preprocessing.labels.smoothing.threshold must be numeric, null, "
                f"or one of {allowed}."
            ) from exc
    return float(value)


def _optional_int(value: Any, dotted_path: str) -> int | None:
    """Convert an optional config value to int without accepting floats."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Invalid experiment config; {dotted_path} must be an integer or null.")
    return int(value)


def _required_int(value: Any, dotted_path: str) -> int:
    """Convert a required config value to int without accepting floats."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Invalid experiment config; {dotted_path} must be an integer.")
    return int(value)


def _optional_bool(value: Any, dotted_path: str, *, default: bool) -> bool:
    """Convert an optional config value to bool with a default."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "none", "null"}:
            return default
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    raise ValueError(f"Invalid experiment config; {dotted_path} must be a boolean.")


def _is_valid_training_device(value: Any) -> bool:
    """Check whether a training device string is accepted.

    Args:
        value: Candidate training device value.

    Returns:
        True for "cpu", "cuda", or "cuda:<non-negative index>";
        otherwise ``False``.
    """
    if not isinstance(value, str):
        return False
    device = value.strip().lower()
    if device in {"cpu", "cuda"}:
        return True
    if not device.startswith("cuda:"):
        return False
    index = device.removeprefix("cuda:")
    return index.isdigit() and int(index) >= 0


def _validate_allowed_values(payload: dict[str, Any]) -> None:
    """Validate enumerated configuration values.

    Args:
        payload: Loaded YAML configuration payload.

    Raises:
        ValueError: If any configured value is outside its allowed set.
    """
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

    training_device = _get_nested(payload, "training.device")
    if not _is_valid_training_device(training_device):
        invalid.append(f"training.device={training_device!r}; allowed values are ['cpu', 'cuda', 'cuda:<index>']")

    if invalid:
        raise ValueError("Invalid experiment config; invalid values: " + "; ".join(invalid))


@dataclass(slots=True)
class ExperimentMetadataConfig:
    name: str

    def __post_init__(self) -> None:
        """Check experiment metadata used for readable run names."""
        self.name = self.name.strip()
        if not self.name:
            raise ValueError("experiment.name must be a non-empty string.")

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None, *, default_name: str) -> "ExperimentMetadataConfig":
        """Build experiment metadata with a backwards-compatible default."""
        if payload is None:
            return cls(name=default_name)
        if not isinstance(payload, dict):
            raise ValueError("Invalid experiment config; experiment must be a mapping.")
        unexpected = sorted(set(payload) - {"name"})
        if unexpected:
            raise ValueError(f"Invalid experiment config; unexpected key(s) in experiment: {unexpected}")
        return cls(name=str(_require_explicit_value(payload.get("name"), "experiment.name")))


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
        """Build a data configuration from a YAML subsection.

        Args:
            payload: data section of the loaded configuration.

        Returns:
            A populated DataConfig instance.
        """
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
    n_df_candidates: int
    orderbook_top_k_levels: int | None = None

    def __post_init__(self) -> None:
        """Check kinematic tokenization settings."""
        self.method = self.method.lower()
        if self.method not in {"basis", "fast"}:
            raise ValueError("preprocessing.kinematic_tokenization.method must be 'basis' or 'fast'.")
        if self.chunk_size <= 0:
            raise ValueError("preprocessing.kinematic_tokenization.chunk_size must be > 0.")
        if self.n_df_candidates <= 0:
            raise ValueError("preprocessing.kinematic_tokenization.n_df_candidates must be > 0.")
        if self.orderbook_top_k_levels is not None and self.orderbook_top_k_levels < 1:
            raise ValueError("preprocessing.kinematic_tokenization.orderbook_top_k_levels must be >= 1 or null.")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "KinematicTokenizationConfig":
        """Build kinematic tokenization settings from a YAML subsection."""
        return cls(
            method=str(_require_explicit_value(payload["method"], "preprocessing.kinematic_tokenization.method")),
            chunk_size=int(
                _require_explicit_value(payload["chunk_size"], "preprocessing.kinematic_tokenization.chunk_size")
            ),
            n_df_candidates=int(
                _require_explicit_value(
                    payload["n_df_candidates"],
                    "preprocessing.kinematic_tokenization.n_df_candidates",
                )
            ),
            orderbook_top_k_levels=_optional_int(
                payload.get("orderbook_top_k_levels"),
                "preprocessing.kinematic_tokenization.orderbook_top_k_levels",
            ),
        )


@dataclass(slots=True)
class FastKinematicConfig:
    n_basis: int
    df: float
    eval_at: float
    selected_smoothing_lambda: float | None = None

    def __post_init__(self) -> None:
        """Check fast B-spline tokenization parameters."""
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
        """Build fast kinematic settings from a YAML subsection."""
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
        """Build basis kinematic settings from a YAML subsection."""
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
        """Build price kinematic settings from a YAML subsection."""
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
        """Build static price scaling settings from a YAML subsection."""
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
        """Build volume kinematic settings from a YAML subsection."""
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
        """Check static volume scaling parameters."""
        if not 0.0 <= self.quantile <= 100.0:
            raise ValueError("preprocessing.volume_static.quantile must be in [0, 100].")
        if not 0.0 < self.target < 1.0:
            raise ValueError("preprocessing.volume_static.target must be in (0, 1).")
        if self.k is not None and self.k <= 0:
            raise ValueError("preprocessing.volume_static.k must be > 0 when fitted.")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VolumeStaticConfig":
        """Build static volume scaling settings from a YAML subsection."""
        return cls(
            enabled=bool(payload["enabled"]),
            columns=_ensure_list(payload["columns"]),
            quantile=float(payload["quantile"]),
            target=float(payload["target"]),
        )


@dataclass(slots=True)
class MicropriceConfig:
    enabled: bool = False
    levels: int = 1

    def __post_init__(self) -> None:
        """Check optional microprice feature settings."""
        if self.levels < 1:
            raise ValueError("preprocessing.microprice.levels must be >= 1.")

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "MicropriceConfig":
        """Build optional microprice settings from a YAML subsection."""
        if payload is None:
            return cls()
        unexpected = sorted(set(payload) - {"enabled", "levels"})
        if unexpected:
            raise ValueError(f"Invalid experiment config; unexpected key(s) in preprocessing.microprice: {unexpected}")
        return cls(
            enabled=bool(payload.get("enabled", False)),
            levels=_required_int(payload.get("levels", 1), "preprocessing.microprice.levels"),
        )


@dataclass(slots=True)
class SampleClockConfig:
    mode: str = "event"
    volume_step_shares: float | None = None
    volume_source: str = "traded"
    trade_type_values: list[int] = field(default_factory=lambda: [4, 5])

    def __post_init__(self) -> None:
        """Check optional event/volume sampling clock settings."""
        self.mode = self.mode.lower()
        self.volume_source = self.volume_source.lower()
        if self.mode not in {"event", "volume"}:
            raise ValueError("preprocessing.sample_clock.mode must be 'event' or 'volume'.")
        if self.volume_source not in {"traded", "message_size"}:
            raise ValueError("preprocessing.sample_clock.volume_source must be 'traded' or 'message_size'.")
        if self.volume_step_shares is not None and self.volume_step_shares <= 0:
            raise ValueError("preprocessing.sample_clock.volume_step_shares must be > 0 when set.")
        if not self.trade_type_values:
            raise ValueError("preprocessing.sample_clock.trade_type_values must contain at least one integer.")
        if self.mode == "volume" and self.volume_step_shares is None:
            raise ValueError("preprocessing.sample_clock.volume_step_shares must be set when mode is 'volume'.")

    @property
    def enabled(self) -> bool:
        """Return whether volume-clock sampling is active."""
        return self.mode == "volume"

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "SampleClockConfig":
        """Build optional sample-clock settings from a YAML subsection."""
        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise ValueError("Invalid experiment config; preprocessing.sample_clock must be a mapping.")
        unexpected = sorted(set(payload) - {"mode", "volume_step_shares", "volume_source", "trade_type_values"})
        if unexpected:
            raise ValueError(f"Invalid experiment config; unexpected key(s) in preprocessing.sample_clock: {unexpected}")

        raw_trade_types = payload.get("trade_type_values", [4, 5])
        if raw_trade_types is None:
            trade_type_values: list[int] = []
        elif isinstance(raw_trade_types, (str, bytes)):
            raise ValueError("preprocessing.sample_clock.trade_type_values must be a list of integers.")
        else:
            trade_type_values = []
            for value in raw_trade_types:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError("preprocessing.sample_clock.trade_type_values must be a list of integers.")
                trade_type_values.append(int(value))

        return cls(
            mode=str(payload.get("mode", "event")),
            volume_step_shares=_optional_float(payload.get("volume_step_shares")),
            volume_source=str(payload.get("volume_source", "traded")),
            trade_type_values=trade_type_values,
        )


@dataclass(slots=True)
class AdaptiveThresholdConfig:
    enabled: bool
    exit_spread_window: int
    volatility_window: int
    round_trip_fees_bps: float
    volatility_lambda: float

    def __post_init__(self) -> None:
        """Check adaptive label-threshold parameters."""
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
        """Build adaptive threshold settings from a YAML subsection."""
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
    threshold: float | str | None
    k: int
    h: int
    bid_column: str
    ask_column: str
    fit_scope: str | None = None
    adaptive_threshold: AdaptiveThresholdConfig | None = None

    def __post_init__(self) -> None:
        """Check smoothing-label horizon parameters."""
        if self.k < 0:
            raise ValueError("preprocessing.labels.smoothing.k must be >= 0.")
        if self.h <= 0:
            raise ValueError("preprocessing.labels.smoothing.h must be > 0.")
        if self.method.upper() == "C" and self.k >= self.h:
            raise ValueError("preprocessing.labels.smoothing method C requires k < h.")
        if self.fit_scope is not None:
            self.fit_scope = self.fit_scope.lower()
            if self.fit_scope not in SMOOTHING_THRESHOLD_FIT_SCOPES:
                allowed = sorted(SMOOTHING_THRESHOLD_FIT_SCOPES)
                raise ValueError(f"preprocessing.labels.smoothing.fit_scope must be one of {allowed}.")
        if isinstance(self.threshold, str):
            self.threshold = self.threshold.lower()
            if self.threshold not in FITTED_SMOOTHING_THRESHOLDS:
                allowed = sorted(FITTED_SMOOTHING_THRESHOLDS)
                raise ValueError(
                    "preprocessing.labels.smoothing.threshold must be numeric, null, "
                    f"or one of {allowed}."
                )
            if self.adaptive_threshold is not None and self.adaptive_threshold.enabled:
                raise ValueError(
                    "preprocessing.labels.smoothing.threshold fitted modes cannot be combined "
                    "with adaptive_threshold.enabled=true."
                )

    def resolved_fit_scope(self) -> str | None:
        """Return where a fitted threshold should be estimated."""
        if not isinstance(self.threshold, str) or self.threshold not in FITTED_SMOOTHING_THRESHOLDS:
            return None
        if self.fit_scope is not None:
            return self.fit_scope
        if self.threshold in TRAIN_FITTED_SMOOTHING_THRESHOLDS:
            return "train"
        return "per_split"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SmoothingLabelConfig":
        """Build smoothing-label settings from a YAML subsection."""
        adaptive_payload = payload.get("adaptive_threshold")
        return cls(
            method=str(payload["method"]),
            threshold=_optional_smoothing_threshold(payload["threshold"]),
            fit_scope=None if payload.get("fit_scope") is None else str(payload["fit_scope"]),
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
        """Build triple-barrier label settings from a YAML subsection."""
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

    def __post_init__(self) -> None:
        """Check that the active labeling strategy has a usable threshold rule."""
        if self.strategy.lower() != "smoothing":
            return
        adaptive_enabled = (
            self.smoothing.adaptive_threshold is not None
            and self.smoothing.adaptive_threshold.enabled
        )
        if self.smoothing.threshold is None and not adaptive_enabled:
            raise ValueError(
                "preprocessing.labels.smoothing.threshold cannot be null when "
                "adaptive_threshold.enabled is false; set a numeric/train-fitted threshold "
                "or enable adaptive_threshold."
            )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LabelConfig":
        """Build label settings from a YAML subsection."""
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
        """Build message preprocessing settings from a YAML subsection."""
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
        """Build temporal feature settings from a YAML subsection."""
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
    derivative_scaling_method: str = "zscore"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NormalizationConfig":
        """Build normalization settings from a YAML subsection."""
        return cls(
            derivatives_stats_dir=str(payload["derivatives_stats_dir"]),
            scope=str(payload["scope"]),
            derivative_scaling_method=str(payload["derivative_scaling_method"]),
        )


def _validate_split_dates(
    context: str,
    *,
    train_dates: list[str],
    validation_dates: list[str],
    test_dates: list[str],
) -> None:
    """Check split completeness, disjointness, and chronological order.

    Args:
        context: Name included in validation error messages.
        train_dates: Dates assigned to the training split.
        validation_dates: Dates assigned to the validation split.
        test_dates: Dates assigned to the optional test split.

    Raises:
        ValueError: If train/validation is empty, if a date appears in multiple
            splits, or if configured splits are not strictly ordered.
    """
    missing = [
        split_name
        for split_name, dates in (
            ("train_dates", train_dates),
            ("validation_dates", validation_dates),
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
    if test_dates and max(validation_dates) >= min(test_dates):
        raise ValueError(f"{context} must have validation dates strictly before test dates.")


@dataclass(slots=True)
class DatasetSplitConfig:
    train_dates: list[str]
    validation_dates: list[str]
    test_dates: list[str]

    def __post_init__(self) -> None:
        """Check the configured dataset split dates.

        Raises:
            ValueError: If train/validation dates are empty, overlapping, or out
                of order.
        """
        _validate_split_dates(
            "dataset_splits",
            train_dates=self.train_dates,
            validation_dates=self.validation_dates,
            test_dates=self.test_dates,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetSplitConfig":
        """Build dataset split settings from a YAML subsection."""
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
        """Check fold identifier and split dates.

        Raises:
            ValueError: If the fold id is blank, or if train/validation dates
                are empty, overlapping, or out of order.
        """
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
        """Build a fold configuration from a YAML mapping."""
        return cls(
            id=str(_require_explicit_value(payload.get("id"), "folds[].id")),
            train_dates=[str(value) for value in payload.get("train_dates", [])],
            validation_dates=[str(value) for value in payload.get("validation_dates", [])],
            test_dates=[str(value) for value in payload.get("test_dates", [])],
        )

    @property
    def has_test_dates(self) -> bool:
        """Return whether this fold declares an in-fold test split."""
        return bool(self.test_dates)

    @classmethod
    def from_dataset_splits(cls, payload: DatasetSplitConfig) -> "FoldConfig":
        """Create the default single fold from top-level dataset splits."""
        return cls(
            id="single",
            train_dates=payload.train_dates,
            validation_dates=payload.validation_dates,
            test_dates=payload.test_dates,
        )


def _folds_from_payload(payload: dict[str, Any], dataset_splits: DatasetSplitConfig) -> list[FoldConfig]:
    """Resolve fold settings from the loaded configuration payload.

    Args:
        payload: Loaded YAML configuration payload.
        dataset_splits: Fallback split configuration used when no folds are provided.

    Returns:
        A non-empty list of fold configurations.

    Raises:
        ValueError: If "folds" is not a list, or if fold ids are duplicated.
    """
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
    microprice: MicropriceConfig = field(default_factory=MicropriceConfig)
    sample_clock: SampleClockConfig = field(default_factory=SampleClockConfig)
    save_processed_dataframes: bool = False

    def __post_init__(self) -> None:
        """Check cross-field preprocessing constraints."""
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
        """Build preprocessing settings from a YAML subsection."""
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
            microprice=MicropriceConfig.from_dict(payload.get("microprice")),
            sample_clock=SampleClockConfig.from_dict(payload.get("sample_clock")),
            save_processed_dataframes=bool(payload.get("save_processed_dataframes", False)),
        )


CLASSIFIER_POOLING_METHODS = {"last", "mean", "max"}


@dataclass(slots=True)
class ClassifierPoolingConfig:
    methods: tuple[str, ...] = ("last",)
    last_k: int = 1

    def __post_init__(self) -> None:
        """Validate the classifier token pooling strategy."""
        normalized = tuple(str(method).strip().lower() for method in self.methods)
        if not normalized:
            raise ValueError("model.classifier_pooling.methods must be a non-empty list.")
        invalid = sorted(set(normalized) - CLASSIFIER_POOLING_METHODS)
        if invalid:
            raise ValueError(
                "model.classifier_pooling.methods must contain only last, mean, max; "
                f"got {invalid}."
            )
        if len(set(normalized)) != len(normalized):
            raise ValueError("model.classifier_pooling.methods must not contain duplicates.")
        if self.last_k < 1:
            raise ValueError("model.classifier_pooling.last_k must be >= 1.")
        self.methods = normalized

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ClassifierPoolingConfig":
        """Build classifier pooling settings from YAML."""
        if payload is None:
            return cls()
        unexpected = sorted(set(payload) - {"methods", "last_k"})
        if unexpected:
            raise ValueError(
                "Invalid experiment config; unexpected key(s) in model.classifier_pooling: "
                f"{unexpected}"
            )
        methods = payload.get("methods", ["last"])
        if isinstance(methods, str) or not isinstance(methods, (list, tuple)):
            raise ValueError("model.classifier_pooling.methods must be a list.")
        return cls(
            methods=tuple(methods),
            last_k=_required_int(payload.get("last_k", 1), "model.classifier_pooling.last_k"),
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
    classifier_pooling: ClassifierPoolingConfig = field(default_factory=ClassifierPoolingConfig)
    num_layers: int = 1
    latent_spatial_embed_dim: int | None = None
    use_moe: bool = True
    max_dt_quantile: float = 95.0
    max_dt: float | None = None

    def __post_init__(self) -> None:
        """Check model-derived scalar settings.

        Raises:
            ValueError: If "max_dt_quantile" or resolved "max_dt" is out of range.
        """
        if self.num_layers < 1:
            raise ValueError("model.num_layers must be >= 1.")
        if self.latent_spatial_embed_dim is not None and self.latent_spatial_embed_dim < 0:
            raise ValueError("model.latent_spatial_embed_dim must be >= 0 or null.")
        if self.num_layers > 1:
            if self.latent_spatial_embed_dim is None or self.latent_spatial_embed_dim <= 0:
                raise ValueError("model.latent_spatial_embed_dim must be > 0 when model.num_layers > 1.")
            if self.d_model % self.latent_spatial_embed_dim != 0:
                raise ValueError("model.d_model must be divisible by model.latent_spatial_embed_dim.")
            if self.latent_spatial_embed_dim % self.num_heads != 0:
                raise ValueError("model.latent_spatial_embed_dim must be divisible by model.num_heads.")
        if not 0.0 <= self.max_dt_quantile <= 100.0:
            raise ValueError("model.max_dt_quantile must be in [0, 100].")
        if self.max_dt is not None and self.max_dt < 0.0:
            raise ValueError("model.max_dt must be >= 0 when resolved.")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelConfig":
        """Build model settings from a YAML subsection."""
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
            classifier_pooling=ClassifierPoolingConfig.from_dict(payload.get("classifier_pooling")),
            num_layers=_required_int(payload.get("num_layers", 1), "model.num_layers"),
            latent_spatial_embed_dim=_optional_int(
                payload.get("latent_spatial_embed_dim"),
                "model.latent_spatial_embed_dim",
            ),
            use_moe=_optional_bool(payload.get("use_moe"), "model.use_moe", default=True),
            max_dt_quantile=float(_require_explicit_value(payload["max_dt_quantile"], "model.max_dt_quantile")),
            max_dt=_optional_float(payload.get("max_dt")),
        )

    def resolved_latent_spatial_embed_dim(self) -> int:
        """Return the latent chunk size used by post-stem spatial attention."""
        if self.latent_spatial_embed_dim is None or self.latent_spatial_embed_dim <= 0:
            raise ValueError("model.latent_spatial_embed_dim must be > 0 when latent layers are enabled.")
        return self.latent_spatial_embed_dim

    def resolved_d_input(self, inferred_feature_count: int | None = None) -> int:
        """Return the configured or inferred model input width.

        Args:
            inferred_feature_count: Feature count inferred from prepared data when
                "d_input" is not explicitly configured.

        Returns:
            The input feature dimension to use when building the model.

        Raises:
            ValueError: If ``d_input`` is unset and no inferred feature count is provided.
        """
        if self.d_input is not None:
            return self.d_input
        if inferred_feature_count is None:
            raise ValueError("d_input is missing from config and could not be inferred from the data.")
        return inferred_feature_count


@dataclass(slots=True)
class TrainingSamplingConfig:
    neutral_to_directional_ratio: float | None

    def __post_init__(self) -> None:
        """Check optional train-time sampling settings."""
        if self.neutral_to_directional_ratio is not None and self.neutral_to_directional_ratio <= 0.0:
            raise ValueError("training.sampling.neutral_to_directional_ratio must be > 0 or null.")

    @property
    def enabled(self) -> bool:
        """Whether neutral downsampling should be applied to train batches."""
        return self.neutral_to_directional_ratio is not None

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "TrainingSamplingConfig":
        """Build training sampling settings from a YAML subsection."""
        if payload is None:
            return cls(neutral_to_directional_ratio=None)
        return cls(
            neutral_to_directional_ratio=_optional_float(payload.get("neutral_to_directional_ratio")),
        )


@dataclass(slots=True)
class TrainingMonitorParamsConfig:
    base_metric: str = "val_directional_macro_f1"
    lambda_ece: float | None = None
    lambda_rate: float | None = None

    def __post_init__(self) -> None:
        """Check optional custom monitor parameters."""
        if self.base_metric is None:
            self.base_metric = "val_directional_macro_f1"
        self.base_metric = str(self.base_metric).strip().lower()
        if self.base_metric not in {"val_macro_f1", "val_directional_macro_f1"}:
            raise ValueError(
                "training.monitor_params.base_metric must be 'val_macro_f1' "
                "or 'val_directional_macro_f1'."
            )
        if self.lambda_ece is not None and self.lambda_ece < 0.0:
            raise ValueError("training.monitor_params.lambda_ece must be >= 0.")
        if self.lambda_rate is not None and self.lambda_rate < 0.0:
            raise ValueError("training.monitor_params.lambda_rate must be >= 0.")

    @property
    def complete(self) -> bool:
        """Whether all tailored_score parameters are present."""
        return self.lambda_ece is not None and self.lambda_rate is not None

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "TrainingMonitorParamsConfig":
        """Build custom monitor parameters from a YAML subsection."""
        if payload is None:
            return cls()
        return cls(
            base_metric=payload.get("base_metric", "val_directional_macro_f1"),
            lambda_ece=_optional_float(payload.get("lambda_ece")),
            lambda_rate=_optional_float(payload.get("lambda_rate")),
        )


@dataclass(slots=True)
class TrainingDirectionalThresholdConfig:
    enabled: bool = False
    method: str = "joint_up_down"
    score: str = "directional_macro_f1"
    min_threshold: float = 0.05
    max_threshold: float = 0.95
    step: float = 0.05
    delta: float = 0.0
    up_precision_floor: float | None = None
    down_precision_floor: float | None = None
    up_quantile: float | None = None
    down_quantile: float | None = None

    def __post_init__(self) -> None:
        """Check optional post-training directional threshold settings."""
        self.method = self.method.lower()
        if self.method not in {"joint_up_down", "precision_floor", "top_x_quantile"}:
            raise ValueError(
                "training.directional_thresholds.method must be 'joint_up_down', "
                "'precision_floor', or 'top_x_quantile'."
            )
        self.score = self.score.lower()
        if self.score not in {"macro_f1", "directional_macro_f1", "tailored_score"}:
            raise ValueError(
                "training.directional_thresholds.score must be 'macro_f1', "
                "'directional_macro_f1', or 'tailored_score'."
            )
        if not 0.0 <= self.min_threshold <= 1.0:
            raise ValueError("training.directional_thresholds.min must be in [0, 1].")
        if not 0.0 <= self.max_threshold <= 1.0:
            raise ValueError("training.directional_thresholds.max must be in [0, 1].")
        if self.min_threshold > self.max_threshold:
            raise ValueError("training.directional_thresholds.min must be <= max.")
        if self.step <= 0.0:
            raise ValueError("training.directional_thresholds.step must be > 0.")
        if self.delta < 0.0:
            raise ValueError("training.directional_thresholds.delta must be >= 0.")
        for name, value in (
            ("up_precision_floor", self.up_precision_floor),
            ("down_precision_floor", self.down_precision_floor),
        ):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"training.directional_thresholds.{name} must be in [0, 1] or null.")
        for name, value in (
            ("up_quantile", self.up_quantile),
            ("down_quantile", self.down_quantile),
        ):
            if value is not None and not 0.0 < value <= 1.0:
                raise ValueError(f"training.directional_thresholds.{name} must be in (0, 1] or null.")
        if self.method in {"joint_up_down", "top_x_quantile"}:
            if self.up_precision_floor is not None or self.down_precision_floor is not None:
                raise ValueError(
                    "training.directional_thresholds up/down precision floors must be null "
                    "unless method is precision_floor."
                )
        if self.method == "precision_floor":
            if self.up_precision_floor is None or self.down_precision_floor is None:
                raise ValueError(
                    "training.directional_thresholds up_precision_floor and down_precision_floor "
                    "must be set when method is precision_floor."
                )
        if self.method == "top_x_quantile":
            if self.up_quantile is None or self.down_quantile is None:
                raise ValueError(
                    "training.directional_thresholds up_quantile and down_quantile "
                    "must be set when method is top_x_quantile."
                )

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "TrainingDirectionalThresholdConfig":
        """Build threshold settings from a YAML subsection."""
        if payload is None:
            return cls()
        defaults = cls()
        return cls(
            enabled=bool(payload.get("enabled", False)),
            method=str(payload.get("method", defaults.method)),
            score=str(payload.get("score", defaults.score)),
            min_threshold=float(payload.get("min", defaults.min_threshold)),
            max_threshold=float(payload.get("max", defaults.max_threshold)),
            step=float(payload.get("step", defaults.step)),
            delta=float(payload.get("delta", defaults.delta)),
            up_precision_floor=_optional_float(payload.get("up_precision_floor")),
            down_precision_floor=_optional_float(payload.get("down_precision_floor")),
            up_quantile=_optional_float(payload.get("up_quantile")),
            down_quantile=_optional_float(payload.get("down_quantile")),
        )


@dataclass(slots=True)
class TrainingTemperatureScalingConfig:
    enabled: bool = False
    class_bias_calibration: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "TrainingTemperatureScalingConfig":
        """Build optional temperature scaling settings from YAML."""
        if payload is None:
            return cls()
        unexpected = sorted(set(payload) - {"enabled", "class_bias_calibration"})
        if unexpected:
            raise ValueError(
                "Invalid experiment config; unexpected key(s) in training.temperature_scaling: "
                f"{unexpected}"
            )
        return cls(
            enabled=bool(payload.get("enabled", False)),
            class_bias_calibration=bool(payload.get("class_bias_calibration", False)),
        )


@dataclass(slots=True)
class TrainingConfig:
    device: str
    epochs: int
    batch_size: int
    eval_batch_size: int
    num_workers: int
    early_stopping_patience: int
    early_stopping_warmup: int
    early_stopping_min_delta: float
    monitor: str
    monitor_mode: str
    monitor_params: TrainingMonitorParamsConfig
    top_k_checkpoints: int
    persistent_workers: bool
    optimizer: str
    learning_rate: float
    weight_decay: float
    focal_gamma: float
    class_weight_beta: float
    class_weight_min: float
    class_weight_max: float
    grad_clip_norm: float
    model_dir: str
    use_amp: bool
    deterministic_torch: bool
    temperature_scaling: TrainingTemperatureScalingConfig
    directional_thresholds: TrainingDirectionalThresholdConfig
    sampling: TrainingSamplingConfig
    class_weights: list[float] | None = None

    def __post_init__(self) -> None:
        """Check training worker settings."""
        if self.batch_size <= 0:
            raise ValueError("training.batch_size must be > 0.")
        if self.eval_batch_size <= 0:
            raise ValueError("training.eval_batch_size must be > 0.")
        if self.num_workers < 0:
            raise ValueError("training.num_workers must be >= 0.")
        if self.early_stopping_patience < 0:
            raise ValueError("training.early_stopping_patience must be >= 0.")
        if self.early_stopping_warmup < 0:
            raise ValueError("training.early_stopping_warmup must be >= 0.")
        if self.early_stopping_min_delta < 0.0:
            raise ValueError("training.early_stopping_min_delta must be >= 0.")
        self.monitor = self.monitor.lower()
        self.monitor_mode = self.monitor_mode.lower()
        self.optimizer = self.optimizer.lower()
        if self.optimizer not in {"adam", "adamw"}:
            raise ValueError("training.optimizer must be 'adam' or 'adamw'.")
        if self.monitor not in {"val_loss", "val_macro_f1", "val_directional_macro_f1", "tailored_score"}:
            raise ValueError(
                "training.monitor must be one of val_loss, val_macro_f1, "
                "val_directional_macro_f1, tailored_score."
            )
        if self.monitor_mode not in {"min", "max"}:
            raise ValueError("training.monitor_mode must be 'min' or 'max'.")
        if self.top_k_checkpoints <= 0:
            raise ValueError("training.top_k_checkpoints must be > 0.")
        if self.monitor == "tailored_score":
            if self.monitor_mode != "max":
                raise ValueError("training.monitor_mode must be 'max' when training.monitor is tailored_score.")
            if not self.monitor_params.complete:
                raise ValueError(
                    "training.monitor_params.lambda_ece and training.monitor_params.lambda_rate "
                    "must be set for tailored_score."
                )
        if (
            self.directional_thresholds.enabled
            and self.directional_thresholds.score == "tailored_score"
            and not self.monitor_params.complete
        ):
            raise ValueError(
                "training.monitor_params.lambda_ece and training.monitor_params.lambda_rate "
                "must be set when training.directional_thresholds.score is tailored_score."
            )
        if self.persistent_workers and self.num_workers == 0:
            raise ValueError("training.persistent_workers requires training.num_workers > 0.")
        if self.class_weight_beta < 0.0:
            raise ValueError("training.class_weight_beta must be >= 0.")
        if self.class_weight_min <= 0.0:
            raise ValueError("training.class_weight_min must be > 0.")
        if self.class_weight_max < self.class_weight_min:
            raise ValueError("training.class_weight_max must be >= training.class_weight_min.")

    @property
    def pin_memory(self) -> bool:
        """Whether data loaders should pin memory for CUDA transfers.

        Returns:
            True when the configured device starts with "cuda".
        """
        return self.device.lower().startswith("cuda")

    @property
    def best_model_path(self) -> Path:
        """Path where the best model checkpoint should be stored.

        Returns:
            model_dir joined with the standard best-model filename.
        """
        return Path(self.model_dir) / BEST_MODEL_FILENAME

    @property
    def checkpoint_dir(self) -> Path:
        """Directory where top-k candidate checkpoints are stored."""
        return Path(self.model_dir) / "checkpoints"

    def checkpoint_path(self, epoch: int) -> Path:
        """Return the candidate checkpoint path for an epoch."""
        return self.checkpoint_dir / f"epoch_{int(epoch):04d}.pth"

    def data_loader_kwargs(self) -> dict[str, bool | int]:
        """Return PyTorch data-loader worker and memory options.

        Returns:
            Keyword arguments for ``DataLoader`` construction.
        """
        return {
            "num_workers": self.num_workers,
            "persistent_workers": self.persistent_workers,
            "pin_memory": self.pin_memory,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingConfig":
        """Build training settings from a YAML subsection."""
        return cls(
            device=str(payload["device"]).lower(),
            epochs=int(payload["epochs"]),
            batch_size=int(payload["batch_size"]),
            eval_batch_size=int(payload["eval_batch_size"]),
            num_workers=int(payload["num_workers"]),
            early_stopping_patience=int(payload["early_stopping_patience"]),
            early_stopping_warmup=int(payload.get("early_stopping_warmup", 0)),
            early_stopping_min_delta=float(payload.get("early_stopping_min_delta", 0.0)),
            monitor=str(payload["monitor"]),
            monitor_mode=str(payload["monitor_mode"]),
            monitor_params=TrainingMonitorParamsConfig.from_dict(payload.get("monitor_params")),
            top_k_checkpoints=int(payload.get("top_k_checkpoints", 1)),
            persistent_workers=bool(payload["persistent_workers"]),
            optimizer=str(payload.get("optimizer", "adamw")),
            learning_rate=float(payload["learning_rate"]),
            weight_decay=float(payload["weight_decay"]),
            focal_gamma=float(payload["focal_gamma"]),
            class_weight_beta=float(payload["class_weight_beta"]),
            class_weight_min=float(payload["class_weight_min"]),
            class_weight_max=float(payload["class_weight_max"]),
            grad_clip_norm=float(payload["grad_clip_norm"]),
            model_dir=str(payload["model_dir"]),
            use_amp=bool(payload["use_amp"]),
            deterministic_torch=bool(payload["deterministic_torch"]),
            temperature_scaling=TrainingTemperatureScalingConfig.from_dict(
                payload.get("temperature_scaling"),
            ),
            directional_thresholds=TrainingDirectionalThresholdConfig.from_dict(
                payload.get("directional_thresholds"),
            ),
            sampling=TrainingSamplingConfig.from_dict(payload.get("sampling")),
        )


@dataclass(slots=True)
class ExperimentConfig:
    path: Path
    experiment: ExperimentMetadataConfig
    seed: int
    data: DataConfig
    dataset_splits: DatasetSplitConfig
    folds: list[FoldConfig]
    preprocessing: PreprocessingConfig
    model: ModelConfig
    training: TrainingConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load and validate an experiment configuration from YAML.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            A fully populated ``ExperimentConfig`` instance.

        Raises:
            ValueError: If the file contents violate the expected schema or value
                constraints.
            yaml.YAMLError: If the YAML file cannot be parsed.
        """
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}

        _validate_required_config(payload, REQUIRED_CONFIG_SCHEMA)
        _validate_allowed_values(payload)

        data_config = DataConfig.from_dict(payload["data"])
        dataset_splits = DatasetSplitConfig.from_dict(payload["dataset_splits"])
        experiment_config = ExperimentMetadataConfig.from_dict(
            payload.get("experiment"),
            default_name=config_path.stem,
        )
        seed = int(_require_explicit_value(payload["seed"], "seed"))
        if seed < 0:
            raise ValueError("Invalid experiment config; seed must be >= 0.")

        return cls(
            path=config_path.resolve(),
            experiment=experiment_config,
            seed=seed,
            data=data_config,
            dataset_splits=dataset_splits,
            folds=_folds_from_payload(payload, dataset_splits),
            preprocessing=PreprocessingConfig.from_dict(payload["preprocessing"], tick_size=data_config.tick_size),
            model=ModelConfig.from_dict(payload["model"]),
            training=TrainingConfig.from_dict(payload["training"]),
        )


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    """Load the project experiment configuration.

    Args:
        path: Optional path to a YAML configuration file. When omitted, the default
            "configs/pipeline_config.yaml" file is used.

    Returns:
        A fully populated "ExperimentConfig" instance.
    """
    default_path = Path(__file__).resolve().parent.parent / "configs" / "pipeline_config.yaml"
    return ExperimentConfig.from_yaml(path or default_path)
