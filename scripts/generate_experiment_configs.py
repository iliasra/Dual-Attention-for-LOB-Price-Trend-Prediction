from __future__ import annotations

import argparse
import copy
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from configuration import ExperimentConfig, TrainingConfig, WandbTrackingConfig, load_config


INTC_FOLD_ID = "intc_dev"
FI2010_FOLD_ID = "fi2010_tlob_h100"
INTC_SPLITS = {
    "train": ("2023-11-01", "2024-04-30"),
    "validation": ("2024-05-01", "2024-05-17"),
    "test": ("2024-05-20", "2024-06-07"),
    "holdout": ("2024-06-10", "2024-06-27"),
}
EXPECTED_SPLIT_COUNTS = {"train": 124, "validation": 13, "test": 14, "holdout": 13}
NYSE_HOLIDAYS = {
    "2023-11-23",
    "2023-12-25",
    "2024-01-01",
    "2024-01-15",
    "2024-02-19",
    "2024-03-29",
    "2024-05-27",
    "2024-06-19",
}
SEEDS = (42, 123, 456)
RESERVE_SEED = 789
ACTION_TARGETS = ["long_net_return_ticks", "short_net_return_ticks"]
STREAM_KEYS = ("microprice", "price_kinematic", "volume_kinematic")
SUPPORTS_DEFERRED_TEST = "evaluate_test_after_fit" in getattr(TrainingConfig, "__dataclass_fields__", {})
SUPPORTS_REQUIRED_WANDB = "required" in getattr(WandbTrackingConfig, "__dataclass_fields__", {})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and validate the staged thesis experiment configurations."
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "configs_runs")
    parser.add_argument("--template", type=Path, default=REPO_ROOT / "configs" / "pipeline_config.yaml")
    parser.add_argument("--fi2010-template", type=Path, default=REPO_ROOT / "configs" / "config_TLOB_F1_2010.yaml")
    parser.add_argument("--h-star", type=int, default=100)
    parser.add_argument("--loss-star", choices=("huber", "mse"), default="huber")
    parser.add_argument("--feature-star", choices=("base", "kin"), default="kin")
    parser.add_argument(
        "--w-star-start-seconds",
        type=int,
        default=None,
        help="Train-only early-session start; when both bounds are set, executable early YAML files are emitted.",
    )
    parser.add_argument(
        "--w-star-end-seconds",
        type=int,
        default=None,
        help="Train-only early-session end; when both bounds are set, executable early YAML files are emitted.",
    )
    parser.add_argument("--raw-data-dir", default="data/INTC_batch_2")
    parser.add_argument("--wandb-project", default="lob-price-trend")
    parser.add_argument("--wandb-entity", default="ilias-ramim-imperial-college-london")
    parser.add_argument(
        "--wandb-mode",
        choices=("online",),
        default="online",
        help="Final-campaign invariant. Offline tracking is debug-only and is not generated here.",
    )
    return parser.parse_args()


def trading_dates(start: str, end: str) -> list[str]:
    current = date.fromisoformat(start)
    final = date.fromisoformat(end)
    values: list[str] = []
    while current <= final:
        token = current.isoformat()
        if current.weekday() < 5 and token not in NYSE_HOLIDAYS:
            values.append(token)
        current += timedelta(days=1)
    return values


def build_splits() -> dict[str, list[str]]:
    splits = {
        split: trading_dates(start, end)
        for split, (start, end) in INTC_SPLITS.items()
    }
    counts = {name: len(values) for name, values in splits.items()}
    if counts != EXPECTED_SPLIT_COUNTS:
        raise AssertionError(f"Unexpected INTC split counts: {counts}; expected {EXPECTED_SPLIT_COUNTS}.")
    all_dates = [value for values in splits.values() for value in values]
    if len(all_dates) != len(set(all_dates)):
        raise AssertionError("INTC train/validation/test/holdout dates must be disjoint.")
    return splits


def relpath_for(config_path: Path, repo_relative_path: str | Path) -> str:
    absolute = (REPO_ROOT / repo_relative_path).resolve()
    return Path(os.path.relpath(absolute, start=config_path.parent.resolve())).as_posix()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    return payload


def dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=120),
        encoding="utf-8",
    )


def configure_paths(
    payload: dict[str, Any],
    *,
    config_path: Path,
    raw_data_dir: str,
    preprocessing_group: str,
) -> None:
    payload["data"].update(
        {
            "raw_data_dir": relpath_for(config_path, raw_data_dir),
            "processed_data_dir": relpath_for(
                config_path, f"data/processed_final/{preprocessing_group}"
            ),
            "sequence_data_dir": relpath_for(
                config_path, f"data/sequences_final/{preprocessing_group}"
            ),
            "logs_dir": relpath_for(config_path, "logs"),
        }
    )
    payload["preprocessing"]["normalization"]["derivatives_stats_dir"] = relpath_for(
        config_path, f"data/normalization_final/{preprocessing_group}"
    )
    payload["training"]["model_dir"] = relpath_for(config_path, "results")


def configure_intc_split(payload: dict[str, Any], splits: dict[str, list[str]]) -> None:
    payload["dataset_splits"] = {
        "train_dates": list(splits["train"]),
        "validation_dates": list(splits["validation"]),
        "test_dates": list(splits["test"]),
    }
    payload["folds"] = [
        {
            "id": INTC_FOLD_ID,
            "train_dates": list(splits["train"]),
            "validation_dates": list(splits["validation"]),
            "test_dates": list(splits["test"]),
        }
    ]


def configure_fixed_model(payload: dict[str, Any], *, d_input: int | None = None) -> None:
    model = payload["model"]
    model.update(
        {
            "d_input": d_input,
            "d_model": 256,
            "feature_embed_dim": 32,
            "feature_num_frequencies": 8,
            "feature_sigma": 1.0,
            "feature_include_raw_value": True,
            "num_layers": 2,
            "latent_spatial_embed_dim": 16,
            "num_heads": 4,
            "max_dt_quantile": 95.0,
            "max_dt": None,
            "use_moe": False,
            "use_spatial_attention": True,
            "use_temporal_attention": True,
            "num_experts": 4,
            "top_k": 2,
            "num_classes": 3,
            "rope_type": "hybrid_crope",
            "rope_base": 10000,
            "attention_dropout": 0.1,
            "moe_dropout": 0.1,
            "moe_expansion_factor": 4,
            "moe_router_noise": 0.01,
            "moe_load_balancing_weight": 0.01,
            "moe_router_z_loss_weight": 0.001,
            "classifier_dropout": 0.1,
            "local_attention_context_tokens": 128,
            "classifier_pooling": {"methods": ["last", "mean"], "last_k": 16},
            "auxiliary_heads": {
                "enabled": False,
                "movement": True,
                "direction": True,
                "hidden_dim": None,
            },
        }
    )


def configure_common_training(payload: dict[str, Any]) -> None:
    training = payload["training"]
    training.update(
        {
            "device": "cuda",
            "epochs": 20,
            "batch_size": 32,
            "gradient_accumulation_steps": 2,
            "eval_batch_size": 128,
            "num_workers": 0,
            "prefetch_factor": None,
            "preload_data_to_memory": False,
            "validate_every_n_batches": "epoch",
            "validate_at_epoch_end": False,
            "early_stopping_patience": 0,
            "early_stopping_warmup": 0,
            "early_stopping_min_delta": 0.0,
            "top_k_checkpoints": 3,
            "persistent_workers": False,
            "optimizer": "adamw",
            "learning_rate": 1.0e-4,
            "weight_decay": 1.0e-4,
            "focal_gamma": 0.0,
            "class_weight_beta": 0.25,
            "class_weight_min": 0.75,
            "class_weight_max": 1.5,
            "grad_clip_norm": 1.0,
            "use_amp": True,
            "deterministic_torch": False,
            "torch_compile": {
                "enabled": True,
                "backend": "inductor",
                "mode": "default",
                "fullgraph": False,
                "dynamic": False,
                "require_cuda": True,
            },
            "temperature_scaling": {"enabled": False, "class_bias_calibration": True},
            "directional_thresholds": {
                "enabled": False,
                "method": "top_x_quantile",
                "score": "precision_at_fixed_rate",
                "min": 0.1,
                "max": 0.95,
                "step": 0.05,
                "delta": 0.0,
                "up_precision_floor": None,
                "down_precision_floor": None,
                "up_quantile": 0.005,
                "down_quantile": 0.005,
            },
            "sampling": {"neutral_to_directional_ratio": 20.0},
            "sequence_supervision": {
                "mode": "token_chunk",
                "loss_warmup_tokens": 128,
                "chunk_stride": 128,
                "neutral_weighting": "loss_weight",
            },
        }
    )
    training["monitor_params"] = {
        "base_metric": "val_directional_macro_f1",
        "lambda_ece": 0.0,
        "lambda_rate": 0.1,
        "fixed_rate": 0.005,
    }
    if SUPPORTS_DEFERRED_TEST:
        training["evaluate_test_after_fit"] = False


def configure_tracking(
    payload: dict[str, Any],
    *,
    run_id: str,
    target: str,
    representation: str,
    project: str,
    entity: str,
    mode: str,
) -> None:
    payload["tracking"] = {
        "wandb": {
            "enabled": True,
            "project": project,
            "entity": entity,
            "mode": mode,
            "dir": None,
            "tags": ["thesis-final", "INTC", target, representation, run_id],
            "log_training_steps": True,
            "log_best_checkpoint": True,
            "log_top_k_checkpoints": False,
        }
    }
    if SUPPORTS_REQUIRED_WANDB:
        payload["tracking"]["wandb"]["required"] = True


def configure_representation(payload: dict[str, Any], representation: str) -> None:
    enabled = representation == "kin"
    payload["preprocessing"]["microprice"]["enabled"] = enabled
    payload["preprocessing"]["price_kinematic"]["enabled"] = enabled
    payload["preprocessing"]["volume_kinematic"]["enabled"] = enabled


def configure_target(payload: dict[str, Any], *, target: str, h_star: int, loss: str) -> None:
    payload["preprocessing"]["common_endpoint_support"] = {"enabled": True}
    labels = payload["preprocessing"]["labels"]
    labels["smoothing"]["h"] = h_star
    labels["smoothing"]["fit_scope"] = None
    labels["smoothing"]["adaptive_threshold"].update(
        {"label_timing": "ex_ante", "include_exante_features": False}
    )
    labels["executable_return"].update(
        {
            "horizon_events": h_star,
            "entry_lag_events": 1,
            "round_trip_fees_bps": 1.5,
            "slippage_ticks_per_side": 0.0,
            "minimum_edge_ticks": 0.0,
            "clip_target_ticks": None,
            "bid_column": "bid_price_1",
            "ask_column": "ask_price_1",
        }
    )
    objective = payload["training"]["objective"]
    objective.update(
        {
            # Classification uses cross-entropy in the trainer; this compatibility
            # field is ignored there and must not vary with the AV loss selection.
            "loss": loss if target == "exec_av" else "huber",
            "huber_delta": 1.0,
            "quantiles": [0.1, 0.5, 0.9],
            "quantile_loss_weight": 0.25,
            "quantile_crossing_weight": 0.1,
            "decision_threshold_ticks": 0.0,
            "fixed_rate": 0.005,
        }
    )

    if target == "broad":
        labels["strategy"] = "smoothing"
        payload["data"]["target_columns"] = None
        payload["data"]["feature_exclude_columns"] = []
        objective["type"] = "classification"
        objective["loss"] = "cross_entropy"
        payload["training"].update({"monitor": "val_directional_macro_f1", "monitor_mode": "max"})
    elif target == "exec_cls":
        labels["strategy"] = "executable_return"
        payload["data"]["target_columns"] = None
        payload["data"]["feature_exclude_columns"] = list(ACTION_TARGETS)
        objective["type"] = "classification"
        objective["loss"] = "cross_entropy"
        payload["training"].update({"monitor": "precision_at_fixed_rate", "monitor_mode": "max"})
    elif target == "exec_av":
        labels["strategy"] = "executable_return"
        payload["data"]["target_columns"] = list(ACTION_TARGETS)
        payload["data"]["feature_exclude_columns"] = []
        objective["type"] = "action_value_regression"
        payload["training"].update({"monitor": "val_rank_ic", "monitor_mode": "max"})
        payload["training"]["sampling"]["neutral_to_directional_ratio"] = None
    else:
        raise ValueError(f"Unknown target: {target}")


def intc_config(
    template: dict[str, Any],
    *,
    config_path: Path,
    run_id: str,
    seed: int,
    target: str,
    representation: str,
    role: str,
    h_star: int,
    loss: str,
    raw_data_dir: str,
    wandb_project: str,
    wandb_entity: str,
    wandb_mode: str,
) -> dict[str, Any]:
    payload = copy.deepcopy(template)
    payload["seed"] = seed
    payload["experiment"] = {"name": run_id}
    configure_intc_split(payload, build_splits())
    configure_fixed_model(payload)
    configure_common_training(payload)
    configure_target(payload, target=target, h_star=h_star, loss=loss)
    configure_representation(payload, representation)
    preprocessing_group = f"{target}_{representation}"
    configure_paths(
        payload,
        config_path=config_path,
        raw_data_dir=raw_data_dir,
        preprocessing_group=preprocessing_group,
    )
    configure_tracking(
        payload,
        run_id=run_id,
        target=target,
        representation=representation,
        project=wandb_project,
        entity=wandb_entity,
        mode=wandb_mode,
    )
    payload["run_metadata"] = {
        "campaign": "thesis_final",
        "config_id": run_id,
        "role": role,
        "target": target,
        "representation": representation,
        "preprocessing_group": preprocessing_group,
        "h_star": h_star,
        "loss": loss if target == "exec_av" else "cross_entropy",
        "selection_status": "provisional",
    }
    return payload


def fi2010_config(
    template: dict[str, Any],
    *,
    config_path: Path,
    run_id: str,
    seed: int,
    role: str,
    wandb_project: str,
    wandb_entity: str,
    wandb_mode: str,
) -> dict[str, Any]:
    payload = copy.deepcopy(template)
    payload["seed"] = seed
    payload["experiment"] = {"name": run_id}
    payload["data"].update(
        {
            "raw_data_dir": relpath_for(config_path, "data/F1_2010"),
            "processed_data_dir": relpath_for(config_path, "data/processed_dataframes_fi2010"),
            "sequence_data_dir": relpath_for(config_path, "data/sequences"),
            "logs_dir": relpath_for(config_path, "logs"),
            "target_columns": None,
            "feature_exclude_columns": [],
            "sequence_window": 128,
        }
    )
    payload["dataset_splits"] = {
        "train_dates": ["0001-train"],
        "validation_dates": ["0002-validation"],
        "test_dates": ["0003-test"],
    }
    payload["folds"] = [
        {
            "id": FI2010_FOLD_ID,
            "train_dates": ["0001-train"],
            "validation_dates": ["0002-validation"],
            "test_dates": ["0003-test"],
        }
    ]
    payload["preprocessing"]["normalization"]["derivatives_stats_dir"] = relpath_for(
        config_path, "data/derivatives_z_scores_fi2010"
    )
    payload['preprocessing']['causal_market_features'] = {
        'enabled': False,
        'volatility_windows': [32, 128, 256],
        'spread_regime_window': 128,
        'imbalance_levels': [1, 5],
        'microprice_levels': 5,
        'ofi_windows': [10, 50, 100],
        'intensity_windows': [10, 50, 100],
        'momentum_windows': [5, 20, 100],
        'trade_type_values': [4, 5],
    }
    payload["preprocessing"].setdefault("microprice", {"enabled": False, "levels": 1})
    payload["preprocessing"]["microprice"]["enabled"] = False
    payload["preprocessing"].setdefault(
        "sample_clock",
        {
            "mode": "event",
            "volume_step_shares": 500,
            "volume_source": "traded",
            "trade_type_values": [4, 5],
        },
    )
    configure_fixed_model(payload, d_input=144)
    configure_common_training(payload)
    payload["training"].update(
        {
            "batch_size": 64,
            "gradient_accumulation_steps": 1,
            "eval_batch_size": 256,
            "monitor": "val_macro_f1",
            "monitor_mode": "max",
            "model_dir": relpath_for(config_path, "results"),
            "sampling": {"neutral_to_directional_ratio": None},
            "sequence_supervision": {
                "mode": "last_window",
                "loss_warmup_tokens": None,
                "chunk_stride": None,
                "neutral_weighting": "none",
            },
            "objective": {
                "type": "classification",
                "loss": "cross_entropy",
                "huber_delta": 1.0,
                "quantiles": [0.1, 0.5, 0.9],
                "quantile_loss_weight": 0.25,
                "quantile_crossing_weight": 0.1,
                "decision_threshold_ticks": 0.0,
                "fixed_rate": 0.005,
            },
        }
    )
    configure_tracking(
        payload,
        run_id=run_id,
        target="fi2010_classification",
        representation="published_features",
        project=wandb_project,
        entity=wandb_entity,
        mode=wandb_mode,
    )
    payload["tracking"]["wandb"]["tags"] = ["thesis-final", "FI2010", role, run_id]
    payload["run_metadata"] = {
        "campaign": "thesis_final_fi2010",
        "config_id": run_id,
        "role": role,
        "target": "fi2010_classification",
        "representation": "published_features",
        "preprocessing_group": FI2010_FOLD_ID,
        "selection_status": "fixed",
    }
    return payload


def training_manifest_entry(run_id: str, config_path: str, *, stage: str, execute: bool = True) -> dict[str, Any]:
    entry = {
        "id": run_id,
        "config": config_path,
        "stage": stage,
        "runner": "scripts/run_training.py",
        "execute": execute,
        "status": "ready" if execute else "do_not_execute",
        "command": [
            "python",
            "scripts/run_training.py",
            "--config",
            config_path,
            "--fold-id",
            INTC_FOLD_ID,
            "--run-stem",
            run_id,
        ],
    }
    return entry


def baseline_manifest_entry(
    run_id: str,
    config_path: str,
    *,
    model: str,
    regression: bool,
    sequence_dir: str,
    loss_star: str,
    fi2010: bool = False,
) -> dict[str, Any]:
    fold_id = FI2010_FOLD_ID if fi2010 else INTC_FOLD_ID
    command: list[Any] = [
        "python",
        "baselines/run_baselines.py",
        "--config",
        config_path,
        "--run-stem",
        run_id,
        "--sequence-dir",
        f"{sequence_dir}/{fold_id}",
        "--output",
        f"metrics/{run_id}/{fold_id}/validation/metrics.json",
        "--model",
        model,
        "--context",
        "last_mean",
        "--endpoint-support",
        "common",
        "--seed",
        42,
        "--max-train-rows",
        200000,
        "--max-eval-rows",
        0,
    ]
    prerequisites: list[str] = []
    if model == "mlp":
        command.extend(["--epochs", 20, "--mlp-layers", 2, "--target-parameters", "<TRANSFORMER_PARAM_COUNT>"])
        prerequisites.append("resolve TRANSFORMER_PARAM_COUNT from the matched Transformer run")
    if model == "xgboost":
        command.extend(
            [
                "--n-estimators",
                500,
                "--max-depth",
                6,
                "--xgb-learning-rate",
                0.05,
                "--xgb-subsample",
                0.8,
                "--xgb-colsample-bytree",
                0.8,
            ]
        )
    if regression:
        command.extend(["--regression-loss", loss_star, "--huber-delta", 1.0])
    model_output = f"results/{run_id}/{fold_id}/baseline_model.pkl"
    command.extend(["--model-output", model_output])
    entry = {
        "id": run_id,
        "config": config_path,
        "stage": "fi2010_baseline" if fi2010 else "ml_baseline",
        "runner": "baselines/run_baselines.py",
        "execute": not prerequisites,
        "status": "blocked_parameter_resolution" if prerequisites else "ready",
        "prerequisites": prerequisites,
        "command": command,
        "frozen_test_command": [
            "python",
            "scripts/evaluate_baseline.py",
            "--artifact",
            model_output,
            "--sequence-dir",
            f"{sequence_dir}/{fold_id}",
            "--split",
            "test",
            "--output",
            f"metrics/{run_id}/{fold_id}/test/metrics.json",
        ],
        "notes": "Training cap is fixed at 200000 rows; validation/test remain uncapped.",
    }
    if fi2010:
        entry["frozen_holdout_command"] = None
        entry["holdout_status"] = "not_applicable_fi2010"
    else:
        entry["frozen_holdout_command"] = [
            "python",
            "scripts/evaluate_baseline.py",
            "--artifact",
            model_output,
            "--sequence-dir",
            f"{sequence_dir}/{fold_id}",
            "--split",
            "holdout",
            "--output",
            f"metrics/{run_id}/{fold_id}/holdout/metrics.json",
        ]
    return entry


def naive_baseline_manifest_entry(
    run_id: str,
    config_path: str,
    *,
    model: str,
    task: str,
    sequence_dir: str,
    window: int,
    h_star: int,
    fi2010: bool = False,
) -> dict[str, Any]:
    blocked = fi2010 and model in {"momentum", "momentum_ma"}
    fold_id = FI2010_FOLD_ID if fi2010 else INTC_FOLD_ID
    command: list[Any] = [
        "python",
        "baselines/run_baselines.py",
        "--config",
        config_path,
        "--run-stem",
        run_id,
        "--sequence-dir",
        sequence_dir,
        "--output",
        f"metrics/{run_id}/{fold_id}/validation/metrics.json",
        "--model",
        model,
        "--endpoint-support",
        "common",
        "--window",
        window,
        "--seed",
        42,
        "--num-classes",
        3,
        "--fixed-rate",
        0.005,
        "--max-train-rows",
        0,
        "--max-eval-rows",
        0,
    ]
    if model == "label_persistence":
        command.extend(["--label-lag", h_star, "--label-horizon", h_star])
    if model == "momentum" and not fi2010:
        command.extend(["--momentum-feature-name", "causal_midprice_momentum_ticks_20"])
    if model == "momentum_ma" and not fi2010:
        command.extend(
            [
                "--momentum-feature-name",
                "bid_price_1",
                "--momentum-feature-name",
                "ask_price_1",
            ]
        )
    if model in {"momentum", "momentum_ma"}:
        if task == "classification":
            if fi2010:
                command.extend(["--up-class", 0, "--neutral-class", 1, "--down-class", 2])
            else:
                command.extend(["--up-class", 2, "--neutral-class", 1, "--down-class", 0])
    reason = None
    if fi2010 and model == "momentum":
        reason = (
            "FI-2010 published features have anonymous indices; no causal raw-price contract can be verified"
        )
    elif fi2010 and model == "momentum_ma":
        reason = (
            "FI-2010 published features have anonymous indices; no named bid/ask midpoint can be verified"
        )
    return {
        "id": run_id,
        "config": config_path,
        "config_reused": True,
        "stage": "fi2010_naive_baseline" if fi2010 else "naive_baseline",
        "runner": "baselines/run_baselines.py",
        "model": model,
        "task": task,
        "execute": not blocked,
        "status": "blocked_feature_resolution" if blocked else "ready",
        "reason": reason,
        "command": command,
    }


def assert_intc_semantics(config: ExperimentConfig, payload: dict[str, Any], *, expected_path: Path) -> None:
    metadata = payload["run_metadata"]
    target = metadata["target"]
    representation = metadata["representation"]
    if config.seed < 0 or len(config.folds) != 1 or config.folds[0].id != INTC_FOLD_ID:
        raise AssertionError(f"Invalid seed/fold in {expected_path}.")
    fold = config.folds[0]
    counts = {
        "train": len(fold.train_dates),
        "validation": len(fold.validation_dates),
        "test": len(fold.test_dates),
    }
    if counts != {"train": 124, "validation": 13, "test": 14}:
        raise AssertionError(f"Invalid split counts in {expected_path}: {counts}")
    if not config.tracking.wandb.enabled or not config.tracking.wandb.log_training_steps:
        raise AssertionError(f"W&B step tracking must be enabled in {expected_path}.")
    if config.tracking.wandb.mode != "online":
        raise AssertionError(f"W&B must be online in {expected_path}.")
    if SUPPORTS_REQUIRED_WANDB and not config.tracking.wandb.required:
        raise AssertionError(f"W&B required=true is missing in {expected_path}.")
    if SUPPORTS_DEFERRED_TEST and config.training.evaluate_test_after_fit:
        raise AssertionError(f"Development test evaluation must be deferred in {expected_path}.")
    if config.training.optimizer != "adamw" or config.training.validate_every_n_batches != "epoch":
        raise AssertionError(f"Invalid optimizer/validation schedule in {expected_path}.")
    is_label_audit = metadata.get("role") == "label_audit_only"
    if not is_label_audit and config.training.epochs != 20:
        raise AssertionError(f"Invalid epoch budget in {expected_path}.")
    if config.training.top_k_checkpoints != 3 or config.training.early_stopping_patience != 0:
        raise AssertionError(f"Invalid checkpoint budget in {expected_path}.")
    model = config.model
    expected_model = (256, 2, 16, 4, False, True, True)
    actual_model = (
        model.d_model,
        model.num_layers,
        model.latent_spatial_embed_dim,
        model.num_heads,
        model.use_moe,
        model.use_spatial_attention,
        model.use_temporal_attention,
    )
    if actual_model != expected_model:
        raise AssertionError(f"Fixed architecture mismatch in {expected_path}: {actual_model}")
    stream_state = tuple(
        bool(payload["preprocessing"][key]["enabled"])
        for key in STREAM_KEYS
    )
    expected_stream_state = (True, True, True) if representation == "kin" else (False, False, False)
    if stream_state != expected_stream_state:
        raise AssertionError(f"Representation mismatch in {expected_path}: {stream_state}")
    if target == "exec_cls":
        if config.data.target_columns is not None or config.data.feature_exclude_columns != ACTION_TARGETS:
            raise AssertionError(f"EXEC_CLS anti-leakage contract failed in {expected_path}.")
        if config.preprocessing.labels.strategy != "executable_return" or config.training.objective.is_regression:
            raise AssertionError(f"EXEC_CLS objective mismatch in {expected_path}.")
        if config.training.objective.loss != "cross_entropy":
            raise AssertionError(f"EXEC_CLS must declare cross_entropy in {expected_path}.")
    elif target == "exec_av":
        if config.data.target_columns != ACTION_TARGETS or not config.training.objective.is_regression:
            raise AssertionError(f"EXEC_AV target contract failed in {expected_path}.")
        if config.training.monitor != "val_rank_ic" or config.training.monitor_mode != "max":
            raise AssertionError(f"EXEC_AV monitor mismatch in {expected_path}.")
    elif target == "broad":
        adaptive = config.preprocessing.labels.smoothing.adaptive_threshold
        if config.preprocessing.labels.strategy != "smoothing" or adaptive is None:
            raise AssertionError(f"BROAD label mismatch in {expected_path}.")
        if adaptive.label_timing != "ex_ante" or adaptive.include_exante_features:
            raise AssertionError(f"BROAD ex-ante contract failed in {expected_path}.")
        if config.training.objective.loss != "cross_entropy":
            raise AssertionError(f"BROAD must declare cross_entropy in {expected_path}.")
    else:
        raise AssertionError(f"Unknown metadata target {target!r} in {expected_path}.")
    if not config.preprocessing.common_endpoint_support.enabled:
        raise AssertionError(f"Common endpoint support must be enabled in {expected_path}.")


def validate_generated_configs(
    written: dict[str, Path],
    *,
    alias_id: str,
    alias_of: str,
) -> None:
    for run_id, path in written.items():
        config = load_config(path)
        payload = load_yaml(path)
        if run_id.startswith("FI-"):
            if config.folds[0].id != FI2010_FOLD_ID or config.training.objective.is_regression:
                raise AssertionError(f"Invalid FI-2010 config: {path}")
            if config.model.num_layers != 2 or config.model.use_moe or config.training.optimizer != "adamw":
                raise AssertionError(f"FI-2010 backbone mismatch: {path}")
            if config.training.objective.loss != "cross_entropy":
                raise AssertionError(f"FI-2010 must declare cross_entropy in {path}")
            if not config.tracking.wandb.enabled:
                raise AssertionError(f"W&B disabled in {path}")
            if config.tracking.wandb.mode != "online":
                raise AssertionError(f"W&B must be online in {path}")
            if SUPPORTS_REQUIRED_WANDB and not config.tracking.wandb.required:
                raise AssertionError(f"W&B required=true is missing in {path}")
            if SUPPORTS_DEFERRED_TEST and config.training.evaluate_test_after_fit:
                raise AssertionError(f"FI-2010 test evaluation must be deferred in {path}")
            continue
        if run_id == alias_id:
            continue
        if payload["experiment"]["name"] != run_id or payload["run_metadata"]["config_id"] != run_id:
            raise AssertionError(f"ID/filename mismatch in {path}.")
        assert_intc_semantics(config, payload, expected_path=path)

    alias_payload = load_yaml(written[alias_id])
    source_payload = load_yaml(written[alias_of])
    if alias_payload != source_payload:
        raise AssertionError(f"{alias_id} must be an exact config alias of {alias_of}.")

    sequence_dirs: dict[str, Path] = {}
    for run_id, path in written.items():
        if run_id.startswith("FI-") or run_id == alias_id:
            continue
        payload = load_yaml(path)
        if payload["run_metadata"].get("role") == "label_audit_only":
            continue
        group = payload["run_metadata"]["preprocessing_group"]
        sequence_dir = (path.parent / payload["data"]["sequence_data_dir"]).resolve()
        existing = sequence_dirs.setdefault(group, sequence_dir)
        if sequence_dir != existing:
            raise AssertionError(f"Preprocessing group {group} has inconsistent sequence roots.")
    if len(sequence_dirs) != 5:
        raise AssertionError(f"Expected five immutable INTC preprocessing groups, got {sequence_dirs}.")


def validate_manifest_entries(runs: list[dict[str, Any]], *, h_star: int) -> None:
    ids = [str(entry["id"]) for entry in runs]
    if len(ids) != len(set(ids)):
        duplicates = sorted({run_id for run_id in ids if ids.count(run_id) > 1})
        raise AssertionError(f"Duplicate run IDs in manifest: {duplicates}")

    by_id = {str(entry["id"]): entry for entry in runs}
    expected_intc = {
        f"N{index}-{task}"
        for index in range(5)
        for task in ("EC", "AV")
    }
    expected_fi = {"FI-N0", "FI-N2", "FI-N3", "FI-N4"}
    missing = sorted((expected_intc | expected_fi | {"N5-ORACLE"}) - set(by_id))
    if missing:
        raise AssertionError(f"Missing naive baseline manifest entries: {missing}")

    for run_id in sorted(expected_intc | expected_fi):
        entry = by_id[run_id]
        index = int(run_id.split("-N")[-1]) if run_id.startswith("FI-N") else int(run_id[1])
        expected_blocked = run_id.startswith("FI-") and index in {3, 4}
        if bool(entry["execute"]) == expected_blocked:
            raise AssertionError(f"Invalid execute flag for {run_id}.")
        expected_status = "blocked_feature_resolution" if expected_blocked else "ready"
        if entry["status"] != expected_status:
            raise AssertionError(f"Invalid status for {run_id}: {entry['status']}")
        if index == 2:
            command = entry["command"]
            lag_index = command.index("--label-lag") + 1
            horizon_index = command.index("--label-horizon") + 1
            expected_lag = 100 if run_id.startswith("FI-") else h_star
            if int(command[lag_index]) != expected_lag or int(command[horizon_index]) != expected_lag:
                raise AssertionError(f"Invalid causal persistence lag for {run_id}.")

    oracle = by_id["N5-ORACLE"]
    if oracle.get("runner") is not None or oracle.get("execute") or oracle.get("status") != "metric_only":
        raise AssertionError("N5-ORACLE must remain a metric-only ex-post ceiling.")

    for entry in runs:
        config_path = entry.get("config")
        if config_path is not None and not (REPO_ROOT / str(config_path)).exists():
            raise AssertionError(f"Manifest references a missing config: {config_path}")


def readme_text(*, h_star: int, loss_star: str, feature_star: str, alias_of: str) -> str:
    deferred_test_note = (
        "Le champ `training.evaluate_test_after_fit: false` est présent : les runs de développement ne "
        "doivent pas ouvrir le test automatiquement."
        if SUPPORTS_DEFERRED_TEST
        else "TODO de compatibilité : le parseur courant n'expose pas encore "
        "`training.evaluate_test_after_fit`; régénérer les configs après son ajout avant la campagne."
    )
    return f"""# Configurations de la campagne finale

Ce dossier est généré par `scripts/generate_experiment_configs.py`. Les YAML sont complets, chargés par
`src.configuration.load_config` puis contrôlés par des assertions de campagne à chaque génération.
Lire aussi `configs_runs/READINESS_PIPELINE.md` : il distingue ce qui est lançable des verrous scientifiques
qui empêchent encore d'exécuter le protocole complet de bout en bout.

## Choix encore provisoires

- `H_STAR={h_star}` : fallback historique, à remplacer après L2 si nécessaire ;
- `LOSS_STAR={loss_star}` : incumbent, à confirmer avec P0/P1 ;
- `FEATURE_STAR={feature_star}` : incumbent, à confirmer sur validation ;
- les configs early-session restent conditionnelles tant que `W_STAR` n'est pas gelée sur les diagnostics train-only.

`AV-K-S42.yaml` est une copie exacte de `{alias_of}.yaml` et un **alias de résultat**. Il ne faut jamais lancer un
fit supplémentaire sous cet ID. Le manifeste porte `execute: false` pour cette entrée.

Pour régénérer après les décisions train/validation :

```bash
python scripts/generate_experiment_configs.py --h-star {h_star} --loss-star {loss_star} --feature-star {feature_star}
```

Les options `--w-star-start-seconds` et `--w-star-end-seconds` produisent les neuf YAML `EC-E`/`AV-E`/`BR-E`.
Le masque est appliqué uniquement aux endpoints supervisés après chargement des journées complètes : les lignes
précédentes restent donc disponibles comme contexte causal et les shards full-session sont réutilisés.

## W&B

Toutes les expériences de développement ont `tracking.wandb.enabled: true` et journalisent chaque batch. La clé
W&B est un secret : elle doit être fournie par `WANDB_API_KEY` ou par `wandb login` sur le HPC et ne doit jamais
être écrite dans un YAML, ce README, un script PBS ou Git. Le générateur n'accepte que le mode final strict
`online`. Un run offline est un smoke test non conforme à la campagne et doit utiliser une config de debug séparée.

## Groupes de preprocessing immuables

Les seeds réutilisent les mêmes shards. Ne préprocesser qu'une config représentative par groupe :

| Groupe | Config représentative |
|---|---|
| `broad_kin` | `BR-K-S42.yaml` |
| `exec_cls_base` | `EC-B-S42.yaml` |
| `exec_cls_kin` | `EC-K-S42.yaml` |
| `exec_av_base` | `AV-B-S42.yaml` |
| `exec_av_kin` | `P0.yaml` ou `P1.yaml` (mêmes targets/features) |

Toutes les configs INTC de cette campagne activent
`preprocessing.common_endpoint_support.enabled: true`. Le preprocessing calcule
les features sur la journée complète et écrit des masques BROAD, EXEC et commun.
Les lignes censurées restent dans le contexte causal mais seule l'intersection
est supervisée pendant l'entraînement. Il faut donc régénérer les shards avec ces
configs ; un cache antérieur ne possède ni le manifeste v2 ni les identifiants
requis pour `scripts/evaluate_common_executable.py`.

Chaque groupe possède ses propres racines `sequences_final`, `processed_final` et `normalization_final`. Le fold
INTC est `intc_dev`; `folds_intc_dev.txt` est prévu pour les PBS :

```bash
qsub -v TRAINING_CONFIG=configs_runs/EC-K-S42.yaml,FOLDS_FILE=configs_runs/folds_intc_dev.txt preprocess.pbs
qsub -v TRAINING_CONFIG=configs_runs/EC-K-S42.yaml,FOLDS_FILE=configs_runs/folds_intc_dev.txt,TRAINING_RUN_STEM=EC-K-S42 run_training.pbs
```

Avant KIN, construire les 248 caches GCV train-only (124 jours × prix/volume). {deferred_test_note} Pour une
ouverture strictement unique du test/holdout, une inférence sur checkpoint gelé reste nécessaire.

## Audits de labels CPU

`L2-H050`, `L2-H100`, `L2-H250` et `L2-H500` matérialisent EXEC_AV/BASE dans quatre racines distinctes et ne
doivent jamais être passés à `run_training.py`. `L3-BROAD-EXANTE` suffit pour L3 :
`scripts/analyze_label_realization.py` calcule conjointement les labels ex-ante et ex-post depuis le même frame
brut. `L3-BROAD-EXPOST` est donc une entrée de manifeste sans YAML, marquée
`computed_jointly_no_duplicate`, afin d'éviter un preprocessing redondant.
Après matérialisation, `scripts/audit_labels.py` orchestre les tableaux de support/censure, comparaison
d'horizons, clusters temporels, temps écoulé, espacement des événements et ESS.

## Manifeste

`runs_manifest.yaml` est la source d'orchestration : il distingue pilotes, fits réels, alias, baselines ML,
réserve S789 et FI-2010. Les commandes MLP restent bloquées jusqu'à substitution de
`<TRANSFORMER_PARAM_COUNT>`. Les quatre configs ML servent au schéma/tracking ; les hyperparamètres du modèle sont
portés par la commande baseline du manifeste.

N0–N4 ne dupliquent aucun YAML : les variantes EXEC_CLS réutilisent `ML1.yaml` et les variantes EXEC_AV
réutilisent `ML2.yaml`, donc la représentation est toujours `FEATURE_STAR`. N0 à N4 sont prêts ; N2 impose
`label_lag >= label_horizon = H_STAR`. N3 résout `causal_midprice_momentum_ticks_20` par nom et N4 résout
`bid_price_1`/`ask_price_1` par nom pour construire le midprice causal. N5 est uniquement l'oracle ex-post
`metric_only`, jamais un runner. FI-N3/FI-N4 restent bloqués car les features FI-2010 publiées sont anonymes.

ML1 à ML4 sauvegardent un artefact fitted avec leurs statistiques de normalisation train-only. Les champs
`frozen_test_command` et `frozen_holdout_command` appellent `scripts/evaluate_baseline.py`, qui ne charge jamais
le train et n'exécute aucun `fit`.

FI-2010 réutilise les shards `data/sequences/{FI2010_FOLD_ID}` et le backbone dense à deux blocs, avec seulement
les dimensions et la supervision adaptées aux arrays publiés.

```bash
python scripts/TLOB/prepare_fi2010_sequences.py --data-root data/F1_2010 --output-dir data/sequences/fi2010_tlob_h100 --horizon 100 --seq-size 128 --train-ratio 0.8
```

```bash
qsub -v TRAINING_CONFIG=configs_runs/fi2010/FI-T-S42.yaml,FOLDS_FILE=configs_runs/folds_fi2010.txt,TRAINING_RUN_STEM=FI-T-S42 run_training.pbs
```

Le manifeste ajoute FI-N0 et FI-N2 ; FI-N3/FI-N4 restent bloqués pour la même absence de contrat causal par nom.
"""


def main() -> None:
    args = parse_args()
    if args.h_star <= 1:
        raise ValueError("--h-star must be > 1 because entry_lag_events=1.")
    w_values = (args.w_star_start_seconds, args.w_star_end_seconds)
    if (w_values[0] is None) != (w_values[1] is None):
        raise ValueError("Set both --w-star-start-seconds and --w-star-end-seconds, or neither.")
    if w_values[0] is not None and not (34200 <= w_values[0] < w_values[1] <= 57600):
        raise ValueError("W_STAR must satisfy 34200 <= start < end <= 57600.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    template = load_yaml(args.template.resolve())
    fi_template = load_yaml(args.fi2010_template.resolve())
    splits = build_splits()
    loss_alias = "P0" if args.loss_star == "huber" else "P1"
    written: dict[str, Path] = {}
    manifest_runs: list[dict[str, Any]] = []

    def add_intc(
        run_id: str,
        *,
        seed: int,
        target: str,
        representation: str,
        role: str,
        loss: str | None = None,
        relative_dir: str = "",
        execute: bool = True,
    ) -> None:
        relative_path = Path(relative_dir) / f"{run_id}.yaml"
        path = output_dir / relative_path
        payload = intc_config(
            template,
            config_path=path,
            run_id=run_id,
            seed=seed,
            target=target,
            representation=representation,
            role=role,
            h_star=args.h_star,
            loss=loss or args.loss_star,
            raw_data_dir=args.raw_data_dir,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_mode=args.wandb_mode,
        )
        dump_yaml(path, payload)
        written[run_id] = path
        config_rel = path.relative_to(REPO_ROOT).as_posix()
        entry = training_manifest_entry(run_id, config_rel, stage=role, execute=execute)
        if role == "optional_reserve":
            entry["status"] = "optional_ready"
        entry["preprocessing_group"] = payload["run_metadata"]["preprocessing_group"]
        manifest_runs.append(entry)

    add_intc("P0", seed=42, target="exec_av", representation="kin", role="loss_pilot", loss="huber")
    add_intc("P1", seed=42, target="exec_av", representation="kin", role="loss_pilot", loss="mse")
    for representation, token in (("base", "B"), ("kin", "K")):
        for seed in SEEDS:
            add_intc(
                f"EC-{token}-S{seed}",
                seed=seed,
                target="exec_cls",
                representation=representation,
                role="core",
            )
    for representation, token in (("base", "B"), ("kin", "K")):
        for seed in SEEDS:
            run_id = f"AV-{token}-S{seed}"
            if run_id == "AV-K-S42":
                alias_path = output_dir / f"{run_id}.yaml"
                source_path = output_dir / f"{loss_alias}.yaml"
                alias_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
                written[run_id] = alias_path
                manifest_runs.append(
                    {
                        **training_manifest_entry(
                            run_id,
                            alias_path.relative_to(REPO_ROOT).as_posix(),
                            stage="core_alias",
                            execute=False,
                        ),
                        "alias_of": loss_alias,
                        "reuse_result": True,
                        "status": "alias_reuse",
                        "preprocessing_group": "exec_av_kin",
                    }
                )
                continue
            add_intc(
                run_id,
                seed=seed,
                target="exec_av",
                representation=representation,
                role="core",
            )
    for seed in SEEDS:
        add_intc(
            f"BR-K-S{seed}",
            seed=seed,
            target="broad",
            representation="kin",
            role="broad_bridge",
        )

    feature_token = "kin" if args.feature_star == "kin" else "base"
    ml_specs = {
        "ML1": ("exec_cls", "mlp", False),
        "ML2": ("exec_av", "mlp", True),
        "ML3": ("exec_cls", "xgboost", False),
        "ML4": ("exec_av", "xgboost", True),
    }
    for run_id, (target, model, regression) in ml_specs.items():
        add_intc(
            run_id,
            seed=42,
            target=target,
            representation=feature_token,
            role="ml_baseline_config",
        )
        manifest_runs.pop()
        group = f"{target}_{feature_token}"
        manifest_runs.append(
            baseline_manifest_entry(
                run_id,
                written[run_id].relative_to(REPO_ROOT).as_posix(),
                model=model,
                regression=regression,
                sequence_dir=f"data/sequences_final/{group}",
                loss_star=args.loss_star,
            )
        )

    naive_models = {
        0: "no_skill",
        1: "time_of_day",
        2: "label_persistence",
        3: "momentum",
        4: "momentum_ma",
    }
    for task_token, task, config_id in (
        ("EC", "classification", "ML1"),
        ("AV", "action_value_regression", "ML2"),
    ):
        target = "exec_cls" if task_token == "EC" else "exec_av"
        sequence_dir = f"data/sequences_final/{target}_{feature_token}/{INTC_FOLD_ID}"
        config_path = written[config_id].relative_to(REPO_ROOT).as_posix()
        for index, model in naive_models.items():
            manifest_runs.append(
                naive_baseline_manifest_entry(
                    f"N{index}-{task_token}",
                    config_path,
                    model=model,
                    task=task,
                    sequence_dir=sequence_dir,
                    window=256,
                    h_star=args.h_star,
                )
            )
    manifest_runs.append(
        {
            "id": "N5-ORACLE",
            "config": None,
            "stage": "naive_baseline",
            "runner": None,
            "execute": False,
            "status": "metric_only",
            "metric": "mean(max(0, V_long, V_short)) and its non-overlap/fixed-budget variants",
            "reason": "ex-post ceiling using future outcomes; it is never a predictive model",
        }
    )

    def add_label_audit(run_id: str, *, target: str, horizon: int, preprocessing_group: str) -> None:
        path = output_dir / f"{run_id}.yaml"
        payload = intc_config(
            template,
            config_path=path,
            run_id=run_id,
            seed=42,
            target=target,
            representation="base",
            role="label_audit_only",
            h_star=horizon,
            loss=args.loss_star,
            raw_data_dir=args.raw_data_dir,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_mode=args.wandb_mode,
        )
        configure_paths(
            payload,
            config_path=path,
            raw_data_dir=args.raw_data_dir,
            preprocessing_group=preprocessing_group,
        )
        payload["training"]["device"] = "cpu"
        payload["training"]["epochs"] = 1
        payload["training"]["use_amp"] = False
        payload["training"]["torch_compile"]["enabled"] = False
        payload["run_metadata"].update(
            {
                "preprocessing_group": preprocessing_group,
                "selection_status": "audit_only",
                "train_model": False,
            }
        )
        payload["tracking"]["wandb"]["tags"] = [
            "thesis-final",
            "INTC",
            "label-audit",
            run_id,
        ]
        dump_yaml(path, payload)
        written[run_id] = path
        config_rel = path.relative_to(REPO_ROOT).as_posix()
        manifest_runs.append(
            {
                "id": run_id,
                "config": config_rel,
                "stage": "label_audit",
                "runner": "scripts/process_data.py",
                "execute": True,
                "status": "audit_only",
                "train_model": False,
                "preprocessing_group": preprocessing_group,
                "command": [
                    "python",
                    "scripts/process_data.py",
                    "--config",
                    config_rel,
                    "--fold-id",
                    INTC_FOLD_ID,
                ],
            }
        )

    for horizon in (50, 100, 250, 500):
        add_label_audit(
            f"L2-H{horizon:03d}",
            target="exec_av",
            horizon=horizon,
            preprocessing_group=f"audit_l2_h{horizon:03d}_exec_av_base",
        )
    add_label_audit(
        "L3-BROAD-EXANTE",
        target="broad",
        horizon=args.h_star,
        preprocessing_group="audit_l3_broad_exante_base",
    )
    manifest_runs[-1]["notes"] = (
        "scripts/analyze_label_realization.py computes ex-ante and ex-post labels jointly from this one config; "
        "L3-BROAD-EXPOST is intentionally not materialized"
    )
    manifest_runs.append(
        {
            "id": "L3-BROAD-EXPOST",
            "config": None,
            "stage": "label_audit",
            "runner": "scripts/analyze_label_realization.py",
            "execute": False,
            "status": "computed_jointly_no_duplicate",
            "reuse_config": "configs_runs/L3-BROAD-EXANTE.yaml",
            "reason": "the analyzer constructs both timing conventions from the same raw frame",
        }
    )

    early_window_is_selected = args.w_star_start_seconds is not None
    for prefix, target in (("EC-E", "exec_cls"), ("AV-E", "exec_av"), ("BR-E", "broad")):
        for seed in SEEDS:
            run_id = f"{prefix}-S{seed}"
            if early_window_is_selected:
                add_intc(
                    run_id,
                    seed=seed,
                    target=target,
                    representation=feature_token if target != "broad" else "kin",
                    role="early_session",
                )
                path = written[run_id]
                payload = load_yaml(path)
                payload["training"]["supervision_time_window"] = {
                    "enabled": True,
                    "start_seconds": float(args.w_star_start_seconds),
                    "end_seconds": float(args.w_star_end_seconds),
                }
                payload["run_metadata"]["supervision_support"] = "early_session_endpoints"
                dump_yaml(path, payload)
                manifest_runs[-1].update(
                    {
                        "w_star_start_seconds": float(args.w_star_start_seconds),
                        "w_star_end_seconds": float(args.w_star_end_seconds),
                        "notes": (
                            "Reuses the full-session preprocessed shards and masks only supervised endpoints; "
                            "the causal sequence context is not truncated."
                        ),
                    }
                )
            else:
                manifest_runs.append(
                    {
                        "id": run_id,
                        "config": None,
                        "stage": "early_session_conditional",
                        "runner": None,
                        "execute": False,
                        "status": "blocked_conditional",
                        "target": target,
                        "seed": seed,
                        "w_star_start_seconds": None,
                        "w_star_end_seconds": None,
                        "reason": (
                            "W_STAR has not been selected from train-only label diagnostics; regenerate with "
                            "--w-star-start-seconds and --w-star-end-seconds to emit executable YAML files."
                        ),
                    }
                )

    for target_token, target, representation in (
        ("EC-B", "exec_cls", "base"),
        ("EC-K", "exec_cls", "kin"),
        ("AV-B", "exec_av", "base"),
        ("AV-K", "exec_av", "kin"),
    ):
        add_intc(
            f"{target_token}-S{RESERVE_SEED}",
            seed=RESERVE_SEED,
            target=target,
            representation=representation,
            role="optional_reserve",
            relative_dir="optional_reserve",
        )

    for seed in SEEDS:
        run_id = f"FI-T-S{seed}"
        relative_path = Path("fi2010") / f"{run_id}.yaml"
        path = output_dir / relative_path
        payload = fi2010_config(
            fi_template,
            config_path=path,
            run_id=run_id,
            seed=seed,
            role="transformer",
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_mode=args.wandb_mode,
        )
        dump_yaml(path, payload)
        written[run_id] = path
        entry = training_manifest_entry(
            run_id,
            path.relative_to(REPO_ROOT).as_posix(),
            stage="fi2010_transformer",
        )
        entry["command"][5] = FI2010_FOLD_ID
        manifest_runs.append(entry)
    for run_id, model in (("FI-MLP", "mlp"), ("FI-XGB", "xgboost")):
        relative_path = Path("fi2010") / f"{run_id}.yaml"
        path = output_dir / relative_path
        payload = fi2010_config(
            fi_template,
            config_path=path,
            run_id=run_id,
            seed=42,
            role=model,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_mode=args.wandb_mode,
        )
        dump_yaml(path, payload)
        written[run_id] = path
        manifest_runs.append(
            baseline_manifest_entry(
                run_id,
                path.relative_to(REPO_ROOT).as_posix(),
                model=model,
                regression=False,
                sequence_dir="data/sequences",
                loss_star=args.loss_star,
                fi2010=True,
            )
        )

    fi_config_path = written["FI-MLP"].relative_to(REPO_ROOT).as_posix()
    for index, model in (
        (0, "no_skill"),
        (2, "label_persistence"),
        (3, "momentum"),
        (4, "momentum_ma"),
    ):
        manifest_runs.append(
            naive_baseline_manifest_entry(
                f"FI-N{index}",
                fi_config_path,
                model=model,
                task="classification",
                sequence_dir=f"data/sequences/{FI2010_FOLD_ID}",
                window=128,
                h_star=100,
                fi2010=True,
            )
        )

    validate_generated_configs(written, alias_id="AV-K-S42", alias_of=loss_alias)
    validate_manifest_entries(manifest_runs, h_star=args.h_star)

    selection = {
        "schema_version": 1,
        "status": "provisional",
        "h_star": {"value": args.h_star, "source": "fallback_pending_train_only_L2"},
        "loss_star": {"value": args.loss_star, "source": "incumbent_pending_P0_P1"},
        "feature_star": {"value": args.feature_star, "source": "incumbent_pending_validation"},
        "w_star": {
            "start_seconds": args.w_star_start_seconds,
            "end_seconds": args.w_star_end_seconds,
            "early_configs_generated": early_window_is_selected,
            "reason": (
                "generated with a supervision-only endpoint mask"
                if early_window_is_selected
                else "W_STAR is not selected yet; pass both window bounds after train-only diagnostics"
            ),
        },
        "split_counts": EXPECTED_SPLIT_COUNTS,
        "holdout_dates": splits["holdout"],
        "aliases": {"AV-K-S42": {"alias_of": loss_alias, "new_fit": False}},
        "deferred_test_control": {
            "supported_by_parser": SUPPORTS_DEFERRED_TEST,
            "evaluate_test_after_fit": False if SUPPORTS_DEFERRED_TEST else None,
            "status": "configured" if SUPPORTS_DEFERRED_TEST else "todo_regenerate_after_schema_merge",
        },
        "strict_wandb_control": {
            "supported_by_parser": SUPPORTS_REQUIRED_WANDB,
            "required": True if SUPPORTS_REQUIRED_WANDB else None,
        },
        "secrets_in_repository": False,
    }
    dump_yaml(output_dir / "campaign_selection.yaml", selection)
    dump_yaml(
        output_dir / "runs_manifest.yaml",
        {
            "schema_version": 1,
            "campaign": "thesis_final",
            "selection_file": "configs_runs/campaign_selection.yaml",
            "folds_file": "configs_runs/folds_intc_dev.txt",
            "fi2010_folds_file": "configs_runs/folds_fi2010.txt",
            "provisional": True,
            "runs": manifest_runs,
            "not_generated": (
                {}
                if early_window_is_selected
                else {
                    "early_session": [
                        "EC-E-S42/123/456",
                        "AV-E-S42/123/456",
                        "BR-E-S42/123/456",
                    ],
                    "reason": "requires a train-only W_STAR selection",
                }
            ),
        },
    )
    (output_dir / "folds_intc_dev.txt").write_text(f"{INTC_FOLD_ID}\n", encoding="utf-8")
    (output_dir / "folds_fi2010.txt").write_text(f"{FI2010_FOLD_ID}\n", encoding="utf-8")
    (output_dir / "README.md").write_text(
        readme_text(
            h_star=args.h_star,
            loss_star=args.loss_star,
            feature_star=args.feature_star,
            alias_of=loss_alias,
        ),
        encoding="utf-8",
    )
    print(f"Generated and validated {len(written)} experiment configs in {output_dir}.")
    print(f"AV-K-S42 is an exact alias of {loss_alias}; do not launch it as a new fit.")


if __name__ == "__main__":
    main()
