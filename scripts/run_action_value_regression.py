from __future__ import annotations

from pathlib import Path
import csv
import sys

import torch
import numpy as np
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from action_value import (
    ACTION_VALUE_METRICS_SCHEMA_VERSION,
    action_value_coverage_curve,
    action_value_policy_frontier,
    action_value_quantile_calibration,
)
from action_value_training import ActionValueTrainer
from configuration import ExperimentConfig
from datasets import EpochShuffledSampler, attach_evaluation_metadata
from model import build_model
from run_logging import model_parameter_summary, save_run_config_snapshot, save_run_summary
from run_training import (
    TRAINING_STATE_FILENAME,
    build_dataset,
    resolve_model_max_dt,
    resolve_resume_checkpoint,
    resolve_test_evaluation_plan,
    resume_wandb_run_id,
)
from utils import load_yaml, seed_torch_worker, set_global_seed
from wandb_tracking import WandbTracker


def flatten_numeric_payload(prefix: str, value: object) -> dict[str, float]:
    """Flatten nested final metrics for W&B while excluding non-finite values."""
    if isinstance(value, dict):
        flattened: dict[str, float] = {}
        for key, item in value.items():
            flattened.update(flatten_numeric_payload(f"{prefix}_{key}", item))
        return flattened
    if isinstance(value, (list, tuple, np.ndarray)):
        flattened = {}
        for index, item in enumerate(value):
            flattened.update(flatten_numeric_payload(f"{prefix}_{index}", item))
        return flattened
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        return {prefix: numeric} if np.isfinite(numeric) else {}
    return {}


def save_policy_frontier_artifacts(
    frontier: list[dict[str, int | float]],
    *,
    output_stem: Path,
) -> dict[str, str]:
    """Write a machine-readable frontier and a PnL-annotated PR diagnostic."""
    if not frontier:
        return {}
    csv_path = output_stem.with_suffix(".csv")
    png_path = output_stem.with_suffix(".png")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frontier[0]))
        writer.writeheader()
        writer.writerows(frontier)

    import matplotlib.pyplot as plt

    coverage = np.asarray([row["actual_total_coverage"] for row in frontier], dtype=float)
    mean_pnl = np.asarray([row["mean_pnl_ticks"] for row in frontier], dtype=float)
    total_pnl = np.asarray([row["total_pnl_ticks"] for row in frontier], dtype=float)
    precision = np.asarray([row["profitable_precision"] for row in frontier], dtype=float)
    recall = np.asarray([row["profitable_recall"] for row in frontier], dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    pnl_axis = axes[0]
    total_axis = pnl_axis.twinx()
    pnl_axis.plot(coverage, mean_pnl, marker="o", color="tab:blue", label="Mean PnL/trade")
    total_axis.plot(coverage, total_pnl, marker="s", color="tab:orange", label="Total PnL")
    pnl_axis.axhline(0.0, color="black", linewidth=0.8)
    pnl_axis.set_xscale("log")
    pnl_axis.set_xlabel("Total decision coverage")
    pnl_axis.set_ylabel("Mean realized PnL (ticks)", color="tab:blue")
    total_axis.set_ylabel("Total realized PnL (ticks)", color="tab:orange")
    pnl_axis.set_title("Economic frontier versus capacity")

    scatter = axes[1].scatter(recall, precision, c=mean_pnl, cmap="coolwarm", edgecolor="black")
    axes[1].plot(recall, precision, color="0.6", linewidth=0.8)
    for index, row in enumerate(frontier):
        axes[1].annotate(
            f"{100.0 * float(row['actual_total_coverage']):.2g}%",
            (recall[index], precision[index]),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )
    axes[1].set_xlabel("Recall of oracle-profitable rows")
    axes[1].set_ylabel("Precision = profitable-trade rate")
    axes[1].set_title(f"PnL-annotated policy PR; AP={float(frontier[0]['policy_ap']):.3f}")
    figure.colorbar(scatter, ax=axes[1], label="Mean realized PnL (ticks)")
    figure.savefig(png_path, dpi=160)
    plt.close(figure)
    return {"csv": str(csv_path), "plot": str(png_path)}


def save_quantile_calibration_artifacts(calibration: dict[str, object], *, output_stem: Path) -> dict[str, str]:
    """Save validation quantile coverage and its reliability diagram."""
    yaml_path = output_stem.with_suffix(".yaml")
    png_path = output_stem.with_suffix(".png")
    save_run_summary(calibration, yaml_path)

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5.5, 5.0), constrained_layout=True)
    levels = np.asarray(calibration["quantile_levels"], dtype=float)
    for action_name, color in (("long", "tab:blue"), ("short", "tab:orange")):
        action = calibration["per_action"][action_name]
        empirical = np.asarray([action[f"{level:g}"]["empirical_cdf"] for level in levels], dtype=float)
        axis.plot(levels, empirical, marker="o", label=action_name, color=color)
    axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="black", linewidth=0.8, label="ideal")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Nominal quantile level")
    axis.set_ylabel("Empirical P(target <= predicted quantile)")
    axis.set_title("Action-value quantile calibration")
    axis.legend()
    figure.savefig(png_path, dpi=160)
    plt.close(figure)
    return {"metrics": str(yaml_path), "plot": str(png_path)}


def train_action_value_fold(
    *,
    config: ExperimentConfig,
    fold_id: str,
    fold_sequence_dir: Path,
    fold_log_dir: Path,
    fold_result_dir: Path,
    run_stem: str,
    seed: int,
    fold_has_test_split: bool,
    resume_latest: bool = False,
    resume_from: Path | None = None,
) -> dict:
    """Train/evaluate one executable action-value regression fold."""
    if not config.training.objective.is_regression:
        raise ValueError("train_action_value_fold requires action_value_regression objective.")
    set_global_seed(seed, deterministic_torch=config.training.deterministic_torch)
    fold_log_dir.mkdir(parents=True, exist_ok=True)
    fold_result_dir.mkdir(parents=True, exist_ok=True)
    config.training.model_dir = str(fold_result_dir)
    training_state_path = fold_result_dir / TRAINING_STATE_FILENAME
    resume_checkpoint_path = resolve_resume_checkpoint(
        fold_result_dir=fold_result_dir,
        resume_latest=resume_latest,
        resume_from=resume_from,
    )
    if resume_checkpoint_path is not None:
        print(f"Fold '{fold_id}' will resume action-value training from: {resume_checkpoint_path}")

    train_dataset = build_dataset(
        fold_sequence_dir,
        "train",
        config,
        preload_to_memory=config.training.preload_data_to_memory,
    )
    validation_dataset = build_dataset(
        fold_sequence_dir,
        "validation",
        config,
        preload_to_memory=config.training.preload_data_to_memory,
    )
    if len(train_dataset) == 0 or len(validation_dataset) == 0:
        raise ValueError("Action-value regression requires non-empty train and validation datasets.")
    test_dataset = None
    open_test_split, test_skip_reason = resolve_test_evaluation_plan(
        evaluate_test_after_fit=config.training.evaluate_test_after_fit,
        fold_has_test_split=fold_has_test_split,
    )
    if not open_test_split and not config.training.evaluate_test_after_fit:
        print(
            f"Fold '{fold_id}' test evaluation disabled by training.evaluate_test_after_fit; "
            "the test dataset will not be opened."
        )
    elif open_test_split:
        test_dataset = build_dataset(
            fold_sequence_dir,
            "test",
            config,
            preload_to_memory=config.training.preload_data_to_memory,
        )
        if len(test_dataset) == 0:
            raise ValueError("Configured action-value test split has no prepared sequences.")
    else:
        print(f"Fold '{fold_id}' has no configured test split; test evaluation will be skipped.")

    max_dt_summary = resolve_model_max_dt(config, train_dataset)
    loader_kwargs = config.training.data_loader_kwargs()
    loader_kwargs["worker_init_fn"] = seed_torch_worker
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        sampler=EpochShuffledSampler(train_dataset, base_seed=seed),
        shuffle=False,
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.training.eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=config.training.eval_batch_size,
            shuffle=False,
            **loader_kwargs,
        )
        if test_dataset is not None
        else None
    )

    sample_features = train_dataset[0][0]
    model = build_model(
        config.model,
        d_input=int(sample_features.shape[-1]),
        output_dim=int(config.training.objective.regression_output_dim),
    )
    run_config_path = fold_log_dir / "config.yaml"
    save_run_config_snapshot(
        config,
        run_config_path,
        fold_id=fold_id,
        model_parameters=model_parameter_summary(model),
    )
    wandb_tracker = WandbTracker.init(
        config.tracking.wandb,
        run_stem=run_stem,
        fold_id=fold_id,
        fold_log_dir=fold_log_dir,
        resume_run_id=resume_wandb_run_id(resume_checkpoint_path),
        config_payload={
            "run_stem": run_stem,
            "fold_id": fold_id,
            "seed": seed,
            "objective": "action_value_regression",
            "config_path": str(config.path),
        },
    )
    wandb_tracker.update_config(load_yaml(run_config_path))
    trainer = ActionValueTrainer(config.training)
    try:
        model, history = trainer.fit(
            model,
            train_loader,
            validation_loader,
            training_state_path=training_state_path,
            resume_checkpoint_path=resume_checkpoint_path,
            wandb_run_id=wandb_tracker.run_id,
            validation_callback=lambda result, global_step, optimizer_step: (
                wandb_tracker.log_action_value_validation(
                    result,
                    global_step=global_step,
                    optimizer_step=optimizer_step,
                )
            ),
            training_step_callback=(
                wandb_tracker.log_training_step
                if wandb_tracker.enabled and config.tracking.wandb.log_training_steps
                else None
            ),
        )
    except Exception:
        wandb_tracker.finish(exit_code=1)
        raise
    validation_loss, validation_metrics = trainer.evaluate(
        model,
        validation_loader,
        description="Selected action-value checkpoint [Validation]",
    )
    if trainer.last_evaluation_outputs is None:
        raise RuntimeError("Validation action-value outputs were not collected.")
    validation_outputs = attach_evaluation_metadata(
        trainer.last_evaluation_outputs,
        validation_dataset,
    )
    validation_quantile_calibration = None
    validation_quantile_artifacts = None
    if "quantile_predictions" in validation_outputs:
        validation_quantile_calibration = action_value_quantile_calibration(
            np.asarray(validation_outputs["quantile_predictions"]),
            np.asarray(validation_outputs["targets"]),
            np.asarray(validation_outputs["quantile_levels"]),
        )
        validation_quantile_artifacts = save_quantile_calibration_artifacts(
            validation_quantile_calibration,
            output_stem=fold_log_dir / "validation_quantile_calibration",
        )
    validation_curve = action_value_coverage_curve(
        np.asarray(validation_outputs["predictions"]),
        np.asarray(validation_outputs["targets"]),
    )
    validation_frontier = action_value_policy_frontier(
        np.asarray(validation_outputs["predictions"]),
        np.asarray(validation_outputs["targets"]),
    )
    validation_frontier_artifacts = save_policy_frontier_artifacts(
        validation_frontier,
        output_stem=fold_log_dir / "validation_ranking_pnl_frontier",
    )
    validation_outputs_path = fold_log_dir / "validation_action_values.npz"
    np.savez_compressed(validation_outputs_path, **validation_outputs)
    test_payload = None
    if test_loader is not None:
        test_loss, test_metrics = trainer.evaluate(
            model,
            test_loader,
            description="Selected action-value checkpoint [Test]",
        )
        test_payload = {"loss": test_loss, "metrics": test_metrics.to_dict()}
        if trainer.last_evaluation_outputs is None:
            raise RuntimeError("Test action-value outputs were not collected.")
        if test_dataset is None:
            raise RuntimeError("Test outputs were collected without a test dataset.")
        test_outputs = attach_evaluation_metadata(
            trainer.last_evaluation_outputs,
            test_dataset,
        )
        if "quantile_predictions" in test_outputs:
            test_payload["quantile_calibration"] = action_value_quantile_calibration(
                np.asarray(test_outputs["quantile_predictions"]),
                np.asarray(test_outputs["targets"]),
                np.asarray(test_outputs["quantile_levels"]),
            )
            test_payload["quantile_calibration_artifacts"] = save_quantile_calibration_artifacts(
                test_payload["quantile_calibration"],
                output_stem=fold_log_dir / "test_quantile_calibration",
            )
        test_payload["coverage_curve"] = action_value_coverage_curve(
            np.asarray(test_outputs["predictions"]),
            np.asarray(test_outputs["targets"]),
        )
        test_payload["ranking_pnl_frontier"] = action_value_policy_frontier(
            np.asarray(test_outputs["predictions"]),
            np.asarray(test_outputs["targets"]),
        )
        test_payload["ranking_pnl_frontier_artifacts"] = save_policy_frontier_artifacts(
            test_payload["ranking_pnl_frontier"],
            output_stem=fold_log_dir / "test_ranking_pnl_frontier",
        )
        test_outputs_path = fold_log_dir / "test_action_values.npz"
        np.savez_compressed(test_outputs_path, **test_outputs)
        test_payload["outputs"] = str(test_outputs_path)

    history_path = fold_log_dir / "action_value_history.yaml"
    metrics_path = fold_log_dir / "action_value_metrics.yaml"
    save_run_summary(
        {
            "metrics_schema_version": ACTION_VALUE_METRICS_SCHEMA_VERSION,
            "epochs": [item.to_dict() for item in history],
        },
        history_path,
    )
    summary = {
        "metrics_schema_version": ACTION_VALUE_METRICS_SCHEMA_VERSION,
        "objective": "action_value_regression",
        "target_columns": list(config.data.target_columns or []),
        "max_dt": max_dt_summary,
        "best_checkpoint": str(trainer.selected_best_model_path or config.training.best_model_path),
        "training_state_path": str(training_state_path),
        "wandb_run_id": wandb_tracker.run_id,
        "top_k_checkpoints": [str(item[2]) for item in trainer.top_checkpoints],
        "test_evaluation_skipped": test_loader is None,
        "test_evaluation_skip_reason": test_skip_reason,
        "validation": {
            "loss": validation_loss,
            "metrics": validation_metrics.to_dict(),
            "coverage_curve": validation_curve,
            "ranking_pnl_frontier": validation_frontier,
            "ranking_pnl_frontier_artifacts": validation_frontier_artifacts,
            "quantile_calibration": validation_quantile_calibration,
            "quantile_calibration_artifacts": validation_quantile_artifacts,
            "outputs": str(validation_outputs_path),
        },
        "test": test_payload,
        "artifacts": {
            "history": str(history_path),
            "metrics": str(metrics_path),
            "config": str(run_config_path),
        },
    }
    save_run_summary(summary, metrics_path)
    final_global_step = int(getattr(trainer, "last_train_epoch_state", {}).get("global_step", 0))
    final_payload: dict[str, object] = {
        "global_step": final_global_step,
        "selected/metrics_schema_version": ACTION_VALUE_METRICS_SCHEMA_VERSION,
        "selected/validation_loss": float(validation_loss),
    }
    for key, value in validation_metrics.to_dict().items():
        if value is not None:
            final_payload[f"selected/validation_{key}"] = value
    if validation_frontier:
        final_payload["selected/validation_policy_ap"] = float(validation_frontier[0]["policy_ap"])
    if validation_quantile_calibration is not None:
        for key, value in flatten_numeric_payload(
            "validation_quantile_calibration",
            validation_quantile_calibration,
        ).items():
            final_payload[f"selected/{key}"] = value
    if test_payload is not None:
        final_payload["selected/test_loss"] = float(test_payload["loss"])
        for key, value in test_payload["metrics"].items():
            if value is not None:
                final_payload[f"selected/test_{key}"] = value
        if test_payload.get("ranking_pnl_frontier"):
            final_payload["selected/test_policy_ap"] = float(
                test_payload["ranking_pnl_frontier"][0]["policy_ap"]
            )
        if test_payload.get("quantile_calibration") is not None:
            for key, value in flatten_numeric_payload(
                "test_quantile_calibration",
                test_payload["quantile_calibration"],
            ).items():
                final_payload[f"selected/{key}"] = value
    wandb_tracker.log_metrics(final_payload)
    artifact_paths = [run_config_path, history_path, metrics_path, validation_outputs_path]
    for artifacts in (validation_frontier_artifacts, validation_quantile_artifacts or {}):
        artifact_paths.extend(Path(path) for path in artifacts.values())
    if test_payload is not None:
        if test_payload.get("outputs"):
            artifact_paths.append(Path(str(test_payload["outputs"])))
        artifact_paths.extend(
            Path(path)
            for path in test_payload.get("ranking_pnl_frontier_artifacts", {}).values()
        )
        artifact_paths.extend(
            Path(path)
            for path in test_payload.get("quantile_calibration_artifacts", {}).values()
        )
    wandb_tracker.log_artifact_files(
        name=f"{run_stem}-{fold_id}-action-value-artifacts",
        artifact_type="training-artifacts",
        paths=artifact_paths,
    )
    if config.tracking.wandb.log_best_checkpoint:
        wandb_tracker.log_artifact_files(
            name=f"{run_stem}-{fold_id}-best-model",
            artifact_type="model",
            paths=[config.training.best_model_path],
        )
    if config.tracking.wandb.log_top_k_checkpoints:
        wandb_tracker.log_artifact_files(
            name=f"{run_stem}-{fold_id}-top-k-checkpoints",
            artifact_type="model",
            paths=[item[2] for item in trainer.top_checkpoints],
        )
    wandb_tracker.finish(exit_code=0)
    return summary


if __name__ == "__main__":
    raise SystemExit(
        "Use scripts/run_training.py --config <config>; it dispatches to action-value regression from training.objective.type."
    )
