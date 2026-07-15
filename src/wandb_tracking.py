from __future__ import annotations

import atexit
import importlib
import os
import re
from pathlib import Path
from typing import Any

try:
    from configuration import WandbTrackingConfig
    from training import ClassificationMetrics, EpochResult
except ImportError:  # pragma: no cover
    from .configuration import WandbTrackingConfig
    from .training import ClassificationMetrics, EpochResult


def wandb_run_id(run_stem: str, fold_id: str) -> str:
    """Return a stable W&B run id for a run/fold pair."""
    token = re.sub(r"[^A-Za-z0-9_.-]+", "-", f"{run_stem}-{fold_id}").strip("-")
    return token or "lob-training-run"


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metrics(prefix: str, metrics: ClassificationMetrics | None) -> dict[str, float]:
    if metrics is None:
        return {}
    values: dict[str, float] = {}
    for name in (
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "directional_macro_f1",
        "directional_precision_at_fixed_rate",
        "directional_precision_at_fixed_rate_k",
        "directional_precision_at_fixed_rate_actual_rate",
        "weighted_f1",
        "balanced_accuracy",
        "expected_calibration_error",
    ):
        value = _optional_float(getattr(metrics, name, None))
        if value is not None:
            values[f"{prefix}/{name}"] = value
    for name in (
        "per_class_pr_ap",
        "per_class_pr_auc",
        "per_class_roc_auc",
        "per_class_precision",
        "per_class_recall",
        "per_class_f1",
        "per_class_expected_calibration_error",
    ):
        sequence = getattr(metrics, name, None)
        if sequence is None:
            continue
        for class_index, item in enumerate(sequence):
            value = _optional_float(item)
            if value is not None:
                values[f"{prefix}/{name}/class_{class_index}"] = value
    return values


def epoch_result_to_wandb_metrics(
    result: EpochResult,
    *,
    monitor_value: float | None = None,
) -> dict[str, Any]:
    """Convert one validation result to a W&B-friendly metric payload."""
    payload: dict[str, Any] = {
        "train/loss": float(result.train_loss),
        "validation/loss": float(result.val_loss),
    }
    for key in ("epoch", "validation_index", "batch_in_epoch", "global_step"):
        value = getattr(result, key)
        if value is not None:
            payload[key] = value
    if result.checkpoint_label is not None:
        payload["validation/checkpoint_label"] = result.checkpoint_label
    if monitor_value is not None:
        payload["validation/monitor_value"] = float(monitor_value)

    payload.update(_metrics("train", result.train_metrics))
    payload.update(_metrics("validation", result.val_metrics))
    if result.test_metrics is not None:
        payload.update(_metrics("test", result.test_metrics))
    if result.val_argmax_metrics is not None:
        payload.update(_metrics("validation/argmax", result.val_argmax_metrics))
    if result.test_argmax_metrics is not None:
        payload.update(_metrics("test/argmax", result.test_argmax_metrics))
    for prefix, values in (
        ("val_threshold", result.val_threshold_metrics),
        ("test_threshold", result.test_threshold_metrics),
    ):
        if not values:
            continue
        for key, value in values.items():
            numeric = _optional_float(value)
            if numeric is not None:
                namespace = "validation" if prefix.startswith("val") else "test"
                payload[f"{namespace}/{prefix}_{key}"] = numeric
    if result.test_pnl_metrics:
        for key, value in result.test_pnl_metrics.items():
            numeric = _optional_float(value)
            if numeric is not None:
                payload[f"test/{key}"] = numeric
    return payload


def action_value_result_to_wandb_metrics(
    result: Any,
    *,
    global_step: int,
    optimizer_step: int,
) -> dict[str, Any]:
    """Convert one action-value validation result to namespaced W&B scalars."""
    payload: dict[str, Any] = {
        "global_step": int(global_step),
        "optimizer_step": int(optimizer_step),
        "epoch": int(result.epoch),
        "train/interval_loss": float(result.train_loss),
        "validation/loss": float(result.validation_loss),
        "validation/monitor_value": float(result.monitor_value),
    }
    for key, value in result.validation_metrics.to_dict().items():
        numeric = _optional_float(value)
        if numeric is not None:
            payload[f"validation/{key}"] = numeric
    for name in ("validation_index", "batch_in_epoch"):
        value = getattr(result, name, None)
        if value is not None:
            payload[name] = int(value)
    checkpoint_label = getattr(result, "checkpoint_label", None)
    if checkpoint_label:
        payload["validation/checkpoint_label"] = str(checkpoint_label)
    return payload


def _namespaced_training_step(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep counters explicit and put per-step diagnostics under ``train/``."""
    counters = {"global_step", "optimizer_step", "epoch", "batch_in_epoch"}
    converted: dict[str, Any] = {key: value for key, value in payload.items() if key in counters}
    aliases = {
        "train_loss_step": "loss_step",
        "train_base_loss_step": "base_loss_step",
        "train_auxiliary_loss_step": "auxiliary_loss_step",
        "train_moe_loss_step": "moe_loss_step",
        "train_central_loss_step": "central_loss_step",
        "train_quantile_loss_step": "quantile_loss_step",
        "train_quantile_crossing_loss_step": "quantile_crossing_loss_step",
    }
    for key, value in payload.items():
        if key in counters:
            continue
        converted[f"train/{aliases.get(key, key)}"] = value
    return converted


class WandbTracker:
    """Small optional W&B adapter that keeps training code independent from wandb."""

    def __init__(
        self,
        run: Any | None = None,
        wandb_module: Any | None = None,
        run_id: str | None = None,
        *,
        log_training_steps: bool = True,
    ) -> None:
        self.run = run
        self._wandb = wandb_module
        self.run_id = run_id
        self.log_training_steps_enabled = bool(log_training_steps)
        self._finished = False
        if self.enabled:
            atexit.register(self._finish_unclosed)

    @property
    def enabled(self) -> bool:
        return self.run is not None and self._wandb is not None

    @classmethod
    def disabled(cls) -> "WandbTracker":
        return cls()

    @classmethod
    def init(
        cls,
        config: WandbTrackingConfig,
        *,
        run_stem: str,
        fold_id: str,
        fold_log_dir: Path,
        resume_run_id: str | None = None,
        config_payload: dict[str, Any] | None = None,
    ) -> "WandbTracker":
        if not config.enabled or config.mode == "disabled":
            return cls.disabled()

        try:
            wandb = importlib.import_module("wandb")
        except ImportError:
            print("W&B tracking requested but wandb is not installed; continuing without W&B.")
            return cls.disabled()

        project = os.environ.get("WANDB_PROJECT") or config.project
        entity = os.environ.get("WANDB_ENTITY") or config.entity
        run_id = resume_run_id or os.environ.get("WANDB_RUN_ID") or wandb_run_id(run_stem, fold_id)
        tracking_dir = Path(config.dir or os.environ.get("WANDB_DIR") or fold_log_dir)
        tracking_dir.mkdir(parents=True, exist_ok=True)

        init_kwargs: dict[str, Any] = {
            "project": project,
            "entity": entity,
            "group": run_stem,
            "name": f"{run_stem}/{fold_id}",
            "id": run_id,
            "resume": "allow",
            "dir": str(tracking_dir),
            "tags": list(config.tags),
            "config": config_payload or {},
        }

        env_mode = os.environ.get("WANDB_MODE")
        requested_mode = env_mode or config.mode
        modes = [requested_mode] if requested_mode != "auto" else ["online", "offline"]
        last_error: Exception | None = None
        for index, mode in enumerate(modes):
            try:
                run = wandb.init(**init_kwargs, mode=mode)
                if index > 0:
                    print("W&B online initialization failed; using offline mode.")
                tracker = cls(
                    run=run,
                    wandb_module=wandb,
                    run_id=run_id,
                    log_training_steps=config.log_training_steps,
                )
                tracker._define_metrics()
                return tracker
            except Exception as exc:  # pragma: no cover - depends on W&B/network state.
                last_error = exc
                if mode != "online":
                    break
                print(f"W&B online initialization failed ({exc}); falling back to offline mode.")
        print(f"W&B initialization failed ({last_error}); continuing without W&B.")
        return cls.disabled()

    def update_config(self, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            self.run.config.update(payload, allow_val_change=True)
        except Exception as exc:  # pragma: no cover - defensive against W&B runtime state.
            print(f"W&B config update failed ({exc}); continuing.")

    def _define_metrics(self) -> None:
        if not self.enabled or not hasattr(self.run, "define_metric"):
            return
        try:
            self.run.define_metric("global_step")
            for namespace in ("train/*", "validation/*", "test/*", "selected/*"):
                self.run.define_metric(namespace, step_metric="global_step")
        except Exception as exc:  # pragma: no cover - depends on W&B runtime state.
            print(f"W&B metric definition failed ({exc}); continuing.")

    def log_metrics(self, payload: dict[str, Any]) -> None:
        """Log one payload without relying on W&B's internal step argument."""
        if not self.enabled:
            return
        try:
            self.run.log(payload)
        except Exception as exc:  # pragma: no cover - defensive against W&B runtime state.
            print(f"W&B metric logging failed ({exc}); continuing.")

    def log_validation(self, result: EpochResult, *, monitor_value: float | None = None) -> None:
        if not self.enabled:
            return
        payload = epoch_result_to_wandb_metrics(result, monitor_value=monitor_value)
        self.log_metrics(payload)

    def log_action_value_validation(
        self,
        result: Any,
        *,
        global_step: int,
        optimizer_step: int,
    ) -> None:
        self.log_metrics(
            action_value_result_to_wandb_metrics(
                result,
                global_step=global_step,
                optimizer_step=optimizer_step,
            )
        )

    def log_training_step(self, payload: dict[str, Any]) -> None:
        if not self.enabled or not self.log_training_steps_enabled:
            return
        self.log_metrics(_namespaced_training_step(payload))

    def log_artifact_files(self, *, name: str, artifact_type: str, paths: list[Path]) -> None:
        if not self.enabled:
            return
        existing_paths = [path for path in paths if path.exists()]
        if not existing_paths:
            return
        try:
            artifact = self._wandb.Artifact(name=name, type=artifact_type)
            for path in existing_paths:
                artifact.add_file(str(path))
            self.run.log_artifact(artifact)
        except Exception as exc:  # pragma: no cover - defensive against W&B runtime state.
            print(f"W&B artifact logging failed ({exc}); continuing.")

    def finish(self, *, exit_code: int | None = None) -> None:
        if not self.enabled or self._finished:
            return
        try:
            if exit_code is None:
                self.run.finish()
            else:
                self.run.finish(exit_code=exit_code)
            self._finished = True
        except Exception as exc:  # pragma: no cover - defensive against W&B runtime state.
            print(f"W&B finish failed ({exc}); continuing.")

    def _finish_unclosed(self) -> None:
        """Mark an uncaught process failure when a caller could not finish explicitly."""
        if self.enabled and not self._finished:
            self.finish(exit_code=1)
