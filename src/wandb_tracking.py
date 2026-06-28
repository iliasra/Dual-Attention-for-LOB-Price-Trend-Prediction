from __future__ import annotations

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
        "weighted_f1",
        "balanced_accuracy",
        "expected_calibration_error",
    ):
        value = _optional_float(getattr(metrics, name, None))
        if value is not None:
            values[f"{prefix}_{name}"] = value
    return values


def epoch_result_to_wandb_metrics(
    result: EpochResult,
    *,
    monitor_value: float | None = None,
) -> dict[str, Any]:
    """Convert one validation result to a W&B-friendly metric payload."""
    payload: dict[str, Any] = {
        "train_loss": float(result.train_loss),
        "val_loss": float(result.val_loss),
    }
    for key in ("epoch", "validation_index", "batch_in_epoch", "global_step", "checkpoint_label"):
        value = getattr(result, key)
        if value is not None:
            payload[key] = value
    if monitor_value is not None:
        payload["monitor_value"] = float(monitor_value)

    payload.update(_metrics("train", result.train_metrics))
    payload.update(_metrics("val", result.val_metrics))
    if result.test_metrics is not None:
        payload.update(_metrics("test", result.test_metrics))
    if result.val_argmax_metrics is not None:
        payload.update(_metrics("val_argmax", result.val_argmax_metrics))
    if result.test_argmax_metrics is not None:
        payload.update(_metrics("test_argmax", result.test_argmax_metrics))
    for prefix, values in (
        ("val_threshold", result.val_threshold_metrics),
        ("test_threshold", result.test_threshold_metrics),
    ):
        if not values:
            continue
        for key, value in values.items():
            numeric = _optional_float(value)
            if numeric is not None:
                payload[f"{prefix}_{key}"] = numeric
    return payload


class WandbTracker:
    """Small optional W&B adapter that keeps training code independent from wandb."""

    def __init__(self, run: Any | None = None, wandb_module: Any | None = None, run_id: str | None = None) -> None:
        self.run = run
        self._wandb = wandb_module
        self.run_id = run_id

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
                return cls(run=run, wandb_module=wandb, run_id=run_id)
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

    def log_validation(self, result: EpochResult, *, monitor_value: float | None = None) -> None:
        if not self.enabled:
            return
        payload = epoch_result_to_wandb_metrics(result, monitor_value=monitor_value)
        step = payload.get("validation_index")
        try:
            self.run.log(payload, step=None if step is None else int(step))
        except Exception as exc:  # pragma: no cover - defensive against W&B runtime state.
            print(f"W&B metric logging failed ({exc}); continuing.")

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
        if not self.enabled:
            return
        try:
            if exit_code is None:
                self.run.finish()
            else:
                self.run.finish(exit_code=exit_code)
        except Exception as exc:  # pragma: no cover - defensive against W&B runtime state.
            print(f"W&B finish failed ({exc}); continuing.")
