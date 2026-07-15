from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from time import perf_counter
from typing import Any, Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from action_value import ActionValueMetrics, action_value_metrics
    from configuration import TrainingConfig
    from training import LobTrainer
    from torch_optimization import (
        compile_model_for_training,
        load_uncompiled_state_dict,
        uncompiled_state_dict,
        unwrap_compiled_model,
    )
except ImportError:  # pragma: no cover
    from .action_value import ActionValueMetrics, action_value_metrics
    from .configuration import TrainingConfig
    from .training import LobTrainer
    from .torch_optimization import (
        compile_model_for_training,
        load_uncompiled_state_dict,
        uncompiled_state_dict,
        unwrap_compiled_model,
    )


@dataclass(frozen=True, slots=True)
class ActionValueEpochResult:
    epoch: int
    train_loss: float
    validation_loss: float
    validation_metrics: ActionValueMetrics
    monitor_value: float
    validation_index: int | None = None
    batch_in_epoch: int | None = None
    global_step: int | None = None
    optimizer_step: int | None = None
    checkpoint_label: str | None = None

    def to_dict(self) -> dict:
        # Manual construction also tolerates epoch-only result objects restored
        # from checkpoints written before intra-epoch validation was supported.
        payload = {
            "epoch": int(self.epoch),
            "train_loss": float(self.train_loss),
            "validation_loss": float(self.validation_loss),
            "monitor_value": float(self.monitor_value),
            "validation_index": getattr(self, "validation_index", None),
            "batch_in_epoch": getattr(self, "batch_in_epoch", None),
            "global_step": getattr(self, "global_step", None),
            "optimizer_step": getattr(self, "optimizer_step", None),
            "checkpoint_label": getattr(self, "checkpoint_label", None),
        }
        payload["validation_metrics"] = self.validation_metrics.to_dict()
        return payload


def _atomic_torch_save(payload: object, path: Path) -> None:
    """Write a torch artifact atomically within its destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def split_action_value_outputs(
    predictions: torch.Tensor,
    config: TrainingConfig,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Split central values and optional quantiles from the regression head."""
    expected = int(config.objective.regression_output_dim)
    if predictions.shape[-1] != expected:
        raise ValueError(f"Action-value head must output {expected} values, got {predictions.shape[-1]}.")
    central = predictions[..., :2]
    if not config.objective.uses_quantiles:
        return central, None
    quantiles = predictions[..., 2:].reshape(*predictions.shape[:-1], len(config.objective.quantiles), 2)
    return central, quantiles


class ActionValueRegressionLoss(nn.Module):
    """Central Huber/MSE loss with optional action-wise pinball heads."""

    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        self.config = config
        self.last_components: dict[str, torch.Tensor] = {}

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        central, quantile_predictions = split_action_value_outputs(predictions, self.config)
        if targets.shape != central.shape or targets.shape[-1] != 2:
            raise ValueError("Action-value targets must have shape [n, 2].")
        if self.config.objective.loss == "mse":
            central_loss = torch.nn.functional.mse_loss(central, targets)
        else:
            central_loss = torch.nn.functional.huber_loss(
                central,
                targets,
                delta=float(self.config.objective.huber_delta),
            )
        if quantile_predictions is None:
            self.last_components = {"central_loss": central_loss.detach()}
            return central_loss

        levels = torch.as_tensor(
            self.config.objective.quantiles,
            dtype=predictions.dtype,
            device=predictions.device,
        ).view(1, -1, 1)
        residual = targets.unsqueeze(1) - quantile_predictions
        pinball = torch.maximum(levels * residual, (levels - 1.0) * residual).mean()
        if quantile_predictions.shape[1] > 1:
            crossing = torch.relu(quantile_predictions[:, :-1] - quantile_predictions[:, 1:]).mean()
        else:
            crossing = predictions.new_zeros(())
        self.last_components = {
            "central_loss": central_loss.detach(),
            "quantile_loss": pinball.detach(),
            "quantile_crossing_loss": crossing.detach(),
        }
        return (
            central_loss
            + float(self.config.objective.quantile_loss_weight) * pinball
            + float(self.config.objective.quantile_crossing_weight) * crossing
        )


class ActionValueTrainer:
    """Train executable action values with optional conditional quantiles."""

    def __init__(self, config: TrainingConfig) -> None:
        if not config.objective.is_regression:
            raise ValueError("ActionValueTrainer requires action_value_regression objective.")
        self.config = config
        self.device = torch.device(config.device)
        self.amp_enabled = bool(config.use_amp and self.device.type == "cuda")
        self.amp_dtype = (
            torch.bfloat16
            if self.amp_enabled and LobTrainer._cuda_supports_bf16(self.device)
            else None
        )
        self.scaler = torch.amp.GradScaler(
            self.device.type,
            enabled=self.amp_enabled and self.amp_dtype is None,
        )
        if self.amp_enabled:
            amp_dtype_name = "bfloat16" if self.amp_dtype is torch.bfloat16 else "float16"
            print(
                "Action-value AMP enabled: "
                f"autocast_dtype={amp_dtype_name}, "
                f"grad_scaler_enabled={self.scaler.is_enabled()}."
            )
        self.top_checkpoints: list[tuple[float, int, Path]] = []
        self.last_evaluation_outputs: dict[str, object] | None = None
        self.selected_best_model_path: Path | None = None

    def _criterion(self) -> nn.Module:
        return ActionValueRegressionLoss(self.config)

    def _amp_context(self):
        if self.amp_dtype is None:
            return torch.autocast(device_type=self.device.type, enabled=self.amp_enabled)
        return torch.autocast(
            device_type=self.device.type,
            enabled=self.amp_enabled,
            dtype=self.amp_dtype,
        )

    @staticmethod
    def _unpack_batch(batch):
        if len(batch) == 3:
            x, t, targets = batch
            return x, t, targets, None, None
        if len(batch) == 5:
            x, t, targets, loss_mask, token_indices = batch
            return x, t, targets, loss_mask, token_indices
        raise ValueError("Expected a 3-tensor window batch or 5-tensor token-chunk batch.")

    @staticmethod
    def _tensor_summary(name: str, tensor: torch.Tensor | None) -> str:
        if tensor is None or tensor.numel() == 0:
            return f"{name}=empty"
        values = tensor.detach().float()
        finite = values[torch.isfinite(values)]
        if finite.numel() == 0:
            return f"{name}=all_non_finite shape={tuple(tensor.shape)}"
        return (
            f"{name}[shape={tuple(tensor.shape)}, min={float(finite.min()):.6g}, "
            f"max={float(finite.max()):.6g}, abs_max={float(finite.abs().max()):.6g}, "
            f"non_finite={int((~torch.isfinite(values)).sum())}]"
        )

    @staticmethod
    def _require_finite_batch(
        *,
        x: torch.Tensor,
        t: torch.Tensor,
        targets: torch.Tensor,
        token_indices: torch.Tensor | None,
        epoch: int,
        batch_index: int,
    ) -> None:
        invalid = [
            name
            for name, tensor in (("features", x), ("times", t), ("targets", targets))
            if not bool(torch.isfinite(tensor).all())
        ]
        if not invalid:
            return
        token_summary = ActionValueTrainer._tensor_summary("token_indices", token_indices)
        summaries = "; ".join(
            ActionValueTrainer._tensor_summary(name, tensor)
            for name, tensor in (("features", x), ("times", t), ("targets", targets))
        )
        raise FloatingPointError(
            f"Non-finite training batch at epoch {epoch}, batch {batch_index}: "
            f"invalid={invalid}; {token_summary}; {summaries}."
        )

    @staticmethod
    def _non_finite_gradient_names(model: nn.Module, *, limit: int = 8) -> list[str]:
        names = []
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                names.append(name)
                if len(names) >= limit:
                    break
        return names

    def _supervised_values(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        targets: torch.Tensor,
        loss_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokenwise = loss_mask is not None
        predictions = model(x, t, tokenwise=tokenwise)
        if loss_mask is not None:
            mask = loss_mask.bool()
            if targets.ndim != 3 or targets.shape[:2] != mask.shape or targets.shape[-1] != 2:
                raise ValueError(
                    "Tokenwise action-value targets must have shape [batch, sequence, 2], "
                    f"got {tuple(targets.shape)} for mask {tuple(mask.shape)}. The prepared label "
                    "shards are likely stale classification targets; rerun preprocessing with "
                    "labels.strategy=executable_return and action_value_regression enabled."
                )
            predictions = predictions[mask]
            targets = targets[mask]
        else:
            if targets.ndim != 2 or targets.shape[-1] != 2:
                raise ValueError(
                    "Window-level action-value targets must have shape [batch, 2], "
                    f"got {tuple(targets.shape)}. The prepared label shards are likely stale "
                    "classification targets; rerun preprocessing with labels.strategy=executable_return."
                )
            predictions = predictions.reshape(-1, int(self.config.objective.regression_output_dim))
        targets = targets.reshape(-1, 2)
        if predictions.shape[0] != targets.shape[0] or targets.shape[-1] != 2:
            raise ValueError(
                "Action-value predictions and targets must share the same sample count, got "
                f"{tuple(predictions.shape)} and {tuple(targets.shape)}."
            )
        return predictions, targets.float()

    def _monitor(self, loss: float, metrics: ActionValueMetrics) -> float:
        if self.config.monitor == "val_loss":
            return float(loss)
        if self.config.monitor == "val_rank_ic":
            return float(metrics.rank_ic_mean)
        if self.config.monitor == "val_pnl":
            if self.config.objective.fixed_rate is not None:
                return float(metrics.fixed_rate_mean_pnl_ticks)
            return float(metrics.mean_pnl_ticks)
        raise ValueError(f"Unsupported action-value monitor: {self.config.monitor}.")

    def _is_better(self, value: float, reference: float) -> bool:
        delta = float(self.config.early_stopping_min_delta)
        if self.config.monitor_mode == "min":
            return value < reference - delta
        return value > reference + delta

    def _save_candidate(
        self,
        model: nn.Module,
        *,
        epoch: int,
        validation_index: int,
        global_step: int | None,
        monitor_value: float,
    ) -> Path:
        candidate_path = self.config.checkpoint_path(epoch, global_step=global_step)
        candidates = [*self.top_checkpoints, (float(monitor_value), int(validation_index), candidate_path)]
        reverse = self.config.monitor_mode == "max"
        candidates.sort(key=lambda item: item[0], reverse=reverse)
        kept = candidates[: int(self.config.top_k_checkpoints)]
        dropped = candidates[int(self.config.top_k_checkpoints) :]
        if any(item[1] == validation_index for item in kept):
            _atomic_torch_save(uncompiled_state_dict(model), candidate_path)
        for _value, _epoch, path in dropped:
            if path.exists():
                path.unlink()
        self.top_checkpoints = kept
        return candidate_path

    @staticmethod
    def _rng_state() -> dict[str, Any]:
        """Capture RNG state so validation-boundary resumes remain reproducible."""
        payload: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            payload["cuda"] = torch.cuda.get_rng_state_all()
        return payload

    @staticmethod
    def _restore_rng_state(payload: dict[str, Any] | None) -> None:
        if not payload:
            return
        if "python" in payload:
            random.setstate(payload["python"])
        if "numpy" in payload:
            np.random.set_state(payload["numpy"])
        if "torch" in payload:
            torch.set_rng_state(payload["torch"].cpu())
        if "cuda" in payload and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload["cuda"])

    def _resume_signature(self) -> dict[str, Any]:
        """Return settings that must not change across an exact resume."""
        return {
            "objective_type": self.config.objective.type,
            "regression_output_dim": int(self.config.objective.regression_output_dim),
            "quantiles": tuple(float(value) for value in self.config.objective.quantiles),
            "gradient_accumulation_steps": int(self.config.gradient_accumulation_steps),
            "sequence_supervision_mode": self.config.sequence_supervision.mode,
            "validate_every_n_batches": self.config.validate_every_n_batches,
            "validate_at_epoch_end": bool(self.config.validate_at_epoch_end),
        }

    def _save_training_state(
        self,
        path: Path,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        history: list[ActionValueEpochResult],
        next_epoch: int,
        next_batch_in_epoch: int,
        global_step: int,
        optimizer_step: int,
        validation_index: int,
        interval_loss_sum: float,
        interval_rows: int,
        best_value: float,
        without_improvement: int,
        best_model_path: Path,
        wandb_run_id: str | None,
    ) -> None:
        """Persist a complete action-value state at a safe optimizer boundary."""
        _atomic_torch_save(
            {
                "state_type": "action_value_regression",
                "model_state_dict": uncompiled_state_dict(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "history": history,
                "next_epoch": int(next_epoch),
                "next_batch_in_epoch": int(next_batch_in_epoch),
                "global_step": int(global_step),
                "optimizer_step": int(optimizer_step),
                "validation_index": int(validation_index),
                "interval_loss_sum": float(interval_loss_sum),
                "interval_rows": int(interval_rows),
                "best_monitor_value": float(best_value),
                "validations_without_improvement": int(without_improvement),
                "best_model_path": str(best_model_path),
                "top_checkpoints": [
                    {"monitor_value": value, "validation_index": index, "path": str(candidate_path)}
                    for value, index, candidate_path in self.top_checkpoints
                ],
                "wandb_run_id": wandb_run_id,
                "rng_state": self._rng_state(),
                "resume_signature": self._resume_signature(),
            },
            path,
        )

    def _load_training_state(
        self,
        path: Path,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> dict[str, Any]:
        """Restore an action-value state saved by ``_save_training_state``."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state.get("state_type") != "action_value_regression":
            raise ValueError(f"Not an action-value regression training state: {path}")
        saved_signature = state.get("resume_signature")
        current_signature = self._resume_signature()
        incompatible_signature = {
            key: (saved_value, current_signature.get(key))
            for key, saved_value in (saved_signature or {}).items()
            if current_signature.get(key) != saved_value
        }
        if incompatible_signature:
            raise ValueError(
                "Cannot resume action-value training with incompatible objective, "
                "accumulation, quantiles, validation schedule, or sequence supervision settings: "
                f"differences={incompatible_signature!r}."
            )
        load_uncompiled_state_dict(model, state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        if state.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(state["scaler_state_dict"])
        self.top_checkpoints = [
            (
                float(item["monitor_value"]),
                int(item.get("validation_index", item.get("epoch", index + 1))),
                Path(item["path"]),
            )
            for index, item in enumerate(state.get("top_checkpoints", []))
        ]
        self._restore_rng_state(state.get("rng_state"))
        return state

    def _run_train_epoch(
        self,
        model: nn.Module,
        data_loader: Iterable,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        epoch: int,
        start_batch_in_epoch: int = 0,
        start_global_step: int = 0,
        start_optimizer_step: int = 0,
        interval_loss_sum: float = 0.0,
        interval_rows: int = 0,
        validation_interval: int | None = None,
        training_step_callback: Callable[[dict[str, Any]], None] | None = None,
        validation_boundary_callback: Callable[[int, int, int, float, int], bool] | None = None,
    ) -> float:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_rows = 0
        accumulation = int(self.config.gradient_accumulation_steps)
        pending = 0
        skipped_amp_steps = 0
        global_step = int(start_global_step)
        optimizer_step = int(start_optimizer_step)
        stopped = False
        last_batch_in_epoch = int(start_batch_in_epoch)
        loader_length = len(data_loader) if hasattr(data_loader, "__len__") else None
        for batch_index, batch in enumerate(tqdm(data_loader, desc=f"Action values epoch {epoch} [Train]"), start=1):
            if batch_index <= start_batch_in_epoch:
                continue
            last_batch_in_epoch = batch_index
            x, t, targets, loss_mask, token_indices = self._unpack_batch(batch)
            x = x.to(self.device, non_blocking=True)
            t = t.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            if loss_mask is not None:
                loss_mask = loss_mask.to(self.device, non_blocking=True)
            if token_indices is not None:
                token_indices = token_indices.to(self.device, non_blocking=True)
            self._require_finite_batch(
                x=x,
                t=t,
                targets=targets,
                token_indices=token_indices,
                epoch=epoch,
                batch_index=batch_index,
            )
            with self._amp_context():
                predictions, supervised_targets = self._supervised_values(model, x, t, targets, loss_mask)
                base_loss = criterion(predictions, supervised_targets)
                moe_loss = getattr(model, "moe_load_balancing_loss", None)
                loss = base_loss if moe_loss is None else base_loss + moe_loss
            if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(predictions).all()):
                raise FloatingPointError(f"Non-finite action-value output/loss at epoch {epoch}, batch {batch_index}.")
            self.scaler.scale(loss).backward()
            pending += 1
            is_last = loader_length is not None and batch_index == loader_length
            validation_boundary = (
                validation_interval is not None
                and (global_step + 1) % int(validation_interval) == 0
            )
            optimizer_step_completed = pending >= accumulation or is_last or validation_boundary
            optimizer_step_applied = False
            optimizer_step_skipped = False
            grad_norm_value: float | None = None
            if optimizer_step_completed:
                self.scaler.unscale_(optimizer)
                if pending > 1:
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(float(pending))
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip_norm)
                if not bool(torch.isfinite(grad_norm)):
                    if self.scaler.is_enabled():
                        old_scale = float(self.scaler.get_scale())
                        # Standard FP16 AMP behavior: unscale_ has recorded the
                        # overflow, so step() skips this update and update()
                        # lowers the dynamic loss scale.
                        self.scaler.step(optimizer)
                        self.scaler.update()
                        skipped_amp_steps += 1
                        optimizer_step_skipped = True
                        print(
                            "Skipped FP16 optimizer step after gradient overflow: "
                            f"epoch={epoch}, batch={batch_index}, "
                            f"scale={old_scale:g}->{float(self.scaler.get_scale()):g}, "
                            f"bad_gradients={self._non_finite_gradient_names(model)}."
                        )
                    else:
                        raise FloatingPointError(
                            f"Non-finite gradient norm at epoch {epoch}, batch {batch_index}; "
                            f"amp_dtype={self.amp_dtype}, "
                            f"bad_gradients={self._non_finite_gradient_names(model)}; "
                            f"{self._tensor_summary('features', x)}; "
                            f"{self._tensor_summary('targets', supervised_targets)}; "
                            f"{self._tensor_summary('predictions', predictions)}; "
                            f"{self._tensor_summary('token_indices', token_indices)}."
                        )
                else:
                    grad_norm_value = float(grad_norm.detach().item())
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer_step += 1
                    optimizer_step_applied = True
                optimizer.zero_grad(set_to_none=True)
                pending = 0
            rows = int(supervised_targets.shape[0])
            base_loss_value = float(base_loss.detach().item())
            loss_value = float(loss.detach().item())
            weighted_loss = base_loss_value * rows
            total_loss += weighted_loss
            total_rows += rows
            interval_loss_sum += weighted_loss
            interval_rows += rows
            global_step += 1
            if training_step_callback is not None:
                payload: dict[str, Any] = {
                    "epoch": int(epoch),
                    "batch_in_epoch": int(batch_index),
                    "global_step": int(global_step),
                    "optimizer_step": int(optimizer_step),
                    "train_loss_step": loss_value,
                    "train_base_loss_step": base_loss_value,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "optimizer_step_completed": bool(optimizer_step_completed),
                    "optimizer_step_applied": bool(optimizer_step_applied),
                    "optimizer_step_skipped": bool(optimizer_step_skipped),
                    "gradient_accumulation_steps": int(accumulation),
                    "supervised_tokens_per_step": rows,
                    "chunks_per_step": int(x.shape[0]),
                    "amp_scale": float(self.scaler.get_scale()),
                }
                if grad_norm_value is not None:
                    payload["gradient_norm"] = grad_norm_value
                if moe_loss is not None:
                    payload["train_moe_loss_step"] = float(moe_loss.detach().item())
                for component_name, component_value in getattr(criterion, "last_components", {}).items():
                    payload[f"train_{component_name}_step"] = float(component_value.item())
                training_step_callback(payload)
            if validation_boundary and validation_boundary_callback is not None:
                stopped = bool(
                    validation_boundary_callback(
                        int(batch_index),
                        int(global_step),
                        int(optimizer_step),
                        float(interval_loss_sum),
                        int(interval_rows),
                    )
                )
                interval_loss_sum = 0.0
                interval_rows = 0
                model.train()
                if stopped:
                    break
        if skipped_amp_steps:
            print(f"Epoch {epoch} skipped {skipped_amp_steps} FP16 optimizer step(s) after scaler overflow.")
        self.last_train_epoch_state = {
            "global_step": int(global_step),
            "optimizer_step": int(optimizer_step),
            "last_batch_in_epoch": int(last_batch_in_epoch),
            "interval_loss_sum": float(interval_loss_sum),
            "interval_rows": int(interval_rows),
            "stopped": bool(stopped),
            "skipped_amp_steps": int(skipped_amp_steps),
        }
        return total_loss / max(total_rows, 1)

    @torch.inference_mode()
    def evaluate(
        self,
        model: nn.Module,
        data_loader: Iterable,
        criterion: nn.Module | None = None,
        *,
        description: str = "Action values [Eval]",
    ) -> tuple[float, ActionValueMetrics]:
        model.eval()
        criterion = criterion or self._criterion()
        prediction_chunks = []
        target_chunks = []
        total_loss = 0.0
        total_rows = 0
        for batch in tqdm(data_loader, desc=description):
            x, t, targets, loss_mask, _token_indices = self._unpack_batch(batch)
            x = x.to(self.device, non_blocking=True)
            t = t.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            if loss_mask is not None:
                loss_mask = loss_mask.to(self.device, non_blocking=True)
            with self._amp_context():
                predictions, supervised_targets = self._supervised_values(model, x, t, targets, loss_mask)
                loss = criterion(predictions, supervised_targets)
            rows = int(supervised_targets.shape[0])
            total_loss += float(loss.item()) * rows
            total_rows += rows
            prediction_chunks.append(predictions.float().cpu())
            target_chunks.append(supervised_targets.float().cpu())
        if not prediction_chunks:
            raise ValueError("Cannot evaluate action values on an empty data loader.")
        raw_predictions = torch.cat(prediction_chunks)
        targets = torch.cat(target_chunks).numpy()
        central_predictions, quantile_predictions = split_action_value_outputs(raw_predictions, self.config)
        predictions = central_predictions.numpy()
        self.last_evaluation_outputs = {"predictions": predictions, "targets": targets}
        if quantile_predictions is not None:
            self.last_evaluation_outputs["quantile_predictions"] = quantile_predictions.numpy()
            self.last_evaluation_outputs["quantile_levels"] = torch.as_tensor(
                self.config.objective.quantiles,
                dtype=torch.float32,
            ).numpy()
        metrics = action_value_metrics(
            predictions,
            targets,
            decision_threshold_ticks=float(self.config.objective.decision_threshold_ticks),
            fixed_rate=self.config.objective.fixed_rate,
        )
        return total_loss / max(total_rows, 1), metrics

    def fit(
        self,
        model: nn.Module,
        train_loader: Iterable,
        validation_loader: Iterable,
        *,
        training_state_path: Path | None = None,
        resume_checkpoint_path: Path | None = None,
        wandb_run_id: str | None = None,
        validation_callback: Callable[[ActionValueEpochResult, int, int], None] | None = None,
        training_step_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[nn.Module, list[ActionValueEpochResult]]:
        started = perf_counter()
        model = model.to(self.device)
        criterion = self._criterion()
        optimizer = LobTrainer(self.config, log_amp_status=False)._optimizer(model)
        model = compile_model_for_training(model, self.config.torch_compile, self.device)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.epochs)
        validate_by_epoch = bool(self.config.validates_by_epoch)
        validation_interval = None if validate_by_epoch else int(self.config.validate_every_n_batches)
        validate_at_epoch_end = bool(validate_by_epoch or self.config.validate_at_epoch_end)
        best_value = float("inf") if self.config.monitor_mode == "min" else -float("inf")
        without_improvement = 0
        history: list[ActionValueEpochResult] = []
        start_epoch = 1
        resume_skip_batches = 0
        global_step = 0
        optimizer_step = 0
        validation_index = 0
        interval_loss_sum = 0.0
        interval_rows = 0
        best_model_path = self.config.best_model_path

        if resume_checkpoint_path is not None:
            state = self._load_training_state(
                resume_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            history = list(state.get("history", []))
            start_epoch = int(state.get("next_epoch", 1))
            resume_skip_batches = max(0, int(state.get("next_batch_in_epoch", 1)) - 1)
            saved_history_steps = [
                int(getattr(item, "global_step", 0) or 0)
                for item in history
            ]
            fallback_global_step = max(saved_history_steps, default=0)
            if fallback_global_step == 0 and start_epoch > 1 and hasattr(train_loader, "__len__"):
                fallback_global_step = (start_epoch - 1) * len(train_loader)
            global_step = int(state.get("global_step", fallback_global_step))
            optimizer_step = int(
                state.get(
                    "optimizer_step",
                    global_step // max(int(self.config.gradient_accumulation_steps), 1),
                )
            )
            validation_index = int(state.get("validation_index", len(history)))
            interval_loss_sum = float(state.get("interval_loss_sum", 0.0))
            interval_rows = int(state.get("interval_rows", 0))
            best_value = float(state.get("best_monitor_value", best_value))
            without_improvement = int(state.get("validations_without_improvement", 0))
            best_model_path = Path(state.get("best_model_path", self.config.best_model_path))
            print(
                f"Resuming action-value training from {resume_checkpoint_path}: "
                f"next_epoch={start_epoch}, next_batch_in_epoch={resume_skip_batches + 1}, "
                f"global_step={global_step}, optimizer_step={optimizer_step}, "
                f"completed_validations={validation_index}."
            )

        def save_training_state(*, next_epoch: int, next_batch_in_epoch: int) -> None:
            if training_state_path is None:
                return
            self._save_training_state(
                training_state_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                history=history,
                next_epoch=next_epoch,
                next_batch_in_epoch=next_batch_in_epoch,
                global_step=global_step,
                optimizer_step=optimizer_step,
                validation_index=validation_index,
                interval_loss_sum=interval_loss_sum,
                interval_rows=interval_rows,
                best_value=best_value,
                without_improvement=without_improvement,
                best_model_path=best_model_path,
                wandb_run_id=wandb_run_id,
            )

        def record_validation(
            *,
            epoch: int,
            batch_in_epoch: int | None,
            current_global_step: int,
            current_optimizer_step: int,
            train_loss_sum: float,
            train_rows: int,
            save_next_epoch: int | None,
            save_next_batch: int | None,
        ) -> bool:
            nonlocal best_value
            nonlocal best_model_path
            nonlocal without_improvement
            nonlocal validation_index
            validation_loss, validation_metrics = self.evaluate(
                model,
                validation_loader,
                criterion,
                description=(
                    f"{self.config.checkpoint_label(epoch, global_step=None if validate_by_epoch else current_global_step)} "
                    "[Validation]"
                ),
            )
            validation_index += 1
            monitor_value = self._monitor(validation_loss, validation_metrics)
            checkpoint_step = None if validate_by_epoch else int(current_global_step)
            checkpoint_label = self.config.checkpoint_label(epoch, global_step=checkpoint_step)
            self._save_candidate(
                model,
                epoch=epoch,
                validation_index=validation_index,
                global_step=checkpoint_step,
                monitor_value=monitor_value,
            )
            improved = self._is_better(monitor_value, best_value)
            if improved:
                best_value = monitor_value
                without_improvement = 0
                _atomic_torch_save(uncompiled_state_dict(model), self.config.best_model_path)
                best_model_path = self.config.best_model_path
            elif validation_index > int(self.config.early_stopping_warmup):
                without_improvement += 1
            result = ActionValueEpochResult(
                epoch=epoch,
                train_loss=float(train_loss_sum / max(train_rows, 1)),
                validation_loss=validation_loss,
                validation_metrics=validation_metrics,
                monitor_value=monitor_value,
                validation_index=validation_index,
                batch_in_epoch=batch_in_epoch,
                global_step=current_global_step,
                optimizer_step=current_optimizer_step,
                checkpoint_label=checkpoint_label,
            )
            history.append(result)
            if save_next_epoch is not None and save_next_batch is not None:
                save_training_state(next_epoch=save_next_epoch, next_batch_in_epoch=save_next_batch)
            if validation_callback is not None:
                validation_callback(result, current_global_step, current_optimizer_step)
            print(
                f"Action values validation {validation_index} ({checkpoint_label}): "
                f"train_loss={result.train_loss:.6f}, val_loss={validation_loss:.6f}, "
                f"rank_ic={validation_metrics.rank_ic_mean:.4f}, "
                f"mean_pnl_ticks={validation_metrics.mean_pnl_ticks:.4f}, "
                f"fixed_rate_mean_pnl_ticks={validation_metrics.fixed_rate_mean_pnl_ticks:.4f}."
            )
            return bool(
                self.config.early_stopping_patience > 0
                and validation_index > self.config.early_stopping_warmup
                and without_improvement >= self.config.early_stopping_patience
            )

        stopped = False
        for epoch in range(start_epoch, int(self.config.epochs) + 1):
            sampler = getattr(train_loader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            def validate_boundary(
                batch_in_epoch: int,
                boundary_global_step: int,
                boundary_optimizer_step: int,
                boundary_loss_sum: float,
                boundary_rows: int,
            ) -> bool:
                nonlocal global_step
                nonlocal optimizer_step
                nonlocal interval_loss_sum
                nonlocal interval_rows
                global_step = boundary_global_step
                optimizer_step = boundary_optimizer_step
                # The completed interval is consumed by this validation, so a
                # resume from its checkpoint starts with an empty accumulator.
                interval_loss_sum = 0.0
                interval_rows = 0
                return record_validation(
                    epoch=epoch,
                    batch_in_epoch=batch_in_epoch,
                    current_global_step=boundary_global_step,
                    current_optimizer_step=boundary_optimizer_step,
                    train_loss_sum=boundary_loss_sum,
                    train_rows=boundary_rows,
                    save_next_epoch=epoch,
                    save_next_batch=batch_in_epoch + 1,
                )

            self._run_train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                epoch=epoch,
                start_batch_in_epoch=resume_skip_batches if epoch == start_epoch else 0,
                start_global_step=global_step,
                start_optimizer_step=optimizer_step,
                interval_loss_sum=interval_loss_sum,
                interval_rows=interval_rows,
                validation_interval=validation_interval,
                training_step_callback=training_step_callback,
                validation_boundary_callback=validate_boundary if validation_interval is not None else None,
            )
            epoch_state = self.last_train_epoch_state
            global_step = int(epoch_state["global_step"])
            optimizer_step = int(epoch_state["optimizer_step"])
            interval_loss_sum = float(epoch_state["interval_loss_sum"])
            interval_rows = int(epoch_state["interval_rows"])
            stopped = bool(epoch_state["stopped"])
            if stopped:
                break
            if validate_at_epoch_end and interval_rows > 0:
                stopped = record_validation(
                    epoch=epoch,
                    batch_in_epoch=int(epoch_state["last_batch_in_epoch"]),
                    current_global_step=global_step,
                    current_optimizer_step=optimizer_step,
                    train_loss_sum=interval_loss_sum,
                    train_rows=interval_rows,
                    save_next_epoch=None,
                    save_next_batch=None,
                )
                interval_loss_sum = 0.0
                interval_rows = 0
            scheduler.step()
            save_training_state(next_epoch=epoch + 1, next_batch_in_epoch=1)
            resume_skip_batches = 0
            if stopped:
                break

        if not history and interval_rows > 0:
            # Ensure a usable selected model even when the interval is larger
            # than the complete training run and epoch-end validation is off.
            record_validation(
                epoch=min(int(self.config.epochs), max(start_epoch, 1)),
                batch_in_epoch=int(self.last_train_epoch_state["last_batch_in_epoch"]),
                current_global_step=global_step,
                current_optimizer_step=optimizer_step,
                train_loss_sum=interval_loss_sum,
                train_rows=interval_rows,
                save_next_epoch=None,
                save_next_batch=None,
            )
            interval_loss_sum = 0.0
            interval_rows = 0
            save_training_state(next_epoch=int(self.config.epochs) + 1, next_batch_in_epoch=1)

        if not best_model_path.exists():
            raise RuntimeError("Action-value training produced no best checkpoint.")
        state = torch.load(best_model_path, map_location=self.device, weights_only=True)
        load_uncompiled_state_dict(model, state)
        self.selected_best_model_path = best_model_path
        print(f"Action-value training finished in {perf_counter() - started:.2f}s.")
        return unwrap_compiled_model(model), history
