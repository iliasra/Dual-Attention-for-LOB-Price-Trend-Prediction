from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from action_value import ActionValueMetrics, action_value_metrics
    from configuration import TrainingConfig
    from training import LobTrainer
except ImportError:  # pragma: no cover
    from .action_value import ActionValueMetrics, action_value_metrics
    from .configuration import TrainingConfig
    from .training import LobTrainer


@dataclass(frozen=True, slots=True)
class ActionValueEpochResult:
    epoch: int
    train_loss: float
    validation_loss: float
    validation_metrics: ActionValueMetrics
    monitor_value: float

    def to_dict(self) -> dict:
        payload = asdict(self)
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
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.top_checkpoints: list[tuple[float, int, Path]] = []
        self.last_evaluation_outputs: dict[str, object] | None = None
        self.selected_best_model_path: Path | None = None

    def _criterion(self) -> nn.Module:
        return ActionValueRegressionLoss(self.config)

    def _amp_context(self):
        return torch.autocast(device_type=self.device.type, enabled=self.amp_enabled)

    @staticmethod
    def _unpack_batch(batch):
        if len(batch) == 3:
            x, t, targets = batch
            return x, t, targets, None
        if len(batch) == 5:
            x, t, targets, loss_mask, _token_indices = batch
            return x, t, targets, loss_mask
        raise ValueError("Expected a 3-tensor window batch or 5-tensor token-chunk batch.")

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

    def _save_candidate(self, model: nn.Module, *, epoch: int, monitor_value: float) -> None:
        candidate_path = self.config.checkpoint_path(epoch)
        candidates = [*self.top_checkpoints, (float(monitor_value), int(epoch), candidate_path)]
        reverse = self.config.monitor_mode == "max"
        candidates.sort(key=lambda item: item[0], reverse=reverse)
        kept = candidates[: int(self.config.top_k_checkpoints)]
        dropped = candidates[int(self.config.top_k_checkpoints) :]
        if any(item[1] == epoch for item in kept):
            _atomic_torch_save(model.state_dict(), candidate_path)
        for _value, _epoch, path in dropped:
            if path.exists():
                path.unlink()
        self.top_checkpoints = kept

    @staticmethod
    def _rng_state() -> dict[str, Any]:
        """Capture RNG state so an epoch-boundary resume remains reproducible."""
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
        best_value: float,
        without_improvement: int,
        best_model_path: Path,
    ) -> None:
        """Persist a complete action-value state at a validated epoch boundary."""
        _atomic_torch_save(
            {
                "state_type": "action_value_regression",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "history": history,
                "next_epoch": int(next_epoch),
                "best_monitor_value": float(best_value),
                "validations_without_improvement": int(without_improvement),
                "best_model_path": str(best_model_path),
                "top_checkpoints": [
                    {"monitor_value": value, "epoch": epoch, "path": str(candidate_path)}
                    for value, epoch, candidate_path in self.top_checkpoints
                ],
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
        if saved_signature != current_signature:
            raise ValueError(
                "Cannot resume action-value training with incompatible objective, "
                "accumulation, quantiles, or sequence supervision settings: "
                f"saved={saved_signature!r}, current={current_signature!r}."
            )
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        if state.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(state["scaler_state_dict"])
        self.top_checkpoints = [
            (float(item["monitor_value"]), int(item["epoch"]), Path(item["path"]))
            for item in state.get("top_checkpoints", [])
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
    ) -> float:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_rows = 0
        accumulation = int(self.config.gradient_accumulation_steps)
        pending = 0
        loader_length = len(data_loader) if hasattr(data_loader, "__len__") else None
        for batch_index, batch in enumerate(tqdm(data_loader, desc=f"Action values epoch {epoch} [Train]"), start=1):
            x, t, targets, loss_mask = self._unpack_batch(batch)
            x = x.to(self.device, non_blocking=True)
            t = t.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            if loss_mask is not None:
                loss_mask = loss_mask.to(self.device, non_blocking=True)
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
            if pending >= accumulation or is_last:
                self.scaler.unscale_(optimizer)
                if pending > 1:
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(float(pending))
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip_norm)
                if not bool(torch.isfinite(grad_norm)):
                    raise FloatingPointError(f"Non-finite gradient norm at epoch {epoch}, batch {batch_index}.")
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
            rows = int(supervised_targets.shape[0])
            total_loss += float(base_loss.detach().item()) * rows
            total_rows += rows
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
            x, t, targets, loss_mask = self._unpack_batch(batch)
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
    ) -> tuple[nn.Module, list[ActionValueEpochResult]]:
        if not self.config.validates_by_epoch:
            raise ValueError("Action-value regression currently requires validate_every_n_batches='epoch'.")
        started = perf_counter()
        model = model.to(self.device)
        criterion = self._criterion()
        optimizer = LobTrainer(self.config)._optimizer(model)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.epochs)
        best_value = float("inf") if self.config.monitor_mode == "min" else -float("inf")
        without_improvement = 0
        history: list[ActionValueEpochResult] = []
        start_epoch = 1
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
            best_value = float(state.get("best_monitor_value", best_value))
            without_improvement = int(state.get("validations_without_improvement", 0))
            best_model_path = Path(state.get("best_model_path", self.config.best_model_path))
            print(
                f"Resuming action-value training from {resume_checkpoint_path}: "
                f"next_epoch={start_epoch}, completed_epochs={len(history)}."
            )

        for epoch in range(start_epoch, int(self.config.epochs) + 1):
            sampler = getattr(train_loader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            train_loss = self._run_train_epoch(model, train_loader, criterion, optimizer, epoch=epoch)
            validation_loss, validation_metrics = self.evaluate(
                model,
                validation_loader,
                criterion,
                description=f"Action values epoch {epoch} [Validation]",
            )
            monitor_value = self._monitor(validation_loss, validation_metrics)
            self._save_candidate(model, epoch=epoch, monitor_value=monitor_value)
            improved = self._is_better(monitor_value, best_value)
            if improved:
                best_value = monitor_value
                without_improvement = 0
                _atomic_torch_save(model.state_dict(), self.config.best_model_path)
                best_model_path = self.config.best_model_path
            elif epoch > int(self.config.early_stopping_warmup):
                without_improvement += 1
            history.append(
                ActionValueEpochResult(
                    epoch=epoch,
                    train_loss=train_loss,
                    validation_loss=validation_loss,
                    validation_metrics=validation_metrics,
                    monitor_value=monitor_value,
                )
            )
            scheduler.step()
            if training_state_path is not None:
                self._save_training_state(
                    training_state_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    history=history,
                    next_epoch=epoch + 1,
                    best_value=best_value,
                    without_improvement=without_improvement,
                    best_model_path=best_model_path,
                )
            print(
                f"Action values epoch {epoch}/{self.config.epochs}: train_loss={train_loss:.6f}, "
                f"val_loss={validation_loss:.6f}, rank_ic={validation_metrics.rank_ic_mean:.4f}, "
                f"mean_pnl_ticks={validation_metrics.mean_pnl_ticks:.4f}, "
                f"fixed_rate_mean_pnl_ticks={validation_metrics.fixed_rate_mean_pnl_ticks:.4f}."
            )
            if (
                self.config.early_stopping_patience > 0
                and epoch > self.config.early_stopping_warmup
                and without_improvement >= self.config.early_stopping_patience
            ):
                break

        if not best_model_path.exists():
            raise RuntimeError("Action-value training produced no best checkpoint.")
        state = torch.load(best_model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state)
        self.selected_best_model_path = best_model_path
        print(f"Action-value training finished in {perf_counter() - started:.2f}s.")
        return model, history
