from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml


MIN_TEMPERATURE = 1e-6
MAX_TEMPERATURE = 1e6


@dataclass(slots=True)
class TemperatureScalingResult:
    temperature: float
    validation_nll_before: float
    validation_nll_after: float
    n_samples: int
    n_classes: int
    optimizer_evaluations: int

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML-friendly calibration summary."""
        return {
            "method": "temperature_scaling",
            "loss": "unweighted_cross_entropy",
            "temperature": float(self.temperature),
            "validation_nll_before": float(self.validation_nll_before),
            "validation_nll_after": float(self.validation_nll_after),
            "n_samples": int(self.n_samples),
            "n_classes": int(self.n_classes),
            "optimizer_evaluations": int(self.optimizer_evaluations),
        }


def _validated_logits_targets(logits: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate collected logits and labels for temperature fitting."""
    logit_array = np.asarray(logits, dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.int64).reshape(-1)
    if logit_array.ndim != 2:
        raise ValueError("temperature scaling requires a 2D logits array.")
    if logit_array.shape[0] == 0:
        raise ValueError("temperature scaling requires at least one validation sample.")
    if logit_array.shape[0] != target_array.shape[0]:
        raise ValueError("temperature scaling logits and targets have inconsistent row counts.")
    if logit_array.shape[1] <= 0:
        raise ValueError("temperature scaling requires at least one class.")
    if not np.all(np.isfinite(logit_array)):
        raise ValueError("temperature scaling logits must be finite.")
    if np.any((target_array < 0) | (target_array >= logit_array.shape[1])):
        raise ValueError("temperature scaling targets contain class ids outside logits columns.")
    return logit_array, target_array


def _resolved_device(device: str | torch.device) -> torch.device:
    """Return a usable torch device for calibration."""
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


def fit_temperature_scaling(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    device: str | torch.device = "cpu",
    max_iter: int = 50,
) -> TemperatureScalingResult:
    """Fit one scalar temperature with unweighted validation cross-entropy."""
    logit_array, target_array = _validated_logits_targets(logits, targets)
    torch_device = _resolved_device(device)
    logits_tensor = torch.as_tensor(logit_array, dtype=torch.float32, device=torch_device)
    targets_tensor = torch.as_tensor(target_array, dtype=torch.long, device=torch_device)
    log_temperature = torch.zeros((), dtype=torch.float32, device=torch_device, requires_grad=True)

    with torch.no_grad():
        nll_before = float(F.cross_entropy(logits_tensor, targets_tensor).item())

    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=int(max_iter),
        line_search_fn="strong_wolfe",
    )
    evaluations = 0

    def closure() -> torch.Tensor:
        nonlocal evaluations
        optimizer.zero_grad(set_to_none=True)
        temperature = torch.exp(log_temperature).clamp(min=MIN_TEMPERATURE, max=MAX_TEMPERATURE)
        loss = F.cross_entropy(logits_tensor / temperature, targets_tensor)
        loss.backward()
        evaluations += 1
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        temperature_tensor = torch.exp(log_temperature).clamp(min=MIN_TEMPERATURE, max=MAX_TEMPERATURE)
        temperature = float(temperature_tensor.item())
        nll_after = float(F.cross_entropy(logits_tensor / temperature_tensor, targets_tensor).item())

    if nll_after > nll_before:
        temperature = 1.0
        nll_after = nll_before

    return TemperatureScalingResult(
        temperature=temperature,
        validation_nll_before=nll_before,
        validation_nll_after=nll_after,
        n_samples=int(logit_array.shape[0]),
        n_classes=int(logit_array.shape[1]),
        optimizer_evaluations=int(evaluations),
    )


def temperature_scaled_probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Convert logits to float32 probabilities after temperature scaling."""
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0.")
    logit_array = np.asarray(logits, dtype=np.float32)
    if logit_array.ndim != 2:
        raise ValueError("temperature scaling requires a 2D logits array.")
    if logit_array.shape[0] == 0:
        return np.empty_like(logit_array, dtype=np.float32)
    scaled = logit_array / np.float32(temperature)
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    exp_scores = np.exp(shifted).astype(np.float32, copy=False)
    denominators = np.sum(exp_scores, axis=1, keepdims=True)
    probabilities = exp_scores / np.maximum(denominators, np.finfo(np.float32).tiny)
    return probabilities.astype(np.float32, copy=False)


def apply_temperature_to_outputs(outputs: dict[str, Any], temperature: float) -> dict[str, Any]:
    """Return prediction outputs with calibrated probabilities and argmax labels."""
    if "logits" not in outputs:
        raise ValueError("prediction outputs do not contain logits for temperature scaling.")
    updated = dict(outputs)
    probabilities = temperature_scaled_probabilities(np.asarray(outputs["logits"], dtype=np.float32), temperature)
    updated["probabilities"] = probabilities
    updated["predictions"] = (
        np.asarray([], dtype=np.int64)
        if probabilities.shape[0] == 0
        else np.argmax(probabilities, axis=1).astype(np.int64, copy=False)
    )
    updated["temperature"] = float(temperature)
    return updated


def save_temperature_scaling_artifact(payload: dict[str, Any], target: Path) -> None:
    """Write fitted temperature scaling metadata to YAML."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
