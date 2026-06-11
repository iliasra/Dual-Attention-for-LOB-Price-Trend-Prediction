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
    class_bias_calibration: bool = False
    class_biases: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML-friendly calibration summary."""
        biases = (
            [0.0] * int(self.n_classes)
            if self.class_biases is None
            else [float(value) for value in self.class_biases]
        )
        method = (
            "temperature_scaling_with_class_bias"
            if self.class_bias_calibration
            else "temperature_scaling"
        )
        probability_source = (
            "temperature_scaled_class_bias_logits"
            if self.class_bias_calibration
            else "temperature_scaled_logits"
        )
        return {
            "method": method,
            "loss": "unweighted_cross_entropy",
            "probability_source": probability_source,
            "temperature": float(self.temperature),
            "class_bias_calibration": bool(self.class_bias_calibration),
            "class_biases": biases,
            "class_bias_sum": float(np.sum(np.asarray(biases, dtype=np.float64))),
            "nll_before": float(self.validation_nll_before),
            "nll_after": float(self.validation_nll_after),
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
    return fit_logit_calibration(
        logits,
        targets,
        device=device,
        max_iter=max_iter,
        class_bias_calibration=False,
    )


def fit_logit_calibration(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    device: str | torch.device = "cpu",
    max_iter: int = 50,
    class_bias_calibration: bool = False,
) -> TemperatureScalingResult:
    """Fit temperature and optional class biases with unweighted validation CE."""
    logit_array, target_array = _validated_logits_targets(logits, targets)
    torch_device = _resolved_device(device)
    logits_tensor = torch.as_tensor(logit_array, dtype=torch.float32, device=torch_device)
    targets_tensor = torch.as_tensor(target_array, dtype=torch.long, device=torch_device)
    log_temperature = torch.zeros((), dtype=torch.float32, device=torch_device, requires_grad=True)
    raw_biases = (
        torch.zeros(logit_array.shape[1], dtype=torch.float32, device=torch_device, requires_grad=True)
        if class_bias_calibration
        else None
    )

    with torch.no_grad():
        nll_before = float(F.cross_entropy(logits_tensor, targets_tensor).item())

    parameters: list[torch.Tensor] = [log_temperature]
    if raw_biases is not None:
        parameters.append(raw_biases)
    optimizer = torch.optim.LBFGS(
        parameters,
        lr=0.1,
        max_iter=int(max_iter),
        line_search_fn="strong_wolfe",
    )
    evaluations = 0

    def calibrated_logits() -> torch.Tensor:
        temperature = torch.exp(log_temperature).clamp(min=MIN_TEMPERATURE, max=MAX_TEMPERATURE)
        adjusted = logits_tensor / temperature
        if raw_biases is not None:
            biases = raw_biases - raw_biases.mean()
            adjusted = adjusted + biases
        return adjusted

    def closure() -> torch.Tensor:
        nonlocal evaluations
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(calibrated_logits(), targets_tensor)
        loss.backward()
        evaluations += 1
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        temperature_tensor = torch.exp(log_temperature).clamp(min=MIN_TEMPERATURE, max=MAX_TEMPERATURE)
        temperature = float(temperature_tensor.item())
        if raw_biases is None:
            class_biases_tensor = torch.zeros(logit_array.shape[1], dtype=torch.float32, device=torch_device)
        else:
            class_biases_tensor = raw_biases - raw_biases.mean()
        nll_after = float(
            F.cross_entropy(logits_tensor / temperature_tensor + class_biases_tensor, targets_tensor).item()
        )

    if not np.isfinite(nll_after) or nll_after > nll_before:
        temperature = 1.0
        nll_after = nll_before
        class_biases = [0.0] * int(logit_array.shape[1])
    else:
        class_biases = [
            float(value)
            for value in class_biases_tensor.detach().cpu().numpy().astype(np.float64, copy=False).tolist()
        ]

    return TemperatureScalingResult(
        temperature=temperature,
        validation_nll_before=nll_before,
        validation_nll_after=nll_after,
        n_samples=int(logit_array.shape[0]),
        n_classes=int(logit_array.shape[1]),
        optimizer_evaluations=int(evaluations),
        class_bias_calibration=bool(class_bias_calibration),
        class_biases=class_biases,
    )


def temperature_scaled_probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Convert logits to float32 probabilities after temperature scaling."""
    return calibrated_probabilities(logits, temperature, class_biases=None)


def calibrated_probabilities(
    logits: np.ndarray,
    temperature: float,
    class_biases: np.ndarray | list[float] | None = None,
) -> np.ndarray:
    """Convert logits to probabilities after temperature and optional bias."""
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0.")
    logit_array = np.asarray(logits, dtype=np.float32)
    if logit_array.ndim != 2:
        raise ValueError("temperature scaling requires a 2D logits array.")
    if logit_array.shape[0] == 0:
        return np.empty_like(logit_array, dtype=np.float32)
    scaled = logit_array / np.float32(temperature)
    if class_biases is not None:
        bias_array = np.asarray(class_biases, dtype=np.float32).reshape(-1)
        if bias_array.shape[0] != logit_array.shape[1]:
            raise ValueError("class_biases length must match the number of logits columns.")
        scaled = scaled + bias_array.reshape(1, -1)
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    exp_scores = np.exp(shifted).astype(np.float32, copy=False)
    denominators = np.sum(exp_scores, axis=1, keepdims=True)
    probabilities = exp_scores / np.maximum(denominators, np.finfo(np.float32).tiny)
    return probabilities.astype(np.float32, copy=False)


def apply_temperature_to_outputs(outputs: dict[str, Any], temperature: float) -> dict[str, Any]:
    """Return prediction outputs with calibrated probabilities and argmax labels."""
    return apply_logit_calibration_to_outputs(outputs, temperature, class_biases=None)


def apply_logit_calibration_to_outputs(
    outputs: dict[str, Any],
    temperature: float,
    class_biases: np.ndarray | list[float] | None = None,
) -> dict[str, Any]:
    """Return outputs with calibrated probabilities and argmax labels."""
    if "logits" not in outputs:
        raise ValueError("prediction outputs do not contain logits for temperature scaling.")
    updated = dict(outputs)
    probabilities = calibrated_probabilities(
        np.asarray(outputs["logits"], dtype=np.float32),
        temperature,
        class_biases=class_biases,
    )
    updated["probabilities"] = probabilities
    updated["predictions"] = (
        np.asarray([], dtype=np.int64)
        if probabilities.shape[0] == 0
        else np.argmax(probabilities, axis=1).astype(np.int64, copy=False)
    )
    updated["temperature"] = float(temperature)
    if class_biases is not None:
        updated["class_biases"] = [float(value) for value in np.asarray(class_biases).reshape(-1).tolist()]
    return updated


def save_temperature_scaling_artifact(payload: dict[str, Any], target: Path) -> None:
    """Write fitted temperature scaling metadata to YAML."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
