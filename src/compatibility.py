"""
This src file is only used to support retro-compatibility
with pytorch=2.1.2 which is the latest available version on the GPU
cluster I will be using.  
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch


class _NoOpGradScaler:
    """Small disabled GradScaler replacement for non-AMP paths."""

    def scale(self, outputs: Any) -> Any:
        return outputs

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        return None

    def step(self, optimizer: torch.optim.Optimizer, *args: Any, **kwargs: Any) -> Any:
        return optimizer.step(*args, **kwargs)

    def update(self, *args: Any, **kwargs: Any) -> None:
        return None

    def is_enabled(self) -> bool:
        return False

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        return None


def torch_device_type(device: str | torch.device) -> str:
    return torch.device(device).type


def make_grad_scaler(device: str | torch.device, enabled: bool):
    device_type = torch_device_type(device)
    enabled = bool(enabled and device_type == "cuda")
    if not enabled:
        return _NoOpGradScaler()

    try:
        return torch.amp.GradScaler(device=device_type, enabled=enabled)
    except (AttributeError, TypeError):
        pass

    try:
        return torch.amp.GradScaler(device_type, enabled=enabled)
    except (AttributeError, TypeError):
        pass

    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device: str | torch.device, enabled: bool):
    device_type = torch_device_type(device)
    enabled = bool(enabled)

    try:
        return torch.amp.autocast(device_type=device_type, enabled=enabled)
    except (AttributeError, TypeError):
        pass

    try:
        return torch.autocast(device_type=device_type, enabled=enabled)
    except (AttributeError, TypeError):
        pass

    if device_type == "cuda":
        return torch.cuda.amp.autocast(enabled=enabled)

    if device_type == "cpu":
        try:
            return torch.cpu.amp.autocast(enabled=enabled)
        except (AttributeError, TypeError):
            return nullcontext()

    return nullcontext()


def load_torch_weights(path: str | Path, map_location: str | torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        return torch.load(path, map_location=map_location)
