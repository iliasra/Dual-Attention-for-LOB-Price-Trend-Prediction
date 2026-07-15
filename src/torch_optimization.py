from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

try:
    from configuration import TorchCompileConfig
except ImportError:  # pragma: no cover
    from .configuration import TorchCompileConfig


def unwrap_compiled_model(model: nn.Module) -> nn.Module:
    """Return the eager module wrapped by ``torch.compile``.

    Keeping this detail out of checkpoints prevents ``_orig_mod.`` prefixes
    and preserves compatibility with runs produced before compilation support.
    """
    current = model
    seen: set[int] = set()
    while hasattr(current, "_orig_mod") and id(current) not in seen:
        seen.add(id(current))
        original = getattr(current, "_orig_mod")
        if not isinstance(original, nn.Module):
            break
        current = original
    return current


def uncompiled_state_dict(model: nn.Module) -> dict[str, Any]:
    """Return a checkpoint-compatible state dict for eager or compiled models."""
    return unwrap_compiled_model(model).state_dict()


def load_uncompiled_state_dict(
    model: nn.Module,
    state_dict: Mapping[str, Any],
    *,
    strict: bool = True,
) -> Any:
    """Load an eager-format state dict into an eager or compiled model."""
    return unwrap_compiled_model(model).load_state_dict(state_dict, strict=strict)


def compile_model_for_training(
    model: nn.Module,
    config: TorchCompileConfig,
    device: torch.device,
) -> nn.Module:
    """Compile only the compute-heavy model path when the runtime supports it."""
    if not config.enabled:
        return model
    if config.require_cuda and device.type != "cuda":
        print(
            "torch.compile requested but skipped because the selected device is "
            f"{device.type!r} and training.torch_compile.require_cuda=true."
        )
        return model
    compile_function = getattr(torch, "compile", None)
    if compile_function is None:
        print("torch.compile requested but unavailable in this PyTorch build; using eager execution.")
        return model

    print(
        "Enabling torch.compile for the model: "
        f"backend={config.backend}, mode={config.mode}, "
        f"fullgraph={config.fullgraph}, dynamic={config.dynamic}. "
        "The first train and validation steps include compilation warm-up."
    )
    try:
        return compile_function(
            model,
            backend=config.backend,
            mode=config.mode,
            fullgraph=config.fullgraph,
            dynamic=config.dynamic,
        )
    except Exception as error:
        print(
            "torch.compile initialization failed; continuing with eager execution: "
            f"{type(error).__name__}: {error}"
        )
        return model
