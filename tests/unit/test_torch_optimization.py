from __future__ import annotations

import torch
import torch.nn as nn

from configuration import TorchCompileConfig, load_config
from torch_optimization import (
    compile_model_for_training,
    load_uncompiled_state_dict,
    uncompiled_state_dict,
    unwrap_compiled_model,
)


class _CompiledLikeModule(nn.Module):
    """Small stand-in exposing the same eager-module attribute as OptimizedModule."""

    def __init__(self, original: nn.Module) -> None:
        super().__init__()
        self._orig_mod = original

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._orig_mod(inputs)


def test_pipeline_enables_conservative_cuda_compile_settings() -> None:
    config = load_config().training.torch_compile

    assert config.enabled is True
    assert config.backend == "inductor"
    assert config.mode == "default"
    assert config.fullgraph is False
    assert config.dynamic is False
    assert config.require_cuda is True


def test_compile_is_skipped_on_cpu_when_cuda_is_required(monkeypatch) -> None:
    model = nn.Linear(3, 2)
    called = False

    def unexpected_compile(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("torch.compile must not run for this CPU configuration")

    monkeypatch.setattr(torch, "compile", unexpected_compile)
    compiled = compile_model_for_training(
        model,
        TorchCompileConfig(enabled=True, require_cuda=True),
        torch.device("cpu"),
    )

    assert compiled is model
    assert called is False


def test_compile_receives_configured_options(monkeypatch) -> None:
    model = nn.Linear(3, 2)
    wrapper = _CompiledLikeModule(model)
    received = {}

    def fake_compile(module, **kwargs):
        received["module"] = module
        received.update(kwargs)
        return wrapper

    monkeypatch.setattr(torch, "compile", fake_compile)
    config = TorchCompileConfig(
        enabled=True,
        backend="inductor",
        mode="reduce-overhead",
        fullgraph=False,
        dynamic=True,
        require_cuda=False,
    )

    compiled = compile_model_for_training(model, config, torch.device("cpu"))

    assert compiled is wrapper
    assert received == {
        "module": model,
        "backend": "inductor",
        "mode": "reduce-overhead",
        "fullgraph": False,
        "dynamic": True,
    }


def test_compiled_wrapper_uses_eager_checkpoint_schema() -> None:
    original = nn.Linear(3, 2)
    wrapper = _CompiledLikeModule(original)
    saved = {name: value.detach().clone() for name, value in uncompiled_state_dict(wrapper).items()}

    assert unwrap_compiled_model(wrapper) is original
    assert set(saved) == {"weight", "bias"}
    assert all(not name.startswith("_orig_mod.") for name in saved)

    with torch.no_grad():
        original.weight.zero_()
        original.bias.zero_()
    load_uncompiled_state_dict(wrapper, saved)

    assert torch.equal(original.weight, saved["weight"])
    assert torch.equal(original.bias, saved["bias"])
