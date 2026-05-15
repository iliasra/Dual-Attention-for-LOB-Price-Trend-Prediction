from __future__ import annotations

import pytest
import torch

from compatibility import autocast_context, cuda_device_index, make_grad_scaler, resolve_torch_device, torch_device_type


class _FakeContext:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return False


def test_resolve_torch_device_keeps_cpu() -> None:
    device = resolve_torch_device("cpu")

    assert device.type == "cpu"
    assert torch_device_type(device) == "cpu"


def test_disabled_grad_scaler_acts_as_passthrough() -> None:
    tensor = torch.tensor(1.0, requires_grad=True)
    scaler = make_grad_scaler("cpu", enabled=False)

    assert scaler.scale(tensor) is tensor
    assert scaler.is_enabled() is False


def test_autocast_context_can_be_disabled_on_cpu() -> None:
    with autocast_context("cpu", enabled=False):
        result = torch.ones(1) + 1

    assert result.item() == 2


def test_grad_scaler_falls_back_when_modern_amp_api_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCudaGradScaler:
        def __init__(self, enabled: bool) -> None:
            self.enabled = enabled

        def is_enabled(self) -> bool:
            return self.enabled

    monkeypatch.delattr(torch.amp, "GradScaler", raising=False)
    monkeypatch.setattr(torch.cuda.amp, "GradScaler", FakeCudaGradScaler)

    scaler = make_grad_scaler("cuda", enabled=True)

    assert isinstance(scaler, FakeCudaGradScaler)
    assert scaler.is_enabled() is True


def test_autocast_falls_back_when_modern_amp_api_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(torch.amp, "autocast", raising=False)
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: _FakeContext(*args, **kwargs))

    context = autocast_context("cuda", enabled=True)

    assert isinstance(context, _FakeContext)
    assert context.kwargs == {"device_type": "cuda", "enabled": True}


def test_resolve_cuda_device_adds_index_for_strict_cuda_memory_apis() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")

    device = resolve_torch_device("cuda")

    assert device.type == "cuda"
    assert device.index is not None
    free, total = torch.cuda.mem_get_info(cuda_device_index(device))
    assert total >= free
