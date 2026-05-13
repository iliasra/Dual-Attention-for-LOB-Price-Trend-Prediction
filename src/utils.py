from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def to_python_type(value: Any) -> Any:
    """Turn NumPy scalars and arrays into YAML-friendly native Python values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def set_global_seed(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible runs."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:  # pragma: no cover - torch is optional for preprocessing-only use.
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def torch_generator_from_seed(seed: int) -> Any:
    """Create a CPU torch.Generator seeded for deterministic DataLoader shuffling."""
    import torch

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def seed_torch_worker(worker_id: int) -> None:
    """Seed NumPy and Python RNGs inside a PyTorch DataLoader worker."""
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def deep_update(original: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge nested dictionaries."""
    for key, value in new.items():
        if key in original and isinstance(original[key], dict) and isinstance(value, dict):
            deep_update(original[key], value)
        else:
            original[key] = value
    return original


def load_yaml(file_path: str | Path) -> dict[str, Any]:
    path = Path(file_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def save_yaml(file_path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, default_flow_style=False, sort_keys=True)


def append_to_yaml(file_path: str | Path, data_to_append: dict[str, Any]) -> None:
    """Append or update entries in a YAML file."""
    existing_data = load_yaml(file_path)
    clean_data = {
        key: {
            nested_key: to_python_type(nested_value)
            for nested_key, nested_value in value.items()
        }
        if isinstance(value, dict)
        else to_python_type(value)
        for key, value in data_to_append.items()
    }
    save_yaml(file_path, deep_update(existing_data, clean_data))
