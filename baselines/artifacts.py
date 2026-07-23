from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


ARTIFACT_SCHEMA_VERSION = 1


def save_baseline_artifact(path: Path, payload: dict[str, Any]) -> None:
    """Persist a trusted local fitted-baseline bundle for inference-only evaluation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"artifact_schema_version": ARTIFACT_SCHEMA_VERSION, **payload}
    with path.open("wb") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_baseline_artifact(path: Path) -> dict[str, Any]:
    """Load a bundle produced by :func:`save_baseline_artifact`.

    Pickle artifacts must only be loaded from a trusted training run.
    """
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or payload.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported baseline artifact schema in {path}.")
    return payload
