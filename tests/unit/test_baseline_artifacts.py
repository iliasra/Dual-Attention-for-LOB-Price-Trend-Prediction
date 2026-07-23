from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import torch
import pytest

from baselines.artifacts import load_baseline_artifact, save_baseline_artifact
from baselines.models import BaselineHead
from scripts.evaluate_baseline import infer_frozen_baseline


@pytest.fixture()
def artifact_dir(request: pytest.FixtureRequest) -> Path:
    path = Path(__file__).resolve().parent / ".test_artifacts" / request.node.name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def test_frozen_mlp_artifact_round_trip_is_inference_only(artifact_dir: Path) -> None:
    model = BaselineHead(2, 2, hidden_dim=3, hidden_layers=1, dropout=0.0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.1)
    path = artifact_dir / "mlp.pkl"
    save_baseline_artifact(
        path,
        {
            "model": "mlp",
            "task": "action_value_regression",
            "output_dim": 2,
            "window": 2,
            "context": "last",
            "input_dim": 2,
            "standardizer_mean": np.asarray([1.0, 2.0]),
            "standardizer_scale": np.asarray([2.0, 4.0]),
            "state_dict": model.state_dict(),
            "hidden_dim": 3,
            "hidden_layers": 1,
            "dropout": 0.0,
        },
    )

    artifact = load_baseline_artifact(path)
    predictions, probabilities = infer_frozen_baseline(
        artifact,
        np.asarray([[1.0, 2.0], [3.0, 6.0]], dtype=np.float32),
        batch_size=2,
        device="cpu",
    )

    assert predictions.shape == (2, 2)
    assert probabilities is None
    assert np.isfinite(predictions).all()
