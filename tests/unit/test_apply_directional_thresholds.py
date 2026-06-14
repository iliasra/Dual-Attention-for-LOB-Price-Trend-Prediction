from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from apply_directional_thresholds import fit_thresholds
from configuration import load_config


def test_replay_fit_thresholds_supports_tailored_score() -> None:
    config = load_config()
    config.training.directional_thresholds.enabled = True
    config.training.directional_thresholds.method = "joint_up_down"
    config.training.directional_thresholds.score = "tailored_score"
    config.training.directional_thresholds.min_threshold = 0.2
    config.training.directional_thresholds.max_threshold = 0.2
    config.training.directional_thresholds.step = 0.2
    config.training.directional_thresholds.delta = 0.0
    config.training.monitor_params.base_metric = "val_macro_f1"
    config.training.monitor_params.lambda_ece = 0.0
    config.training.monitor_params.lambda_rate = 0.5
    probabilities = np.asarray(
        [
            [0.184, 0.288, 0.528],
            [0.501, 0.192, 0.307],
            [0.257, 0.474, 0.269],
            [0.260, 0.590, 0.151],
            [0.405, 0.055, 0.541],
        ],
        dtype=np.float32,
    )
    targets = np.asarray([1, 2, 0, 0, 2])

    selection, refinements = fit_thresholds(config, targets, probabilities)

    assert refinements == (0.01, 0.005, 0.002, 0.001)
    assert (selection.threshold_down, selection.threshold_up) == pytest.approx((0.2, 0.2))
    assert selection.score_details["tailored_lambda_rate"] == pytest.approx(0.5)
