from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from monitoring import (
    directional_precision_at_fixed_rate,
    directional_class_ids,
    epoch_monitor_value,
    tailored_score_components,
)


def _metrics() -> SimpleNamespace:
    """Return validation metrics with a known confusion matrix."""
    return SimpleNamespace(
        macro_f1=0.4,
        directional_macro_f1=0.8,
        per_class_expected_calibration_error=[0.2, 0.1, 0.4],
        confusion_matrix=[
            [6, 2, 2],
            [1, 15, 4],
            [3, 2, 5],
        ],
    )


def test_tailored_score_uses_directional_ece_and_class_rates() -> None:
    components = tailored_score_components(
        _metrics(),
        lambda_ece=0.5,
        lambda_rate=0.25,
        label_mapping={-1: 0, 0: 1, 1: 2},
    )

    total = 40
    expected_ece_dir = (0.2 + 0.4) / 2.0
    expected_rate_penalty = abs((6 + 1 + 3) / total - 10 / total) + abs((2 + 4 + 5) / total - 10 / total)
    expected_score = 0.8 - 0.5 * expected_ece_dir - 0.25 * expected_rate_penalty

    assert components.ece_dir == pytest.approx(expected_ece_dir)
    assert components.pred_rate_down == pytest.approx(10 / total)
    assert components.true_rate_down == pytest.approx(10 / total)
    assert components.pred_rate_up == pytest.approx(11 / total)
    assert components.true_rate_up == pytest.approx(10 / total)
    assert components.rate_penalty == pytest.approx(expected_rate_penalty)
    assert components.score == pytest.approx(expected_score)


def test_tailored_score_can_use_macro_f1_as_base_metric() -> None:
    components = tailored_score_components(
        _metrics(),
        lambda_ece=0.5,
        lambda_rate=0.25,
        base_metric="val_macro_f1",
        label_mapping={-1: 0, 0: 1, 1: 2},
    )

    total = 40
    expected_ece_dir = (0.2 + 0.4) / 2.0
    expected_rate_penalty = abs((6 + 1 + 3) / total - 10 / total) + abs((2 + 4 + 5) / total - 10 / total)
    expected_score = 0.4 - 0.5 * expected_ece_dir - 0.25 * expected_rate_penalty

    assert components.base_metric == "val_macro_f1"
    assert components.base_value == pytest.approx(0.4)
    assert components.score == pytest.approx(expected_score)


def test_tailored_score_requires_down_and_up_classes() -> None:
    with pytest.raises(ValueError, match="raw labels -1 and 1"):
        directional_class_ids({-1: 0, 0: 1}, num_classes=3)


def test_epoch_monitor_value_supports_tailored_score() -> None:
    result = SimpleNamespace(val_loss=0.3, val_metrics=_metrics())
    params = SimpleNamespace(lambda_ece=0.5, lambda_rate=0.25)

    value = epoch_monitor_value(
        result,
        monitor="tailored_score",
        monitor_params=params,
        label_mapping={-1: 0, 0: 1, 1: 2},
    )

    assert value == pytest.approx(0.64375)


def test_directional_precision_at_fixed_rate_uses_top_down_and_up_scores_separately() -> None:
    probabilities = np.asarray(
        [
            [0.99, 0.00, 0.01],
            [0.98, 0.00, 0.02],
            [0.97, 0.00, 0.03],
            [0.01, 0.03, 0.96],
            [0.02, 0.03, 0.95],
        ],
        dtype=np.float32,
    )
    targets = np.asarray([0, 0, 0, 1, 2], dtype=np.int64)

    components = directional_precision_at_fixed_rate(
        probabilities,
        targets,
        fixed_rate=0.4,
    )

    assert components.k == 2
    assert components.actual_rate == pytest.approx(0.4)
    assert components.precision == pytest.approx(0.75)


def test_epoch_monitor_value_supports_precision_at_fixed_rate() -> None:
    result = SimpleNamespace(
        val_loss=0.3,
        val_metrics=SimpleNamespace(directional_precision_at_fixed_rate=0.75),
    )

    value = epoch_monitor_value(
        result,
        monitor="precision_at_fixed_rate",
        monitor_params=SimpleNamespace(fixed_rate=0.01),
    )

    assert value == pytest.approx(0.75)
