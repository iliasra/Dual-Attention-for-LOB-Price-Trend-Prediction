from __future__ import annotations

import numpy as np
import pytest

from pr_metrics import (
    average_precision,
    best_f1_threshold,
    per_class_average_precision,
    per_class_pr_auc,
    per_class_ranking_metrics,
    per_class_roc_auc,
    precision_recall_auc,
    precision_recall_curve,
    roc_auc,
    roc_curve,
)


def test_average_precision_matches_manual_example() -> None:
    scores = np.asarray([0.9, 0.8, 0.7, 0.1])
    positives = np.asarray([True, False, True, False])

    ap = average_precision(scores, positives)

    assert ap == pytest.approx(0.5 * 1.0 + 0.5 * (2.0 / 3.0))


def test_precision_recall_curve_selects_best_f1_threshold() -> None:
    scores = np.asarray([0.9, 0.8, 0.7, 0.1])
    positives = np.asarray([True, False, True, False])

    curve = precision_recall_curve(scores, positives)
    threshold = best_f1_threshold(curve)

    assert curve["threshold"].tolist() == pytest.approx([0.9, 0.8, 0.7, 0.1])
    assert threshold["threshold"] == pytest.approx(0.7)
    assert threshold["f1"] == pytest.approx(0.8)


def test_precision_recall_auc_is_trapezoidal_and_bounded() -> None:
    scores = np.asarray([0.9, 0.8, 0.7, 0.1])
    positives = np.asarray([True, False, True, False])

    auc = precision_recall_auc(precision_recall_curve(scores, positives))

    assert auc == pytest.approx(0.5 + 0.5 * ((0.5 + (2.0 / 3.0)) / 2.0))
    assert 0.0 <= auc <= 1.0


def test_roc_auc_is_trapezoidal_and_bounded() -> None:
    scores = np.asarray([0.9, 0.8, 0.7, 0.1])
    positives = np.asarray([True, False, True, False])

    curve = roc_curve(scores, positives)
    auc = roc_auc(scores, positives)

    assert curve["threshold"].tolist() == pytest.approx([0.9, 0.8, 0.7, 0.1])
    assert auc == pytest.approx(0.75)
    assert 0.0 <= auc <= 1.0


def test_per_class_average_precision_is_bounded() -> None:
    probabilities = np.asarray(
        [
            [0.9, 0.1, 0.0],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.3, 0.5, 0.2],
        ]
    )
    targets = np.asarray([0, 1, 2, 1])

    values = per_class_average_precision(probabilities, targets, num_classes=3)

    assert len(values) == 3
    assert all(0.0 <= value <= 1.0 for value in values)
    assert values[0] == pytest.approx(1.0)


def test_per_class_roc_auc_is_bounded() -> None:
    probabilities = np.asarray(
        [
            [0.9, 0.1, 0.0],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.3, 0.5, 0.2],
        ]
    )
    targets = np.asarray([0, 1, 2, 1])

    values = per_class_roc_auc(probabilities, targets, num_classes=3)

    assert len(values) == 3
    assert all(0.0 <= value <= 1.0 for value in values)
    assert values[0] == pytest.approx(1.0)


def test_per_class_pr_auc_is_bounded() -> None:
    probabilities = np.asarray(
        [
            [0.9, 0.1, 0.0],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.3, 0.5, 0.2],
        ]
    )
    targets = np.asarray([0, 1, 2, 1])

    values = per_class_pr_auc(probabilities, targets, num_classes=3)

    assert len(values) == 3
    assert all(0.0 <= value <= 1.0 for value in values)
    assert values[0] == pytest.approx(1.0)


def test_per_class_ranking_metrics_returns_curves_and_all_scores() -> None:
    probabilities = np.asarray(
        [
            [0.9, 0.1, 0.0],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.3, 0.5, 0.2],
        ],
        dtype=np.float32,
    )
    targets = np.asarray([0, 1, 2, 1])

    metrics = per_class_ranking_metrics(
        probabilities,
        targets,
        num_classes=3,
        class_names=["down", "neutral", "up"],
        include_curves=True,
    )

    assert metrics["pr_ap"][0] == pytest.approx(1.0)
    assert metrics["pr_auc"][0] == pytest.approx(1.0)
    assert metrics["roc_auc"][0] == pytest.approx(1.0)
    assert set(metrics["pr_curves"]) == {"down", "neutral", "up"}
