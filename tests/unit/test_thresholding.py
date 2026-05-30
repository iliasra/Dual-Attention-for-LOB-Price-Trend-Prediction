from __future__ import annotations

import numpy as np
import pytest

from thresholding import (
    DirectionalThresholdSelection,
    _is_better_selection,
    apply_directional_threshold_policy,
    directional_macro_f1_from_predictions,
    optimize_directional_thresholds,
    threshold_candidates,
    thresholded_metric_summary,
)


def test_directional_threshold_policy_handles_neutral_and_tie_break() -> None:
    probabilities = np.asarray(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.2, 0.7],
            [0.4, 0.4, 0.2],
            [0.65, 0.0, 0.62],
            [0.62, 0.0, 0.66],
        ],
        dtype=np.float32,
    )

    predictions = apply_directional_threshold_policy(
        probabilities,
        threshold_down=0.6,
        threshold_up=0.6,
        down_id=0,
        neutral_id=1,
        up_id=2,
    )

    assert predictions.tolist() == [0, 2, 1, 0, 2]


def test_threshold_candidates_are_inclusive() -> None:
    candidates = threshold_candidates(0.05, 0.95, 0.05)

    assert len(candidates) == 19
    assert candidates[0] == pytest.approx(0.05)
    assert candidates[-1] == pytest.approx(0.95)


def test_directional_macro_f1_from_predictions() -> None:
    targets = np.asarray([0, 0, 2, 2, 1])
    predictions = np.asarray([0, 1, 2, 1, 1])

    score = directional_macro_f1_from_predictions(targets, predictions, down_id=0, up_id=2)

    assert score == pytest.approx((2 / 3 + 2 / 3) / 2)


def test_optimize_directional_thresholds_finds_best_pair() -> None:
    probabilities = np.asarray(
        [
            [0.7, 0.2, 0.1],
            [0.55, 0.4, 0.05],
            [0.05, 0.2, 0.7],
            [0.05, 0.4, 0.55],
            [0.2, 0.7, 0.1],
        ],
        dtype=np.float32,
    )
    targets = np.asarray([0, 0, 2, 2, 1])

    selection = optimize_directional_thresholds(
        probabilities,
        targets,
        down_candidates=np.asarray([0.5, 0.6]),
        up_candidates=np.asarray([0.5, 0.6]),
        down_id=0,
        neutral_id=1,
        up_id=2,
    )

    assert selection.threshold_down == pytest.approx(0.5)
    assert selection.threshold_up == pytest.approx(0.5)
    assert selection.score == pytest.approx(1.0)
    assert selection.n_candidates == 4


def test_optimize_directional_thresholds_prefers_rate_penalty_on_score_tie() -> None:
    probabilities = np.asarray(
        [
            [0.23198403, 0.55470207, 0.2133139],
            [0.45777693, 0.13530646, 0.40691661],
            [0.16729683, 0.21569457, 0.6170086],
        ],
        dtype=np.float32,
    )
    targets = np.asarray([2, 2, 1])

    selection = optimize_directional_thresholds(
        probabilities,
        targets,
        down_candidates=np.asarray([0.3]),
        up_candidates=np.asarray([0.3, 0.7]),
        down_id=0,
        neutral_id=1,
        up_id=2,
    )

    assert selection.threshold_down == pytest.approx(0.3)
    assert selection.threshold_up == pytest.approx(0.3)
    assert selection.score == pytest.approx(0.0)
    assert selection.rate_penalty == pytest.approx(2 / 3)


def test_threshold_selection_prefers_min_precision_before_high_thresholds() -> None:
    best = DirectionalThresholdSelection(
        threshold_down=0.9,
        threshold_up=0.9,
        score=0.5,
        rate_penalty=0.1,
        min_directional_precision=0.2,
        n_candidates=2,
    )
    candidate = DirectionalThresholdSelection(
        threshold_down=0.1,
        threshold_up=0.1,
        score=0.5,
        rate_penalty=0.1,
        min_directional_precision=0.3,
        n_candidates=2,
    )

    assert _is_better_selection(candidate, best)


def test_optimize_directional_thresholds_prefers_high_thresholds_on_full_tie() -> None:
    probabilities = np.asarray(
        [
            [0.1, 0.8, 0.1],
            [0.2, 0.7, 0.1],
        ],
        dtype=np.float32,
    )
    targets = np.asarray([1, 1])

    selection = optimize_directional_thresholds(
        probabilities,
        targets,
        down_candidates=np.asarray([0.3, 0.5]),
        up_candidates=np.asarray([0.3, 0.5]),
        down_id=0,
        neutral_id=1,
        up_id=2,
    )

    assert selection.threshold_down == pytest.approx(0.5)
    assert selection.threshold_up == pytest.approx(0.5)
    assert selection.score == pytest.approx(0.0)
    assert selection.rate_penalty == pytest.approx(0.0)


def test_thresholded_metric_summary_contains_rates() -> None:
    targets = np.asarray([0, 0, 1, 2])
    predictions = np.asarray([0, 1, 1, 2])

    summary = thresholded_metric_summary(targets, predictions, down_id=0, neutral_id=1, up_id=2)

    assert summary["accuracy"] == pytest.approx(0.75)
    assert summary["pred_rate_neutral"] == pytest.approx(0.5)
    assert summary["true_rate_down"] == pytest.approx(0.5)
    assert summary["up_f1"] == pytest.approx(1.0)
