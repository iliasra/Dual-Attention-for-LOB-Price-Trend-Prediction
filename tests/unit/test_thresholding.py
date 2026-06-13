from __future__ import annotations

import numpy as np
import pytest

from thresholding import (
    DirectionalThresholdSelection,
    _is_better_selection,
    apply_directional_threshold_policy,
    directional_macro_f1_from_predictions,
    optimize_directional_thresholds,
    optimize_precision_floor_thresholds,
    optimize_top_quantile_thresholds,
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


def test_directional_threshold_policy_uses_logit_margin_delta_for_double_hits() -> None:
    probabilities = np.asarray(
        [
            [0.65, 0.0, 0.64],
            [0.65, 0.0, 0.64],
            [0.64, 0.0, 0.65],
        ],
        dtype=np.float32,
    )

    low_delta = apply_directional_threshold_policy(
        probabilities,
        threshold_down=0.6,
        threshold_up=0.6,
        down_id=0,
        neutral_id=1,
        up_id=2,
        delta=0.0,
    )
    high_delta = apply_directional_threshold_policy(
        probabilities,
        threshold_down=0.6,
        threshold_up=0.6,
        down_id=0,
        neutral_id=1,
        up_id=2,
        delta=0.2,
    )

    assert low_delta.tolist() == [0, 0, 2]
    assert high_delta.tolist() == [1, 1, 1]


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


def test_optimize_directional_thresholds_can_use_macro_f1_score() -> None:
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
    candidates = np.asarray([0.2, 0.4, 0.6, 0.8])

    directional_selection = optimize_directional_thresholds(
        probabilities,
        targets,
        down_candidates=candidates,
        up_candidates=candidates,
        down_id=0,
        neutral_id=1,
        up_id=2,
        score="directional_macro_f1",
    )
    macro_selection = optimize_directional_thresholds(
        probabilities,
        targets,
        down_candidates=candidates,
        up_candidates=candidates,
        down_id=0,
        neutral_id=1,
        up_id=2,
        score="macro_f1",
    )

    assert (directional_selection.threshold_down, directional_selection.threshold_up) == pytest.approx((0.2, 0.2))
    assert (macro_selection.threshold_down, macro_selection.threshold_up) == pytest.approx((0.2, 0.8))
    assert directional_selection.score == pytest.approx(0.45)
    assert macro_selection.score == pytest.approx(5 / 9)


def test_optimize_directional_thresholds_can_use_tailored_score() -> None:
    probabilities = np.asarray(
        [
            [0.90, 0.05, 0.05],
            [0.80, 0.10, 0.10],
            [0.70, 0.20, 0.10],
            [0.10, 0.20, 0.70],
        ],
        dtype=np.float32,
    )
    targets = np.asarray([0, 0, 1, 2])

    selection = optimize_directional_thresholds(
        probabilities,
        targets,
        down_candidates=np.asarray([0.6]),
        up_candidates=np.asarray([0.6]),
        down_id=0,
        neutral_id=1,
        up_id=2,
        score="tailored_score",
        monitor_params={
            "base_metric": "val_macro_f1",
            "lambda_ece": 0.0,
            "lambda_rate": 0.5,
        },
    )

    assert selection.threshold_down == pytest.approx(0.6)
    assert selection.threshold_up == pytest.approx(0.6)
    assert selection.rate_penalty == pytest.approx(0.25)
    assert selection.score_details["tailored_base_metric"] == "val_macro_f1"
    assert selection.score_details["tailored_base_value"] == pytest.approx(0.6)
    assert selection.score_details["tailored_lambda_rate"] == pytest.approx(0.5)
    assert selection.score == pytest.approx(0.6 - 0.5 * 0.25)


def test_tailored_threshold_score_can_change_selected_pair() -> None:
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
    candidates = np.asarray([0.2, 0.4, 0.6, 0.8])

    macro_selection = optimize_directional_thresholds(
        probabilities,
        targets,
        down_candidates=candidates,
        up_candidates=candidates,
        down_id=0,
        neutral_id=1,
        up_id=2,
        score="macro_f1",
    )
    tailored_selection = optimize_directional_thresholds(
        probabilities,
        targets,
        down_candidates=candidates,
        up_candidates=candidates,
        down_id=0,
        neutral_id=1,
        up_id=2,
        score="tailored_score",
        monitor_params={
            "base_metric": "val_macro_f1",
            "lambda_ece": 0.0,
            "lambda_rate": 0.5,
        },
    )

    assert (macro_selection.threshold_down, macro_selection.threshold_up) == pytest.approx((0.2, 0.8))
    assert (tailored_selection.threshold_down, tailored_selection.threshold_up) == pytest.approx((0.2, 0.2))
    assert tailored_selection.rate_penalty < macro_selection.rate_penalty


def test_optimize_directional_thresholds_refines_around_best_region() -> None:
    probabilities = np.asarray(
        [
            [0.56, 0.34, 0.10],
            [0.57, 0.33, 0.10],
            [0.10, 0.34, 0.56],
            [0.10, 0.33, 0.57],
            [0.52, 0.38, 0.10],
            [0.10, 0.38, 0.52],
        ],
        dtype=np.float32,
    )
    targets = np.asarray([0, 0, 2, 2, 1, 1])

    coarse = optimize_directional_thresholds(
        probabilities,
        targets,
        down_candidates=np.asarray([0.5, 0.6]),
        up_candidates=np.asarray([0.5, 0.6]),
        down_id=0,
        neutral_id=1,
        up_id=2,
    )
    refined = optimize_directional_thresholds(
        probabilities,
        targets,
        down_candidates=np.asarray([0.5, 0.6]),
        up_candidates=np.asarray([0.5, 0.6]),
        down_id=0,
        neutral_id=1,
        up_id=2,
        refinement_steps=(0.01, 0.005),
    )

    assert coarse.score < 1.0
    assert refined.score == pytest.approx(1.0)
    assert 0.53 <= refined.threshold_down <= 0.56
    assert 0.53 <= refined.threshold_up <= 0.56
    assert refined.n_candidates > coarse.n_candidates
    assert len(refined.stage_summaries) == 3
    assert refined.stage_summaries[0]["name"] == "coarse"
    assert refined.stage_summaries[-1]["name"] == "refine_0.005"


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


def test_optimize_precision_floor_thresholds_maximizes_recall_under_floor() -> None:
    probabilities = np.asarray(
        [
            [0.1, 0.0, 0.90],
            [0.1, 0.0, 0.80],
            [0.1, 0.0, 0.70],
            [0.8, 0.0, 0.20],
            [0.7, 0.0, 0.10],
        ],
        dtype=np.float32,
    )
    targets = np.asarray([2, 2, 1, 0, 1])

    selection = optimize_precision_floor_thresholds(
        probabilities,
        targets,
        down_candidates=np.asarray([0.5, 0.75]),
        up_candidates=np.asarray([0.5, 0.75, 0.85]),
        down_precision_floor=0.5,
        up_precision_floor=0.67,
        down_id=0,
        neutral_id=1,
        up_id=2,
    )

    assert selection.threshold_up == pytest.approx(0.75)
    assert selection.threshold_down == pytest.approx(0.75)
    assert selection.up_enabled is True
    assert selection.down_enabled is True
    assert selection.selection_details["up"]["precision"] == pytest.approx(1.0)
    assert selection.selection_details["up"]["recall"] == pytest.approx(1.0)


def test_optimize_precision_floor_thresholds_disables_class_when_floor_is_unreachable() -> None:
    probabilities = np.asarray(
        [
            [0.9, 0.0, 0.1],
            [0.8, 0.0, 0.1],
            [0.1, 0.0, 0.9],
        ],
        dtype=np.float32,
    )
    targets = np.asarray([1, 0, 2])

    selection = optimize_precision_floor_thresholds(
        probabilities,
        targets,
        down_candidates=np.asarray([0.5]),
        up_candidates=np.asarray([0.5]),
        down_precision_floor=0.9,
        up_precision_floor=0.9,
        down_id=0,
        neutral_id=1,
        up_id=2,
    )

    assert selection.down_enabled is False
    assert selection.threshold_down is None
    assert selection.up_enabled is True
    assert selection.selection_details["down"]["fallback"] == "disabled_no_candidate_meets_precision_floor"


def test_optimize_top_quantile_thresholds_selects_top_probability_mass() -> None:
    probabilities = np.asarray(
        [
            [0.90, 0.00, 0.10],
            [0.80, 0.00, 0.20],
            [0.70, 0.00, 0.30],
            [0.10, 0.00, 0.90],
            [0.20, 0.00, 0.80],
            [0.30, 0.00, 0.70],
        ],
        dtype=np.float32,
    )
    targets = np.asarray([0, 0, 1, 2, 2, 1])

    selection = optimize_top_quantile_thresholds(
        probabilities,
        targets,
        down_quantile=0.5,
        up_quantile=0.5,
        down_id=0,
        neutral_id=1,
        up_id=2,
    )

    assert selection.threshold_down == pytest.approx(0.7)
    assert selection.threshold_up == pytest.approx(0.7)
    assert selection.selection_details["down"]["requested_count"] == 3
    assert selection.selection_details["up"]["selected_count"] == 3
    assert selection.n_candidates == 2


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
