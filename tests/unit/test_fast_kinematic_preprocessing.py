from __future__ import annotations

import numpy as np

from fast_kinematic_preprocessing import (
    effective_degrees_of_freedom,
    lambda_for_effective_degrees_of_freedom,
    mean_gcv_score,
    optimize_smoothing_lambda_gcv,
)


def test_effective_degrees_of_freedom_decreases_with_lambda() -> None:
    df_low_lambda = effective_degrees_of_freedom(window=20, n_basis=8, smoothing_lambda=0.0)
    df_high_lambda = effective_degrees_of_freedom(window=20, n_basis=8, smoothing_lambda=1e6)

    assert df_high_lambda < df_low_lambda


def test_lambda_for_effective_degrees_of_freedom_matches_target() -> None:
    target_df = 5.0
    smoothing_lambda = lambda_for_effective_degrees_of_freedom(
        target_df=target_df,
        window=20,
        n_basis=8,
    )

    selected_df = effective_degrees_of_freedom(
        window=20,
        n_basis=8,
        smoothing_lambda=smoothing_lambda,
    )

    np.testing.assert_allclose(selected_df, target_df, atol=1e-5)


def test_gcv_optimization_returns_finite_lambda_and_score() -> None:
    ticks = np.arange(28, dtype=np.float64)
    values = np.column_stack(
        [
            100.0 + 0.1 * ticks + 0.001 * ticks**2,
            101.0 + 0.08 * ticks + 0.002 * ticks**2,
        ]
    )

    result = optimize_smoothing_lambda_gcv(
        values_by_day=[values],
        window=12,
        n_basis=6,
        max_df=5.0,
        chunk_size=4,
        n_df_candidates=4,
    )
    gcv_score, selected_df = mean_gcv_score(
        values_by_day=[values],
        window=12,
        n_basis=6,
        smoothing_lambda=result.smoothing_lambda,
        chunk_size=4,
    )

    assert result.smoothing_lambda >= 0.0
    assert np.isfinite(result.gcv_score)
    assert np.isfinite(result.effective_df)
    np.testing.assert_allclose(result.gcv_score, gcv_score)
    np.testing.assert_allclose(result.effective_df, selected_df)


def test_gcv_score_can_center_price_windows_before_smoothing() -> None:
    ticks = np.arange(24, dtype=np.float64)
    values = np.column_stack(
        [
            10_000.0 + 0.5 * ticks,
            10_001.0 + 0.25 * ticks,
        ]
    )
    centers = values[:, 0] - 1.0
    shifted_values = values + 1_000_000.0
    shifted_centers = centers + 1_000_000.0

    base_score, base_df = mean_gcv_score(
        values_by_day=[values],
        centers_by_day=[centers],
        scale=100.0,
        window=8,
        n_basis=6,
        smoothing_lambda=1.0,
        chunk_size=4,
    )
    shifted_score, shifted_df = mean_gcv_score(
        values_by_day=[shifted_values],
        centers_by_day=[shifted_centers],
        scale=100.0,
        window=8,
        n_basis=6,
        smoothing_lambda=1.0,
        chunk_size=4,
    )

    np.testing.assert_allclose(shifted_score, base_score)
    np.testing.assert_allclose(shifted_df, base_df)
