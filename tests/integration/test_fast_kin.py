from __future__ import annotations

import numpy as np
import pytest

from configuration import load_config
from fast_kinematic_preprocessing import PenalizedBSplineKinematicTokenizer


FAST_CONFIG_PATHS = ("price_kinematic", "volume_kinematic")


def _configured_tokenizer(fast_config_path: str) -> tuple[PenalizedBSplineKinematicTokenizer, int, float]:
    config = load_config()
    preprocessing = config.preprocessing
    fast_config = getattr(preprocessing, fast_config_path).fast
    window = preprocessing.snapshot_window

    tokenizer = PenalizedBSplineKinematicTokenizer(
        window=window,
        n_basis=fast_config.n_basis,
        smoothing_lambda=fast_config.smoothing_lambda,
        eval_at=fast_config.eval_at,
        chunk_size=preprocessing.kinematic_tokenization.chunk_size,
        dtype=np.float64,
    )
    return tokenizer, window, fast_config.eval_at


def _tokens_for_signal(
    signal: np.ndarray,
    fast_config_path: str,
) -> tuple[np.ndarray, int, float]:
    tokenizer, window, eval_at = _configured_tokenizer(fast_config_path)
    tokens = tokenizer.transform_values(signal[:, None])[:, 0, :]
    assert np.isfinite(tokens).all()
    return tokens, window, eval_at


@pytest.mark.parametrize("fast_config_path", FAST_CONFIG_PATHS)
def test_configured_fast_kinematic_constant_signal_has_zero_derivatives(fast_config_path: str) -> None:
    tokenizer, window, _ = _configured_tokenizer(fast_config_path)
    signal = np.full(window + 7, 42.0, dtype=np.float64)

    tokens = tokenizer.transform_values(signal[:, None])[:, 0, :]

    np.testing.assert_allclose(tokens[:, 1:], 0.0, atol=1e-6)


@pytest.mark.parametrize("fast_config_path", FAST_CONFIG_PATHS)
def test_configured_fast_kinematic_linear_signal_has_expected_derivatives(fast_config_path: str) -> None:
    slope = 0.125
    _, window, _ = _configured_tokenizer(fast_config_path)
    ticks = np.arange(window + 7, dtype=np.float64)
    signal = 17.0 + slope * ticks

    tokens, _, _ = _tokens_for_signal(signal, fast_config_path)

    np.testing.assert_allclose(tokens[:, 1], slope, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(tokens[:, 2:], 0.0, atol=1e-6)


@pytest.mark.parametrize("fast_config_path", FAST_CONFIG_PATHS)
def test_configured_fast_kinematic_quadratic_signal_has_expected_curvature(fast_config_path: str) -> None:
    acceleration = 0.02
    _, window, _ = _configured_tokenizer(fast_config_path)
    ticks = np.arange(window + 7, dtype=np.float64)
    signal = 17.0 + 0.125 * ticks + 0.5 * acceleration * ticks**2

    tokens, window, eval_at = _tokens_for_signal(signal, fast_config_path)
    expected_velocity = 0.125 + acceleration * (np.arange(len(tokens), dtype=np.float64) + eval_at * (window - 1))

    np.testing.assert_allclose(tokens[:, 1], expected_velocity, rtol=2e-2, atol=2e-3)
    np.testing.assert_allclose(tokens[:, 2], acceleration, rtol=2e-2, atol=2e-4)
    np.testing.assert_allclose(tokens[:, 3], 0.0, atol=5e-4)
