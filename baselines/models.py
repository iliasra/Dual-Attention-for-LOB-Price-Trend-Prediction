from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def context_features(features: np.ndarray, *, window: int, mode: str) -> np.ndarray:
    """Build cheap causal last-token or last+rolling-mean baseline inputs."""
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("features must have shape [rows, features].")
    if window <= 0 or len(values) < window:
        raise ValueError("window must be positive and no longer than the day.")
    mode = str(mode).strip().lower()
    last = values[window - 1 :]
    if mode == "last":
        return last.copy()
    if mode != "last_mean":
        raise ValueError("context mode must be 'last' or 'last_mean'.")
    cumulative = np.vstack(
        [np.zeros((1, values.shape[1]), dtype=np.float64), np.cumsum(values, axis=0, dtype=np.float64)]
    )
    rolling_sum = cumulative[window:] - cumulative[:-window]
    rolling_mean = (rolling_sum / float(window)).astype(np.float32)
    return np.concatenate([last, rolling_mean], axis=1)


def sampled_context_sequences(features: np.ndarray, *, window: int, steps: int) -> np.ndarray:
    """Return causal, evenly sampled windows without first building a dense sliding view."""
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("features must have shape [rows, features].")
    if window <= 0 or len(values) < window:
        raise ValueError("window must be positive and no longer than the day.")
    if steps <= 1 or steps > window:
        raise ValueError("steps must be in [2, window].")
    end_indices = np.arange(window - 1, len(values), dtype=np.int64)
    offsets = np.rint(np.linspace(-(window - 1), 0, num=steps)).astype(np.int64)
    return values[end_indices[:, None] + offsets[None, :]].copy()


def momentum_signal(
    features: np.ndarray,
    *,
    window: int,
    feature_index: int,
    mode: str,
    lookback: int = 10,
    short_window: int = 5,
    long_window: int = 20,
) -> np.ndarray:
    """Build an ex-ante price-momentum signal aligned with windows ending at time t."""
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("features must have shape [rows, features].")
    if not 0 <= feature_index < values.shape[1]:
        raise ValueError("feature_index is outside the feature matrix.")
    if window <= 1 or len(values) < window:
        raise ValueError("window must be > 1 and no longer than the day.")
    price = values[:, feature_index].astype(np.float64, copy=False)
    end = np.arange(window - 1, len(price), dtype=np.int64)
    mode = str(mode).strip().lower()
    if mode == "difference":
        if not 1 <= lookback < window:
            raise ValueError("lookback must be in [1, window).")
        signal = price[end] - price[end - lookback]
    elif mode == "ma_crossover":
        if not 1 <= short_window < long_window <= window:
            raise ValueError("momentum windows must satisfy 1 <= short < long <= window.")
        cumulative = np.concatenate([[0.0], np.cumsum(price, dtype=np.float64)])
        short_mean = (cumulative[end + 1] - cumulative[end + 1 - short_window]) / float(short_window)
        long_mean = (cumulative[end + 1] - cumulative[end + 1 - long_window]) / float(long_window)
        signal = short_mean - long_mean
    else:
        raise ValueError("momentum mode must be 'difference' or 'ma_crossover'.")
    return np.asarray(signal, dtype=np.float32)


class BaselineHead(nn.Module):
    """Linear or one-hidden-layer baseline for classification/regression."""

    def __init__(self, input_dim: int, output_dim: int, *, hidden_dim: int | None = None) -> None:
        super().__init__()
        if hidden_dim is None:
            self.network = nn.Linear(input_dim, output_dim)
        else:
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class LSTMBaseline(nn.Module):
    """Small sequence baseline using only the final recurrent state."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0 or hidden_dim <= 0 or num_layers <= 0:
            raise ValueError("LSTM dimensions and layer count must be positive.")
        effective_dropout = float(dropout) if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("LSTM inputs must have shape [batch, time, features].")
        sequence, _state = self.lstm(inputs)
        return self.output(sequence[:, -1])
