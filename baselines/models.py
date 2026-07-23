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
    if mode == "direct":
        signal = price[end]
    elif mode == "difference":
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
        raise ValueError("momentum mode must be 'direct', 'difference' or 'ma_crossover'.")
    return np.asarray(signal, dtype=np.float32)


def dense_mlp_parameter_count(input_dim: int, output_dim: int, hidden_dim: int, hidden_layers: int) -> int:
    """Return the exact parameter count of an equal-width dense MLP."""
    if min(input_dim, output_dim, hidden_dim, hidden_layers) <= 0:
        raise ValueError("MLP dimensions and hidden_layers must be positive.")
    first_layer = input_dim * hidden_dim + hidden_dim
    hidden_tail = (hidden_layers - 1) * (hidden_dim * hidden_dim + hidden_dim)
    output_layer = hidden_dim * output_dim + output_dim
    return int(first_layer + hidden_tail + output_layer)


def resolve_mlp_hidden_dim(
    input_dim: int,
    output_dim: int,
    *,
    hidden_layers: int,
    target_parameters: int,
) -> int:
    """Choose the positive integer width closest to a parameter budget."""
    if min(input_dim, output_dim, hidden_layers, target_parameters) <= 0:
        raise ValueError("MLP dimensions, hidden_layers and target_parameters must be positive.")

    # P(h) = (L - 1) h^2 + (input + output + L) h + output.
    linear_coefficient = input_dim + output_dim + hidden_layers
    if hidden_layers == 1:
        continuous_width = (target_parameters - output_dim) / float(linear_coefficient)
    else:
        quadratic_coefficient = hidden_layers - 1
        discriminant = linear_coefficient**2 + 4 * quadratic_coefficient * (target_parameters - output_dim)
        continuous_width = (
            -linear_coefficient + np.sqrt(max(float(discriminant), 0.0))
        ) / (2.0 * quadratic_coefficient)

    center = max(1, int(round(continuous_width)))
    candidates = range(max(1, center - 2), center + 3)
    return min(
        candidates,
        key=lambda width: (
            abs(dense_mlp_parameter_count(input_dim, output_dim, width, hidden_layers) - target_parameters),
            width,
        ),
    )


class BaselineHead(nn.Module):
    """Linear or configurable equal-width MLP for classification/regression."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int | None = None,
        hidden_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("Baseline input_dim and output_dim must be positive.")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("MLP dropout must be in [0, 1).")
        if hidden_dim is None:
            self.network = nn.Linear(input_dim, output_dim)
        else:
            if hidden_dim <= 0 or hidden_layers <= 0:
                raise ValueError("MLP hidden_dim and hidden_layers must be positive.")
            layers: list[nn.Module] = []
            current_dim = input_dim
            for _ in range(hidden_layers):
                layers.extend([nn.Linear(current_dim, hidden_dim), nn.GELU()])
                if dropout > 0.0:
                    layers.append(nn.Dropout(float(dropout)))
                current_dim = hidden_dim
            layers.append(nn.Linear(current_dim, output_dim))
            self.network = nn.Sequential(*layers)

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


class RecurrentBaseline(nn.Module):
    """Small vanilla-RNN or GRU baseline using the last recurrent output."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        cell: str,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0 or hidden_dim <= 0 or num_layers <= 0:
            raise ValueError("Recurrent dimensions and layer count must be positive.")
        cell = str(cell).strip().lower()
        recurrent_type = {"rnn": nn.RNN, "gru": nn.GRU}.get(cell)
        if recurrent_type is None:
            raise ValueError("cell must be 'rnn' or 'gru'.")
        effective_dropout = float(dropout) if num_layers > 1 else 0.0
        self.cell = cell
        self.recurrent = recurrent_type(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("Recurrent inputs must have shape [batch, time, features].")
        sequence, _state = self.recurrent(inputs)
        return self.output(sequence[:, -1])
