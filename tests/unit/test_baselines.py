from __future__ import annotations

from pathlib import Path
import json
import shutil
import sys

import numpy as np
import pytest
import torch

from baselines.models import (
    BaselineHead,
    LSTMBaseline,
    RecurrentBaseline,
    context_features,
    momentum_signal,
    sampled_context_sequences,
)
from baselines.run_baselines import load_delayed_target_split, load_split, main as run_baselines_main, momentum_predictions


@pytest.fixture()
def artifact_dir(request: pytest.FixtureRequest) -> Path:
    path = Path(__file__).resolve().parent / ".test_artifacts" / request.node.name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def test_last_mean_context_is_causal_and_has_expected_shape() -> None:
    values = np.asarray([[1.0], [2.0], [4.0], [8.0]], dtype=np.float32)

    result = context_features(values, window=2, mode="last_mean")

    np.testing.assert_allclose(result, [[2.0, 1.5], [4.0, 3.0], [8.0, 6.0]])


def test_baseline_head_supports_two_action_values() -> None:
    model = BaselineHead(4, 2, hidden_dim=8)

    outputs = model(torch.ones((3, 4)))

    assert outputs.shape == (3, 2)
    assert torch.isfinite(outputs).all()


def test_sampled_lstm_context_is_causal_and_includes_last_snapshot() -> None:
    values = np.arange(1, 7, dtype=np.float32)[:, None]

    sequences = sampled_context_sequences(values, window=4, steps=3)

    np.testing.assert_allclose(sequences[:, :, 0], [[1.0, 2.0, 4.0], [2.0, 3.0, 5.0], [3.0, 4.0, 6.0]])


def test_lstm_baseline_supports_classification_and_regression_shape() -> None:
    model = LSTMBaseline(input_dim=4, output_dim=2, hidden_dim=8)

    outputs = model(torch.ones((3, 5, 4)))

    assert outputs.shape == (3, 2)
    assert torch.isfinite(outputs).all()


@pytest.mark.parametrize("cell", ["rnn", "gru"])
def test_recurrent_baselines_support_classification_and_regression_shape(cell: str) -> None:
    model = RecurrentBaseline(input_dim=4, output_dim=2, cell=cell, hidden_dim=8)

    outputs = model(torch.ones((3, 5, 4)))

    assert outputs.shape == (3, 2)
    assert torch.isfinite(outputs).all()


def test_momentum_difference_and_moving_average_are_causal() -> None:
    values = np.asarray([[1.0], [2.0], [4.0], [8.0], [16.0]], dtype=np.float32)

    difference = momentum_signal(values, window=4, feature_index=0, mode="difference", lookback=2)
    crossover = momentum_signal(
        values,
        window=4,
        feature_index=0,
        mode="ma_crossover",
        short_window=2,
        long_window=4,
    )

    np.testing.assert_allclose(difference, [6.0, 12.0])
    np.testing.assert_allclose(crossover, [2.25, 4.5])


def test_momentum_classifier_fits_neutral_zone_on_train_only() -> None:
    train_signal = np.asarray([-3.0, -0.1, 0.1, 2.0], dtype=np.float32)
    train_targets = np.asarray([2, 1, 1, 0], dtype=np.int64)
    validation_signal = np.asarray([-4.0, 0.0, 3.0], dtype=np.float32)

    predictions, probabilities = momentum_predictions(
        train_signal,
        train_targets,
        validation_signal,
        neutral_quantile="auto",
        up_class=0,
        neutral_class=1,
        down_class=2,
        output_dim=3,
    )

    np.testing.assert_array_equal(predictions, [2, 1, 0])
    assert probabilities is not None
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_momentum_regression_fits_two_train_only_action_values() -> None:
    train_signal = np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)
    train_targets = np.column_stack([1.0 + 2.0 * train_signal, -1.0 - train_signal]).astype(np.float32)

    predictions, probabilities = momentum_predictions(
        train_signal,
        train_targets,
        np.asarray([2.0], dtype=np.float32),
        neutral_quantile="auto",
        up_class=0,
        neutral_class=1,
        down_class=2,
        output_dim=2,
    )

    np.testing.assert_allclose(predictions, [[5.0, -3.0]], atol=1e-6)
    assert probabilities is None


def test_sequence_loader_samples_windows_after_row_selection(artifact_dir: Path) -> None:
    split_dir = artifact_dir / "train"
    split_dir.mkdir()
    features = np.arange(12, dtype=np.float32).reshape(6, 2)
    labels = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
    np.save(split_dir / "day_features.npy", features)
    np.save(split_dir / "day_labels.npy", labels)

    inputs, targets = load_split(
        artifact_dir,
        "train",
        window=4,
        context="last",
        max_rows=0,
        seed=1,
        sequence_steps=3,
    )

    assert inputs.shape == (3, 3, 2)
    np.testing.assert_array_equal(inputs[:, -1], features[3:])
    np.testing.assert_array_equal(targets, labels[3:])


def test_delayed_label_loader_never_crosses_day_boundary(artifact_dir: Path) -> None:
    split_dir = artifact_dir / "validation"
    split_dir.mkdir()
    for day, labels in (("a", [0, 1, 2, 0]), ("b", [2, 2, 1, 0])):
        np.save(split_dir / f"{day}_features.npy", np.ones((4, 1), dtype=np.float32))
        np.save(split_dir / f"{day}_labels.npy", np.asarray(labels, dtype=np.int64))

    delayed, actual = load_delayed_target_split(
        artifact_dir,
        "validation",
        window=2,
        lag=2,
        max_rows=0,
        seed=1,
    )

    np.testing.assert_array_equal(delayed, [0, 1, 2, 2])
    np.testing.assert_array_equal(actual, [2, 0, 1, 0])


@pytest.mark.parametrize(
    "model_name",
    ["no_skill", "momentum", "momentum_ma", "label_persistence", "linear", "mlp", "lstm"],
)
def test_baseline_runner_smoke_for_dependency_free_models(
    artifact_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
) -> None:
    for split in ("train", "validation"):
        split_dir = artifact_dir / split
        split_dir.mkdir()
        time = np.arange(24, dtype=np.float32)
        features = np.column_stack([time, np.sin(time)]).astype(np.float32)
        labels = np.asarray(([0, 1, 2] * 8), dtype=np.int64)
        np.save(split_dir / "day_features.npy", features)
        np.save(split_dir / "day_labels.npy", labels)
    output = artifact_dir / f"{model_name}.json"
    argv = [
        "run_baselines.py",
        "--sequence-dir",
        str(artifact_dir),
        "--output",
        str(output),
        "--model",
        model_name,
        "--window",
        "4",
        "--max-rows",
        "20",
        "--epochs",
        "1",
        "--batch-size",
        "8",
        "--device",
        "cpu",
    ]
    if model_name in {"momentum", "momentum_ma"}:
        argv.extend(
            [
                "--momentum-lookback",
                "2",
                "--momentum-short-window",
                "2",
                "--momentum-long-window",
                "4",
                "--up-class",
                "2",
                "--neutral-class",
                "1",
                "--down-class",
                "0",
            ]
        )
    if model_name == "label_persistence":
        argv.extend(["--label-lag", "2"])
    if model_name == "lstm":
        argv.extend(["--lstm-steps", "3", "--hidden-dim", "4"])
    monkeypatch.setattr(sys, "argv", argv)

    run_baselines_main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["model"] == model_name
    assert payload["validation_rows"] > 0
    assert "metrics" in payload
