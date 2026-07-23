from __future__ import annotations

from pathlib import Path
import json
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import baselines.run_baselines as baseline_runner

from baselines.models import (
    BaselineHead,
    LSTMBaseline,
    RecurrentBaseline,
    context_features,
    dense_mlp_parameter_count,
    momentum_signal,
    resolve_mlp_hidden_dim,
    sampled_context_sequences,
)
from baselines.run_baselines import (
    baseline_regression_loss,
    fit_classical_model,
    infer_baseline_task,
    label_persistence_predictions,
    load_delayed_target_split,
    load_split,
    main as run_baselines_main,
    momentum_predictions,
    no_skill_predictions,
    resolve_feature_indices,
    resolve_row_caps,
    time_of_day_predictions,
    train_head,
)


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


def test_momentum_features_are_resolved_by_schema_name(artifact_dir: Path) -> None:
    (artifact_dir / "feature_schema.yaml").write_text(
        "ordered_feature_columns:\n- bid_price_1\n- ask_price_1\n- causal_midprice_momentum_ticks_20\n",
        encoding="utf-8",
    )

    assert resolve_feature_indices(
        artifact_dir,
        feature_names=["causal_midprice_momentum_ticks_20"],
        feature_index=None,
        feature_schema=None,
    ) == (2,)
    assert resolve_feature_indices(
        artifact_dir,
        feature_names=["bid_price_1", "ask_price_1"],
        feature_index=None,
        feature_schema=None,
    ) == (0, 1)


def test_direct_momentum_uses_precomputed_signal_without_differencing() -> None:
    values = np.asarray([[1.0], [2.0], [4.0], [8.0]], dtype=np.float32)

    np.testing.assert_allclose(
        momentum_signal(values, window=3, feature_index=0, mode="direct"),
        [4.0, 8.0],
    )


def test_baseline_head_supports_two_action_values() -> None:
    model = BaselineHead(4, 2, hidden_dim=8)

    outputs = model(torch.ones((3, 4)))

    assert outputs.shape == (3, 2)
    assert torch.isfinite(outputs).all()


def test_multilayer_mlp_resolves_width_near_parameter_budget() -> None:
    target = 25_000
    width = resolve_mlp_hidden_dim(12, 3, hidden_layers=3, target_parameters=target)
    model = BaselineHead(12, 3, hidden_dim=width, hidden_layers=3, dropout=0.2)
    actual = sum(parameter.numel() for parameter in model.parameters())

    assert actual == dense_mlp_parameter_count(12, 3, width, 3)
    assert abs(actual - target) <= abs(dense_mlp_parameter_count(12, 3, width + 1, 3) - target)
    if width > 1:
        assert abs(actual - target) <= abs(dense_mlp_parameter_count(12, 3, width - 1, 3) - target)
    assert sum(isinstance(module, torch.nn.Dropout) for module in model.modules()) == 3


def test_row_cap_defaults_and_legacy_alias() -> None:
    modern = SimpleNamespace(max_rows=None, max_train_rows=None, max_eval_rows=None)
    legacy = SimpleNamespace(max_rows=123, max_train_rows=None, max_eval_rows=None)
    override = SimpleNamespace(max_rows=123, max_train_rows=10, max_eval_rows=None)

    assert resolve_row_caps(modern) == (200_000, 0)
    assert resolve_row_caps(legacy) == (123, 123)
    assert resolve_row_caps(override) == (10, 123)


def test_classification_output_dim_is_configured_not_inferred_from_held_out() -> None:
    regression, output_dim = infer_baseline_task(
        np.asarray([0, 1, 1], dtype=np.int64),
        np.asarray([2, 2], dtype=np.int64),
        num_classes=3,
    )

    assert regression is False
    assert output_dim == 3
    with pytest.raises(ValueError, match="outside configured"):
        infer_baseline_task(
            np.asarray([0, 1], dtype=np.int64),
            np.asarray([2], dtype=np.int64),
            num_classes=2,
        )


def test_neural_regression_loss_supports_huber_delta_and_mse() -> None:
    outputs = torch.tensor([[0.0, 0.0]])
    targets = torch.tensor([[1.0, 3.0]])

    huber = baseline_regression_loss(outputs, targets, loss_name="huber", huber_delta=0.5)
    mse = baseline_regression_loss(outputs, targets, loss_name="mse", huber_delta=0.5)

    assert huber.item() == pytest.approx(torch.nn.functional.huber_loss(outputs, targets, delta=0.5).item())
    assert mse.item() == pytest.approx(5.0)


def test_no_skill_classification_returns_train_prevalence_probabilities() -> None:
    predictions, probabilities = no_skill_predictions(
        np.asarray([0, 1, 1, 1, 2], dtype=np.int64),
        evaluation_rows=4,
        output_dim=3,
    )

    np.testing.assert_array_equal(predictions, np.full(4, 1))
    assert probabilities is not None
    np.testing.assert_allclose(probabilities, np.tile([0.2, 0.6, 0.2], (4, 1)))


def test_time_of_day_classification_uses_train_bins_and_laplace_smoothing() -> None:
    predictions, probabilities = time_of_day_predictions(
        np.asarray([60.0, 120.0, 180.0, 1_000.0]),
        np.asarray([0, 0, 1, 2], dtype=np.int64),
        np.asarray([300.0, 1_200.0, 4_000.0, 86_700.0]),
        bin_minutes=15.0,
        laplace_alpha=1.0,
        output_dim=3,
    )

    assert probabilities is not None
    np.testing.assert_allclose(probabilities[0], [3.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0])
    np.testing.assert_allclose(probabilities[1], [0.25, 0.25, 0.5])
    np.testing.assert_allclose(probabilities[2], [3.0 / 7.0, 2.0 / 7.0, 2.0 / 7.0])
    np.testing.assert_allclose(probabilities[3], probabilities[0])
    np.testing.assert_array_equal(predictions, [0, 2, 0, 0])


def test_time_of_day_regression_uses_global_train_fallback_for_unseen_bin() -> None:
    train_values = np.asarray([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]], dtype=np.float32)
    predictions, probabilities = time_of_day_predictions(
        np.asarray([60.0, 120.0, 1_000.0]),
        train_values,
        np.asarray([300.0, 4_000.0, 86_700.0]),
        bin_minutes=15.0,
        laplace_alpha=1.0,
        output_dim=2,
    )

    assert probabilities is None
    np.testing.assert_allclose(predictions[0], [2.0, 3.0])
    np.testing.assert_allclose(predictions[1], train_values.mean(axis=0))
    np.testing.assert_allclose(predictions[2], predictions[0])


def test_label_persistence_probabilities_are_fit_from_train_transitions() -> None:
    predictions, probabilities = label_persistence_predictions(
        np.asarray([0, 0, 1, 1], dtype=np.int64),
        np.asarray([0, 0, 2, 2], dtype=np.int64),
        np.asarray([0, 1, 2], dtype=np.int64),
        output_dim=3,
        laplace_alpha=1.0,
    )

    np.testing.assert_allclose(probabilities[0], [0.6, 0.2, 0.2])
    np.testing.assert_allclose(probabilities[1], [0.2, 0.2, 0.6])
    np.testing.assert_allclose(probabilities[2], [1.0 / 3.0] * 3)
    np.testing.assert_array_equal(predictions, [0, 2, 0])


@pytest.mark.parametrize("regression", [False, True])
def test_neural_baseline_logs_every_training_batch(regression: bool) -> None:
    class FakeTracker:
        def __init__(self) -> None:
            self.payloads: list[dict[str, object]] = []

        def log_training_step(self, payload: dict[str, object]) -> None:
            self.payloads.append(payload)

    x = np.arange(20, dtype=np.float32).reshape(10, 2)
    if regression:
        y = np.column_stack([x[:, 0], -x[:, 0]]).astype(np.float32)
        output_dim = 2
    else:
        y = np.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], dtype=np.int64)
        output_dim = 3
    args = SimpleNamespace(device="cpu", batch_size=4, learning_rate=1e-3, epochs=2)
    tracker = FakeTracker()

    global_step = train_head(
        BaselineHead(2, output_dim, hidden_dim=4),
        x,
        y,
        regression=regression,
        args=args,
        wandb_tracker=tracker,  # type: ignore[arg-type]
    )

    assert global_step == 6
    assert len(tracker.payloads) == 6
    assert [payload["global_step"] for payload in tracker.payloads] == list(range(1, 7))
    assert all("train_loss_step" in payload for payload in tracker.payloads)


def test_xgboost_baseline_logs_every_boosting_round(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeXGB:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def fit(self, *_args: object, **_kwargs: object) -> None:
            return None

        def evals_result(self) -> dict[str, dict[str, list[float]]]:
            return {
                "validation_0": {"mlogloss": [1.0, 0.8]},
                "validation_1": {"mlogloss": [1.1, 0.9]},
            }

        def predict_proba(self, values: np.ndarray) -> np.ndarray:
            return np.full((len(values), 3), 1.0 / 3.0, dtype=np.float32)

    class FakeTracker:
        log_training_steps_enabled = True

        def __init__(self) -> None:
            self.payloads: list[dict[str, object]] = []

        def log_metrics(self, payload: dict[str, object]) -> None:
            self.payloads.append(payload)

    monkeypatch.setitem(sys.modules, "xgboost", SimpleNamespace(XGBClassifier=FakeXGB, XGBRegressor=FakeXGB))
    tracker = FakeTracker()
    args = SimpleNamespace(
        n_estimators=2,
        max_depth=3,
        xgb_learning_rate=0.1,
        xgb_subsample=1.0,
        xgb_colsample_bytree=1.0,
        n_jobs=1,
        seed=42,
    )
    train_x = np.ones((6, 2), dtype=np.float32)
    train_y = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
    validation_x = np.ones((3, 2), dtype=np.float32)
    validation_y = np.asarray([0, 1, 2], dtype=np.int64)

    predictions, probabilities, fitted = fit_classical_model(
        "xgboost",
        train_x,
        train_y,
        validation_x,
        validation_y,
        regression=False,
        args=args,
        wandb_tracker=tracker,  # type: ignore[arg-type]
    )

    assert predictions.shape == (3,)
    assert probabilities is not None
    assert fitted is not None
    assert [payload["global_step"] for payload in tracker.payloads] == [1, 2]
    assert tracker.payloads[-1]["validation/xgboost_mlogloss"] == 0.9


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


def test_sequence_loader_applies_endpoint_mask_without_removing_context(artifact_dir: Path) -> None:
    split_dir = artifact_dir / "train"
    split_dir.mkdir()
    features = np.arange(6, dtype=np.float32)[:, None]
    labels = np.arange(6, dtype=np.int64)
    np.save(split_dir / "day_features.npy", features)
    np.save(split_dir / "day_labels.npy", labels)
    np.save(split_dir / "day_supervision_mask.npy", [False, False, True, False, True, False])

    inputs, targets = load_split(
        artifact_dir,
        "train",
        window=3,
        context="last",
        max_rows=0,
        seed=1,
    )

    np.testing.assert_array_equal(inputs[:, 0], [2.0, 4.0])
    np.testing.assert_array_equal(targets, [2, 4])


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
    ["no_skill", "time_of_day", "momentum", "momentum_ma", "label_persistence", "linear", "mlp", "lstm"],
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
        np.save(split_dir / "day_times.npy", 34_200.0 + time)
        np.save(split_dir / "day_labels.npy", labels)
    (artifact_dir / "feature_schema.yaml").write_text(
        "ordered_feature_columns:\n- bid_price_1\n- ask_price_1\n",
        encoding="utf-8",
    )
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
        if model_name == "momentum":
            argv.extend(["--momentum-feature-name", "bid_price_1"])
        else:
            argv.extend(
                [
                    "--momentum-feature-name",
                    "bid_price_1",
                    "--momentum-feature-name",
                    "ask_price_1",
                ]
            )
    if model_name == "label_persistence":
        argv.extend(["--label-lag", "2", "--label-horizon", "2"])
    if model_name == "lstm":
        argv.extend(["--lstm-steps", "3", "--hidden-dim", "4"])
    monkeypatch.setattr(sys, "argv", argv)

    run_baselines_main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["model"] == model_name
    assert payload["validation_rows"] > 0
    assert "metrics" in payload


def test_runner_uses_full_eval_default_and_reports_resolved_mlp_parameters(
    artifact_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for split, rows in (("train", 30), ("validation", 40)):
        split_dir = artifact_dir / split
        split_dir.mkdir()
        time = np.arange(rows, dtype=np.float32)
        np.save(split_dir / "day_features.npy", np.column_stack([time, np.sin(time)]).astype(np.float32))
        np.save(split_dir / "day_labels.npy", (np.arange(rows) % 3).astype(np.int64))
    output = artifact_dir / "parameter_matched_mlp.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_baselines.py",
            "--sequence-dir",
            str(artifact_dir),
            "--output",
            str(output),
            "--model",
            "mlp",
            "--window",
            "4",
            "--max-train-rows",
            "7",
            "--mlp-layers",
            "2",
            "--mlp-dropout",
            "0.1",
            "--target-parameters",
            "1000",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--device",
            "cpu",
        ],
    )

    run_baselines_main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["train_rows"] == 7
    assert payload["evaluation_rows"] == 37
    assert payload["parameters"]["max_eval_rows"] == 0
    assert payload["parameters"]["resolved_hidden_dim"] is not None
    assert payload["parameters"]["actual_parameter_count"] == payload["model_parameters"]["total"]
    assert payload["model_parameters"]["total"] > 0


def test_baseline_runner_uses_configured_wandb_for_steps_and_final_artifact(
    artifact_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTracker:
        enabled = True
        log_training_steps_enabled = True

        def __init__(self) -> None:
            self.steps: list[dict[str, object]] = []
            self.metrics: list[dict[str, object]] = []
            self.artifacts: list[list[Path]] = []
            self.exit_code: int | None = None

        def log_training_step(self, payload: dict[str, object]) -> None:
            self.steps.append(payload)

        def log_metrics(self, payload: dict[str, object]) -> None:
            self.metrics.append(payload)

        def log_artifact_files(self, *, name: str, artifact_type: str, paths: list[Path]) -> None:
            del name, artifact_type
            self.artifacts.append(paths)

        def finish(self, *, exit_code: int | None = None) -> None:
            self.exit_code = exit_code

    for split in ("train", "validation"):
        split_dir = artifact_dir / split
        split_dir.mkdir()
        features = np.arange(48, dtype=np.float32).reshape(24, 2)
        labels = np.asarray(([0, 1, 2] * 8), dtype=np.int64)
        np.save(split_dir / "day_features.npy", features)
        np.save(split_dir / "day_labels.npy", labels)
    output = artifact_dir / "mlp_wandb.json"
    tracker = FakeTracker()
    monkeypatch.setattr(
        baseline_runner,
        "load_config",
        lambda _path: SimpleNamespace(tracking=SimpleNamespace(wandb=SimpleNamespace())),
    )
    monkeypatch.setattr(baseline_runner.WandbTracker, "init", lambda *_args, **_kwargs: tracker)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_baselines.py",
            "--config",
            "dummy.yaml",
            "--run-stem",
            "baseline-test",
            "--sequence-dir",
            str(artifact_dir),
            "--output",
            str(output),
            "--model",
            "mlp",
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
        ],
    )

    run_baselines_main()

    assert len(tracker.steps) == 3
    assert any("selected/validation_macro_f1" in payload for payload in tracker.metrics)
    assert tracker.artifacts == [[output]]
    assert tracker.exit_code == 0
