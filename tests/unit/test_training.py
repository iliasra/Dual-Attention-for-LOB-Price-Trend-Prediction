from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pytest
import torch
import torch.nn as nn

from configuration import load_config
from training import (
    ClassificationMetricAccumulator,
    EvaluationResult,
    LobTrainer,
    class_weights_from_class_counts,
    class_weights_from_sequence_labels,
)


@pytest.fixture()
def artifact_dir(request: pytest.FixtureRequest) -> Path:
    path = Path(__file__).resolve().parent / ".test_artifacts" / request.node.name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


@pytest.mark.filterwarnings("ignore:Detected call of.*lr_scheduler\\.step.*:UserWarning")
def test_lob_trainer_stops_after_patience_without_val_improvement(
    artifact_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config().training
    config.epochs = 10
    config.early_stopping_patience = 2
    config.monitor = "val_loss"
    config.monitor_mode = "min"
    config.model_dir = str(artifact_dir)
    trainer = LobTrainer(config)
    metrics = ClassificationMetricAccumulator._zero_metrics(num_classes=3)
    validation_losses = iter([1.0, 1.1, 1.2, 0.5])

    def fake_train_epoch(*args, **kwargs) -> EvaluationResult:
        return EvaluationResult(loss=0.25, metrics=metrics)

    def fake_evaluate(*args, **kwargs) -> EvaluationResult:
        return EvaluationResult(loss=next(validation_losses), metrics=metrics)

    monkeypatch.setattr(trainer, "_run_epoch", fake_train_epoch)
    monkeypatch.setattr(trainer, "evaluate", fake_evaluate)

    _, history = trainer.fit(nn.Linear(1, 3), train_loader=[], val_loader=[])

    assert len(history) == 3
    assert [result.val_loss for result in history] == [1.0, 1.1, 1.2]
    assert config.best_model_path.exists()


@pytest.mark.filterwarnings("ignore:Detected call of.*lr_scheduler\\.step.*:UserWarning")
def test_lob_trainer_fit_uses_only_train_and_validation_loaders(
    artifact_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config().training
    config.epochs = 3
    config.early_stopping_patience = 0
    config.monitor = "val_loss"
    config.monitor_mode = "min"
    config.model_dir = str(artifact_dir)
    trainer = LobTrainer(config)
    metrics = ClassificationMetricAccumulator._zero_metrics(num_classes=3)
    validation_losses = iter([1.0, 0.9, 1.1])
    val_loader = object()
    evaluated_loaders: list[object] = []

    def fake_train_epoch(*args, **kwargs) -> EvaluationResult:
        return EvaluationResult(loss=0.25, metrics=metrics)

    def fake_evaluate(*args, **kwargs) -> EvaluationResult:
        data_loader = kwargs["data_loader"]
        evaluated_loaders.append(data_loader)
        if data_loader is val_loader:
            return EvaluationResult(loss=next(validation_losses), metrics=metrics)
        raise AssertionError("Unexpected data loader passed to evaluate.")

    monkeypatch.setattr(trainer, "_run_epoch", fake_train_epoch)
    monkeypatch.setattr(trainer, "evaluate", fake_evaluate)

    _, history = trainer.fit(nn.Linear(1, 3), train_loader=[], val_loader=val_loader)

    assert evaluated_loaders == [val_loader, val_loader, val_loader]
    assert [result.val_loss for result in history] == [1.0, 0.9, 1.1]
    assert [result.test_loss for result in history] == [None, None, None]


@pytest.mark.filterwarnings("ignore:Detected call of.*lr_scheduler\\.step.*:UserWarning")
def test_lob_trainer_sets_epoch_on_train_sampler(
    artifact_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummySampler:
        def __init__(self) -> None:
            self.epochs: list[int] = []

        def set_epoch(self, epoch: int) -> None:
            self.epochs.append(epoch)

    class DummyLoader:
        def __init__(self) -> None:
            self.sampler = DummySampler()

    config = load_config().training
    config.epochs = 3
    config.early_stopping_patience = 0
    config.monitor = "val_loss"
    config.monitor_mode = "min"
    config.model_dir = str(artifact_dir)
    trainer = LobTrainer(config)
    metrics = ClassificationMetricAccumulator._zero_metrics(num_classes=3)
    validation_losses = iter([1.0, 0.9, 0.8])
    train_loader = DummyLoader()

    def fake_train_epoch(*args, **kwargs) -> EvaluationResult:
        return EvaluationResult(loss=0.25, metrics=metrics)

    def fake_evaluate(*args, **kwargs) -> EvaluationResult:
        return EvaluationResult(loss=next(validation_losses), metrics=metrics)

    monkeypatch.setattr(trainer, "_run_epoch", fake_train_epoch)
    monkeypatch.setattr(trainer, "evaluate", fake_evaluate)

    trainer.fit(nn.Linear(1, 3), train_loader=train_loader, val_loader=[])

    assert train_loader.sampler.epochs == [0, 1, 2]


@pytest.mark.filterwarnings("ignore:Detected call of.*lr_scheduler\\.step.*:UserWarning")
def test_lob_trainer_can_monitor_directional_macro_f1(
    artifact_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config().training
    config.epochs = 3
    config.early_stopping_patience = 0
    config.monitor = "val_directional_macro_f1"
    config.monitor_mode = "max"
    config.model_dir = str(artifact_dir)
    trainer = LobTrainer(config)
    train_metrics = ClassificationMetricAccumulator._zero_metrics(num_classes=3)
    validation_scores = iter([0.1, 0.8, 0.4])

    def fake_train_epoch(*args, **kwargs) -> EvaluationResult:
        return EvaluationResult(loss=0.25, metrics=train_metrics)

    def fake_evaluate(*args, **kwargs) -> EvaluationResult:
        metrics = ClassificationMetricAccumulator._zero_metrics(num_classes=3)
        metrics.directional_macro_f1 = next(validation_scores)
        return EvaluationResult(loss=1.0, metrics=metrics)

    monkeypatch.setattr(trainer, "_run_epoch", fake_train_epoch)
    monkeypatch.setattr(trainer, "evaluate", fake_evaluate)

    _, history = trainer.fit(nn.Linear(1, 3), train_loader=[], val_loader=[])

    assert [result.val_metrics.directional_macro_f1 for result in history if result.val_metrics] == [0.1, 0.8, 0.4]
    assert config.best_model_path.exists()


def test_class_weights_from_sequence_labels_uses_balanced_clipped_weights() -> None:
    weights, counts = class_weights_from_sequence_labels(
        np.asarray([0, 0, 0, 0, 1, 1, 2]),
        num_classes=3,
        beta=0.25,
        min_weight=0.5,
        max_weight=3.0,
    )

    raw = 7 / (3 * np.asarray([4, 2, 1], dtype=float))
    expected = raw ** 0.25
    expected = expected / expected.mean()
    expected = np.clip(expected, 0.5, 3.0)

    assert counts == [4, 2, 1]
    assert weights == pytest.approx(expected.tolist())


def test_directional_macro_f1_averages_down_and_up_only() -> None:
    accumulator = ClassificationMetricAccumulator(device=torch.device("cpu"))
    logits = torch.tensor(
        [
            [4.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 4.0],
            [4.0, 0.0, 0.0],
        ]
    )
    targets = torch.tensor([0, 0, 1, 1, 2, 2])

    accumulator.update(logits, targets)
    metrics = accumulator.compute()

    down_f1 = metrics.per_class_f1[0]
    up_f1 = metrics.per_class_f1[2]
    assert metrics.directional_macro_f1 == pytest.approx((down_f1 + up_f1) / 2.0)
    assert metrics.directional_macro_f1 != pytest.approx(metrics.macro_f1)


def test_class_weights_from_class_counts_uses_sampled_counts() -> None:
    weights, counts = class_weights_from_class_counts(
        [2, 4, 2],
        beta=0.25,
        min_weight=0.5,
        max_weight=3.0,
    )

    raw = 8 / (3 * np.asarray([2, 4, 2], dtype=float))
    expected = raw ** 0.25
    expected = expected / expected.mean()
    expected = np.clip(expected, 0.5, 3.0)

    assert counts == [2, 4, 2]
    assert weights == pytest.approx(expected.tolist())


def test_class_weights_from_class_counts_uses_configurable_clip_bounds() -> None:
    weights, _ = class_weights_from_class_counts(
        [100, 1, 1],
        beta=1.0,
        min_weight=0.8,
        max_weight=1.5,
    )

    assert min(weights) >= 0.8
    assert max(weights) <= 1.5
