from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pytest
import torch.nn as nn

from configuration import load_config
from training import (
    ClassificationMetricAccumulator,
    EvaluationResult,
    LobTrainer,
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
def test_lob_trainer_evaluates_test_loader_only_once_on_best_model(
    artifact_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config().training
    config.epochs = 3
    config.early_stopping_patience = 0
    config.model_dir = str(artifact_dir)
    trainer = LobTrainer(config)
    metrics = ClassificationMetricAccumulator._zero_metrics(num_classes=3)
    validation_losses = iter([1.0, 0.9, 1.1])
    val_loader = object()
    test_loader = object()
    test_call_count = 0

    def fake_train_epoch(*args, **kwargs) -> EvaluationResult:
        return EvaluationResult(loss=0.25, metrics=metrics)

    def fake_evaluate(*args, **kwargs) -> EvaluationResult:
        nonlocal test_call_count
        data_loader = kwargs["data_loader"]
        if data_loader is test_loader:
            test_call_count += 1
            return EvaluationResult(loss=0.77, metrics=metrics)
        if data_loader is val_loader:
            return EvaluationResult(loss=next(validation_losses), metrics=metrics)
        raise AssertionError("Unexpected data loader passed to evaluate.")

    monkeypatch.setattr(trainer, "_run_epoch", fake_train_epoch)
    monkeypatch.setattr(trainer, "evaluate", fake_evaluate)

    _, history = trainer.fit(nn.Linear(1, 3), train_loader=[], val_loader=val_loader, test_loader=test_loader)

    assert test_call_count == 1
    assert [result.val_loss for result in history] == [1.0, 0.9, 1.1]
    assert [result.test_loss for result in history] == [None, 0.77, None]


def test_class_weights_from_sequence_labels_uses_balanced_clipped_weights() -> None:
    weights, counts = class_weights_from_sequence_labels(
        np.asarray([0, 0, 0, 0, 1, 1, 2]),
        num_classes=3,
        gamma_mode=True,
    )

    raw = 7 / (3 * np.asarray([4, 2, 1], dtype=float))
    expected = np.sqrt(raw)
    expected = expected / expected.mean()
    expected = np.clip(expected, 0.5, 3.0)

    assert counts == [4, 2, 1]
    assert weights == pytest.approx(expected.tolist())
