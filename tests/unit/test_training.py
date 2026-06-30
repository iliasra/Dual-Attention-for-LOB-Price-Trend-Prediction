from __future__ import annotations

import contextlib
from pathlib import Path
import shutil

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from configuration import load_config
from training import (
    ClassificationMetricAccumulator,
    EvaluationResult,
    LobTrainer,
    MovementDirectionAuxiliaryLoss,
    class_weights_from_class_counts,
    class_weights_from_sequence_labels,
    classification_metrics_from_predictions,
)


@pytest.fixture()
def artifact_dir(request: pytest.FixtureRequest) -> Path:
    path = Path(__file__).resolve().parent / ".test_artifacts" / request.node.name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def _training_config():
    config = load_config().training
    config.validate_every_n_batches = "epoch"
    return config


@pytest.mark.filterwarnings("ignore:Detected call of.*lr_scheduler\\.step.*:UserWarning")
def test_lob_trainer_stops_after_patience_without_val_improvement(
    artifact_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _training_config()
    config.epochs = 10
    config.early_stopping_patience = 2
    config.early_stopping_warmup = 0
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
def test_lob_trainer_does_not_check_early_stopping_during_warmup(
    artifact_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _training_config()
    config.epochs = 10
    config.early_stopping_patience = 1
    config.early_stopping_warmup = 3
    config.monitor = "val_loss"
    config.monitor_mode = "min"
    config.model_dir = str(artifact_dir)
    trainer = LobTrainer(config)
    metrics = ClassificationMetricAccumulator._zero_metrics(num_classes=3)
    validation_losses = iter([1.0, 1.1, 1.2, 1.3, 0.5])

    def fake_train_epoch(*args, **kwargs) -> EvaluationResult:
        return EvaluationResult(loss=0.25, metrics=metrics)

    def fake_evaluate(*args, **kwargs) -> EvaluationResult:
        return EvaluationResult(loss=next(validation_losses), metrics=metrics)

    monkeypatch.setattr(trainer, "_run_epoch", fake_train_epoch)
    monkeypatch.setattr(trainer, "evaluate", fake_evaluate)

    _, history = trainer.fit(nn.Linear(1, 3), train_loader=[], val_loader=[])

    assert len(history) == 4
    assert [result.val_loss for result in history] == [1.0, 1.1, 1.2, 1.3]
    assert config.best_model_path.exists()


def test_run_epoch_raises_on_non_finite_logits(artifact_dir: Path) -> None:
    config = _training_config()
    config.device = "cpu"
    config.use_amp = False
    config.model_dir = str(artifact_dir)
    trainer = LobTrainer(config)

    class NonFiniteLogitModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))

        def forward(self, x_batch: torch.Tensor, t_batch: torch.Tensor) -> torch.Tensor:
            logits = torch.zeros((x_batch.shape[0], 3), dtype=x_batch.dtype, device=x_batch.device)
            logits[0, 0] = float("nan")
            return logits + self.weight * 0.0

    model = NonFiniteLogitModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = [
        (
            torch.zeros((2, 3, 2), dtype=torch.float32),
            torch.zeros((2, 3), dtype=torch.float32),
            torch.tensor([0, 1], dtype=torch.long),
        )
    ]

    with pytest.raises(FloatingPointError, match="Non-finite logits"):
        trainer._run_epoch(
            model,
            loader,
            nn.CrossEntropyLoss(),
            optimizer,
            description="BadTrain",
        )


def test_lob_trainer_uses_bf16_amp_when_cuda_supports_it(
    artifact_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _training_config()
    config.device = "cuda"
    config.use_amp = True
    config.model_dir = str(artifact_dir)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", lambda _index: contextlib.nullcontext())

    trainer = LobTrainer(config)

    assert trainer.amp_enabled is True
    assert trainer.amp_dtype is torch.bfloat16
    assert trainer.scaler.is_enabled() is False


@pytest.mark.filterwarnings("ignore:Detected call of.*lr_scheduler\\.step.*:UserWarning")
def test_lob_trainer_fit_uses_only_train_and_validation_loaders(
    artifact_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _training_config()
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
def test_lob_trainer_fit_tracks_validation_ranking_metrics_without_outputs(
    artifact_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _training_config()
    config.epochs = 2
    config.early_stopping_patience = 0
    config.monitor = "val_loss"
    config.monitor_mode = "min"
    config.model_dir = str(artifact_dir)
    trainer = LobTrainer(config)
    metrics = ClassificationMetricAccumulator._zero_metrics(num_classes=3)
    evaluate_flags: list[tuple[bool, bool, bool]] = []

    def fake_train_epoch(*args, **kwargs) -> EvaluationResult:
        return EvaluationResult(loss=0.25, metrics=metrics)

    def fake_evaluate(*args, **kwargs) -> EvaluationResult:
        evaluate_flags.append(
            (
                bool(kwargs.get("collect_outputs", False)),
                bool(kwargs.get("track_pr_metrics", False)),
                bool(kwargs.get("track_expert_usage", False)),
            )
        )
        return EvaluationResult(loss=1.0, metrics=metrics)

    monkeypatch.setattr(trainer, "_run_epoch", fake_train_epoch)
    monkeypatch.setattr(trainer, "evaluate", fake_evaluate)

    trainer.fit(nn.Linear(1, 3), train_loader=[], val_loader=[])

    assert evaluate_flags == [(False, True, False), (False, True, False)]


@pytest.mark.filterwarnings("ignore:Detected call of.*lr_scheduler\\.step.*:UserWarning")
def test_lob_trainer_saves_only_configured_top_k_checkpoints(
    artifact_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _training_config()
    config.epochs = 4
    config.early_stopping_patience = 0
    config.monitor = "val_loss"
    config.monitor_mode = "min"
    config.top_k_checkpoints = 2
    config.device = "cpu"
    config.model_dir = str(artifact_dir)
    trainer = LobTrainer(config)
    metrics = ClassificationMetricAccumulator._zero_metrics(num_classes=3)
    validation_losses = iter([1.0, 0.8, 1.2, 0.7])

    def fake_train_epoch(*args, **kwargs) -> EvaluationResult:
        return EvaluationResult(loss=0.25, metrics=metrics)

    def fake_evaluate(*args, **kwargs) -> EvaluationResult:
        loss = next(validation_losses)
        return EvaluationResult(loss=loss, metrics=metrics)

    monkeypatch.setattr(trainer, "_run_epoch", fake_train_epoch)
    monkeypatch.setattr(trainer, "evaluate", fake_evaluate)

    trainer.fit(nn.Linear(1, 3), train_loader=[], val_loader=[])

    assert [candidate.epoch for candidate in trainer.top_checkpoint_candidates] == [4, 2]
    assert config.checkpoint_path(4).exists()
    assert config.checkpoint_path(2).exists()
    assert not config.checkpoint_path(1).exists()
    assert not config.checkpoint_path(3).exists()
    assert config.best_model_path.exists()


@pytest.mark.filterwarnings("ignore:Detected call of.*lr_scheduler\\.step.*:UserWarning")
def test_lob_trainer_recovers_missing_best_model_from_top_checkpoint(
    artifact_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _training_config()
    config.epochs = 2
    config.early_stopping_patience = 0
    config.monitor = "val_loss"
    config.monitor_mode = "min"
    config.top_k_checkpoints = 1
    config.device = "cpu"
    config.model_dir = str(artifact_dir)
    trainer = LobTrainer(config)
    metrics = ClassificationMetricAccumulator._zero_metrics(num_classes=3)
    validation_losses = iter([1.0, 1.2])
    validation_calls = 0

    def fake_train_epoch(*args, **kwargs) -> EvaluationResult:
        return EvaluationResult(loss=0.25, metrics=metrics)

    def fake_evaluate(*args, **kwargs) -> EvaluationResult:
        nonlocal validation_calls
        validation_calls += 1
        if validation_calls == 2 and config.best_model_path.exists():
            config.best_model_path.unlink()
        return EvaluationResult(loss=next(validation_losses), metrics=metrics)

    monkeypatch.setattr(trainer, "_run_epoch", fake_train_epoch)
    monkeypatch.setattr(trainer, "evaluate", fake_evaluate)

    trainer.fit(nn.Linear(1, 3), train_loader=[], val_loader=[])

    assert config.checkpoint_path(1).exists()
    assert config.best_model_path.exists()
    assert [candidate.epoch for candidate in trainer.top_checkpoint_candidates] == [1]


class TinySequenceClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 3)

    def forward(self, x_batch: torch.Tensor, t_batch: torch.Tensor) -> torch.Tensor:
        del t_batch
        return self.linear(x_batch[:, -1, :])


def _tiny_batches(count: int) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    return [
        (
            torch.full((2, 2, 1), float(index), dtype=torch.float32),
            torch.zeros((2, 2), dtype=torch.float32),
            torch.tensor([0, 2], dtype=torch.long),
        )
        for index in range(count)
    ]


@pytest.mark.filterwarnings("ignore:Detected call of.*lr_scheduler\\.step.*:UserWarning")
def test_lob_trainer_validates_every_n_batches_and_at_epoch_end(
    artifact_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _training_config()
    config.epochs = 1
    config.validate_every_n_batches = 2
    config.early_stopping_patience = 0
    config.monitor = "val_loss"
    config.monitor_mode = "min"
    config.device = "cpu"
    config.use_amp = False
    config.model_dir = str(artifact_dir)
    trainer = LobTrainer(config)
    metrics = ClassificationMetricAccumulator._zero_metrics(num_classes=3)
    validation_losses = iter([1.0, 0.9, 0.8])

    def fake_evaluate(*args, **kwargs) -> EvaluationResult:
        return EvaluationResult(loss=next(validation_losses), metrics=metrics)

    monkeypatch.setattr(trainer, "evaluate", fake_evaluate)

    _, history = trainer.fit(TinySequenceClassifier(), train_loader=_tiny_batches(5), val_loader=[])

    assert [(item.epoch, item.batch_in_epoch, item.global_step, item.validation_index) for item in history] == [
        (1, 2, 2, 1),
        (1, 4, 4, 2),
        (1, 5, 5, 3),
    ]
    assert [item.checkpoint_label for item in history] == [
        "epoch_0001_step_00000002",
        "epoch_0001_step_00000004",
        "epoch_0001_step_00000005",
    ]
    assert config.checkpoint_path(1, global_step=5).exists()
    assert config.best_model_path.exists()


@pytest.mark.filterwarnings("ignore:Detected call of.*lr_scheduler\\.step.*:UserWarning")
def test_lob_trainer_logs_training_step_callback_for_each_batch(
    artifact_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _training_config()
    config.epochs = 1
    config.validate_every_n_batches = 10
    config.early_stopping_patience = 0
    config.monitor = "val_loss"
    config.monitor_mode = "min"
    config.device = "cpu"
    config.use_amp = False
    config.model_dir = str(artifact_dir)
    trainer = LobTrainer(config)
    metrics = ClassificationMetricAccumulator._zero_metrics(num_classes=3)

    def fake_evaluate(*args, **kwargs) -> EvaluationResult:
        return EvaluationResult(loss=1.0, metrics=metrics)

    monkeypatch.setattr(trainer, "evaluate", fake_evaluate)
    logged_steps: list[dict[str, object]] = []

    trainer.fit(
        TinySequenceClassifier(),
        train_loader=_tiny_batches(3),
        val_loader=[],
        training_step_callback=logged_steps.append,
    )

    assert [item["global_step"] for item in logged_steps] == [1, 2, 3]
    assert [item["batch_in_epoch"] for item in logged_steps] == [1, 2, 3]
    assert all("train_loss_step" in item for item in logged_steps)
    assert all("learning_rate" in item for item in logged_steps)


@pytest.mark.filterwarnings("ignore:Detected call of.*lr_scheduler\\.step.*:UserWarning")
def test_lob_trainer_early_stopping_counts_validation_intervals(
    artifact_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _training_config()
    config.epochs = 2
    config.validate_every_n_batches = 2
    config.early_stopping_patience = 2
    config.early_stopping_warmup = 0
    config.monitor = "val_loss"
    config.monitor_mode = "min"
    config.device = "cpu"
    config.use_amp = False
    config.model_dir = str(artifact_dir)
    trainer = LobTrainer(config)
    metrics = ClassificationMetricAccumulator._zero_metrics(num_classes=3)
    validation_losses = iter([1.0, 1.1, 1.2, 0.5])

    def fake_evaluate(*args, **kwargs) -> EvaluationResult:
        return EvaluationResult(loss=next(validation_losses), metrics=metrics)

    monkeypatch.setattr(trainer, "evaluate", fake_evaluate)

    _, history = trainer.fit(TinySequenceClassifier(), train_loader=_tiny_batches(10), val_loader=[])

    assert len(history) == 3
    assert [item.global_step for item in history] == [2, 4, 6]
    assert [item.val_loss for item in history] == [1.0, 1.1, 1.2]


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

    config = _training_config()
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
    config = _training_config()
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


@pytest.mark.filterwarnings("ignore:Detected call of.*lr_scheduler\\.step.*:UserWarning")
def test_lob_trainer_can_monitor_tailored_score(
    artifact_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _training_config()
    config.epochs = 3
    config.early_stopping_patience = 0
    config.monitor = "tailored_score"
    config.monitor_mode = "max"
    config.monitor_params.lambda_ece = 0.5
    config.monitor_params.lambda_rate = 0.5
    config.model_dir = str(artifact_dir)
    trainer = LobTrainer(config)
    train_metrics = ClassificationMetricAccumulator._zero_metrics(num_classes=3)
    validation_scores = iter([0.9, 0.85, 0.7])

    def fake_train_epoch(*args, **kwargs) -> EvaluationResult:
        return EvaluationResult(loss=0.25, metrics=train_metrics)

    def fake_evaluate(*args, **kwargs) -> EvaluationResult:
        model = kwargs["model"]
        metrics = ClassificationMetricAccumulator._zero_metrics(num_classes=3)
        score = next(validation_scores)
        with torch.no_grad():
            model.bias.fill_(score)
        metrics.directional_macro_f1 = score
        metrics.per_class_expected_calibration_error = [0.1, 0.0, 0.1]
        metrics.confusion_matrix = [[8, 1, 1], [1, 18, 1], [1, 1, 8]]
        return EvaluationResult(loss=1.0, metrics=metrics)

    monkeypatch.setattr(trainer, "_run_epoch", fake_train_epoch)
    monkeypatch.setattr(trainer, "evaluate", fake_evaluate)

    model, history = trainer.fit(nn.Linear(1, 3), train_loader=[], val_loader=[])

    assert [result.val_metrics.directional_macro_f1 for result in history if result.val_metrics] == [0.9, 0.85, 0.7]
    assert model.bias[0].item() == pytest.approx(0.9)
    assert config.best_model_path.exists()


@pytest.mark.filterwarnings("ignore:Detected call of.*lr_scheduler\\.step.*:UserWarning")
def test_lob_trainer_can_monitor_precision_at_fixed_rate(
    artifact_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _training_config()
    config.epochs = 3
    config.early_stopping_patience = 0
    config.monitor = "precision_at_fixed_rate"
    config.monitor_mode = "max"
    config.monitor_params.fixed_rate = 0.01
    config.device = "cpu"
    config.model_dir = str(artifact_dir)
    trainer = LobTrainer(config)
    train_metrics = ClassificationMetricAccumulator._zero_metrics(num_classes=3)
    validation_scores = iter([0.2, 0.9, 0.4])

    def fake_train_epoch(*args, **kwargs) -> EvaluationResult:
        return EvaluationResult(loss=0.25, metrics=train_metrics)

    def fake_evaluate(*args, **kwargs) -> EvaluationResult:
        model = kwargs["model"]
        metrics = ClassificationMetricAccumulator._zero_metrics(num_classes=3)
        score = next(validation_scores)
        with torch.no_grad():
            model.bias.fill_(score)
        metrics.directional_precision_at_fixed_rate = score
        metrics.directional_precision_at_fixed_rate_k = 1
        metrics.directional_precision_at_fixed_rate_actual_rate = 0.01
        return EvaluationResult(loss=1.0, metrics=metrics)

    monkeypatch.setattr(trainer, "_run_epoch", fake_train_epoch)
    monkeypatch.setattr(trainer, "evaluate", fake_evaluate)

    model, history = trainer.fit(nn.Linear(1, 3), train_loader=[], val_loader=[])

    assert [
        result.val_metrics.directional_precision_at_fixed_rate
        for result in history
        if result.val_metrics
    ] == [0.2, 0.9, 0.4]
    assert model.bias[0].item() == pytest.approx(0.9)
    assert config.best_model_path.exists()


def test_lob_trainer_uses_configured_adam_optimizer() -> None:
    config = _training_config()
    config.device = "cpu"
    config.optimizer = "adam"
    trainer = LobTrainer(config)

    optimizer = trainer._optimizer(nn.Linear(1, 3))

    assert isinstance(optimizer, torch.optim.Adam)
    assert not isinstance(optimizer, torch.optim.AdamW)


@pytest.mark.filterwarnings("ignore:Detected call of.*lr_scheduler\\.step.*:UserWarning")
def test_lob_trainer_min_delta_filters_tiny_val_loss_improvements(
    artifact_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _training_config()
    config.epochs = 2
    config.early_stopping_patience = 0
    config.early_stopping_min_delta = 0.002
    config.monitor = "val_loss"
    config.monitor_mode = "min"
    config.device = "cpu"
    config.model_dir = str(artifact_dir)
    trainer = LobTrainer(config)
    metrics = ClassificationMetricAccumulator._zero_metrics(num_classes=3)
    validation_losses = iter([1.0, 0.999])

    def fake_train_epoch(*args, **kwargs) -> EvaluationResult:
        return EvaluationResult(loss=0.25, metrics=metrics)

    def fake_evaluate(*args, **kwargs) -> EvaluationResult:
        model = kwargs["model"]
        loss = next(validation_losses)
        with torch.no_grad():
            model.bias.fill_(loss)
        return EvaluationResult(loss=loss, metrics=metrics)

    monkeypatch.setattr(trainer, "_run_epoch", fake_train_epoch)
    monkeypatch.setattr(trainer, "evaluate", fake_evaluate)

    model, _history = trainer.fit(nn.Linear(1, 3), train_loader=[], val_loader=[])

    assert model.bias[0].item() == pytest.approx(1.0)


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


def test_metric_accumulator_computes_directional_precision_at_fixed_rate() -> None:
    accumulator = ClassificationMetricAccumulator(
        device=torch.device("cpu"),
        track_pr_metrics=True,
        directional_precision_fixed_rate=0.5,
    )
    logits = torch.tensor(
        [
            [5.0, 0.0, 0.0],
            [0.0, 0.2, 4.0],
            [4.5, 0.0, 0.0],
            [0.0, 0.1, 4.3],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 1, 2, 2])

    accumulator.update(logits, targets)
    metrics = accumulator.compute()

    assert metrics.directional_precision_at_fixed_rate_k == 2
    assert metrics.directional_precision_at_fixed_rate_actual_rate == pytest.approx(0.5)
    assert metrics.directional_precision_at_fixed_rate == pytest.approx(0.5)


def test_metric_accumulator_skips_non_finite_logits_and_invalid_targets() -> None:
    accumulator = ClassificationMetricAccumulator(device=torch.device("cpu"))
    logits = torch.tensor(
        [
            [4.0, 0.0, 0.0],
            [float("nan"), 0.0, 0.0],
            [0.0, float("inf"), 0.0],
            [0.0, 0.0, 4.0],
            [0.0, 4.0, 0.0],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 1, 1, 2, 3])

    accumulator.update(logits, targets)
    metrics = accumulator.compute()

    confusion = np.asarray(metrics.confusion_matrix)
    assert confusion.sum() == 2
    assert confusion[0, 0] == 1
    assert confusion[2, 2] == 1


def test_per_class_ece_uses_one_vs_rest_probabilities() -> None:
    accumulator = ClassificationMetricAccumulator(device=torch.device("cpu"), num_calibration_bins=2)
    probabilities = torch.tensor(
        [
            [0.2, 0.8],
            [0.4, 0.6],
            [0.6, 0.4],
            [0.8, 0.2],
        ],
        dtype=torch.float32,
    )
    logits = torch.log(probabilities)
    targets = torch.tensor([0, 1, 0, 1])

    accumulator.update(logits, targets)
    metrics = accumulator.compute()

    assert metrics.per_class_expected_calibration_error == pytest.approx([0.2, 0.2])


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


def test_movement_direction_auxiliary_loss_uses_label_mapping_and_masks_neutral() -> None:
    criterion = MovementDirectionAuxiliaryLoss(
        up_id=2,
        neutral_id=1,
        down_id=0,
        movement_weight=0.5,
        direction_weight=0.25,
        movement_pos_weight=2.0,
        direction_class_weights=torch.tensor([1.0, 3.0]),
    )
    class_logits = torch.zeros((3, 3))
    targets = torch.tensor([0, 1, 2])
    auxiliary_outputs = {
        "movement_logit": torch.tensor([0.0, 0.0, 0.0]),
        "direction_logits": torch.tensor([[0.0, 1.0], [3.0, 0.0], [1.0, 0.0]]),
    }

    loss = criterion(
        class_logits=class_logits,
        targets=targets,
        auxiliary_outputs=auxiliary_outputs,
    )

    expected_movement = torch.nn.functional.binary_cross_entropy_with_logits(
        auxiliary_outputs["movement_logit"],
        torch.tensor([1.0, 0.0, 1.0]),
        pos_weight=torch.tensor(2.0),
    )
    expected_direction = torch.nn.functional.cross_entropy(
        auxiliary_outputs["direction_logits"][[0, 2]],
        torch.tensor([1, 0]),
        weight=torch.tensor([1.0, 3.0]),
    )
    assert float(loss) == pytest.approx(float(0.5 * expected_movement + 0.25 * expected_direction))


def test_movement_direction_auxiliary_loss_handles_batch_without_directional_examples() -> None:
    criterion = MovementDirectionAuxiliaryLoss(
        up_id=2,
        neutral_id=1,
        down_id=0,
        movement_weight=0.0,
        direction_weight=1.0,
    )

    loss = criterion(
        class_logits=torch.zeros((2, 3)),
        targets=torch.tensor([1, 1]),
        auxiliary_outputs={"direction_logits": torch.zeros((2, 2))},
    )

    assert loss.item() == 0.0


@pytest.mark.filterwarnings("ignore:Detected call of.*lr_scheduler\\.step.*:UserWarning")
def test_auxiliary_loss_is_train_only_and_evaluate_keeps_primary_loss(
    artifact_dir: Path,
) -> None:
    class AuxModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bias = nn.Parameter(torch.zeros(()))
            self.auxiliary_outputs = {}

        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            del t
            batch_size = x.shape[0]
            logits = self.bias * torch.ones((batch_size, 3), device=x.device)
            self.auxiliary_outputs = {
                "movement_logit": self.bias * torch.ones(batch_size, device=x.device),
                "direction_logits": self.bias * torch.ones((batch_size, 2), device=x.device),
            }
            return logits

    config = _training_config()
    config.device = "cpu"
    config.focal_gamma = 0.0
    config.class_weights = None
    config.model_dir = str(artifact_dir)
    auxiliary = MovementDirectionAuxiliaryLoss(
        up_id=2,
        neutral_id=1,
        down_id=0,
        movement_weight=1.0,
        direction_weight=1.0,
    )
    trainer = LobTrainer(config, auxiliary_criterion=auxiliary)
    model = AuxModel()
    data_loader = [
        (
            torch.zeros((3, 2, 1)),
            torch.zeros((3, 2)),
            torch.tensor([0, 1, 2]),
        )
    ]
    criterion = trainer._criterion()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    train_result = trainer._run_epoch(model, data_loader, criterion, optimizer, "Aux train")
    eval_result = trainer.evaluate(model, data_loader, criterion=criterion, description="Aux eval")

    assert train_result.loss == pytest.approx(np.log(3.0) + 2.0 * np.log(2.0), rel=1e-5)
    assert eval_result.loss == pytest.approx(np.log(3.0), rel=1e-5)


def test_lob_trainer_evaluate_collects_moe_expert_usage() -> None:
    class RoutingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(()))
            self.moe_routing = None

        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            del t
            batch_size, sequence_length, _ = x.shape
            self.moe_routing = {
                "topk_indices": torch.tensor(
                    [
                        [[0, 1], [1, 2]],
                        [[2, 1], [0, 2]],
                    ],
                    device=x.device,
                ),
                "topk_weights": torch.full((batch_size, sequence_length, 2), 0.5, device=x.device),
                "router_probabilities": torch.full((batch_size, sequence_length, 3), 1.0 / 3.0, device=x.device),
                "num_experts": 3,
                "top_k": 2,
            }
            return torch.tensor(
                [
                    [4.0, 0.0, 0.0],
                    [0.0, 0.0, 4.0],
                ],
                device=x.device,
            )

    config = _training_config()
    config.device = "cpu"
    config.class_weights = None
    trainer = LobTrainer(config)
    data_loader = [
        (
            torch.zeros((2, 2, 1)),
            torch.zeros((2, 2)),
            torch.tensor([0, 2]),
        )
    ]

    result = trainer.evaluate(
        RoutingModel(),
        data_loader,
        description="Expert usage test",
        track_expert_usage=True,
    )
    usage = result.expert_usage

    assert usage is not None
    assert usage["num_experts"] == 3
    assert usage["top_k"] == 2
    assert usage["tokens"] == 4
    assert usage["assignments"] == 8
    assert usage["selected_counts"] == [2, 3, 3]
    assert usage["primary_counts"] == [2, 1, 1]
    assert usage["by_true_class"]["0"]["selected_counts"] == [1, 2, 1]
    assert usage["by_true_class"]["2"]["selected_counts"] == [1, 1, 2]


def test_lob_trainer_evaluate_without_moe_routing_has_no_expert_usage() -> None:
    class DenseOnlyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(()))
            self.moe_routing = None

        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            del x, t
            return torch.tensor([[4.0, 0.0, 0.0]], device=self.dummy.device)

    config = _training_config()
    config.device = "cpu"
    config.class_weights = None
    trainer = LobTrainer(config)
    data_loader = [
        (
            torch.zeros((1, 2, 1)),
            torch.zeros((1, 2)),
            torch.tensor([0]),
        )
    ]

    result = trainer.evaluate(
        DenseOnlyModel(),
        data_loader,
        description="No MoE expert usage test",
        track_expert_usage=True,
    )

    assert result.expert_usage is None


def test_lob_trainer_evaluate_can_collect_probability_outputs() -> None:
    class SimpleModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(()))

        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            del x, t
            return torch.tensor(
                [
                    [4.0, 0.0, 0.0],
                    [0.0, 4.0, 0.0],
                    [0.0, 0.0, 4.0],
                ]
            )

    config = _training_config()
    config.device = "cpu"
    config.class_weights = None
    config.monitor_params.fixed_rate = 1.0
    trainer = LobTrainer(config)
    data_loader = [
        (
            torch.zeros((3, 2, 1)),
            torch.zeros((3, 2)),
            torch.tensor([0, 1, 2]),
        )
    ]

    result = trainer.evaluate(
        SimpleModel(),
        data_loader,
        description="Probability output test",
        collect_outputs=True,
        track_pr_metrics=True,
    )

    assert result.prediction_outputs is not None
    assert result.prediction_outputs["logits"].shape == (3, 3)
    assert result.prediction_outputs["logits"].dtype == np.float32
    assert result.prediction_outputs["probabilities"].shape == (3, 3)
    assert result.prediction_outputs["probabilities"].dtype == np.float32
    assert result.prediction_outputs["targets"].tolist() == [0, 1, 2]
    assert result.metrics.per_class_pr_ap == pytest.approx([1.0, 1.0, 1.0])
    assert result.metrics.per_class_pr_auc == pytest.approx([1.0, 1.0, 1.0])
    assert result.metrics.per_class_roc_auc == pytest.approx([1.0, 1.0, 1.0])
    assert result.metrics.directional_precision_at_fixed_rate == pytest.approx(1 / 3)
    assert result.metrics.directional_precision_at_fixed_rate_k == 3
    assert result.metrics.directional_precision_at_fixed_rate_actual_rate == pytest.approx(1.0)


def test_lob_trainer_evaluate_skips_optional_expensive_tracking_by_default() -> None:
    class SimpleModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(()))

        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            del x, t
            return torch.tensor([[4.0, 0.0, 0.0]])

    config = _training_config()
    config.device = "cpu"
    config.class_weights = None
    trainer = LobTrainer(config)
    data_loader = [
        (
            torch.zeros((1, 2, 1)),
            torch.zeros((1, 2)),
            torch.tensor([0]),
        )
    ]

    result = trainer.evaluate(SimpleModel(), data_loader, description="Default lightweight eval")

    assert result.expert_usage is None
    assert result.prediction_outputs is None
    assert result.metrics.per_class_pr_ap is None
    assert result.metrics.per_class_pr_auc is None
    assert result.metrics.per_class_roc_auc is None


def test_classification_metrics_from_predictions_uses_fixed_decisions() -> None:
    targets = np.asarray([0, 2, 1])
    predictions = np.asarray([0, 1, 1])
    probabilities = np.asarray(
        [
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.8, 0.1],
        ],
        dtype=np.float32,
    )

    metrics = classification_metrics_from_predictions(
        targets,
        predictions,
        num_classes=3,
        probabilities=probabilities,
    )

    assert metrics.confusion_matrix == [[1, 0, 0], [0, 1, 0], [0, 1, 0]]
    assert metrics.accuracy == pytest.approx(2 / 3)
    assert metrics.directional_macro_f1 == pytest.approx(0.5)
    assert metrics.per_class_pr_ap is not None
    assert metrics.per_class_roc_auc is not None
