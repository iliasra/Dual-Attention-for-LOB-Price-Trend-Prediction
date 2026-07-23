from __future__ import annotations

import importlib
import sys
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import numpy as np

pytest.importorskip("torch")

from configuration import WandbTrackingConfig
from action_value import action_value_metrics
from training import ClassificationMetrics, EpochResult
from wandb_tracking import (
    WandbTracker,
    action_value_result_to_wandb_metrics,
    epoch_result_to_wandb_metrics,
    wandb_run_id,
)


class FakeArtifact:
    def __init__(self, *, name: str, type: str) -> None:
        self.name = name
        self.type = type
        self.files: list[str] = []

    def add_file(self, path: str) -> None:
        self.files.append(path)


class FakeRun:
    def __init__(self) -> None:
        self.config = {}
        self.logged: list[tuple[dict[str, object], int | None]] = []
        self.artifacts: list[FakeArtifact] = []
        self.metric_definitions: list[tuple[str, str | None]] = []
        self.finished = False
        self.exit_code: int | None = None

    def define_metric(self, name: str, step_metric: str | None = None) -> None:
        self.metric_definitions.append((name, step_metric))

    def log(self, payload: dict[str, object], step: int | None = None) -> None:
        self.logged.append((payload, step))

    def log_artifact(self, artifact: FakeArtifact) -> None:
        self.artifacts.append(artifact)

    def finish(self, exit_code: int | None = None) -> None:
        self.finished = True
        self.exit_code = exit_code


class ConfigDict(dict):
    def update(self, payload: dict[str, object], allow_val_change: bool = False) -> None:  # type: ignore[override]
        super().update(payload)


@pytest.fixture()
def artifact_dir(request: pytest.FixtureRequest) -> Path:
    path = Path(__file__).resolve().parent / ".test_artifacts" / request.node.name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def _metrics() -> ClassificationMetrics:
    return ClassificationMetrics(
        accuracy=0.9,
        macro_precision=0.8,
        macro_recall=0.7,
        macro_f1=0.75,
        directional_macro_f1=0.6,
        weighted_f1=0.76,
        balanced_accuracy=0.7,
        expected_calibration_error=0.05,
        per_class_expected_calibration_error=[0.0, 0.0, 0.0],
        per_class_pr_ap=[1.0, 1.0, 1.0],
        per_class_pr_auc=[1.0, 1.0, 1.0],
        per_class_roc_auc=[1.0, 1.0, 1.0],
        per_class_precision=[1.0, 1.0, 1.0],
        per_class_recall=[1.0, 1.0, 1.0],
        per_class_f1=[1.0, 1.0, 1.0],
        confusion_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        normalized_confusion_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        directional_precision_at_fixed_rate=0.66,
        directional_precision_at_fixed_rate_k=10,
        directional_precision_at_fixed_rate_actual_rate=0.01,
    )


def test_epoch_result_to_wandb_metrics_uses_validation_step_fields() -> None:
    result = EpochResult(
        train_loss=1.2,
        val_loss=0.8,
        train_metrics=_metrics(),
        val_metrics=_metrics(),
        epoch=2,
        batch_in_epoch=500,
        global_step=1500,
        validation_index=3,
        checkpoint_label="epoch_0002_step_00001500",
    )

    payload = epoch_result_to_wandb_metrics(result, monitor_value=0.75)

    assert payload["validation_index"] == 3
    assert payload["global_step"] == 1500
    assert payload["validation/macro_f1"] == 0.75
    assert payload["validation/directional_macro_f1"] == 0.6
    assert payload["validation/directional_precision_at_fixed_rate"] == 0.66
    assert payload["validation/directional_precision_at_fixed_rate_k"] == 10.0
    assert payload["validation/directional_precision_at_fixed_rate_actual_rate"] == 0.01
    assert payload["validation/monitor_value"] == 0.75


def test_wandb_tracker_initializes_and_logs_with_fake_module(
    monkeypatch,
    artifact_dir: Path,
) -> None:
    fake_run = FakeRun()
    fake_run.config = ConfigDict()
    init_calls: list[dict[str, object]] = []

    def fake_init(**kwargs: object) -> FakeRun:
        init_calls.append(dict(kwargs))
        return fake_run

    fake_wandb = SimpleNamespace(init=fake_init, Artifact=FakeArtifact)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    tracker = WandbTracker.init(
        WandbTrackingConfig(enabled=True, mode="offline", tags=["unit"]),
        run_stem="run a",
        fold_id="fold/001",
        fold_log_dir=artifact_dir,
        config_payload={"seed": 42},
    )

    assert tracker.enabled
    assert init_calls[0]["project"] == "lob-price-trend"
    assert init_calls[0]["group"] == "run a"
    assert init_calls[0]["name"] == "run a/fold/001"
    assert init_calls[0]["id"] == wandb_run_id("run a", "fold/001")
    assert init_calls[0]["resume"] == "allow"
    assert init_calls[0]["mode"] == "offline"
    assert ("train/*", "global_step") in fake_run.metric_definitions
    assert ("validation/*", "global_step") in fake_run.metric_definitions

    result = EpochResult(train_loss=1.0, val_loss=0.5, val_metrics=_metrics(), validation_index=4)
    tracker.log_validation(result, monitor_value=0.5)
    assert fake_run.logged[0][1] is None
    assert fake_run.logged[0][0]["validation/macro_f1"] == 0.75

    stepped_result = EpochResult(
        train_loss=0.9,
        val_loss=0.4,
        val_metrics=_metrics(),
        validation_index=5,
        global_step=128,
    )
    tracker.log_validation(stepped_result, monitor_value=0.4)
    assert fake_run.logged[1][1] is None
    assert fake_run.logged[1][0]["validation/loss"] == 0.4

    tracker.log_training_step(
        {
            "epoch": 1,
            "batch_in_epoch": 8,
            "global_step": 129,
            "optimizer_step": 64,
            "gradient_accumulation_steps": 2,
            "effective_batch_size_chunks": 64,
            "optimizer_step_completed": True,
            "train_loss_step": 0.123,
            "learning_rate": 1e-4,
        }
    )
    assert fake_run.logged[2][1] is None
    assert fake_run.logged[2][0]["train/loss_step"] == 0.123
    assert fake_run.logged[2][0]["optimizer_step"] == 64
    assert fake_run.logged[2][0]["train/gradient_accumulation_steps"] == 2
    assert fake_run.logged[2][0]["train/optimizer_step_completed"] is True

    artifact_file = artifact_dir / "metrics.csv"
    artifact_file.write_text("epoch,val_loss\n1,0.5\n", encoding="utf-8")
    tracker.log_artifact_files(name="files", artifact_type="training-artifacts", paths=[artifact_file])
    assert fake_run.artifacts[0].files == [str(artifact_file)]
    tracker.finish()
    assert fake_run.finished


def test_action_value_metrics_are_namespaced_on_explicit_global_step() -> None:
    metrics = action_value_metrics(
        np.zeros((2, 2)),
        np.asarray([[2.0, -1.0], [-4.0, -3.0]]),
    )
    result = SimpleNamespace(
        epoch=2,
        train_loss=0.7,
        validation_loss=0.5,
        monitor_value=1.25,
        validation_index=3,
        batch_in_epoch=5000,
        checkpoint_label="epoch_0002_step_00030000",
        validation_metrics=metrics,
    )

    payload = action_value_result_to_wandb_metrics(result, global_step=30000, optimizer_step=15000)

    assert payload["global_step"] == 30000
    assert payload["optimizer_step"] == 15000
    assert payload["train/interval_loss"] == 0.7
    assert payload["validation/rank_ic_mean"] == 0.0
    assert payload["validation/fixed_rate_mean_pnl_ticks"] == 0.0
    assert payload["validation/oracle_mean_pnl_ticks"] == pytest.approx(1.0)
    assert payload["validation/forced_oracle_mean_pnl_ticks"] == pytest.approx(-0.5)


def test_unclosed_wandb_tracker_marks_process_failure() -> None:
    run = FakeRun()
    tracker = WandbTracker(run=run, wandb_module=SimpleNamespace(), run_id="failed-run")

    tracker._finish_unclosed()

    assert run.finished is True
    assert run.exit_code == 1


def test_required_wandb_raises_when_package_is_missing(monkeypatch, artifact_dir: Path) -> None:
    real_import_module = importlib.import_module

    def missing_wandb(name: str, package: str | None = None):
        if name == 'wandb':
            raise ImportError('missing for test')
        return real_import_module(name, package)

    monkeypatch.setattr('wandb_tracking.importlib.import_module', missing_wandb)

    with pytest.raises(RuntimeError, match='W&B tracking is required.*not installed'):
        WandbTracker.init(
            WandbTrackingConfig(enabled=True, required=True, mode='online'),
            run_stem='strict',
            fold_id='fold',
            fold_log_dir=artifact_dir,
        )


def test_required_auto_attempts_online_only_and_raises(monkeypatch, artifact_dir: Path) -> None:
    attempted_modes: list[str] = []
    monkeypatch.setenv('WANDB_MODE', 'offline')

    def failing_init(**kwargs: object) -> FakeRun:
        attempted_modes.append(str(kwargs['mode']))
        raise ConnectionError('offline network')

    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=failing_init, Artifact=FakeArtifact))

    with pytest.raises(RuntimeError, match='W&B tracking is required.*initialization failed'):
        WandbTracker.init(
            WandbTrackingConfig(enabled=True, required=True, mode='auto'),
            run_stem='strict',
            fold_id='fold',
            fold_log_dir=artifact_dir,
        )

    assert attempted_modes == ['online']


def test_optional_auto_still_falls_back_to_offline(monkeypatch, artifact_dir: Path) -> None:
    fake_run = FakeRun()
    attempted_modes: list[str] = []

    def fallback_init(**kwargs: object) -> FakeRun:
        mode = str(kwargs['mode'])
        attempted_modes.append(mode)
        if mode == 'online':
            raise ConnectionError('offline network')
        return fake_run

    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=fallback_init, Artifact=FakeArtifact))

    tracker = WandbTracker.init(
        WandbTrackingConfig(enabled=True, mode='auto'),
        run_stem='optional',
        fold_id='fold',
        fold_log_dir=artifact_dir,
    )

    assert tracker.enabled is True
    assert attempted_modes == ['online', 'offline']
