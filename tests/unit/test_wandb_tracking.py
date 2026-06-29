from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from configuration import WandbTrackingConfig
from training import ClassificationMetrics, EpochResult
from wandb_tracking import WandbTracker, epoch_result_to_wandb_metrics, wandb_run_id


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
        self.finished = False

    def log(self, payload: dict[str, object], step: int | None = None) -> None:
        self.logged.append((payload, step))

    def log_artifact(self, artifact: FakeArtifact) -> None:
        self.artifacts.append(artifact)

    def finish(self, exit_code: int | None = None) -> None:
        self.finished = True


class ConfigDict(dict):
    def update(self, payload: dict[str, object], allow_val_change: bool = False) -> None:  # type: ignore[override]
        super().update(payload)


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
    assert payload["val_macro_f1"] == 0.75
    assert payload["val_directional_macro_f1"] == 0.6
    assert payload["val_directional_precision_at_fixed_rate"] == 0.66
    assert payload["val_directional_precision_at_fixed_rate_k"] == 10.0
    assert payload["val_directional_precision_at_fixed_rate_actual_rate"] == 0.01
    assert payload["monitor_value"] == 0.75


def test_wandb_tracker_initializes_and_logs_with_fake_module(
    monkeypatch,
    tmp_path: Path,
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
        fold_log_dir=tmp_path,
        config_payload={"seed": 42},
    )

    assert tracker.enabled
    assert init_calls[0]["project"] == "lob-price-trend"
    assert init_calls[0]["group"] == "run a"
    assert init_calls[0]["name"] == "run a/fold/001"
    assert init_calls[0]["id"] == wandb_run_id("run a", "fold/001")
    assert init_calls[0]["resume"] == "allow"
    assert init_calls[0]["mode"] == "offline"

    result = EpochResult(train_loss=1.0, val_loss=0.5, val_metrics=_metrics(), validation_index=4)
    tracker.log_validation(result, monitor_value=0.5)
    assert fake_run.logged[0][1] == 4
    assert fake_run.logged[0][0]["val_macro_f1"] == 0.75

    artifact_file = tmp_path / "metrics.csv"
    artifact_file.write_text("epoch,val_loss\n1,0.5\n", encoding="utf-8")
    tracker.log_artifact_files(name="files", artifact_type="training-artifacts", paths=[artifact_file])
    assert fake_run.artifacts[0].files == [str(artifact_file)]
    tracker.finish()
    assert fake_run.finished
