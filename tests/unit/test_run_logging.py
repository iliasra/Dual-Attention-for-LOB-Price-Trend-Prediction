from __future__ import annotations

from pathlib import Path
import csv
import shutil

import numpy as np
import pytest
import yaml

pytest.importorskip("torch")

from configuration import load_config
from run_logging import (
    save_best_pr_artifacts,
    save_confusion_matrices,
    save_epoch_history,
    save_probability_outputs,
)
from training import ClassificationMetricAccumulator, EpochResult


@pytest.fixture()
def artifact_dir(request: pytest.FixtureRequest) -> Path:
    """Return a local writable test artifact directory."""
    path = Path(__file__).resolve().parent / ".test_artifacts" / request.node.name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def test_epoch_history_contains_pr_columns(artifact_dir: Path) -> None:
    config = load_config()
    metrics = ClassificationMetricAccumulator._zero_metrics(num_classes=3)
    metrics.per_class_pr_ap = [0.1, 0.2, 0.3]
    metrics.per_class_pr_auc = [0.4, 0.5, 0.6]
    metrics.per_class_roc_auc = [0.7, 0.8, 0.9]
    metrics.confusion_matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    target = artifact_dir / "metrics.csv"

    save_epoch_history(
        [EpochResult(train_loss=1.0, val_loss=0.5, val_metrics=metrics)],
        target,
        config=config,
        fold="fold_001",
    )

    with target.open("r", newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))

    assert row["val_pr_ap_down"] == "0.1"
    assert row["val_pr_ap_neutral"] == "0.2"
    assert row["val_pr_ap_up"] == "0.3"
    assert row["val_pr_auc_down"] == "0.4"
    assert row["val_pr_auc_neutral"] == "0.5"
    assert row["val_pr_auc_up"] == "0.6"
    assert row["val_roc_auc_down"] == "0.7"
    assert row["val_roc_auc_neutral"] == "0.8"
    assert row["val_roc_auc_up"] == "0.9"
    assert float(row["val_pred_directional_rate"]) == pytest.approx(30 / 45)
    assert "val_tailored_score" in row


def test_epoch_history_contains_tailored_monitor_columns(artifact_dir: Path) -> None:
    config = load_config()
    metrics = ClassificationMetricAccumulator._zero_metrics(num_classes=3)
    metrics.directional_macro_f1 = 0.8
    metrics.per_class_expected_calibration_error = [0.2, 0.1, 0.4]
    metrics.confusion_matrix = [[6, 2, 2], [1, 15, 4], [3, 2, 5]]
    target = artifact_dir / "metrics.csv"

    save_epoch_history(
        [EpochResult(train_loss=1.0, val_loss=0.5, val_metrics=metrics)],
        target,
        config=config,
        fold="fold_001",
    )

    with target.open("r", newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))

    assert float(row["val_tailored_score"]) == pytest.approx(0.6375)
    assert row["val_tailored_ece_dir"] == "0.3"
    assert row["val_tailored_rate_penalty"] == "0.025"


def test_epoch_history_writes_intra_epoch_metadata(artifact_dir: Path) -> None:
    config = load_config()
    config.training.monitor = "val_loss"
    config.training.monitor_mode = "min"
    metrics = ClassificationMetricAccumulator._zero_metrics(num_classes=3)
    target = artifact_dir / "metrics.csv"

    save_epoch_history(
        [
            EpochResult(
                train_loss=1.0,
                val_loss=0.5,
                val_metrics=metrics,
                epoch=1,
                batch_in_epoch=5000,
                global_step=5000,
                validation_index=1,
                checkpoint_label="epoch_0001_step_00005000",
            )
        ],
        target,
        config=config,
        fold="fold_001",
    )

    with target.open("r", newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))

    assert row["epoch"] == "1"
    assert row["validation_index"] == "1"
    assert row["batch_in_epoch"] == "5000"
    assert row["global_step"] == "5000"
    assert row["checkpoint_label"] == "epoch_0001_step_00005000"


def test_confusion_matrices_include_selected_best_block(artifact_dir: Path) -> None:
    interval_train_metrics = ClassificationMetricAccumulator._zero_metrics(num_classes=3)
    interval_train_metrics.confusion_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    selected_train_metrics = ClassificationMetricAccumulator._zero_metrics(num_classes=3)
    selected_train_metrics.confusion_matrix = [[9, 0, 0], [0, 8, 0], [0, 0, 7]]
    selected_val_metrics = ClassificationMetricAccumulator._zero_metrics(num_classes=3)
    selected_val_metrics.confusion_matrix = [[2, 1, 0], [0, 3, 0], [1, 0, 2]]
    selected_test_metrics = ClassificationMetricAccumulator._zero_metrics(num_classes=3)
    selected_test_metrics.confusion_matrix = [[4, 0, 1], [0, 5, 0], [1, 0, 4]]
    target = artifact_dir / "confusion_matrices.yaml"

    save_confusion_matrices(
        [
            EpochResult(
                train_loss=1.0,
                val_loss=0.5,
                train_metrics=interval_train_metrics,
                val_metrics=selected_val_metrics,
                epoch=1,
                batch_in_epoch=5000,
                global_step=5000,
                validation_index=1,
                checkpoint_label="epoch_0001_step_00005000",
            )
        ],
        target,
        fold="fold_001",
        selected_best_result=EpochResult(
            train_loss=0.1,
            val_loss=0.5,
            test_loss=0.4,
            train_metrics=selected_train_metrics,
            val_metrics=selected_val_metrics,
            test_metrics=selected_test_metrics,
            epoch=1,
            batch_in_epoch=5000,
            global_step=5000,
            validation_index=1,
            checkpoint_label="epoch_0001_step_00005000",
        ),
        selected_best_label="epoch_0001_step_00005000",
    )

    with target.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    fold_payload = payload["folds"]["fold_001"]
    assert fold_payload["epoch_0001_step_00005000"]["train"]["raw"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert fold_payload["selected_best_checkpoint"]["checkpoint_label"] == "epoch_0001_step_00005000"
    assert fold_payload["selected_best_checkpoint"]["train"]["raw"] == [[9, 0, 0], [0, 8, 0], [0, 0, 7]]
    assert fold_payload["selected_best_checkpoint"]["validation"]["raw"] == [[2, 1, 0], [0, 3, 0], [1, 0, 2]]
    assert fold_payload["selected_best_checkpoint"]["test"]["raw"] == [[4, 0, 1], [0, 5, 0], [1, 0, 4]]


def test_epoch_history_contains_threshold_columns(artifact_dir: Path) -> None:
    config = load_config()
    metrics = ClassificationMetricAccumulator._zero_metrics(num_classes=3)
    argmax_metrics = ClassificationMetricAccumulator._zero_metrics(num_classes=3)
    argmax_metrics.accuracy = 0.2
    argmax_metrics.macro_f1 = 0.3
    argmax_metrics.directional_macro_f1 = 0.4
    argmax_metrics.confusion_matrix = [[1, 2, 0], [0, 3, 1], [1, 0, 2]]
    threshold_metrics = {
        "directional_macro_f1": 0.42,
        "macro_f1": 0.4,
        "accuracy": 0.7,
        "down_precision": 0.5,
        "down_recall": 0.6,
        "down_f1": 0.55,
        "up_precision": 0.7,
        "up_recall": 0.8,
        "up_f1": 0.75,
        "pred_rate_down": 0.1,
        "pred_rate_up": 0.2,
        "pred_rate_neutral": 0.7,
        "true_rate_down": 0.15,
        "true_rate_up": 0.25,
        "true_rate_neutral": 0.6,
    }
    target = artifact_dir / "metrics.csv"

    save_epoch_history(
        [
            EpochResult(
                train_loss=1.0,
                val_loss=0.5,
                val_metrics=metrics,
                val_threshold_metrics=threshold_metrics,
                test_threshold_metrics=threshold_metrics,
                val_argmax_metrics=argmax_metrics,
                test_argmax_metrics=argmax_metrics,
            )
        ],
        target,
        config=config,
        fold="fold_001",
    )

    with target.open("r", newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))

    assert row["val_threshold_directional_macro_f1"] == "0.42"
    assert row["test_threshold_directional_macro_f1"] == "0.42"
    assert row["val_threshold_pred_rate_neutral"] == "0.7"
    assert row["test_threshold_true_rate_up"] == "0.25"
    assert row["val_argmax_accuracy"] == "0.2"
    assert row["test_argmax_directional_macro_f1"] == "0.4"
    assert row["val_argmax_pred_rate_down"] == "0.2"


def test_best_pr_artifacts_and_probabilities_are_written(artifact_dir: Path) -> None:
    config = load_config()
    outputs = {
        "sample_index": np.asarray([0, 1, 2]),
        "targets": np.asarray([0, 1, 2]),
        "predictions": np.asarray([0, 1, 2]),
        "probabilities": np.asarray(
            [
                [0.8, 0.1, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7],
            ]
        ),
    }

    probabilities_path = artifact_dir / "probabilities" / "validation_best_epoch_2.csv"
    save_probability_outputs(outputs, probabilities_path, config)
    artifacts = save_best_pr_artifacts(
        outputs,
        test_outputs=outputs,
        curves_dir=artifact_dir / "pr_curves",
        thresholds_path=artifact_dir / "pr_thresholds.yaml",
        config=config,
        best_epoch=2,
        fold="fold_001",
    )

    assert probabilities_path.exists()
    with probabilities_path.open("r", newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert header == ["sample_index", "true_label", "pred_label", "p_down", "p_neutral", "p_up"]
    assert (artifact_dir / "pr_thresholds.yaml").exists()
    assert (artifact_dir / "pr_curves" / "validation_best_epoch_2_down.csv").exists()
    assert (artifact_dir / "pr_curves" / "test_best_epoch_2_down.csv").exists()
    assert artifacts["thresholds"]["fold"] == "fold_001"
    assert artifacts["thresholds"]["selection_rule"] == "max_f1"
    assert artifacts["thresholds"]["selection_split"] == "validation"
    assert artifacts["thresholds"]["evaluated_splits"] == ["validation", "test"]
    assert artifacts["thresholds"]["classes"]["down"]["pr_auc"] == pytest.approx(1.0)
    assert artifacts["thresholds"]["classes"]["down"]["roc_auc"] == pytest.approx(1.0)
    assert artifacts["thresholds"]["splits"]["test"]["classes"]["down"]["curve_csv"].endswith(
        "test_best_epoch_2_down.csv"
    )
    assert set(artifacts["thresholds"]["classes"]) == {"down", "neutral", "up"}
