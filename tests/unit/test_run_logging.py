from __future__ import annotations

from pathlib import Path
import csv
import shutil

import numpy as np
import pytest

from configuration import load_config
from run_logging import save_best_pr_artifacts, save_epoch_history, save_probability_outputs
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
    assert artifacts["thresholds"]["fold"] == "fold_001"
    assert artifacts["thresholds"]["selection_rule"] == "max_f1"
    assert artifacts["thresholds"]["classes"]["down"]["pr_auc"] == pytest.approx(1.0)
    assert artifacts["thresholds"]["classes"]["down"]["roc_auc"] == pytest.approx(1.0)
    assert set(artifacts["thresholds"]["classes"]) == {"down", "neutral", "up"}
