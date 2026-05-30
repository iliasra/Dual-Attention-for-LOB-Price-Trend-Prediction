from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace
import shutil
import sys

import pytest
import yaml

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from configuration import load_config
from evaluate_model import (
    extract_checkpoint_state_dict,
    load_config_for_evaluation,
    write_evaluation_outputs,
)


@pytest.fixture()
def artifact_dir(request: pytest.FixtureRequest) -> Path:
    """Return a clean writable test artifact directory."""
    path = Path(__file__).resolve().parent / ".test_artifacts" / request.node.name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def _config_payload() -> dict:
    """Return the default config payload."""
    config = load_config()
    return yaml.safe_load(config.path.read_text(encoding="utf-8"))


def test_load_config_for_evaluation_uses_model_max_dt(artifact_dir: Path) -> None:
    payload = _config_payload()
    payload["model"]["max_dt"] = 1.25
    config_path = artifact_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    config = load_config_for_evaluation(config_path)

    assert config.model.max_dt == 1.25


def test_load_config_for_evaluation_uses_snapshot_fallbacks(artifact_dir: Path) -> None:
    payload = _config_payload()
    payload["run_metadata"] = {
        "model_max_dt": {"resolved_max_dt": 2.5},
        "class_weights": [0.5, 1.0, 1.5],
    }
    config_path = artifact_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    config = load_config_for_evaluation(config_path)

    assert config.model.max_dt == 2.5
    assert config.training.class_weights == [0.5, 1.0, 1.5]


def test_load_config_for_evaluation_requires_max_dt(artifact_dir: Path) -> None:
    config_path = artifact_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(_config_payload()), encoding="utf-8")

    with pytest.raises(ValueError, match="model.max_dt is required"):
        load_config_for_evaluation(config_path)


def test_extract_checkpoint_state_dict_accepts_common_formats() -> None:
    state_dict = {"weight": torch.tensor([1.0])}

    assert extract_checkpoint_state_dict(state_dict) is state_dict
    assert extract_checkpoint_state_dict({"state_dict": state_dict}) is state_dict
    assert extract_checkpoint_state_dict({"model_state_dict": state_dict}) is state_dict


def test_write_evaluation_outputs_creates_metrics_and_logs(artifact_dir: Path) -> None:
    payload = _config_payload()
    payload["model"]["max_dt"] = 1.0
    config_path = artifact_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = load_config_for_evaluation(config_path)
    metrics = SimpleNamespace(
        accuracy=0.7,
        macro_precision=0.6,
        macro_recall=0.5,
        macro_f1=0.55,
        directional_macro_f1=0.52,
        weighted_f1=0.58,
        balanced_accuracy=0.51,
        expected_calibration_error=0.12,
        per_class_expected_calibration_error=[0.1, 0.2, 0.3],
        per_class_pr_ap=[0.4, 0.5, 0.6],
        per_class_pr_auc=[0.45, 0.55, 0.65],
        per_class_roc_auc=[0.7, 0.8, 0.9],
        per_class_precision=[0.3, 0.4, 0.5],
        per_class_recall=[0.6, 0.7, 0.8],
        per_class_f1=[0.4, 0.5, 0.6],
        confusion_matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        normalized_confusion_matrix=[[0.1, 0.2, 0.7], [0.4, 0.5, 0.1], [0.2, 0.3, 0.5]],
    )
    result = SimpleNamespace(loss=0.9, metrics=metrics, expert_usage={"total_sequences": 3}, prediction_outputs=None)

    row = write_evaluation_outputs(
        result=result,
        config=config,
        output_dir=artifact_dir / "evaluation",
        split="holdout",
        num_samples=3,
        checkpoint=artifact_dir / "model.pth",
        config_path=config_path,
        sequence_path=artifact_dir / "sequences",
        device="cpu",
        batch_size=256,
        duration_seconds=1.5,
        save_probabilities=False,
    )

    output_dir = artifact_dir / "evaluation"
    assert row["pr_ap_down"] == pytest.approx(0.4)
    assert (output_dir / "metrics.csv").exists()
    assert (output_dir / "metrics.yaml").exists()
    assert (output_dir / "confusion_matrix.yaml").exists()
    assert (output_dir / "expert_usage.yaml").exists()
    assert (output_dir / "evaluation.log").exists()
    with (output_dir / "metrics.csv").open("r", newline="", encoding="utf-8") as handle:
        csv_row = next(csv.DictReader(handle))
    assert csv_row["split"] == "holdout"
    assert csv_row["roc_auc_up"] == "0.9"
