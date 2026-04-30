from __future__ import annotations

from pathlib import Path
import shutil

import pytest
import yaml

from configuration import ExperimentConfig, load_config


@pytest.fixture()
def artifact_dir(request: pytest.FixtureRequest) -> Path:
    path = Path(__file__).resolve().parent / ".test_artifacts" / request.node.name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def test_config_loader_reports_missing_yaml_parameter(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    del payload["model"]["d_model"]

    broken_config_path = artifact_dir / "missing_model_param.yaml"
    broken_config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="model\\.d_model"):
        ExperimentConfig.from_yaml(broken_config_path)


def test_config_loader_reports_invalid_allowed_value(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["preprocessing"]["labels"]["strategy"] = "made_up_strategy"

    broken_config_path = artifact_dir / "invalid_label_strategy.yaml"
    broken_config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="preprocessing\\.labels\\.strategy"):
        ExperimentConfig.from_yaml(broken_config_path)


def test_tick_size_is_inherited_from_data_config() -> None:
    config = load_config()

    assert config.data.tick_size == 100.0
    assert config.preprocessing.message.tick_size == config.data.tick_size
    assert config.preprocessing.price_kinematic.tick_size == config.data.tick_size
    assert config.preprocessing.price_static.tick_size == config.data.tick_size

