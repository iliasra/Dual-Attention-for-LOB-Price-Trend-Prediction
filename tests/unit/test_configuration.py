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
    assert config.data.logs_dir == "../logs"
    assert config.preprocessing.message.tick_size == config.data.tick_size
    assert config.preprocessing.price_kinematic.tick_size == config.data.tick_size
    assert config.preprocessing.price_static.tick_size == config.data.tick_size


def test_fast_kinematic_config_values_are_loaded() -> None:
    config = load_config()

    assert config.preprocessing.kinematic_tokenization.method in {"basis", "fast"}
    assert config.preprocessing.kinematic_tokenization.chunk_size == 100000
    assert config.preprocessing.price_kinematic.basis.alpha == 5.0
    assert config.preprocessing.price_kinematic.fast.n_basis == 20
    assert config.preprocessing.price_kinematic.fast.df == 20.0
    assert config.preprocessing.price_kinematic.fast.eval_at == 1.0
    assert config.preprocessing.volume_kinematic.basis.alpha == 5.0
    assert config.preprocessing.volume_kinematic.fast.n_basis == 20
    assert config.preprocessing.volume_kinematic.fast.df == 20.0


def test_explicit_folds_are_loaded() -> None:
    config = load_config()

    assert [fold.id for fold in config.folds] == ["fold_001"]
    assert config.folds[0].train_dates == ["2012-06-21"]


def test_folds_fallback_to_dataset_splits_when_missing(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload.pop("folds", None)

    config_path = artifact_dir / "single_fold_fallback.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    loaded = ExperimentConfig.from_yaml(config_path)

    assert len(loaded.folds) == 1
    assert loaded.folds[0].id == "single"
    assert loaded.folds[0].train_dates == loaded.dataset_splits.train_dates


def test_fold_config_rejects_overlapping_split_dates(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["folds"] = [
        {
            "id": "bad_fold",
            "train_dates": ["2012-06-21"],
            "validation_dates": ["2012-06-21"],
            "test_dates": [],
        }
    ]

    config_path = artifact_dir / "overlapping_fold.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="multiple splits"):
        ExperimentConfig.from_yaml(config_path)


def test_fold_config_requires_chronological_order(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["folds"] = [
        {
            "id": "bad_fold",
            "train_dates": ["2012-06-22"],
            "validation_dates": ["2012-06-21"],
            "test_dates": [],
        }
    ]

    config_path = artifact_dir / "bad_fold_order.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="strictly before validation"):
        ExperimentConfig.from_yaml(config_path)


def test_basis_kinematic_config_requires_explicit_values(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["preprocessing"]["volume_kinematic"]["basis"]["alpha"] = None

    broken_config_path = artifact_dir / "missing_basis_param.yaml"
    broken_config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="preprocessing\\.volume_kinematic\\.basis\\.alpha"):
        ExperimentConfig.from_yaml(broken_config_path)


def test_fast_kinematic_config_requires_explicit_values(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["preprocessing"]["price_kinematic"]["fast"]["n_basis"] = None

    broken_config_path = artifact_dir / "missing_fast_param.yaml"
    broken_config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="preprocessing\\.price_kinematic\\.fast\\.n_basis"):
        ExperimentConfig.from_yaml(broken_config_path)


def test_fast_kinematic_method_requires_tick_reference(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["preprocessing"]["kinematic_tokenization"]["method"] = "fast"
    payload["preprocessing"]["volume_kinematic"]["reference"] = "time"

    broken_config_path = artifact_dir / "fast_time_reference.yaml"
    broken_config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="volume_kinematic\\.reference"):
        ExperimentConfig.from_yaml(broken_config_path)


def test_training_data_loader_settings_are_loaded() -> None:
    config = load_config()

    assert config.training.num_workers == 0
    assert config.training.early_stopping_patience == 8
    assert config.training.persistent_workers is False
    assert config.training.pin_memory is False
    assert config.training.data_loader_kwargs() == {
        "num_workers": 0,
        "persistent_workers": False,
        "pin_memory": False,
    }


def test_training_pin_memory_is_enabled_for_cuda(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["training"]["device"] = "cuda"

    cuda_config_path = artifact_dir / "cuda_config.yaml"
    cuda_config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    cuda_config = ExperimentConfig.from_yaml(cuda_config_path)

    assert cuda_config.training.pin_memory is True


def test_persistent_workers_requires_positive_num_workers(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["training"]["num_workers"] = 0
    payload["training"]["persistent_workers"] = True

    broken_config_path = artifact_dir / "invalid_workers.yaml"
    broken_config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="training\\.persistent_workers"):
        ExperimentConfig.from_yaml(broken_config_path)
