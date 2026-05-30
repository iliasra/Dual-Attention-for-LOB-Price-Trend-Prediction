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

    assert config.seed == 42
    assert config.data.tick_size == 100.0
    assert config.data.logs_dir == "../logs"
    assert config.preprocessing.normalization.derivative_scaling_method == "robust_mad"
    assert config.preprocessing.message.tick_size == config.data.tick_size
    assert config.preprocessing.price_kinematic.tick_size == config.data.tick_size
    assert config.preprocessing.price_static.tick_size == config.data.tick_size


def test_config_loader_reports_negative_seed(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["seed"] = -1

    broken_config_path = artifact_dir / "negative_seed.yaml"
    broken_config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="seed"):
        ExperimentConfig.from_yaml(broken_config_path)


def test_config_loader_accepts_zscore_derivative_scaling(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["preprocessing"]["normalization"]["derivative_scaling_method"] = "zscore"

    config_path = artifact_dir / "zscore_derivative_scaling.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    loaded = ExperimentConfig.from_yaml(config_path)

    assert loaded.preprocessing.normalization.derivative_scaling_method == "zscore"


def test_config_loader_rejects_unknown_derivative_scaling(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["preprocessing"]["normalization"]["derivative_scaling_method"] = "made_up"

    broken_config_path = artifact_dir / "bad_derivative_scaling.yaml"
    broken_config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="preprocessing\\.normalization\\.derivative_scaling_method"):
        ExperimentConfig.from_yaml(broken_config_path)


def test_training_sampling_ratio_is_loaded() -> None:
    config = load_config()

    assert config.training.sampling.enabled
    assert config.training.sampling.neutral_to_directional_ratio == 2.0


def test_training_class_weight_parameters_are_loaded() -> None:
    config = load_config()

    assert config.training.class_weight_beta == 0.25
    assert config.training.class_weight_min == 0.5
    assert config.training.class_weight_max == 3.0
    assert config.training.deterministic_torch is False


def test_training_class_weight_parameters_are_validated(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["training"]["class_weight_beta"] = -0.1

    broken_config_path = artifact_dir / "bad_class_weight_beta.yaml"
    broken_config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="class_weight_beta"):
        ExperimentConfig.from_yaml(broken_config_path)

    payload["training"]["class_weight_beta"] = 0.25
    payload["training"]["class_weight_min"] = 2.0
    payload["training"]["class_weight_max"] = 1.0
    broken_config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="class_weight_max"):
        ExperimentConfig.from_yaml(broken_config_path)


def test_training_sampling_rejects_non_positive_ratio(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["training"]["sampling"]["neutral_to_directional_ratio"] = 0

    broken_config_path = artifact_dir / "bad_sampling_ratio.yaml"
    broken_config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="neutral_to_directional_ratio"):
        ExperimentConfig.from_yaml(broken_config_path)


def test_training_sampling_null_disables_sampler(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["training"]["sampling"]["neutral_to_directional_ratio"] = None

    config_path = artifact_dir / "disabled_sampling.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    loaded = ExperimentConfig.from_yaml(config_path)

    assert not loaded.training.sampling.enabled
    assert loaded.training.sampling.neutral_to_directional_ratio is None


def test_fast_kinematic_config_values_are_loaded() -> None:
    config = load_config()

    assert config.preprocessing.kinematic_tokenization.method in {"basis", "fast"}
    assert config.preprocessing.kinematic_tokenization.chunk_size == 100000
    assert config.preprocessing.kinematic_tokenization.n_df_candidates == 25
    assert config.preprocessing.price_kinematic.basis.alpha == 5.0
    assert config.preprocessing.price_kinematic.fast.n_basis == 20
    assert config.preprocessing.price_kinematic.fast.df == 12.0
    assert config.preprocessing.price_kinematic.fast.eval_at == 0.95
    assert config.preprocessing.volume_kinematic.basis.alpha == 5.0
    assert config.preprocessing.volume_kinematic.fast.n_basis == 20
    assert config.preprocessing.volume_kinematic.fast.df == 8.0
    assert config.preprocessing.volume_kinematic.fast.eval_at == 0.95


def test_price_static_plgs_train_fitted_config_values_are_loaded() -> None:
    config = load_config()

    assert config.preprocessing.price_static.tau_start == 2.0
    assert config.preprocessing.price_static.tau_clip is None
    assert config.preprocessing.price_static.tau_max is None


def test_volume_static_exp_scaling_train_fitted_config_values_are_loaded() -> None:
    config = load_config()

    assert config.preprocessing.volume_static.quantile == 95.0
    assert config.preprocessing.volume_static.target == 0.5
    assert config.preprocessing.volume_static.k is None


def test_model_max_dt_quantile_is_loaded_and_max_dt_is_resolved_later() -> None:
    config = load_config()

    assert config.model.max_dt_quantile == 95.0
    assert config.model.max_dt is None


def test_model_max_dt_quantile_is_validated(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["model"]["max_dt_quantile"] = 101.0

    config_path = artifact_dir / "bad_model_max_dt_quantile.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="model\\.max_dt_quantile"):
        ExperimentConfig.from_yaml(config_path)


def test_volume_static_config_validates_quantile(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["preprocessing"]["volume_static"]["quantile"] = 101.0

    config_path = artifact_dir / "bad_volume_static_quantile.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="volume_static\\.quantile"):
        ExperimentConfig.from_yaml(config_path)


def test_volume_static_config_validates_target(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["preprocessing"]["volume_static"]["target"] = 1.0

    config_path = artifact_dir / "bad_volume_static_target.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="volume_static\\.target"):
        ExperimentConfig.from_yaml(config_path)


def test_adaptive_threshold_config_values_are_loaded() -> None:
    config = load_config()
    adaptive = config.preprocessing.labels.smoothing.adaptive_threshold

    assert config.preprocessing.labels.smoothing.threshold is None
    assert adaptive is not None
    assert adaptive.enabled is True
    assert adaptive.exit_spread_window == 100
    assert adaptive.volatility_window == 256
    assert adaptive.round_trip_fees_bps == 1.5
    assert adaptive.volatility_lambda == 1.0


def test_smoothing_threshold_string_none_is_loaded_as_none_with_adaptive_threshold_enabled(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["preprocessing"]["labels"]["smoothing"]["threshold"] = "None"
    payload["preprocessing"]["labels"]["smoothing"]["adaptive_threshold"]["enabled"] = True

    config_path = artifact_dir / "string_none_threshold.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    loaded = ExperimentConfig.from_yaml(config_path)

    assert loaded.preprocessing.labels.smoothing.threshold is None
    assert loaded.preprocessing.labels.smoothing.adaptive_threshold is not None
    assert loaded.preprocessing.labels.smoothing.adaptive_threshold.enabled is True


def test_smoothing_config_validates_non_negative_k(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["preprocessing"]["labels"]["smoothing"]["k"] = -1

    config_path = artifact_dir / "bad_smoothing_k.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="smoothing\\.k"):
        ExperimentConfig.from_yaml(config_path)


def test_smoothing_config_validates_positive_h(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["preprocessing"]["labels"]["smoothing"]["h"] = 0

    config_path = artifact_dir / "bad_smoothing_h.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="smoothing\\.h"):
        ExperimentConfig.from_yaml(config_path)


def test_smoothing_method_c_requires_k_strictly_less_than_h(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["preprocessing"]["labels"]["smoothing"]["method"] = "C"
    payload["preprocessing"]["labels"]["smoothing"]["k"] = 5
    payload["preprocessing"]["labels"]["smoothing"]["h"] = 5

    config_path = artifact_dir / "bad_smoothing_c_windows.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="method C requires k < h"):
        ExperimentConfig.from_yaml(config_path)


def test_adaptive_threshold_config_is_optional_for_legacy_configs(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["preprocessing"]["labels"]["smoothing"].pop("adaptive_threshold", None)

    config_path = artifact_dir / "no_adaptive_threshold.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    loaded = ExperimentConfig.from_yaml(config_path)

    assert loaded.preprocessing.labels.smoothing.adaptive_threshold is None


def test_adaptive_threshold_config_validates_window_lengths(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["preprocessing"]["labels"]["smoothing"]["adaptive_threshold"]["exit_spread_window"] = 0

    config_path = artifact_dir / "bad_adaptive_threshold_window.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="adaptive_threshold\\.exit_spread_window"):
        ExperimentConfig.from_yaml(config_path)


def test_adaptive_threshold_config_validates_non_negative_costs(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["preprocessing"]["labels"]["smoothing"]["adaptive_threshold"]["round_trip_fees_bps"] = -1.0

    config_path = artifact_dir / "bad_adaptive_threshold_fees.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="adaptive_threshold\\.round_trip_fees_bps"):
        ExperimentConfig.from_yaml(config_path)


def test_adaptive_threshold_config_validates_non_negative_lambda(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["preprocessing"]["labels"]["smoothing"]["adaptive_threshold"]["volatility_lambda"] = -1.0

    config_path = artifact_dir / "bad_adaptive_threshold_lambda.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="adaptive_threshold\\.volatility_lambda"):
        ExperimentConfig.from_yaml(config_path)


def test_explicit_folds_are_loaded() -> None:
    config = load_config()

    assert [fold.id for fold in config.folds] == [
        "fold_001",
        "fold_002",
        "fold_003",
        "fold_004",
        "fold_005",
        "fold_006",
    ]
    assert config.folds[0].train_dates == ["2024-03-04", "2024-03-05", "2024-03-06"]
    assert config.folds[0].validation_dates == ["2024-03-07"]
    assert config.folds[0].test_dates == ["2024-03-08"]
    assert config.folds[-1].validation_dates == ["2024-03-14"]
    assert config.folds[-1].test_dates == ["2024-03-15"]


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
    assert loaded.folds[0].validation_dates == loaded.dataset_splits.validation_dates
    assert loaded.folds[0].test_dates == loaded.dataset_splits.test_dates


def test_dataset_split_fallback_accepts_empty_test_dates(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload.pop("folds", None)
    payload["dataset_splits"]["test_dates"] = []

    config_path = artifact_dir / "single_fold_without_test.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    loaded = ExperimentConfig.from_yaml(config_path)

    assert loaded.dataset_splits.test_dates == []
    assert loaded.folds[0].test_dates == []
    assert loaded.folds[0].has_test_dates is False


def test_fold_config_accepts_missing_test_dates(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["folds"] = [
        {
            "id": "fold_without_test",
            "train_dates": ["2012-06-21"],
            "validation_dates": ["2012-06-22"],
        }
    ]

    config_path = artifact_dir / "fold_without_test.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    loaded = ExperimentConfig.from_yaml(config_path)

    assert loaded.folds[0].test_dates == []
    assert loaded.folds[0].has_test_dates is False


def test_fold_config_rejects_overlapping_split_dates(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["folds"] = [
        {
            "id": "bad_fold",
            "train_dates": ["2012-06-21"],
            "validation_dates": ["2012-06-21"],
            "test_dates": ["2012-06-23"],
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
            "test_dates": ["2012-06-23"],
        }
    ]

    config_path = artifact_dir / "bad_fold_order.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="strictly before validation"):
        ExperimentConfig.from_yaml(config_path)


def test_fold_config_requires_validation_dates_but_allows_missing_test_dates(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["folds"] = [
        {
            "id": "bad_fold",
            "train_dates": ["2012-06-21"],
            "validation_dates": [],
            "test_dates": [],
        }
    ]

    config_path = artifact_dir / "missing_fold_validation_test.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="validation_dates"):
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
    pin_memory = config.training.device.startswith("cuda")

    assert config.training.num_workers >= 0
    assert config.training.early_stopping_patience >= 0
    assert config.training.monitor == "val_directional_macro_f1"
    assert config.training.monitor_mode == "max"
    assert config.training.eval_batch_size == 256
    assert config.training.class_weights is None
    assert config.training.pin_memory is pin_memory
    assert config.training.data_loader_kwargs() == {
        "num_workers": config.training.num_workers,
        "persistent_workers": config.training.persistent_workers,
        "pin_memory": pin_memory,
    }


def test_training_eval_batch_size_rejects_non_positive_value(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["training"]["eval_batch_size"] = 0

    broken_config_path = artifact_dir / "bad_eval_batch_size.yaml"
    broken_config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="training\\.eval_batch_size"):
        ExperimentConfig.from_yaml(broken_config_path)


def test_training_class_weights_yaml_parameter_is_rejected(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["training"]["class_weights"] = [1.0, 1.0, 1.0]

    config_path = artifact_dir / "unexpected_class_weights.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="training\\.class_weights"):
        ExperimentConfig.from_yaml(config_path)


def test_training_monitor_values_are_validated(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["training"]["monitor"] = "not_a_metric"

    config_path = artifact_dir / "bad_monitor.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="training\\.monitor"):
        ExperimentConfig.from_yaml(config_path)

    payload["training"]["monitor"] = "val_loss"
    payload["training"]["monitor_mode"] = "middle"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="training\\.monitor_mode"):
        ExperimentConfig.from_yaml(config_path)


def test_training_pin_memory_is_enabled_for_cuda(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["training"]["device"] = "cuda"

    cuda_config_path = artifact_dir / "cuda_config.yaml"
    cuda_config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    cuda_config = ExperimentConfig.from_yaml(cuda_config_path)

    assert cuda_config.training.pin_memory is True


def test_training_pin_memory_is_enabled_for_indexed_cuda(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["training"]["device"] = "cuda:0"

    cuda_config_path = artifact_dir / "indexed_cuda_config.yaml"
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
