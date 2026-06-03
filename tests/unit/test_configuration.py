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
    assert config.preprocessing.normalization.derivative_scaling_method == "quantile_scaling"
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


def test_config_loader_accepts_quantile_derivative_scaling(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["preprocessing"]["normalization"]["derivative_scaling_method"] = "quantile_scaling"

    config_path = artifact_dir / "quantile_derivative_scaling.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    loaded = ExperimentConfig.from_yaml(config_path)

    assert loaded.preprocessing.normalization.derivative_scaling_method == "quantile_scaling"


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
    assert config.training.sampling.neutral_to_directional_ratio == 8.0


def test_training_class_weight_parameters_are_loaded() -> None:
    config = load_config()

    assert config.training.class_weight_beta == 0.1
    assert config.training.class_weight_min == 0.5
    assert config.training.class_weight_max == 3.0
    assert config.training.deterministic_torch is False
    assert config.training.optimizer == "adamw"
    assert config.training.early_stopping_min_delta == 0.0


def test_tlob_fi2010_config_loads() -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "config_TLOB_F1_2010.yaml"
    config = ExperimentConfig.from_yaml(config_path)

    assert config.folds[0].id == "fi2010_tlob"
    assert config.experiment.name
    assert config.data.sequence_window == 128
    assert config.data.label_mapping == {-1: 2, 0: 1, 1: 0}
    assert config.model.d_input == 144
    assert config.model.d_model == 128
    assert config.training.optimizer == "adam"
    assert config.training.early_stopping_min_delta >= 0.0
    assert config.training.monitor == "val_loss"
    assert isinstance(config.training.temperature_scaling.enabled, bool)


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


def test_training_optimizer_and_min_delta_are_validated(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["training"]["optimizer"] = "rmsprop"

    broken_config_path = artifact_dir / "bad_optimizer.yaml"
    broken_config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="training\\.optimizer"):
        ExperimentConfig.from_yaml(broken_config_path)

    payload["training"]["optimizer"] = "adam"
    payload["training"]["early_stopping_min_delta"] = -0.001
    broken_config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="early_stopping_min_delta"):
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
    assert config.preprocessing.kinematic_tokenization.orderbook_top_k_levels == 5
    assert config.preprocessing.sample_clock.mode == "event"
    assert config.preprocessing.sample_clock.enabled is False
    assert config.preprocessing.sample_clock.volume_step_shares is None
    assert config.preprocessing.sample_clock.volume_source == "traded"
    assert config.preprocessing.sample_clock.trade_type_values == [4, 5]
    assert config.preprocessing.microprice.enabled is True
    assert config.preprocessing.microprice.levels == 5
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


def test_model_max_dt_can_be_loaded_when_present(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["model"]["max_dt"] = 1.25

    config_path = artifact_dir / "resolved_model_max_dt.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    loaded = ExperimentConfig.from_yaml(config_path)

    assert loaded.model.max_dt == 1.25


def test_config_loader_accepts_run_metadata_snapshot_section(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["run_metadata"] = {
        "model_max_dt": {
            "quantile": 95.0,
            "resolved_max_dt": 1.5,
        },
        "class_weights": [1.0, 1.0, 1.0],
    }

    config_path = artifact_dir / "snapshot_config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    loaded = ExperimentConfig.from_yaml(config_path)

    assert loaded.path == config_path.resolve()


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


def test_train_fitted_smoothing_threshold_modes_are_loaded(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["preprocessing"]["labels"]["smoothing"]["adaptive_threshold"]["enabled"] = False

    for mode in ("mean_spread", "mean_pct"):
        payload["preprocessing"]["labels"]["smoothing"]["threshold"] = mode
        config_path = artifact_dir / f"{mode}.yaml"
        config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        loaded = ExperimentConfig.from_yaml(config_path)

        assert loaded.preprocessing.labels.smoothing.threshold == mode


def test_train_fitted_smoothing_threshold_rejects_bad_modes(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["preprocessing"]["labels"]["smoothing"]["adaptive_threshold"]["enabled"] = False
    payload["preprocessing"]["labels"]["smoothing"]["threshold"] = "mean_magic"

    config_path = artifact_dir / "bad_smoothing_threshold.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="smoothing\\.threshold"):
        ExperimentConfig.from_yaml(config_path)


def test_train_fitted_smoothing_threshold_rejects_adaptive_combo(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["preprocessing"]["labels"]["smoothing"]["threshold"] = "mean_pct"
    payload["preprocessing"]["labels"]["smoothing"]["adaptive_threshold"]["enabled"] = True

    config_path = artifact_dir / "bad_smoothing_threshold_adaptive.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="adaptive_threshold"):
        ExperimentConfig.from_yaml(config_path)


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

    assert [fold.id for fold in config.folds] == ["fold_001", "fold_002", "fold_003", "fold_004"]
    assert config.folds[0].train_dates == ["2024-03-04", "2024-03-05", "2024-03-06"]
    assert config.folds[0].validation_dates == ["2024-03-07", "2024-03-08"]
    assert config.folds[0].test_dates == []
    assert config.folds[-1].validation_dates == ["2024-03-12", "2024-03-13"]
    assert config.folds[-1].test_dates == []
    assert all(not fold.has_test_dates for fold in config.folds)


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


def test_kinematic_top_k_and_microprice_config_are_optional_and_validated(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["preprocessing"]["kinematic_tokenization"].pop("orderbook_top_k_levels", None)
    payload["preprocessing"].pop("microprice", None)

    config_path = artifact_dir / "legacy_without_top_k_microprice.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    loaded = ExperimentConfig.from_yaml(config_path)

    assert loaded.preprocessing.kinematic_tokenization.orderbook_top_k_levels is None
    assert loaded.preprocessing.microprice.enabled is False
    assert loaded.preprocessing.microprice.levels == 1

    payload["preprocessing"]["kinematic_tokenization"]["orderbook_top_k_levels"] = 0
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="orderbook_top_k_levels"):
        ExperimentConfig.from_yaml(config_path)

    payload["preprocessing"]["kinematic_tokenization"]["orderbook_top_k_levels"] = 2
    payload["preprocessing"]["microprice"] = {"enabled": True, "levels": 0}
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="preprocessing\\.microprice\\.levels"):
        ExperimentConfig.from_yaml(config_path)

    payload["preprocessing"]["microprice"] = {"enabled": True, "levels": 2.5}
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="preprocessing\\.microprice\\.levels"):
        ExperimentConfig.from_yaml(config_path)


def test_sample_clock_config_is_optional_and_validated(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["preprocessing"].pop("sample_clock", None)

    config_path = artifact_dir / "legacy_without_sample_clock.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    loaded = ExperimentConfig.from_yaml(config_path)

    assert loaded.preprocessing.sample_clock.mode == "event"
    assert loaded.preprocessing.sample_clock.enabled is False
    assert loaded.preprocessing.sample_clock.volume_step_shares is None
    assert loaded.preprocessing.sample_clock.volume_source == "traded"
    assert loaded.preprocessing.sample_clock.trade_type_values == [4, 5]

    payload["preprocessing"]["sample_clock"] = {
        "mode": "volume",
        "volume_step_shares": 1000,
        "volume_source": "traded",
        "trade_type_values": [4, 5],
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    loaded = ExperimentConfig.from_yaml(config_path)

    assert loaded.preprocessing.sample_clock.enabled is True
    assert loaded.preprocessing.sample_clock.volume_step_shares == 1000.0

    payload["preprocessing"]["sample_clock"]["volume_step_shares"] = None
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="volume_step_shares"):
        ExperimentConfig.from_yaml(config_path)

    payload["preprocessing"]["sample_clock"]["volume_step_shares"] = 1000
    payload["preprocessing"]["sample_clock"]["mode"] = "clock"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="sample_clock\\.mode"):
        ExperimentConfig.from_yaml(config_path)

    payload["preprocessing"]["sample_clock"]["mode"] = "volume"
    payload["preprocessing"]["sample_clock"]["volume_source"] = "not_a_source"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="sample_clock\\.volume_source"):
        ExperimentConfig.from_yaml(config_path)

    payload["preprocessing"]["sample_clock"]["volume_source"] = "traded"
    payload["preprocessing"]["sample_clock"]["trade_type_values"] = []
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="trade_type_values"):
        ExperimentConfig.from_yaml(config_path)


def test_fast_kinematic_method_requires_tick_reference(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["preprocessing"]["kinematic_tokenization"]["method"] = "fast"
    payload["preprocessing"]["volume_kinematic"]["reference"] = "time"

    broken_config_path = artifact_dir / "fast_time_reference.yaml"
    broken_config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="volume_kinematic\\.reference"):
        ExperimentConfig.from_yaml(broken_config_path)


def test_preprocessing_processed_dataframe_saving_defaults_to_false(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))

    assert config.preprocessing.save_processed_dataframes is False

    payload["preprocessing"].pop("save_processed_dataframes", None)
    config_path = artifact_dir / "legacy_without_processed_dataframe_flag.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    loaded = ExperimentConfig.from_yaml(config_path)

    assert loaded.preprocessing.save_processed_dataframes is False

    payload["preprocessing"]["save_processed_dataframes"] = True
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    loaded = ExperimentConfig.from_yaml(config_path)

    assert loaded.preprocessing.save_processed_dataframes is True


def test_training_data_loader_settings_are_loaded() -> None:
    config = load_config()
    pin_memory = config.training.device.startswith("cuda")

    assert config.training.num_workers >= 0
    assert config.training.early_stopping_patience >= 0
    assert config.training.early_stopping_warmup == 3
    assert config.training.monitor == "tailored_score"
    assert config.training.monitor_mode == "max"
    assert config.training.monitor_params.lambda_ece == 0.1
    assert config.training.monitor_params.lambda_rate == 0.2
    assert isinstance(config.training.temperature_scaling.enabled, bool)
    assert config.training.directional_thresholds.enabled is True
    assert config.training.directional_thresholds.method == "joint_up_down"
    assert config.training.directional_thresholds.min_threshold == 0.05
    assert config.training.directional_thresholds.max_threshold == 0.95
    assert config.training.directional_thresholds.step == 0.05
    assert config.training.directional_thresholds.delta == 0.0
    assert config.training.directional_thresholds.up_precision_floor is None
    assert config.training.directional_thresholds.down_precision_floor is None
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


def test_training_early_stopping_warmup_is_optional_and_validated(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["training"].pop("early_stopping_warmup", None)

    config_path = artifact_dir / "legacy_without_early_stopping_warmup.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    loaded = ExperimentConfig.from_yaml(config_path)

    assert loaded.training.early_stopping_warmup == 0

    payload["training"]["early_stopping_warmup"] = -1
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="early_stopping_warmup"):
        ExperimentConfig.from_yaml(config_path)


def test_training_tailored_monitor_requires_params_and_max_mode(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["training"]["monitor"] = "tailored_score"
    payload["training"]["monitor_mode"] = "min"

    config_path = artifact_dir / "bad_tailored_mode.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="monitor_mode"):
        ExperimentConfig.from_yaml(config_path)

    payload["training"]["monitor_mode"] = "max"
    payload["training"].pop("monitor_params", None)
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="monitor_params"):
        ExperimentConfig.from_yaml(config_path)

    payload["training"]["monitor_params"] = {"lambda_ece": -0.1, "lambda_rate": 0.5}
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="lambda_ece"):
        ExperimentConfig.from_yaml(config_path)


def test_legacy_monitors_do_not_require_monitor_params(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["training"]["monitor"] = "val_loss"
    payload["training"]["monitor_mode"] = "min"
    payload["training"].pop("monitor_params", None)

    config_path = artifact_dir / "legacy_monitor_without_params.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    loaded = ExperimentConfig.from_yaml(config_path)

    assert loaded.training.monitor == "val_loss"
    assert not loaded.training.monitor_params.complete


def test_directional_threshold_config_is_optional(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["training"].pop("directional_thresholds", None)

    config_path = artifact_dir / "no_directional_thresholds.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    loaded = ExperimentConfig.from_yaml(config_path)

    assert loaded.training.directional_thresholds.enabled is False
    assert loaded.training.directional_thresholds.method == "joint_up_down"
    assert loaded.training.directional_thresholds.min_threshold == 0.05
    assert loaded.training.directional_thresholds.max_threshold == 0.95
    assert loaded.training.directional_thresholds.step == 0.05
    assert loaded.training.directional_thresholds.delta == 0.0


def test_temperature_scaling_config_is_optional(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["training"].pop("temperature_scaling", None)

    config_path = artifact_dir / "no_temperature_scaling.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    loaded = ExperimentConfig.from_yaml(config_path)

    assert loaded.training.temperature_scaling.enabled is False

    payload["training"]["temperature_scaling"] = {"enabled": True}
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    loaded = ExperimentConfig.from_yaml(config_path)

    assert loaded.training.temperature_scaling.enabled is True


def test_directional_threshold_config_uses_grid_defaults(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["training"]["directional_thresholds"] = {"enabled": False}

    config_path = artifact_dir / "directional_threshold_defaults.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    loaded = ExperimentConfig.from_yaml(config_path)

    assert loaded.training.directional_thresholds.enabled is False
    assert loaded.training.directional_thresholds.method == "joint_up_down"
    assert loaded.training.directional_thresholds.min_threshold == 0.05
    assert loaded.training.directional_thresholds.max_threshold == 0.95
    assert loaded.training.directional_thresholds.step == 0.05


def test_directional_threshold_config_validates_grid(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["training"]["directional_thresholds"] = {
        "enabled": True,
        "min": 0.8,
        "max": 0.2,
        "step": 0.05,
    }

    config_path = artifact_dir / "bad_directional_threshold_grid.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="directional_thresholds\\.min"):
        ExperimentConfig.from_yaml(config_path)

    payload["training"]["directional_thresholds"] = {
        "enabled": True,
        "min": 0.05,
        "max": 0.95,
        "step": 0.0,
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="directional_thresholds\\.step"):
        ExperimentConfig.from_yaml(config_path)


def test_directional_threshold_config_validates_methods_and_floors(artifact_dir: Path) -> None:
    config = load_config()
    payload = yaml.safe_load(config.path.read_text(encoding="utf-8"))
    payload["training"]["directional_thresholds"] = {
        "enabled": True,
        "method": "joint_up_down",
        "min": 0.05,
        "max": 0.95,
        "step": 0.05,
        "delta": 0.0,
        "up_precision_floor": 0.6,
        "down_precision_floor": None,
    }

    config_path = artifact_dir / "bad_joint_threshold_floor.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="precision floors must be null"):
        ExperimentConfig.from_yaml(config_path)

    payload["training"]["directional_thresholds"] = {
        "enabled": True,
        "method": "precision_floor",
        "min": 0.05,
        "max": 0.95,
        "step": 0.05,
        "delta": 0.0,
        "up_precision_floor": 0.6,
        "down_precision_floor": 0.6,
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    loaded = ExperimentConfig.from_yaml(config_path)

    assert loaded.training.directional_thresholds.method == "precision_floor"
    assert loaded.training.directional_thresholds.up_precision_floor == pytest.approx(0.6)
    assert loaded.training.directional_thresholds.down_precision_floor == pytest.approx(0.6)

    payload["training"]["directional_thresholds"]["up_precision_floor"] = None
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must be set when method is precision_floor"):
        ExperimentConfig.from_yaml(config_path)

    payload["training"]["directional_thresholds"]["up_precision_floor"] = 1.2
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="up_precision_floor"):
        ExperimentConfig.from_yaml(config_path)

    payload["training"]["directional_thresholds"]["up_precision_floor"] = 0.6
    payload["training"]["directional_thresholds"]["delta"] = -0.1
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="delta"):
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
