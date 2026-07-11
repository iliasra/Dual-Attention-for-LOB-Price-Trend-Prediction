from __future__ import annotations

import sys
from pathlib import Path
import shutil

import numpy as np
import pytest
import yaml

pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_training import (
    build_auxiliary_criterion,
    build_train_sampler,
    monitor_value_after_postprocessing,
    resolve_resume_checkpoint,
    resume_wandb_run_id,
    select_checkpoint_after_validation_postprocessing,
    evaluate_best_model_on_validation_and_test_splits,
    fold_artifact_paths,
    resolve_class_weights,
    sequence_label_values,
    sequence_paths,
    sequence_time_span_quantile,
    train_fold,
)
from configuration import load_config
from training import CheckpointCandidate, ClassificationMetrics, EpochResult, EvaluationResult


@pytest.fixture()
def artifact_dir(request: pytest.FixtureRequest) -> Path:
    path = Path(__file__).resolve().parent / ".test_artifacts" / request.node.name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def test_fold_artifact_paths_are_scoped_by_fold() -> None:
    paths = fold_artifact_paths(
        sequence_dir=Path("data/sequences"),
        run_log_dir=Path("logs/run_7"),
        run_result_dir=Path("results/run_7"),
        fold_id="fold_003",
    )

    assert paths["sequence_dir"] == Path("data/sequences/fold_003")
    assert paths["log_dir"] == Path("logs/run_7/fold_003")
    assert paths["result_dir"] == Path("results/run_7/fold_003")


def test_sequence_paths_uses_manifest_and_ignores_stale_glob_files(artifact_dir: Path) -> None:
    split_dir = artifact_dir / "train"
    split_dir.mkdir()
    for stem in ("kept", "stale"):
        np.save(split_dir / f"{stem}_features.npy", np.zeros((2, 1), dtype=np.float32))
        np.save(split_dir / f"{stem}_times.npy", np.zeros(2, dtype=np.float64))
        np.save(split_dir / f"{stem}_labels.npy", np.zeros(2, dtype=np.int64))
    manifest = {
        "version": 1,
        "splits": {
            "train": [
                {
                    "features": "train/kept_features.npy",
                    "times": "train/kept_times.npy",
                    "labels": "train/kept_labels.npy",
                }
            ]
        },
    }
    (artifact_dir / "sequence_manifest.yaml").write_text(
        yaml.safe_dump(manifest),
        encoding="utf-8",
    )

    x_paths, t_paths, y_paths = sequence_paths(artifact_dir, "train")

    assert [Path(path).name for path in x_paths] == ["kept_features.npy"]
    assert [Path(path).name for path in t_paths] == ["kept_times.npy"]
    assert [Path(path).name for path in y_paths] == ["kept_labels.npy"]


def test_resolve_resume_checkpoint_prefers_explicit_path(artifact_dir: Path) -> None:
    latest = artifact_dir / "fold" / "training_state_latest.pth"
    explicit = artifact_dir / "manual_resume.pth"
    latest.parent.mkdir(parents=True)
    latest.write_bytes(b"latest")
    explicit.write_bytes(b"explicit")

    assert resolve_resume_checkpoint(
        fold_result_dir=latest.parent,
        resume_latest=True,
        resume_from=explicit,
    ) == explicit
    assert resolve_resume_checkpoint(
        fold_result_dir=latest.parent,
        resume_latest=True,
        resume_from=None,
    ) == latest
    assert resolve_resume_checkpoint(
        fold_result_dir=artifact_dir / "missing",
        resume_latest=True,
        resume_from=None,
    ) is None


def test_resume_wandb_run_id_reads_complete_training_state(artifact_dir: Path) -> None:
    import torch

    checkpoint_path = artifact_dir / "training_state_latest.pth"
    torch.save({"wandb_run_id": "abc123"}, checkpoint_path)

    assert resume_wandb_run_id(checkpoint_path) == "abc123"


def test_build_auxiliary_criterion_uses_auto_clipped_weights() -> None:
    config = load_config()
    config.model.auxiliary_heads.enabled = True
    config.model.auxiliary_heads.movement = True
    config.model.auxiliary_heads.direction = True
    config.training.auxiliary_losses.movement_weight = 0.05
    config.training.auxiliary_losses.direction_weight = 0.05
    config.training.auxiliary_losses.movement_pos_weight = "auto_clipped"
    config.training.auxiliary_losses.movement_pos_weight_min = 0.5
    config.training.auxiliary_losses.movement_pos_weight_max = 5.0
    config.training.auxiliary_losses.direction_class_weight_beta = 1.0
    config.training.class_weight_min = 0.1
    config.training.class_weight_max = 10.0

    criterion, summary = build_auxiliary_criterion(config, class_counts=[10, 60, 30])

    assert criterion is not None
    assert summary["enabled"] is True
    assert summary["movement_pos_weight"] == pytest.approx(1.5)
    assert summary["direction_class_weights"] == pytest.approx([0.5, 1.5])
    assert float(criterion.movement_pos_weight) == pytest.approx(1.5)
    assert criterion.direction_class_weights.tolist() == pytest.approx([0.5, 1.5])


def test_sequence_time_span_quantile_uses_train_window_duration() -> None:
    summary = sequence_time_span_quantile(
        [np.asarray([0.0, 1.0, 3.0, 6.0])],
        sequence_window=3,
        quantile=50.0,
    )

    assert summary["max_dt"] == pytest.approx(4.0)
    assert summary["n_windows"] == 2
    assert summary["min_span"] == pytest.approx(3.0)
    assert summary["max_span"] == pytest.approx(5.0)


def test_sequence_time_span_quantile_rejects_non_monotonic_times() -> None:
    with pytest.raises(ValueError, match="not non-decreasing"):
        sequence_time_span_quantile(
            [np.asarray([0.0, 2.0, 1.0])],
            sequence_window=3,
            quantile=95.0,
        )


def test_sequence_label_values_uses_sequence_end_labels() -> None:
    class DummyDataset:
        sequence_window = 3
        y_data = [
            np.asarray([9, 0, 1, 2]),
            np.asarray([0, 2, 1]),
        ]

    labels = sequence_label_values(DummyDataset())

    assert labels.tolist() == [1, 2, 1]


def test_build_train_sampler_uses_configured_sampling_ratio(artifact_dir: Path) -> None:
    from datasets import LOBDataset

    x_path = artifact_dir / "sample_features.npy"
    t_path = artifact_dir / "sample_times.npy"
    y_path = artifact_dir / "sample_labels.npy"
    labels = np.asarray([0, 2, *([1] * 6)], dtype=np.int64)
    np.save(x_path, np.ones((len(labels), 1), dtype=np.float32))
    np.save(t_path, np.arange(len(labels), dtype=np.float32))
    np.save(y_path, labels)
    dataset = LOBDataset([str(x_path)], [str(t_path)], [str(y_path)], sequence_window=1)
    config = load_config()
    config.training.sampling.neutral_to_directional_ratio = 2.0

    sampler, summary = build_train_sampler(config, dataset, seed=123)

    assert sampler is not None
    assert summary["enabled"] is True
    assert sampler.sampled_class_counts(config.model.num_classes) == [1, 4, 1]


def test_epoch_shuffled_sampler_is_epoch_deterministic() -> None:
    from datasets import EpochShuffledSampler

    sampler = EpochShuffledSampler(list(range(10)), base_seed=7)
    sampler.set_epoch(3)
    first = list(iter(sampler))
    sampler.set_epoch(3)
    second = list(iter(sampler))
    sampler.set_epoch(4)
    third = list(iter(sampler))

    assert first == second
    assert sorted(first) == list(range(10))
    assert third != first


def test_resolve_class_weights_can_use_sampled_class_counts() -> None:
    config = load_config()

    summary = resolve_class_weights(config, train_dataset=object(), sampled_class_counts=[1, 4, 1])

    assert summary["source"] == "sampled_train_per_epoch"
    assert summary["counts"] == [1, 4, 1]
    assert config.training.class_weights == summary["weights"]


def _dummy_metrics() -> ClassificationMetrics:
    return ClassificationMetrics(
        accuracy=1.0,
        macro_precision=1.0,
        macro_recall=1.0,
        macro_f1=1.0,
        directional_macro_f1=1.0,
        weighted_f1=1.0,
        balanced_accuracy=1.0,
        expected_calibration_error=0.0,
        per_class_expected_calibration_error=[0.0, 0.0, 0.0],
        per_class_pr_ap=[1.0, 1.0, 1.0],
        per_class_pr_auc=[1.0, 1.0, 1.0],
        per_class_roc_auc=[1.0, 1.0, 1.0],
        per_class_precision=[1.0, 1.0, 1.0],
        per_class_recall=[1.0, 1.0, 1.0],
        per_class_f1=[1.0, 1.0, 1.0],
        confusion_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        normalized_confusion_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )


def test_best_model_evaluation_can_skip_missing_test_split() -> None:
    class FakeTrainer:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.flags: list[tuple[bool, bool, bool]] = []

        def evaluate(self, **kwargs: object) -> EvaluationResult:
            self.calls.append(str(kwargs["description"]))
            self.flags.append(
                (
                    bool(kwargs.get("collect_outputs", False)),
                    bool(kwargs.get("track_pr_metrics", False)),
                    bool(kwargs.get("track_expert_usage", False)),
                )
            )
            return EvaluationResult(
                loss=0.25,
                metrics=_dummy_metrics(),
                prediction_outputs={
                    "sample_index": np.asarray([0], dtype=np.int64),
                    "targets": np.asarray([0], dtype=np.int64),
                    "predictions": np.asarray([0], dtype=np.int64),
                    "probabilities": np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
                },
            )

    config = load_config()
    config.training.monitor = "val_loss"
    config.training.monitor_mode = "min"
    history = [EpochResult(train_loss=1.0, val_loss=0.5)]
    trainer = FakeTrainer()

    evaluation = evaluate_best_model_on_validation_and_test_splits(
        config=config,
        trainer=trainer,  # type: ignore[arg-type]
        model=object(),
        validation_loader=object(),  # type: ignore[arg-type]
        test_loader=None,
        history=history,
    )

    assert trainer.calls == ["Best epoch 1 [Validation artifacts]"]
    assert trainer.flags == [(True, True, True)]
    assert evaluation["test_seconds"] is None
    assert evaluation["test_outputs"] is None
    assert history[0].test_loss is None


def test_checkpoint_selection_can_prefer_postprocessed_monitor(
    artifact_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config()
    config.training.monitor = "val_macro_f1"
    config.training.monitor_mode = "max"
    config.training.top_k_checkpoints = 2
    checkpoint_1 = artifact_dir / "epoch_0001.pth"
    checkpoint_2 = artifact_dir / "epoch_0002.pth"
    checkpoint_1.write_bytes(b"candidate-1")
    checkpoint_2.write_bytes(b"candidate-2")

    class FakeTrainer:
        top_checkpoint_candidates = [
            CheckpointCandidate(epoch=1, monitor_value=0.9, path=checkpoint_1),
            CheckpointCandidate(epoch=2, monitor_value=0.8, path=checkpoint_2),
        ]

    def fake_evaluate_candidate(**kwargs: object) -> dict[str, object]:
        candidate = kwargs["candidate"]
        assert isinstance(candidate, CheckpointCandidate)
        postprocessed = 0.7 if candidate.epoch == 1 else 0.95
        return {
            "epoch": candidate.epoch,
            "checkpoint_path": candidate.path,
            "raw_monitor_value": candidate.monitor_value,
            "postprocessed_monitor_value": postprocessed,
            "validation_loss": 1.0,
            "validation_metrics": _dummy_metrics(),
            "validation_expert_usage": None,
            "validation_outputs": {},
            "postprocessed_validation_metrics": _dummy_metrics(),
            "postprocessed_validation_outputs": {},
            "validation_seconds": 0.01,
            "postprocessing_seconds": 0.02,
            "candidate_dir": kwargs["candidate_dir"],
            "temperature_scaling": {"enabled": False},
            "temperature_scaling_path": None,
            "directional_thresholds": {"enabled": False},
            "directional_thresholds_path": None,
        }

    monkeypatch.setattr("run_training.evaluate_checkpoint_candidate_on_validation", fake_evaluate_candidate)

    selection = select_checkpoint_after_validation_postprocessing(
        config=config,
        trainer=FakeTrainer(),  # type: ignore[arg-type]
        model=object(),
        validation_loader=object(),  # type: ignore[arg-type]
        history=[
            EpochResult(train_loss=1.0, val_loss=1.0, val_metrics=_dummy_metrics()),
            EpochResult(train_loss=1.0, val_loss=1.0, val_metrics=_dummy_metrics()),
        ],
        fold="fold_001",
        selection_dir=artifact_dir / "checkpoint_selection",
    )

    assert selection["selected"]["epoch"] == 2
    assert selection["summary"]["selected_epoch"] == 2
    assert selection["summary"]["raw_monitor_value"] == pytest.approx(0.8)
    assert selection["summary"]["postprocessed_monitor_value"] == pytest.approx(0.95)


def test_checkpoint_selection_distinguishes_intra_epoch_candidates(
    artifact_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config()
    config.training.monitor = "val_macro_f1"
    config.training.monitor_mode = "max"
    first = artifact_dir / "epoch_0001_step_00005000.pth"
    second = artifact_dir / "epoch_0001_step_00010000.pth"
    first.write_bytes(b"candidate-1")
    second.write_bytes(b"candidate-2")

    class FakeTrainer:
        top_checkpoint_candidates = [
            CheckpointCandidate(
                epoch=1,
                batch_in_epoch=5000,
                global_step=5000,
                validation_index=1,
                checkpoint_label="epoch_0001_step_00005000",
                monitor_value=0.9,
                path=first,
            ),
            CheckpointCandidate(
                epoch=1,
                batch_in_epoch=10000,
                global_step=10000,
                validation_index=2,
                checkpoint_label="epoch_0001_step_00010000",
                monitor_value=0.8,
                path=second,
            ),
        ]

    def fake_evaluate_candidate(**kwargs: object) -> dict[str, object]:
        candidate = kwargs["candidate"]
        assert isinstance(candidate, CheckpointCandidate)
        return {
            "epoch": candidate.epoch,
            "batch_in_epoch": candidate.batch_in_epoch,
            "global_step": candidate.global_step,
            "validation_index": candidate.validation_index,
            "checkpoint_label": candidate.checkpoint_label,
            "checkpoint_path": candidate.path,
            "raw_monitor_value": candidate.monitor_value,
            "postprocessed_monitor_value": 0.7 if candidate.validation_index == 1 else 0.95,
            "validation_loss": 1.0,
            "validation_metrics": _dummy_metrics(),
            "validation_expert_usage": None,
            "validation_outputs": {},
            "postprocessed_validation_metrics": _dummy_metrics(),
            "postprocessed_validation_outputs": {},
            "validation_seconds": 0.01,
            "postprocessing_seconds": 0.02,
            "candidate_dir": kwargs["candidate_dir"],
            "temperature_scaling": {"enabled": False},
            "temperature_scaling_path": None,
            "directional_thresholds": {"enabled": False},
            "directional_thresholds_path": None,
        }

    monkeypatch.setattr("run_training.evaluate_checkpoint_candidate_on_validation", fake_evaluate_candidate)

    selection = select_checkpoint_after_validation_postprocessing(
        config=config,
        trainer=FakeTrainer(),  # type: ignore[arg-type]
        model=object(),
        validation_loader=object(),  # type: ignore[arg-type]
        history=[
            EpochResult(
                train_loss=1.0,
                val_loss=1.0,
                val_metrics=_dummy_metrics(),
                epoch=1,
                batch_in_epoch=5000,
                global_step=5000,
                validation_index=1,
                checkpoint_label="epoch_0001_step_00005000",
            ),
            EpochResult(
                train_loss=1.0,
                val_loss=1.0,
                val_metrics=_dummy_metrics(),
                epoch=1,
                batch_in_epoch=10000,
                global_step=10000,
                validation_index=2,
                checkpoint_label="epoch_0001_step_00010000",
            ),
        ],
        fold="fold_001",
        selection_dir=artifact_dir / "checkpoint_selection",
    )

    assert selection["summary"]["selected_epoch"] == 1
    assert selection["summary"]["selected_validation_index"] == 2
    assert selection["summary"]["selected_checkpoint_label"] == "epoch_0001_step_00010000"
    assert selection["summary"]["candidates"][1]["global_step"] == 10000


def test_checkpoint_selection_keeps_raw_validation_loss_monitor() -> None:
    config = load_config()
    config.training.monitor = "val_loss"
    config.training.monitor_mode = "min"

    value = monitor_value_after_postprocessing(
        config=config,
        raw_monitor_value=0.42,
        val_loss=99.0,
        val_metrics=_dummy_metrics(),
    )

    assert value == pytest.approx(0.42)


def test_train_fold_rejects_missing_validation_sequences(
    artifact_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDataset:
        def __init__(self, length: int) -> None:
            self.length = length

        def __len__(self) -> int:
            return self.length

    def fake_build_dataset(
        _sequence_dir: Path,
        split: str,
        _sequence_window: int,
        **_kwargs,
    ) -> FakeDataset:
        return FakeDataset(1 if split == "train" else 0)

    monkeypatch.setattr("run_training.build_dataset", fake_build_dataset)
    config = load_config()

    with pytest.raises(ValueError, match="No validation sequences"):
        train_fold(
            config=config,
            fold_id="fold_001",
            fold_sequence_dir=artifact_dir / "sequences" / "fold_001",
            fold_log_dir=artifact_dir / "logs" / "fold_001",
            fold_result_dir=artifact_dir / "results" / "fold_001",
            run_stem="run_1",
        )
