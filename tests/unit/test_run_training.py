from __future__ import annotations

import sys
from pathlib import Path
import shutil

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_training import (
    build_train_sampler,
    fold_artifact_paths,
    resolve_class_weights,
    sequence_label_values,
    sequence_time_span_quantile,
    train_fold,
)
from configuration import load_config


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

    sampler, summary = build_train_sampler(config, dataset, seed=123)

    assert sampler is not None
    assert summary["enabled"] is True
    assert sampler.sampled_class_counts(config.model.num_classes) == [1, 4, 1]


def test_resolve_class_weights_can_use_sampled_class_counts() -> None:
    config = load_config()

    summary = resolve_class_weights(config, train_dataset=object(), sampled_class_counts=[1, 4, 1])

    assert summary["source"] == "sampled_train_per_epoch"
    assert summary["counts"] == [1, 4, 1]
    assert config.training.class_weights == summary["weights"]


def test_train_fold_rejects_missing_validation_sequences(
    artifact_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDataset:
        def __init__(self, length: int) -> None:
            self.length = length

        def __len__(self) -> int:
            return self.length

    def fake_build_dataset(_sequence_dir: Path, split: str, _sequence_window: int) -> FakeDataset:
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
