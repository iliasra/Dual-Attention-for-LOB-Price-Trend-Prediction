from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
import torch

from configuration import DataConfig
from datasets import DailySequenceBuilder, EpochNeutralDownsamplingSampler, LOBDataset, LOBTokenChunkDataset


def make_data_config(
    sequence_window: int = 64,
    feature_exclude_columns: list[str] | None = None,
) -> DataConfig:
    return DataConfig(
        raw_data_dir="",
        processed_data_dir="",
        sequence_data_dir="",
        logs_dir="",
        tick_size=1.0,
        time_column="time",
        label_column="trend_label",
        label_mapping={-1: 0, 0: 1, 1: 2},
        price_columns=None,
        volume_columns=None,
        feature_exclude_columns=feature_exclude_columns or [],
        sequence_window=sequence_window,
    )


@pytest.fixture()
def artifact_dir(request: pytest.FixtureRequest) -> Path:
    path = Path(__file__).resolve().parent / ".test_artifacts" / request.node.name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def test_feature_columns_excludes_time_label_and_configured_columns() -> None:
    df = pd.DataFrame(
        {
            "time": [1.0],
            "trend_label": [0],
            "keep_a": [10.0],
            "drop_me": [20.0],
            "keep_b": [30.0],
        }
    )
    config = make_data_config(feature_exclude_columns=["drop_me"])

    assert DailySequenceBuilder(config).feature_columns(df) == ["keep_a", "keep_b"]


def test_build_creates_compact_feature_time_and_label_arrays() -> None:
    df = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_a": [10, 20, 30, 40, 50],
            "feature_b": [1, 2, 3, 4, 5],
            "trend_label": [-1, 0, 1, -1, 0],
        }
    )
    config = make_data_config(sequence_window=3)

    features, times, labels = DailySequenceBuilder(config).build(df)

    assert features.shape == (5, 2)
    assert times.shape == (5,)
    assert labels.tolist() == [0, 1, 2, 0, 1]
    np.testing.assert_allclose(features, [[10, 1], [20, 2], [30, 3], [40, 4], [50, 5]])
    np.testing.assert_allclose(times, [1.0, 2.0, 3.0, 4.0, 5.0])


def test_build_creates_two_regression_targets_without_feature_leakage() -> None:
    df = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0],
            "feature": [4.0, 5.0, 6.0],
            "trend_label": [1, -1, 0],
            "long_net_return_ticks": [1.5, -0.5, 0.1],
            "short_net_return_ticks": [-2.0, 0.25, -0.2],
        }
    )
    config = make_data_config(sequence_window=2)
    config.target_columns = ["long_net_return_ticks", "short_net_return_ticks"]

    features, _times, targets = DailySequenceBuilder(config).build(df)

    assert features.shape == (3, 1)
    np.testing.assert_allclose(features[:, 0], df["feature"])
    assert targets.dtype == np.float32
    np.testing.assert_allclose(targets, [[1.5, -2.0], [-0.5, 0.25], [0.1, -0.2]])


def test_lob_dataset_getitem_returns_sequence_window_starting_at_idx(artifact_dir: Path) -> None:
    df = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_a": [10, 20, 30, 40, 50],
            "feature_b": [1, 2, 3, 4, 5],
            "trend_label": [-1, 0, 1, -1, 0],
        }
    )
    config = make_data_config(sequence_window=3)
    x_path, t_path, y_path = DailySequenceBuilder(config).save(df, artifact_dir / "toy_day")
    dataset = LOBDataset([str(x_path)], [str(t_path)], [str(y_path)], sequence_window=config.sequence_window)

    assert x_path.name == "toy_day_features.npy"
    assert t_path.name == "toy_day_times.npy"
    assert y_path.name == "toy_day_labels.npy"
    assert len(dataset) == 3

    x_seq, t_seq, y_label = dataset[1]

    assert tuple(x_seq.shape) == (3, 2)  # sequence_window events, 2 feature columns.
    assert tuple(t_seq.shape) == (3,)
    np.testing.assert_allclose(x_seq.numpy(), [[20, 2], [30, 3], [40, 4]])
    np.testing.assert_allclose(t_seq.numpy(), [0.0, 1.0, 2.0])
    assert y_label.item() == 0


def test_lob_dataset_masks_only_endpoints_and_keeps_censored_rows_as_context(artifact_dir: Path) -> None:
    x_path = artifact_dir / "masked_features.npy"
    t_path = artifact_dir / "masked_times.npy"
    y_path = artifact_dir / "masked_labels.npy"
    mask_path = artifact_dir / "masked_supervision_mask.npy"
    broad_mask_path = artifact_dir / "masked_broad_supervision_mask.npy"
    np.save(x_path, np.arange(6, dtype=np.float32)[:, None])
    np.save(t_path, np.arange(6, dtype=np.float64))
    np.save(y_path, np.arange(6, dtype=np.int64))
    np.save(mask_path, np.asarray([False, False, False, True, False, True]))
    np.save(broad_mask_path, np.asarray([False, False, False, True, True, True]))

    common = LOBDataset([str(x_path)], [str(t_path)], [str(y_path)], sequence_window=3)
    broad = LOBDataset(
        [str(x_path)],
        [str(t_path)],
        [str(y_path)],
        sequence_window=3,
        supervision_support="broad",
    )

    assert len(common) == 2
    assert common.supervised_labels().tolist() == [3, 5]
    np.testing.assert_allclose(common[0][0].numpy().ravel(), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(common[1][0].numpy().ravel(), [3.0, 4.0, 5.0])
    assert broad.supervised_labels().tolist() == [3, 4, 5]


def test_lob_dataset_early_window_masks_endpoints_without_truncating_context(artifact_dir: Path) -> None:
    x_path = artifact_dir / "early_features.npy"
    t_path = artifact_dir / "early_times.npy"
    y_path = artifact_dir / "early_labels.npy"
    np.save(x_path, np.arange(7, dtype=np.float32)[:, None])
    np.save(t_path, np.asarray([34198.0, 34199.0, 34200.0, 34201.0, 34202.0, 34203.0, 34204.0]))
    np.save(y_path, np.arange(7, dtype=np.int64))

    dataset = LOBDataset(
        [str(x_path)],
        [str(t_path)],
        [str(y_path)],
        sequence_window=3,
        supervision_time_window=(34200.0, 34202.0),
    )

    assert dataset.supervised_labels().tolist() == [2, 3, 4]
    np.testing.assert_allclose(dataset[0][0].numpy().ravel(), [0.0, 1.0, 2.0])


def test_lob_dataset_preserves_microsecond_deltas_at_absolute_market_times(artifact_dir: Path) -> None:
    x_path = artifact_dir / "precise_features.npy"
    t_path = artifact_dir / "precise_times.npy"
    y_path = artifact_dir / "precise_labels.npy"
    np.save(x_path, np.ones((4, 2), dtype=np.float32))
    np.save(
        t_path,
        np.asarray([34_200.0, 34_200.000_001, 34_200.000_003, 34_200.000_008], dtype=np.float64),
    )
    np.save(y_path, np.asarray([0, 1, 2, 1], dtype=np.int64))

    dataset = LOBDataset([str(x_path)], [str(t_path)], [str(y_path)], sequence_window=4)
    _x, relative_times, _target = dataset[0]

    assert relative_times.dtype == torch.float32
    assert relative_times[0].item() == 0.0
    assert torch.all(relative_times[1:] > relative_times[:-1])
    np.testing.assert_allclose(
        relative_times.numpy(),
        [0.0, 1e-6, 3e-6, 8e-6],
        rtol=2e-4,
        atol=1e-10,
    )


def test_lob_dataset_can_preload_compact_arrays_to_memory(artifact_dir: Path) -> None:
    labels = [0, 1, 2, 1]
    x_path, t_path, y_path = save_compact_arrays(artifact_dir, labels)

    mmap_dataset = LOBDataset([str(x_path)], [str(t_path)], [str(y_path)], sequence_window=1)
    preloaded_dataset = LOBDataset(
        [str(x_path)],
        [str(t_path)],
        [str(y_path)],
        sequence_window=1,
        preload_to_memory=True,
    )

    assert isinstance(mmap_dataset.X_data[0], np.memmap)
    assert not isinstance(preloaded_dataset.X_data[0], np.memmap)
    assert preloaded_dataset.arrays_nbytes == (
        preloaded_dataset.X_data[0].nbytes
        + preloaded_dataset.T_data[0].nbytes
        + preloaded_dataset.y_data[0].nbytes
    )


def test_lob_token_chunk_dataset_supervises_tail_tokens_once(artifact_dir: Path) -> None:
    labels = list(range(10))
    x_path, t_path, y_path = save_compact_arrays(artifact_dir, labels)
    dataset = LOBTokenChunkDataset(
        [str(x_path)],
        [str(t_path)],
        [str(y_path)],
        sequence_window=6,
        loss_warmup_tokens=3,
        chunk_stride=3,
    )

    assert len(dataset) == 3
    assert dataset.supervised_labels().tolist() == labels[3:]

    _, _, y_0, mask_0, ids_0 = dataset[0]
    _, _, y_1, mask_1, ids_1 = dataset[1]
    _, _, y_2, mask_2, ids_2 = dataset[2]

    assert y_0[mask_0].tolist() == [3, 4, 5]
    assert y_1[mask_1].tolist() == [6, 7, 8]
    assert y_2[mask_2].tolist() == [9]
    supervised_ids = [
        *ids_0[mask_0].tolist(),
        *ids_1[mask_1].tolist(),
        *ids_2[mask_2].tolist(),
    ]
    assert supervised_ids == list(range(3, 10))


def test_lob_token_chunk_dataset_intersects_loss_mask_with_endpoint_mask(artifact_dir: Path) -> None:
    labels = list(range(10))
    x_path, t_path, y_path = save_compact_arrays(artifact_dir, labels)
    mask_path = artifact_dir / "sample_supervision_mask.npy"
    np.save(mask_path, np.asarray([False, False, False, True, False, True, False, False, False, True]))
    dataset = LOBTokenChunkDataset(
        [str(x_path)],
        [str(t_path)],
        [str(y_path)],
        sequence_window=6,
        loss_warmup_tokens=3,
        chunk_stride=3,
    )

    supervised = []
    for index in range(len(dataset)):
        _x, _t, targets, mask, _ids = dataset[index]
        supervised.extend(targets[mask].tolist())
    assert supervised == [3, 5, 9]
    assert dataset.supervised_labels().tolist() == [3, 5, 9]


def test_lob_dataset_rejects_inconsistent_feature_widths(artifact_dir: Path) -> None:
    x_path_1 = artifact_dir / "day_1_features.npy"
    x_path_2 = artifact_dir / "day_2_features.npy"
    t_path_1 = artifact_dir / "day_1_times.npy"
    t_path_2 = artifact_dir / "day_2_times.npy"
    y_path_1 = artifact_dir / "day_1_labels.npy"
    y_path_2 = artifact_dir / "day_2_labels.npy"

    np.save(x_path_1, np.ones((4, 2), dtype=np.float32))
    np.save(x_path_2, np.ones((4, 3), dtype=np.float32))
    np.save(t_path_1, np.arange(4, dtype=np.float32))
    np.save(t_path_2, np.arange(4, dtype=np.float32))
    np.save(y_path_1, np.zeros(4, dtype=np.int64))
    np.save(y_path_2, np.zeros(4, dtype=np.int64))

    with pytest.raises(ValueError, match="index 1.*expected 2"):
        LOBDataset(
            [str(x_path_1), str(x_path_2)],
            [str(t_path_1), str(t_path_2)],
            [str(y_path_1), str(y_path_2)],
            sequence_window=2,
        )


def test_build_rejects_unknown_labels() -> None:
    df = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0],
            "feature": [10, 20, 30],
            "trend_label": [-1, 7, 1],
        }
    )
    config = make_data_config(sequence_window=2)

    with pytest.raises(ValueError, match="label_mapping"):
        DailySequenceBuilder(config).build(df)


def save_compact_arrays(artifact_dir: Path, labels: list[int]) -> tuple[Path, Path, Path]:
    x_path = artifact_dir / "sample_features.npy"
    t_path = artifact_dir / "sample_times.npy"
    y_path = artifact_dir / "sample_labels.npy"
    np.save(x_path, np.arange(len(labels), dtype=np.float32).reshape(-1, 1))
    np.save(t_path, np.arange(len(labels), dtype=np.float32))
    np.save(y_path, np.asarray(labels, dtype=np.int64))
    return x_path, t_path, y_path


def test_epoch_neutral_downsampling_sampler_keeps_directional_and_limits_neutral(
    artifact_dir: Path,
) -> None:
    labels = [0, 1, 1, 1, 1, 1, 2, 1]
    x_path, t_path, y_path = save_compact_arrays(artifact_dir, labels)
    dataset = LOBDataset([str(x_path)], [str(t_path)], [str(y_path)], sequence_window=1)
    sampler = EpochNeutralDownsamplingSampler(
        dataset,
        label_mapping={-1: 0, 0: 1, 1: 2},
        neutral_to_directional_ratio=2.0,
        base_seed=7,
    )

    sampled_indices = list(sampler)

    assert {0, 6}.issubset(set(sampled_indices))
    assert sum(labels[index] == 1 for index in sampled_indices) == 4
    assert len(sampled_indices) == 6


def test_epoch_neutral_downsampling_sampler_changes_neutrals_by_epoch(
    artifact_dir: Path,
) -> None:
    labels = [0, 2, *([1] * 20)]
    x_path, t_path, y_path = save_compact_arrays(artifact_dir, labels)
    dataset = LOBDataset([str(x_path)], [str(t_path)], [str(y_path)], sequence_window=1)
    sampler = EpochNeutralDownsamplingSampler(
        dataset,
        label_mapping={-1: 0, 0: 1, 1: 2},
        neutral_to_directional_ratio=1.0,
        base_seed=11,
    )

    sampler.set_epoch(0)
    epoch_0_neutrals = {index for index in sampler if labels[index] == 1}
    sampler.set_epoch(1)
    epoch_1_neutrals = {index for index in sampler if labels[index] == 1}

    assert epoch_0_neutrals != epoch_1_neutrals


def test_epoch_neutral_downsampling_sampler_is_reproducible_for_same_epoch(
    artifact_dir: Path,
) -> None:
    labels = [0, 2, *([1] * 10)]
    x_path, t_path, y_path = save_compact_arrays(artifact_dir, labels)
    dataset = LOBDataset([str(x_path)], [str(t_path)], [str(y_path)], sequence_window=1)
    kwargs = {
        "label_mapping": {-1: 0, 0: 1, 1: 2},
        "neutral_to_directional_ratio": 1.0,
        "base_seed": 13,
    }
    sampler_a = EpochNeutralDownsamplingSampler(dataset, **kwargs)
    sampler_b = EpochNeutralDownsamplingSampler(dataset, **kwargs)

    sampler_a.set_epoch(3)
    sampler_b.set_epoch(3)

    assert list(sampler_a) == list(sampler_b)
