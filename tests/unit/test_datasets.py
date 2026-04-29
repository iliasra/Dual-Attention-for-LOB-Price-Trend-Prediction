from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from configuration import DataConfig
from datasets import DailySequenceBuilder, LOBDataset


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
    config = DataConfig(feature_exclude_columns=["drop_me"])

    assert DailySequenceBuilder(config).feature_columns(df) == ["keep_a", "keep_b"]


def test_build_creates_sliding_feature_time_and_label_windows() -> None:
    df = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_a": [10, 20, 30, 40, 50],
            "feature_b": [1, 2, 3, 4, 5],
            "trend_label": [-1, 0, 1, -1, 0],
        }
    )
    config = DataConfig(sequence_window=3)

    features, times, labels = DailySequenceBuilder(config).build(df)

    assert features.shape == (3, 3, 2) # (3 sliding windows, 3 timesteps per window, 2 feature columns (feature_a, feature_b))
    assert times.shape == (3, 3) #(3 sliding windows, 3 timesteps per window)
    assert labels.tolist() == [2, 0, 1] #(1 label per window)
    np.testing.assert_allclose(features[0], [[10, 1], [20, 2], [30, 3]])
    np.testing.assert_allclose(times[0], [1.0, 2.0, 3.0])


def test_lob_dataset_getitem_returns_sequence_window_starting_at_idx(artifact_dir: Path) -> None:
    df = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_a": [10, 20, 30, 40, 50],
            "feature_b": [1, 2, 3, 4, 5],
            "trend_label": [-1, 0, 1, -1, 0],
        }
    )
    config = DataConfig(sequence_window=3)
    x_path, t_path, y_path = DailySequenceBuilder(config).save(df, artifact_dir / "toy_day")
    dataset = LOBDataset([str(x_path)], [str(t_path)], [str(y_path)], sequence_window=config.sequence_window)

    x_seq, t_seq, y_label = dataset[1]

    assert tuple(x_seq.shape) == (3, 2)  # sequence_window events, 2 feature columns.
    assert tuple(t_seq.shape) == (3,)
    np.testing.assert_allclose(x_seq.numpy(), [[20, 2], [30, 3], [40, 4]])
    np.testing.assert_allclose(t_seq.numpy(), [2.0, 3.0, 4.0])
    assert y_label.item() == 0


def test_build_rejects_unknown_labels() -> None:
    df = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0],
            "feature": [10, 20, 30],
            "trend_label": [-1, 7, 1],
        }
    )
    config = DataConfig(sequence_window=2)

    with pytest.raises(ValueError, match="label_mapping"):
        DailySequenceBuilder(config).build(df)
