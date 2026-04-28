from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from configuration import DataConfig
from datasets import DailySequenceBuilder


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
