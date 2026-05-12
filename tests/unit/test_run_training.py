from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_training import fold_artifact_paths, sequence_label_values, sequence_time_span_quantile


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
