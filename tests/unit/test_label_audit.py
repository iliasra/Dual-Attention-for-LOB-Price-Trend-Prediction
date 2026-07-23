from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.audit_labels import effective_sample_size, label_temporal_tables


def test_label_temporal_tables_report_cluster_elapsed_time_and_event_spacing() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02"] * 6,
            "raw_event_index": np.arange(6),
            "decision_time": [10.0, 10.1, 10.4, 11.0, 11.2, 12.0],
            "broad_trend_label": [1, 1, 1, 0, 0, -1],
            "exec_trend_label": [0, 1, 1, 1, -1, -1],
        }
    )

    clusters, ess = label_temporal_tables(frame, max_acf_lag=3)

    broad_up = clusters[(clusters["label_family"] == "broad") & (clusters["class"] == 1)].iloc[0]
    assert broad_up["events"] == 3
    assert broad_up["elapsed_seconds"] == pytest.approx(0.4)
    assert broad_up["mean_seconds_between_events"] == pytest.approx(0.2)
    assert set(ess["class"]) == {-1, 0, 1}


def test_effective_sample_size_is_bounded_by_observations() -> None:
    ess, _lags = effective_sample_size(np.asarray([1, 1, 0, 0, 1, 1], dtype=float), 5)
    assert 0.0 < ess <= 6.0
