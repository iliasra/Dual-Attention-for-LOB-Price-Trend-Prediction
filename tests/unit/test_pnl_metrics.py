from __future__ import annotations

from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd

from pnl_metrics import add_pnl_columns, build_pnl_samples, non_overlapping_subset, supervised_y_positions


def _config(*, tick_size: float = 0.01, token_chunk: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        data=SimpleNamespace(
            raw_data_dir="",
            tick_size=tick_size,
            sequence_window=5,
            label_mapping={-1: 0, 0: 1, 1: 2},
        ),
        path=None,
        preprocessing=SimpleNamespace(
            sample_clock=SimpleNamespace(enabled=False),
            snapshot_window=2,
            temporal_features=SimpleNamespace(
                market_open_seconds=0.0,
                market_close_seconds=100.0,
                start_offset_minutes=0.0,
                end_offset_minutes=0.0,
            ),
            labels=SimpleNamespace(
                strategy="smoothing",
                smoothing=SimpleNamespace(
                    method="C",
                    threshold=0.0,
                    k=0,
                    h=2,
                    bid_column="bid_price_1",
                    ask_column="ask_price_1",
                    adaptive_threshold=None,
                ),
            ),
        ),
        training=SimpleNamespace(
            sequence_supervision=SimpleNamespace(
                token_chunk_enabled=token_chunk,
                loss_warmup_tokens=2,
            ),
        ),
    )


def test_add_pnl_columns_uses_crossing_prices_and_fees() -> None:
    samples = pd.DataFrame(
        {
            "prediction": [2, 0, 1],
            "entry_bid": [100.00, 100.00, 100.00],
            "entry_ask": [100.01, 100.01, 100.01],
            "entry_mid": [100.005, 100.005, 100.005],
            "exit_bid": [100.04, 99.96, 100.10],
            "exit_ask": [100.05, 99.97, 100.11],
            "exit_mid": [100.045, 99.965, 100.105],
        }
    )

    result = add_pnl_columns(samples, config=_config(), fees_bps=1.0)

    np.testing.assert_array_equal(result["position"], [1, -1, 0])
    np.testing.assert_allclose(result["cross_pnl_ticks"], [3.0, 3.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(result["net_cross_pnl_ticks"], [1.99995, 1.99995, 0.0], atol=1e-10)
    np.testing.assert_allclose(result["mid_pnl_ticks"], [4.0, 4.0, 0.0], atol=1e-10)


def test_non_overlapping_subset_keeps_first_non_overlapping_trades() -> None:
    trades = pd.DataFrame(
        {
            "date": ["2024-03-04"] * 4,
            "entry_raw_index": [10, 12, 16, 17],
            "exit_raw_index": [15, 18, 20, 21],
        }
    )

    result = non_overlapping_subset(trades)

    assert result["entry_raw_index"].tolist() == [10, 16]


def test_supervised_y_positions_match_last_window_and_token_chunk_offsets() -> None:
    last_window = supervised_y_positions(8, _config(token_chunk=False))
    token_chunk = supervised_y_positions(8, _config(token_chunk=True))

    np.testing.assert_array_equal(last_window, [4, 5, 6, 7])
    np.testing.assert_array_equal(token_chunk, [2, 3, 4, 5, 6, 7])


def test_build_pnl_samples_aligns_last_window_predictions_to_raw_rows() -> None:
    artifact_dir = Path(__file__).resolve().parent / ".test_artifacts" / "pnl_alignment"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True)
    raw_dir = artifact_dir / "raw"
    raw_dir.mkdir()
    message_path = raw_dir / "TEST_2024-03-04_0_100_message_10.csv"
    orderbook_path = raw_dir / "TEST_2024-03-04_0_100_orderbook_10.csv"
    message_path.write_text("\n".join(f"{idx},0,0,0,0,0" for idx in range(8)), encoding="utf-8")
    orderbook_path.write_text(
        "\n".join(f"{100.01 + idx:.2f},1,{99.99 + idx:.2f},1" for idx in range(8)),
        encoding="utf-8",
    )
    labels_path = artifact_dir / "TEST_2024-03-04_labels.npy"
    np.save(labels_path, np.asarray([2, 2, 2, 2, 2], dtype=np.int64))
    outputs = {
        "targets": np.asarray([2, 2, 2], dtype=np.int64),
        "predictions": np.asarray([2, 1, 0], dtype=np.int64),
    }
    config = _config(token_chunk=False)
    config.data.sequence_window = 3

    samples, metadata = build_pnl_samples(
        config=config,
        outputs=outputs,
        y_paths=[labels_path],
        raw_dir=raw_dir,
        split="test",
    )

    assert metadata["true_label_match_rate"] == 1.0
    assert metadata["invalid_exit_count"] == 0
    assert samples["entry_raw_index"].tolist() == [3, 4, 5]
    assert samples["exit_raw_index"].tolist() == [5, 6, 7]
