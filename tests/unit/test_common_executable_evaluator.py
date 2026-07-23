from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from datasets import LOBDataset, attach_evaluation_metadata
from configuration import load_config
from processing import LobFilePair, LobFileSegment, LobProcessingPipeline, ProcessedDay
from common_executable_evaluator import (
    evaluate_common_models,
    load_classification_predictions,
    save_common_evaluation,
    select_daily_fixed_budget,
    summarize_support_censoring,
)


@pytest.fixture
def artifact_dir(request: pytest.FixtureRequest):
    path = Path(__file__).resolve().parent / ".test_artifacts" / request.node.name
    path.mkdir(parents=True, exist_ok=True)
    return path


def prediction_frame(model: str, raw_indices: list[int]) -> pd.DataFrame:
    raw = np.asarray(raw_indices, dtype=np.int64)
    long = np.where(raw % 3 == 0, 1.0, -0.2)
    short = np.where(raw % 3 == 1, 0.8, -0.3)
    labels = np.where(long > 0.0, "up", np.where(short > 0.0, "down", "neutral"))
    return pd.DataFrame(
        {
            "date": "2024-05-20",
            "raw_event_index": raw,
            "entry_index": raw + 1,
            "exit_index": raw + 3,
            "realized_long": long,
            "realized_short": short,
            "score_long": long + np.linspace(0.0, 0.01, len(raw)),
            "score_short": short + np.linspace(0.0, 0.01, len(raw)),
            "true_label": labels,
            "pred_label": labels,
            "model": model,
        }
    )


def test_common_evaluation_uses_exact_intersection_and_keeps_exec_full() -> None:
    broad = prediction_frame("BROAD", [2, 3, 4, 5, 6, 7])
    exec_cls = prediction_frame("EXEC_CLS", [1, 2, 3, 4, 5, 6])
    exec_av = prediction_frame("EXEC_AV", [6, 5, 4, 3, 2, 1])

    result = evaluate_common_models(broad, exec_cls, exec_av, budgets=[0.2], seed=7)

    assert result.support_audit.loc[0, "intersection_rows"] == 5
    triple = result.daily_metrics[result.daily_metrics["support"] == "triple_common"]
    assert set(triple["model"]) == {"BROAD", "EXEC_CLS", "EXEC_AV"}
    assert set(triple["eligible_rows"]) == {5}
    exec_full = result.daily_metrics[result.daily_metrics["support"] == "exec_full"]
    assert set(exec_full["eligible_rows"]) == {6}


def test_common_evaluation_writes_tables_and_summary(artifact_dir) -> None:
    broad = prediction_frame("BROAD", list(range(12)))
    exec_cls = prediction_frame("EXEC_CLS", list(range(12)))
    exec_av = prediction_frame("EXEC_AV", list(range(12)))

    result = evaluate_common_models(broad, exec_cls, exec_av, budgets=[0.1], seed=3)
    artifacts = save_common_evaluation(result, artifact_dir)

    assert artifacts["summary"].exists()
    assert artifacts["daily_metrics"].exists()
    assert artifacts["paired_differences"].exists()
    assert artifacts["label_confusion"].exists()


def test_broad_native_artifact_needs_no_executable_outcomes(artifact_dir) -> None:
    broad_path = artifact_dir / "broad.csv"
    pd.DataFrame(
        {
            "date": ["2024-05-20", "2024-05-20"],
            "raw_event_index": [1, 2],
            "true_label": ["up", "down"],
            "pred_label": ["up", "down"],
            "p_up": [0.8, 0.1],
            "p_down": [0.1, 0.7],
        }
    ).to_csv(broad_path, index=False)
    broad = load_classification_predictions(
        broad_path,
        model="BROAD",
        require_economic=False,
    )
    exec_cls = prediction_frame("EXEC_CLS", [1, 2])
    exec_av = prediction_frame("EXEC_AV", [1, 2])

    result = evaluate_common_models(broad, exec_cls, exec_av, budgets=[0.5])

    assert set(result.daily_metrics["support"]) == {"triple_common", "exec_full"}
    assert result.native_metrics.loc[result.native_metrics["model"] == "BROAD", "rows"].item() == 2
    av_metrics = result.native_metrics.loc[result.native_metrics["model"] == "EXEC_AV"].iloc[0]
    assert np.isclose(av_metrics["mae_long"], 0.005)
    assert "profitable_ap_short" in result.native_metrics.columns


def test_exec_outcome_mismatch_fails_closed() -> None:
    exec_cls = prediction_frame("EXEC_CLS", [1, 2, 3])
    exec_av = prediction_frame("EXEC_AV", [1, 2, 3])
    exec_av.loc[1, "realized_long"] += 1.0

    with pytest.raises(ValueError, match="disagree on 'realized_long'"):
        evaluate_common_models(exec_cls.copy().assign(model="BROAD"), exec_cls, exec_av, budgets=[0.1])


def test_exec_class_must_be_reconstructible_from_action_values() -> None:
    exec_cls = prediction_frame("EXEC_CLS", [1, 2, 3])
    exec_av = prediction_frame("EXEC_AV", [1, 2, 3])
    exec_cls.loc[0, "true_label"] = "up" if exec_cls.loc[0, "true_label"] != "up" else "down"

    with pytest.raises(ValueError, match="not exactly reconstructible"):
        evaluate_common_models(exec_cls.copy().assign(model="BROAD"), exec_cls, exec_av, budgets=[0.1])


def test_fixed_budget_is_side_exclusive_and_has_no_overlapping_intervals() -> None:
    frame = prediction_frame("BROAD", list(range(10)))
    # Every interval overlaps its immediate neighbours; both side scores are high
    # for the same best row, which must still yield at most one accepted action.
    trades, metrics = select_daily_fixed_budget(frame, budget=0.3, seed=42, support="test")

    assert not trades["raw_event_index"].duplicated().any()
    ordered = trades.sort_values("entry_index")
    assert all(
        int(right.entry_index) > int(left.exit_index)
        for left, right in zip(ordered.itertuples(), ordered.iloc[1:].itertuples())
    )
    assert metrics["long_trades"] <= 3
    assert metrics["short_trades"] <= 3
    assert metrics["executed_trades"] < metrics["requested_trades"]


def test_support_censor_summary_reports_day_and_time_bin() -> None:
    audit = pd.DataFrame(
        {
            "date": ["2024-05-20"] * 4,
            "raw_event_index": [1, 2, 3, 4],
            "decision_time": [34201.0, 34202.0, 35101.0, 35102.0],
            "broad_valid": [False, True, True, True],
            "exec_valid": [True, True, False, True],
            "feature_history_valid": [False, True, True, True],
            "broad_trend_label": [np.nan, 1, -1, 1],
            "exec_trend_label": [1, 1, np.nan, -1],
            "long_net_return_ticks": [0.2, 0.3, np.nan, -0.1],
            "short_net_return_ticks": [-0.1, -0.2, np.nan, 0.4],
        }
    )

    summary = summarize_support_censoring(audit)

    assert len(summary) == 2
    assert summary["intersection_rows"].sum() == 2
    assert summary["broad_only_rows"].sum() == 1
    assert summary["exec_only_rows"].sum() == 1
    assert np.isclose(summary["intersection_profitable_rate"].dropna().mean(), 1.0)
    assert "exec_only_oracle_mean_pnl_ticks" in summary


def test_dataset_metadata_is_sliced_in_evaluation_order(artifact_dir) -> None:
    prefix = artifact_dir / "INTC_2024-05-20"
    x_path = prefix.with_name(prefix.name + "_features.npy")
    t_path = prefix.with_name(prefix.name + "_times.npy")
    y_path = prefix.with_name(prefix.name + "_labels.npy")
    mask_path = prefix.with_name(prefix.name + "_supervision_mask.npy")
    metadata_path = prefix.with_name(prefix.name + "_endpoint_metadata.npz")
    np.save(x_path, np.ones((5, 2), dtype=np.float32))
    np.save(t_path, np.arange(5, dtype=np.float64))
    np.save(y_path, np.asarray([0, 1, 2, 1, 0], dtype=np.int64))
    np.save(mask_path, np.asarray([False, False, True, False, True]))
    np.savez_compressed(
        metadata_path,
        date=np.asarray(["2024-05-20"] * 5),
        raw_event_index=np.arange(10, 15, dtype=np.int64),
    )
    dataset = LOBDataset(
        [str(x_path)], [str(t_path)], [str(y_path)], sequence_window=3
    )

    enriched = attach_evaluation_metadata(
        {"targets": np.asarray([2, 0]), "predictions": np.zeros((2, 2))},
        dataset,
    )

    assert enriched["raw_event_index"].tolist() == [12, 14]
    assert enriched["date"].tolist() == ["2024-05-20"] * 2


def test_common_preprocessing_calculates_features_before_target_filtering() -> None:
    config = load_config(Path(__file__).resolve().parents[2] / "configs" / "pipeline_config.yaml")
    config.preprocessing.common_endpoint_support.enabled = True
    pipeline = LobProcessingPipeline(config)
    n = 500
    sampled = pd.DataFrame(
        {
            "time": np.arange(n, dtype=float),
            "raw_event_index": np.arange(n, dtype=np.int64),
            "bid_price_1": 10000.0 + np.sin(np.arange(n) / 20.0),
            "ask_price_1": 10001.0 + np.sin(np.arange(n) / 20.0),
        }
    )
    seen_rows = []

    def identity_features(frame):
        seen_rows.append(len(frame))
        return frame.copy()

    pipeline._message_features_from_labeled = identity_features
    pair = LobFilePair(
        symbol="INTC",
        date="2024-05-20",
        segments=(LobFileSegment(Path("message.csv"), Path("book.csv")),),
    )
    day = ProcessedDay(
        split="train",
        pair=pair,
        raw=sampled,
        joined=sampled,
        labeled=None,
        message_features=None,
    )

    result = pipeline._build_common_endpoint_day(day, sampled)

    assert seen_rows == [n]
    assert len(result.message_features) == n
    assert result.message_features["raw_event_index"].is_unique
    assert 0 < int(result.message_features["common_endpoint_valid"].sum()) < n
    assert result.message_features.loc[
        result.message_features["common_endpoint_valid"],
        ["trend_label", "long_net_return_ticks", "short_net_return_ticks"],
    ].notna().all().all()
    feature_columns = pipeline.sequence_builder.feature_columns(result.message_features)
    assert "long_net_return_ticks" not in feature_columns
    assert "short_net_return_ticks" not in feature_columns
    assert "raw_event_index" not in feature_columns
    assert "broad_trend_label" not in feature_columns
    assert "exec_trend_label" not in feature_columns

    sequence_frame, masks = pipeline._sequence_output_frame(result.message_features, pair.label)
    assert len(sequence_frame) == n
    assert set(masks) == {"common", "broad", "exec"}
    assert masks["common"].sum() < masks["broad"].sum() or masks["common"].sum() < masks["exec"].sum()
    # Invalid supervised endpoints remain physically present as causal context.
    assert sequence_frame.loc[~masks["common"], "raw_event_index"].notna().all()
    metadata = pipeline._endpoint_metadata_payload(result.message_features, pair.date)
    assert len(metadata["raw_event_index"]) == n
    assert metadata["entry_index"][-1] == -1
    assert metadata["broad_label"].dtype == np.int8
