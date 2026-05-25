from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from configuration import FoldConfig
from processing import LobFilePair, LobFileSegment, LobProcessingPipeline


def _pair(date: str) -> LobFilePair:
    return LobFilePair(
        symbol="TEST",
        date=date,
        segments=(
            LobFileSegment(
                message_path=Path(f"TEST_{date}_message_10.csv"),
                orderbook_path=Path(f"TEST_{date}_orderbook_10.csv"),
            ),
        ),
    )


def test_processing_pipeline_runs_only_selected_fold_dates() -> None:
    folds = [
        FoldConfig(
            id="fold_001",
            train_dates=["2020-01-05"],
            validation_dates=["2020-01-06"],
            test_dates=["2020-01-07"],
        ),
        FoldConfig(
            id="fold_002",
            train_dates=["2020-01-01", "2020-01-02"],
            validation_dates=["2020-01-03"],
            test_dates=["2020-01-04"],
        ),
    ]
    pairs = [
        _pair(date)
        for date in (
            "2020-01-01",
            "2020-01-02",
            "2020-01-03",
            "2020-01-04",
            "2020-01-05",
            "2020-01-06",
            "2020-01-07",
        )
    ]
    pipeline = object.__new__(LobProcessingPipeline)
    pipeline.config = SimpleNamespace(folds=folds)
    run_fold_dates: list[dict[str, list[str]]] = []

    pipeline.discover_pairs = lambda: pairs

    def run_fold(fold: FoldConfig, split_pairs: dict[str, list[LobFilePair]]):
        run_fold_dates.append({split: [pair.date for pair in split_pairs[split]] for split in split_pairs})
        return {split: {pair.output_stem: (0, 0) for pair in split_pairs[split]} for split in split_pairs}

    pipeline.run_fold = run_fold

    summary = pipeline.run(selected_fold_ids={"fold_002"})

    assert list(summary) == ["fold_002"]
    assert run_fold_dates == [
        {
            "train": ["2020-01-01", "2020-01-02"],
            "validation": ["2020-01-03"],
            "test": ["2020-01-04"],
        }
    ]
    assert "TEST_2020-01-04" in summary["fold_002"]["test"]


def test_processing_pipeline_rejects_unknown_selected_fold() -> None:
    pipeline = object.__new__(LobProcessingPipeline)
    pipeline.config = SimpleNamespace(
        folds=[
            FoldConfig(
                id="fold_001",
                train_dates=["2020-01-01"],
                validation_dates=["2020-01-02"],
                test_dates=["2020-01-03"],
            )
        ]
    )
    pipeline.discover_pairs = lambda: []

    with pytest.raises(ValueError, match="Unknown fold id"):
        pipeline.run(selected_fold_ids={"fold_999"})
