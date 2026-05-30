from __future__ import annotations

from pathlib import Path
import shutil
from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip("torch")

from configuration import FoldConfig
from processing import LobFilePair, LobFileSegment, LobProcessingPipeline, ProcessedDay


@pytest.fixture()
def artifact_dir(request: pytest.FixtureRequest) -> Path:
    path = Path(__file__).resolve().parent / ".test_artifacts" / request.node.name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


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


class _FakeSequenceBuilder:
    def save(self, _df: pd.DataFrame, save_prefix: str | Path) -> tuple[Path, Path, Path]:
        prefix = Path(save_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        paths = (
            prefix.with_name(f"{prefix.name}_features.npy"),
            prefix.with_name(f"{prefix.name}_times.npy"),
            prefix.with_name(f"{prefix.name}_labels.npy"),
        )
        for path in paths:
            path.write_text("placeholder", encoding="utf-8")
        return paths


def _pipeline_for_output_saving(artifact_dir: Path, *, save_processed_dataframes: bool) -> LobProcessingPipeline:
    pipeline = object.__new__(LobProcessingPipeline)
    pipeline.config = SimpleNamespace(
        preprocessing=SimpleNamespace(save_processed_dataframes=save_processed_dataframes),
    )
    pipeline.processed_dir = artifact_dir / "processed"
    pipeline.sequence_dir = artifact_dir / "sequences"
    pipeline.sequence_builder = _FakeSequenceBuilder()
    return pipeline


def _processed_day() -> ProcessedDay:
    return ProcessedDay(
        split="train",
        pair=_pair("2020-01-01"),
        raw=None,
        joined=None,
        labeled=None,
        message_features=None,
        normalized=pd.DataFrame({"time": [1.0], "trend_label": [0], "feature": [2.0]}),
    )


def test_save_split_outputs_skips_processed_csv_when_disabled(artifact_dir: Path) -> None:
    pipeline = _pipeline_for_output_saving(artifact_dir, save_processed_dataframes=False)
    day = _processed_day()

    pipeline.save_split_outputs({"train": [day]})

    assert day.processed_csv_path is None
    assert not (artifact_dir / "processed").exists()
    assert (artifact_dir / "sequences" / "train" / "TEST_2020-01-01_features.npy").exists()


def test_save_split_outputs_writes_processed_csv_when_enabled(artifact_dir: Path) -> None:
    pipeline = _pipeline_for_output_saving(artifact_dir, save_processed_dataframes=True)
    day = _processed_day()

    pipeline.save_split_outputs({"train": [day]})

    expected_csv = artifact_dir / "processed" / "train" / "TEST_2020-01-01_processed.csv"
    assert day.processed_csv_path == expected_csv
    assert expected_csv.exists()
    assert (artifact_dir / "sequences" / "train" / "TEST_2020-01-01_features.npy").exists()


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
