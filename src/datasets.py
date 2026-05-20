from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from configuration import DataConfig
except ImportError:  # pragma: no cover
    from .configuration import DataConfig


@dataclass(slots=True)
class DailySequenceBuilder:
    config: DataConfig

    def feature_columns(self, df: pd.DataFrame) -> list[str]:
        excluded = {self.config.time_column, self.config.label_column, *self.config.feature_exclude_columns}
        return [column for column in df.columns if column not in excluded]

    def build(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract compact feature, timestamp, and label arrays from a dataframe."""
        sequence_length = self.config.sequence_window
        if len(df) < sequence_length:
            raise ValueError(
                f"Dataframe length ({len(df)}) must be >= sequence length ({sequence_length})."
            )

        feature_columns = self.feature_columns(df)
        times = df[self.config.time_column].to_numpy()
        labels = df[self.config.label_column].map(self.config.label_mapping).to_numpy(dtype=np.float32)
        if np.isnan(labels).any():
            raise ValueError("Encountered labels missing from the configured label_mapping.")
        labels = labels.astype(np.int64)
        features = df[feature_columns].to_numpy(dtype=np.float32)

        return features, times, labels

    def save(self, df: pd.DataFrame, save_prefix: str | Path) -> tuple[Path, Path, Path]:
        """Save a dataframe into 3 .npy files, divided in features/time/labels."""
        prefix = Path(save_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        features, times, labels = self.build(df)

        x_path = prefix.with_name(f"{prefix.name}_features.npy")
        t_path = prefix.with_name(f"{prefix.name}_times.npy")
        y_path = prefix.with_name(f"{prefix.name}_labels.npy")

        np.save(x_path, features)
        np.save(t_path, times)
        np.save(y_path, labels)
        return x_path, t_path, y_path


def process_daily_dataframe(
    df: pd.DataFrame,
    T_window: int = 64,
    time_col: str = "timestamp",
    label_col: str = "trend_label",
    save_prefix: str = "day_1",
) -> tuple[Path, Path, Path]:
    config = DataConfig(
        raw_data_dir="",
        processed_data_dir="",
        sequence_data_dir="",
        logs_dir="",
        tick_size=0.0,
        time_column=time_col,
        label_column=label_col,
        label_mapping={-1: 0, 0: 1, 1: 2},
        price_columns=None,
        volume_columns=None,
        feature_exclude_columns=[],
        sequence_window=T_window,
    )
    return DailySequenceBuilder(config).save(df, save_prefix)


class LOBDataset(Dataset):
    def __init__(
        self,
        x_paths: list[str],
        t_paths: list[str],
        y_paths: list[str],
        sequence_window: int | None = None,
    ):
        if sequence_window is None:
            raise ValueError("sequence_window is required to reconstruct compact feature arrays.")
        if sequence_window <= 0:
            raise ValueError("sequence_window must be > 0.")

        self.X_data = [np.load(path, mmap_mode="r") for path in x_paths] #mmap_mode="r" allows to partially load 
        self.T_data = [np.load(path, mmap_mode="r") for path in t_paths] #the files in the RAM, not the full array.
        self.y_data = [np.load(path, mmap_mode="r") for path in y_paths]
        self.sequence_window = int(sequence_window)
        self._validate_arrays()

        self.lengths = [len(labels) - self.sequence_window + 1 for labels in self.y_data]
        self.cumulative_lengths = np.cumsum(self.lengths)

    def _validate_arrays(self) -> None:
        """Check consistency across files length/dimensions."""
        if not (len(self.X_data) == len(self.T_data) == len(self.y_data)):
            raise ValueError("x_paths, t_paths, and y_paths must contain the same number of files.")

        expected_num_features: int | None = None
        for day_idx, (features, times, labels) in enumerate(zip(self.X_data, self.T_data, self.y_data)):
            if features.ndim != 2:
                raise ValueError(
                    f"Feature file at index {day_idx} must contain compact features with shape "
                    "[num_rows, num_features]."
                )
            num_features = int(features.shape[1])
            if expected_num_features is None:
                expected_num_features = num_features
            elif num_features != expected_num_features:
                raise ValueError(
                    f"Feature file at index {day_idx} has {num_features} feature columns, "
                    f"expected {expected_num_features}. All features.npy files must share shape[1]."
                )
            if times.ndim != 1:
                raise ValueError(
                    f"Time file at index {day_idx} must contain compact times with shape [num_rows]."
                )
            if len(features) != len(times) or len(features) != len(labels):
                raise ValueError(f"Feature, time, and label files at index {day_idx} must contain the same rows.")
            if len(labels) < self.sequence_window:
                raise ValueError(
                    f"Files at index {day_idx} contain {len(labels)} rows, "
                    f"but sequence_window is {self.sequence_window}."
                )

    def __len__(self) -> int:
        return int(self.cumulative_lengths[-1]) if len(self.cumulative_lengths) else 0

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns one sliding sequence window and its target label."""
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} is out of bounds for dataset of length {len(self)}.")

        day_idx = int(np.searchsorted(self.cumulative_lengths, idx, side="right"))
        local_idx = idx if day_idx == 0 else idx - int(self.cumulative_lengths[day_idx - 1])
        end_idx = local_idx + self.sequence_window

        # To leverage on mmap_mode="r", we access sequences through slicing.
        x_seq = torch.from_numpy(np.array(self.X_data[day_idx][local_idx:end_idx], dtype=np.float32, copy=True))
        t_seq = torch.from_numpy(np.array(self.T_data[day_idx][local_idx:end_idx], dtype=np.float32, copy=True))
        y_label = torch.tensor(int(self.y_data[day_idx][end_idx - 1]), dtype=torch.long)

        return x_seq, t_seq, y_label
