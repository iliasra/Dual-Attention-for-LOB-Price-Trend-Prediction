from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
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

        feature_windows = sliding_window_view(features, window_shape=sequence_length, axis=0)
        feature_windows = np.swapaxes(feature_windows, 1, 2).copy()
        time_windows = sliding_window_view(times, window_shape=sequence_length, axis=0).copy()
        label_windows = labels[sequence_length - 1 :].copy()

        return feature_windows, time_windows, label_windows

    def save(self, df: pd.DataFrame, save_prefix: str | Path) -> tuple[Path, Path, Path]:
        prefix = Path(save_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        features, times, labels = self.build(df)

        x_path = prefix.with_name(f"{prefix.name}_X.npy")
        t_path = prefix.with_name(f"{prefix.name}_T.npy")
        y_path = prefix.with_name(f"{prefix.name}_y.npy")

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
    config = DataConfig(time_column=time_col, label_column=label_col, sequence_window=T_window)
    return DailySequenceBuilder(config).save(df, save_prefix)


class LOBDataset(Dataset):
    def __init__(self, x_paths: list[str], t_paths: list[str], y_paths: list[str]):
        self.X_data = [np.load(path, mmap_mode="r") for path in x_paths]
        self.T_data = [np.load(path, mmap_mode="r") for path in t_paths]
        self.y_data = [np.load(path, mmap_mode="r") for path in y_paths]

        self.lengths = [len(labels) for labels in self.y_data]
        self.cumulative_lengths = np.cumsum(self.lengths)

    def __len__(self) -> int:
        return int(self.cumulative_lengths[-1]) if len(self.cumulative_lengths) else 0

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        day_idx = int(np.searchsorted(self.cumulative_lengths, idx, side="right"))
        local_idx = idx if day_idx == 0 else idx - int(self.cumulative_lengths[day_idx - 1])

        x_seq = torch.from_numpy(np.array(self.X_data[day_idx][local_idx], dtype=np.float32, copy=True))
        t_seq = torch.from_numpy(np.array(self.T_data[day_idx][local_idx], dtype=np.float32, copy=True))
        y_label = torch.tensor(int(self.y_data[day_idx][local_idx]), dtype=torch.long)

        return x_seq, t_seq, y_label
