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
    config = DataConfig(
        raw_data_dir="",
        processed_data_dir="",
        sequence_data_dir="",
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
        self.X_data = [np.load(path, mmap_mode="r") for path in x_paths]
        self.T_data = [np.load(path, mmap_mode="r") for path in t_paths]
        self.y_data = [np.load(path, mmap_mode="r") for path in y_paths]
        self.sequence_window = sequence_window
        self._validate_arrays()

        self.lengths = [len(labels) for labels in self.y_data]
        self.cumulative_lengths = np.cumsum(self.lengths)

    def _validate_arrays(self) -> None:
        if not (len(self.X_data) == len(self.T_data) == len(self.y_data)):
            raise ValueError("x_paths, t_paths, and y_paths must contain the same number of files.")

        for day_idx, (features, times, labels) in enumerate(zip(self.X_data, self.T_data, self.y_data)):
            if features.ndim != 3:
                raise ValueError(
                    f"X file at index {day_idx} must contain pre-built windows with shape "
                    "[num_windows, sequence_window, num_features]."
                )
            if times.ndim != 2:
                raise ValueError(
                    f"T file at index {day_idx} must contain time windows with shape "
                    "[num_windows, sequence_window]."
                )
            if len(features) != len(times) or len(features) != len(labels):
                raise ValueError(f"X, T, and y files at index {day_idx} must contain the same number of windows.")
            if self.sequence_window is not None and features.shape[1] != self.sequence_window:
                raise ValueError(
                    f"X file at index {day_idx} has sequence length {features.shape[1]}, "
                    f"expected {self.sequence_window}."
                )
            if self.sequence_window is not None and times.shape[1] != self.sequence_window:
                raise ValueError(
                    f"T file at index {day_idx} has sequence length {times.shape[1]}, "
                    f"expected {self.sequence_window}."
                )

    def __len__(self) -> int:
        return int(self.cumulative_lengths[-1]) if len(self.cumulative_lengths) else 0

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        day_idx = int(np.searchsorted(self.cumulative_lengths, idx, side="right"))
        local_idx = idx if day_idx == 0 else idx - int(self.cumulative_lengths[day_idx - 1])

        x_seq = torch.from_numpy(np.array(self.X_data[day_idx][local_idx], dtype=np.float32, copy=True))
        t_seq = torch.from_numpy(np.array(self.T_data[day_idx][local_idx], dtype=np.float32, copy=True))
        y_label = torch.tensor(int(self.y_data[day_idx][local_idx]), dtype=torch.long)

        return x_seq, t_seq, y_label
