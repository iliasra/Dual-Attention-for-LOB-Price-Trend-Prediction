from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

try:
    from configuration import DataConfig
except ImportError:  # pragma: no cover
    from .configuration import DataConfig


ENDPOINT_METADATA_SUFFIX = "_endpoint_metadata.npz"
SUPERVISION_MASK_SUFFIX = "_supervision_mask.npy"
EVALUATION_METADATA_COLUMNS = frozenset(
    {
        "raw_event_index",
        "decision_time",
        "entry_index",
        "exit_index",
        "broad_valid",
        "exec_valid",
        "feature_history_valid",
        "common_endpoint_valid",
        "broad_trend_label",
        "exec_trend_label",
        "long_net_return_ticks",
        "short_net_return_ticks",
        "censor_reason_code",
    }
)


def relative_time_tensor(time_values: np.ndarray) -> torch.Tensor:
    """Return window-relative float32 times without quantizing absolute timestamps.

    LOBSTER timestamps are seconds since midnight. Casting values around 34,000--
    57,000 seconds directly to float32 destroys sub-millisecond event spacing.
    Subtracting the window origin in float64 first retains that spacing while
    keeping the model input compact.
    """
    absolute_times = np.asarray(time_values, dtype=np.float64)
    if absolute_times.ndim != 1:
        raise ValueError("time_values must be one-dimensional.")
    if absolute_times.size == 0:
        return torch.empty(0, dtype=torch.float32)
    if not np.isfinite(absolute_times).all():
        raise ValueError("time_values contains non-finite timestamps.")
    relative_times = absolute_times - absolute_times[0]
    return torch.from_numpy(relative_times.astype(np.float32, copy=False))


@dataclass(slots=True)
class DailySequenceBuilder:
    config: DataConfig

    def feature_columns(self, df: pd.DataFrame) -> list[str]:
        excluded = {
            self.config.time_column,
            self.config.label_column,
            *self.config.feature_exclude_columns,
            *(self.config.target_columns or []),
            *EVALUATION_METADATA_COLUMNS,
        }
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
        if self.config.target_columns:
            missing_targets = sorted(set(self.config.target_columns) - set(df.columns))
            if missing_targets:
                raise ValueError(f"Missing configured regression target columns: {missing_targets}.")
            labels = df.loc[:, self.config.target_columns].to_numpy(dtype=np.float32)
            if not np.isfinite(labels).all():
                raise ValueError("Encountered non-finite regression targets.")
        else:
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
        *,
        preload_to_memory: bool = False,
        supervision_support: str = "common",
        supervision_time_window: tuple[float, float] | None = None,
    ):
        if sequence_window is None:
            raise ValueError("sequence_window is required to reconstruct compact feature arrays.")
        if sequence_window <= 0:
            raise ValueError("sequence_window must be > 0.")

        self.preload_to_memory = bool(preload_to_memory)
        self.x_paths = [Path(path) for path in x_paths]
        self.t_paths = [Path(path) for path in t_paths]
        self.y_paths = [Path(path) for path in y_paths]
        mmap_mode = None if self.preload_to_memory else "r"
        self.X_data = [np.load(path, mmap_mode=mmap_mode) for path in x_paths]
        self.T_data = [np.load(path, mmap_mode=mmap_mode) for path in t_paths]
        self.y_data = [np.load(path, mmap_mode=mmap_mode) for path in y_paths]
        self.supervision_support = supervision_support
        self.supervision_masks = _load_supervision_masks(
            self.y_paths,
            self.y_data,
            mmap_mode=mmap_mode,
            support=supervision_support,
        )
        self.supervision_masks = _apply_supervision_time_window(
            self.supervision_masks,
            self.T_data,
            supervision_time_window,
        )
        self.sequence_window = int(sequence_window)
        self._validate_arrays()

        self.endpoint_positions = [
            np.flatnonzero(
                np.asarray(mask, dtype=bool)
                & (np.arange(len(mask), dtype=np.int64) >= self.sequence_window - 1)
            )
            for mask in self.supervision_masks
        ]
        self.lengths = [len(positions) for positions in self.endpoint_positions]
        self.cumulative_lengths = np.cumsum(self.lengths)

    @property
    def arrays_nbytes(self) -> int:
        """Return bytes held by compact feature/time/label arrays."""
        return int(
            sum(array.nbytes for array in self.X_data)
            + sum(array.nbytes for array in self.T_data)
            + sum(array.nbytes for array in self.y_data)
        )

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
        endpoint = int(self.endpoint_positions[day_idx][local_idx])
        start_idx = endpoint - self.sequence_window + 1
        end_idx = endpoint + 1

        # Access compact arrays by slicing, then copy into a tensor-owned batch sample.
        x_seq = torch.from_numpy(np.array(self.X_data[day_idx][start_idx:end_idx], dtype=np.float32, copy=True))
        t_seq = relative_time_tensor(self.T_data[day_idx][start_idx:end_idx])
        raw_target = self.y_data[day_idx][endpoint]
        y_label = (
            torch.from_numpy(np.asarray(raw_target, dtype=np.float32).copy())
            if np.asarray(raw_target).ndim > 0
            else torch.tensor(int(raw_target), dtype=torch.long)
        )

        return x_seq, t_seq, y_label

    def supervised_labels(self) -> np.ndarray:
        """Return labels at valid endpoints while retaining all preceding rows as context."""
        parts = [
            np.asarray(labels)[positions]
            for labels, positions in zip(self.y_data, self.endpoint_positions)
            if len(positions)
        ]
        return np.concatenate(parts) if parts else np.asarray([], dtype=np.int64)


class LOBTokenChunkDataset(Dataset):
    """Return non-overlapping supervised tails inside longer causal chunks."""

    def __init__(
        self,
        x_paths: list[str],
        t_paths: list[str],
        y_paths: list[str],
        sequence_window: int | None = None,
        *,
        loss_warmup_tokens: int,
        chunk_stride: int,
        preload_to_memory: bool = False,
        supervision_support: str = "common",
        supervision_time_window: tuple[float, float] | None = None,
    ) -> None:
        if sequence_window is None:
            raise ValueError("sequence_window is required to reconstruct compact feature arrays.")
        if sequence_window <= 0:
            raise ValueError("sequence_window must be > 0.")
        if loss_warmup_tokens < 0 or loss_warmup_tokens >= sequence_window:
            raise ValueError("loss_warmup_tokens must be in [0, sequence_window).")
        if chunk_stride <= 0:
            raise ValueError("chunk_stride must be > 0.")

        self.preload_to_memory = bool(preload_to_memory)
        self.x_paths = [Path(path) for path in x_paths]
        self.t_paths = [Path(path) for path in t_paths]
        self.y_paths = [Path(path) for path in y_paths]
        mmap_mode = None if self.preload_to_memory else "r"
        self.X_data = [np.load(path, mmap_mode=mmap_mode) for path in x_paths]
        self.T_data = [np.load(path, mmap_mode=mmap_mode) for path in t_paths]
        self.y_data = [np.load(path, mmap_mode=mmap_mode) for path in y_paths]
        self.supervision_support = supervision_support
        self.supervision_masks = _load_supervision_masks(
            self.y_paths,
            self.y_data,
            mmap_mode=mmap_mode,
            support=supervision_support,
        )
        self.supervision_masks = _apply_supervision_time_window(
            self.supervision_masks,
            self.T_data,
            supervision_time_window,
        )
        self.sequence_window = int(sequence_window)
        self.loss_warmup_tokens = int(loss_warmup_tokens)
        self.chunk_stride = int(chunk_stride)
        self._validate_arrays()

        self.day_row_offsets = np.cumsum([0, *[len(labels) for labels in self.y_data[:-1]]]).astype(np.int64)
        self.chunks: list[tuple[int, int, int]] = []
        for day_idx, labels in enumerate(self.y_data):
            self.chunks.extend(self._day_chunks(day_idx, len(labels), self.supervision_masks[day_idx]))

    @property
    def arrays_nbytes(self) -> int:
        """Return bytes held by compact feature/time/label arrays."""
        return int(
            sum(array.nbytes for array in self.X_data)
            + sum(array.nbytes for array in self.T_data)
            + sum(array.nbytes for array in self.y_data)
        )

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

    def _day_chunks(
        self,
        day_idx: int,
        num_rows: int,
        supervision_mask: np.ndarray,
    ) -> list[tuple[int, int, int]]:
        if num_rows < self.sequence_window:
            return []
        last_start = num_rows - self.sequence_window
        starts = list(range(0, last_start + 1, self.chunk_stride))
        if not starts or starts[-1] != last_start:
            starts.append(last_start)
        unique_starts = sorted(set(starts))

        chunks: list[tuple[int, int, int]] = []
        covered_until = self.loss_warmup_tokens
        for start in unique_starts:
            supervise_from = max(start + self.loss_warmup_tokens, covered_until)
            supervise_to = start + self.sequence_window
            if supervise_from >= supervise_to:
                continue
            if np.asarray(supervision_mask[supervise_from:supervise_to], dtype=bool).any():
                chunks.append((int(day_idx), int(start), int(supervise_from)))
            covered_until = supervise_to
        return chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one chunk, all labels in the chunk, a supervised-token mask, and token ids."""
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} is out of bounds for dataset of length {len(self)}.")

        day_idx, start, supervise_from = self.chunks[idx]
        end = start + self.sequence_window
        row_positions = np.arange(start, end, dtype=np.int64)
        token_indices = row_positions + int(self.day_row_offsets[day_idx])
        loss_mask = (row_positions >= supervise_from) & np.asarray(
            self.supervision_masks[day_idx][start:end],
            dtype=bool,
        )

        x_seq = torch.from_numpy(np.array(self.X_data[day_idx][start:end], dtype=np.float32, copy=True))
        t_seq = relative_time_tensor(self.T_data[day_idx][start:end])
        target_dtype = np.float32 if self.y_data[day_idx].ndim > 1 else np.int64
        y_seq = torch.from_numpy(np.array(self.y_data[day_idx][start:end], dtype=target_dtype, copy=True))
        mask = torch.from_numpy(loss_mask.astype(np.bool_, copy=False))
        ids = torch.from_numpy(token_indices.astype(np.int64, copy=False))
        return x_seq, t_seq, y_seq, mask, ids

    def supervised_labels(self) -> np.ndarray:
        """Return labels for tokens that are supervised exactly once."""
        labels_by_day = []
        for labels, mask in zip(self.y_data, self.supervision_masks):
            if len(labels) < self.sequence_window:
                continue
            positions = np.arange(len(labels), dtype=np.int64) >= self.loss_warmup_tokens
            positions &= np.asarray(mask, dtype=bool)
            labels_by_day.append(np.asarray(labels, dtype=np.int64)[positions])
        return np.concatenate(labels_by_day) if labels_by_day else np.asarray([], dtype=np.int64)

    def supervised_class_counts(self, num_classes: int) -> list[int]:
        labels = self.supervised_labels()
        return np.bincount(labels, minlength=num_classes)[:num_classes].astype(int).tolist()


def sequence_window_labels(dataset: Dataset) -> np.ndarray:
    """Return one label per sliding sequence window."""
    if hasattr(dataset, "supervised_labels"):
        return dataset.supervised_labels()  # type: ignore[no-any-return]
    labels_by_day = [
        np.asarray(labels, dtype=np.int64)[dataset.sequence_window - 1 :]
        for labels in dataset.y_data
        if len(labels) >= dataset.sequence_window
    ]
    return np.concatenate(labels_by_day) if labels_by_day else np.asarray([], dtype=np.int64)


def endpoint_metadata_path(label_path: str | Path) -> Path:
    """Return the metadata artifact paired with one compact label shard."""
    path = Path(label_path)
    suffix = "_labels.npy"
    if not path.name.endswith(suffix):
        raise ValueError(f"Expected a label shard ending with {suffix!r}, got {path}.")
    return path.with_name(path.name[: -len(suffix)] + ENDPOINT_METADATA_SUFFIX)


def supervision_mask_path(label_path: str | Path, support: str = "common") -> Path:
    """Return the optional endpoint-supervision mask paired with one label shard."""
    path = Path(label_path)
    suffix = "_labels.npy"
    if not path.name.endswith(suffix):
        raise ValueError(f"Expected a label shard ending with {suffix!r}, got {path}.")
    suffixes = {
        "common": SUPERVISION_MASK_SUFFIX,
        "broad": "_broad_supervision_mask.npy",
        "exec": "_exec_supervision_mask.npy",
    }
    if support not in suffixes:
        raise ValueError(f"Unknown supervision support {support!r}; expected one of {sorted(suffixes)}.")
    return path.with_name(path.name[: -len(suffix)] + suffixes[support])


def _load_supervision_masks(
    y_paths: list[Path],
    y_data: list[np.ndarray],
    *,
    mmap_mode: str | None,
    support: str,
) -> list[np.ndarray]:
    """Load strict endpoint masks, falling back to all rows for legacy caches."""
    paths = [supervision_mask_path(path, support=support) for path in y_paths]
    existing = [path.exists() for path in paths]
    if any(existing) and not all(existing):
        missing = [str(path) for path, exists in zip(paths, existing) if not exists]
        raise FileNotFoundError(f"Endpoint supervision masks are only partially available: {missing}")
    if not any(existing) and support == "common":
        return [np.ones(len(labels), dtype=bool) for labels in y_data]
    if not any(existing):
        raise FileNotFoundError(
            f"The requested {support!r} supervision support is unavailable; rerun common-support preprocessing."
        )
    masks = [np.load(path, mmap_mode=mmap_mode) for path in paths]
    for path, mask, labels in zip(paths, masks, y_data):
        if mask.ndim != 1 or len(mask) != len(labels):
            raise ValueError(f"Supervision mask {path} must have shape [{len(labels)}].")
    return masks


def _apply_supervision_time_window(
    masks: list[np.ndarray],
    times: list[np.ndarray],
    window: tuple[float, float] | None,
) -> list[np.ndarray]:
    """Intersect endpoint masks with an inclusive wall-clock window without truncating context."""
    if window is None:
        return masks
    start, end = map(float, window)
    if not 0.0 <= start < end <= 86_400.0:
        raise ValueError("supervision_time_window must satisfy 0 <= start < end <= 86400.")
    result: list[np.ndarray] = []
    for mask, day_times in zip(masks, times):
        absolute_times = np.asarray(day_times, dtype=np.float64)
        if len(mask) != len(absolute_times):
            raise ValueError("Supervision mask and time shard have different row counts.")
        in_window = np.isfinite(absolute_times) & (absolute_times >= start) & (absolute_times <= end)
        result.append(np.asarray(mask, dtype=bool) & in_window)
    return result


def evaluation_metadata_for_dataset(dataset: Dataset) -> dict[str, np.ndarray] | None:
    """Return endpoint metadata in exactly the order emitted during evaluation.

    Old preprocessing caches do not contain metadata.  They remain loadable, but
    cannot participate in strict common-endpoint evaluation.
    """
    y_paths = getattr(dataset, "y_paths", None)
    y_data = getattr(dataset, "y_data", None)
    if y_paths is None or y_data is None:
        return None

    metadata_paths = [endpoint_metadata_path(path) for path in y_paths]
    if not any(path.exists() for path in metadata_paths):
        return None
    missing = [str(path) for path in metadata_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Endpoint metadata is only partially available: {missing}")

    chunks: dict[str, list[np.ndarray]] = {}
    expected_keys: set[str] | None = None
    for day_index, (path, labels) in enumerate(zip(metadata_paths, y_data)):
        with np.load(path, allow_pickle=False) as payload:
            keys = set(payload.files)
            if expected_keys is None:
                expected_keys = keys
            elif keys != expected_keys:
                raise ValueError(
                    f"Endpoint metadata schema mismatch in {path}: expected {sorted(expected_keys)}, "
                    f"found {sorted(keys)}."
                )
            if any(len(payload[key]) != len(labels) for key in keys):
                raise ValueError(f"Endpoint metadata row count does not match labels in {path}.")
            for key in keys:
                values = np.asarray(payload[key])
                if isinstance(dataset, LOBTokenChunkDataset):
                    positions = np.arange(len(labels), dtype=np.int64) >= dataset.loss_warmup_tokens
                    positions &= np.asarray(dataset.supervision_masks[day_index], dtype=bool)
                    chunks.setdefault(key, []).append(values[positions])
                else:
                    chunks.setdefault(key, []).append(values[dataset.endpoint_positions[day_index]])

    if expected_keys is None:
        return None
    result: dict[str, np.ndarray] = {}
    for key in sorted(expected_keys):
        parts = chunks.get(key, [])
        result[key] = np.concatenate(parts) if parts else np.asarray([])
    return result


def attach_evaluation_metadata(outputs: dict[str, object], dataset: Dataset) -> dict[str, object]:
    """Attach stable endpoint keys and realized outcomes to collected predictions."""
    metadata = evaluation_metadata_for_dataset(dataset)
    if metadata is None:
        return dict(outputs)
    row_count = len(np.asarray(outputs.get("targets", outputs.get("predictions", []))))
    lengths = {key: len(values) for key, values in metadata.items()}
    if any(length != row_count for length in lengths.values()):
        raise ValueError(
            "Evaluation outputs and endpoint metadata have different row counts: "
            f"outputs={row_count}, metadata={lengths}."
        )
    enriched = dict(outputs)
    enriched.update(metadata)
    return enriched


class EpochShuffledSampler(Sampler[int]):
    """Globally shuffle dataset indices with a deterministic epoch-specific seed."""

    def __init__(self, dataset: Dataset, *, base_seed: int) -> None:
        self.dataset = dataset
        self.base_seed = int(base_seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used to seed deterministic shuffling."""
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.base_seed + self.epoch)
        indices = np.arange(len(self.dataset), dtype=np.int64)
        rng.shuffle(indices)
        return iter(indices.astype(int).tolist())

    def __len__(self) -> int:
        return len(self.dataset)


class EpochNeutralDownsamplingSampler(Sampler[int]):
    """Keep all directional windows and resample neutral windows each epoch."""

    def __init__(
        self,
        dataset: LOBDataset,
        *,
        label_mapping: dict[int, int],
        neutral_to_directional_ratio: float,
        base_seed: int,
    ) -> None:
        if neutral_to_directional_ratio <= 0.0:
            raise ValueError("neutral_to_directional_ratio must be > 0 when sampling is enabled.")
        missing_raw_labels = [label for label in (-1, 0, 1) if label not in label_mapping]
        if missing_raw_labels:
            raise ValueError(f"label_mapping must define -1, 0, and 1 for sampling; missing {missing_raw_labels}.")

        self.dataset = dataset
        self.neutral_to_directional_ratio = float(neutral_to_directional_ratio)
        self.base_seed = int(base_seed)
        self.epoch = 0
        self.down_class = int(label_mapping[-1])
        self.neutral_class = int(label_mapping[0])
        self.up_class = int(label_mapping[1])

        labels = sequence_window_labels(dataset)
        self.labels = labels
        self.down_indices = np.flatnonzero(labels == self.down_class).astype(np.int64)
        self.neutral_indices = np.flatnonzero(labels == self.neutral_class).astype(np.int64)
        self.up_indices = np.flatnonzero(labels == self.up_class).astype(np.int64)
        self.directional_indices = np.concatenate([self.down_indices, self.up_indices])
        if self.directional_indices.size == 0:
            raise ValueError("Cannot downsample neutral windows because the train split has no up/down windows.")

        self.sampled_neutral_count = min(
            int(self.neutral_indices.size),
            math.floor(self.neutral_to_directional_ratio * int(self.directional_indices.size)),
        )
        self.sampled_length = int(self.directional_indices.size + self.sampled_neutral_count)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used to seed deterministic neutral resampling."""
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.base_seed + self.epoch)
        if self.sampled_neutral_count < self.neutral_indices.size:
            neutral_indices = rng.choice(
                self.neutral_indices,
                size=self.sampled_neutral_count,
                replace=False,
            )
        else:
            neutral_indices = self.neutral_indices.copy()

        indices = np.concatenate([self.directional_indices, neutral_indices.astype(np.int64, copy=False)])
        rng.shuffle(indices)
        return iter(indices.astype(int).tolist())

    def __len__(self) -> int:
        return self.sampled_length

    def sampled_class_counts(self, num_classes: int) -> list[int]:
        """Return expected per-class counts after one epoch of sampling."""
        counts = np.bincount(self.labels, minlength=num_classes)[:num_classes].astype(int)
        counts[self.neutral_class] = self.sampled_neutral_count
        return counts.tolist()

    def summary(self, num_classes: int) -> dict[str, object]:
        """Return stable metadata describing the sampler configuration."""
        full_counts = np.bincount(self.labels, minlength=num_classes)[:num_classes].astype(int)
        sampled_counts = np.asarray(self.sampled_class_counts(num_classes), dtype=int)
        directional_count = int(self.directional_indices.size)
        return {
            "enabled": True,
            "method": "epoch_neutral_downsampling",
            "neutral_to_directional_ratio": self.neutral_to_directional_ratio,
            "base_seed": self.base_seed,
            "epoch_seed_rule": "base_seed + epoch",
            "labels": {
                "down": self.down_class,
                "neutral": self.neutral_class,
                "up": self.up_class,
            },
            "full_counts": {
                "total": int(full_counts.sum()),
                "down": int(full_counts[self.down_class]),
                "neutral": int(full_counts[self.neutral_class]),
                "up": int(full_counts[self.up_class]),
                "directional": directional_count,
                "by_class": full_counts.tolist(),
            },
            "sampled_counts_per_epoch": {
                "total": int(sampled_counts.sum()),
                "down": int(sampled_counts[self.down_class]),
                "neutral": int(sampled_counts[self.neutral_class]),
                "up": int(sampled_counts[self.up_class]),
                "directional": directional_count,
                "by_class": sampled_counts.tolist(),
            },
        }
