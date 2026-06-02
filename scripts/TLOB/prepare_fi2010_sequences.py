from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "F1_2010"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "sequences" / "fi2010_tlob"
FOLD_ID = "fi2010_tlob"
FEATURE_COUNT = 144
HORIZON_TO_LABEL_ROW = {10: 144, 20: 145, 30: 146, 50: 147, 100: 148}
CLASS_SEMANTICS = {
    0: "up",
    1: "neutral",
    2: "down",
}
TRAIN_RELATIVE_PATH = Path(
    "NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_7.txt"
)
TEST_RELATIVE_PATHS = [
    Path("NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_7.txt"),
    Path("NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_8.txt"),
    Path("NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_9.txt"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert FI-2010 NoAuction z-score files to project .npy sequences.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--horizon", type=int, choices=sorted(HORIZON_TO_LABEL_ROW), default=10)
    parser.add_argument("--seq-size", type=int, default=128)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    return parser.parse_args()


def load_fi2010_text(path: Path) -> np.ndarray:
    """Load one FI-2010 text file and validate its row layout."""
    if not path.exists():
        raise FileNotFoundError(f"FI-2010 file not found: {path}")
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[0] < FEATURE_COUNT + len(HORIZON_TO_LABEL_ROW):
        raise ValueError(f"Expected FI-2010 data with at least 149 rows, got shape {data.shape} from {path}.")
    return data


def validate_label_rows(data: np.ndarray) -> None:
    """Ensure all FI-2010 horizon label rows contain integer classes 1, 2, 3."""
    for row in HORIZON_TO_LABEL_ROW.values():
        values = data[row]
        rounded = np.rint(values)
        if not np.allclose(values, rounded):
            raise ValueError(f"FI-2010 label row {row} contains non-integer values.")
        unique = set(rounded.astype(np.int64).tolist())
        if not unique <= {1, 2, 3}:
            raise ValueError(f"FI-2010 label row {row} contains labels outside {{1, 2, 3}}: {sorted(unique)}")


def convert_matrix_to_arrays(data: np.ndarray, *, horizon: int, seq_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a FI-2010 matrix into features, dummy times, and zero-based labels."""
    if seq_size <= 1:
        raise ValueError("seq_size must be > 1 to build normalized dummy times.")
    if horizon not in HORIZON_TO_LABEL_ROW:
        raise ValueError(f"Unsupported horizon {horizon}; expected one of {sorted(HORIZON_TO_LABEL_ROW)}.")
    validate_label_rows(data)
    features = data[:FEATURE_COUNT, :].T.astype(np.float32, copy=False)
    raw_labels = np.rint(data[HORIZON_TO_LABEL_ROW[horizon], :]).astype(np.int64)
    labels = (raw_labels - 1).astype(np.int64, copy=False)
    times = (np.arange(features.shape[0], dtype=np.float32) / float(seq_size - 1)).astype(np.float32, copy=False)
    return features, times, labels


def split_train_validation(
    features: np.ndarray,
    times: np.ndarray,
    labels: np.ndarray,
    *,
    train_ratio: float,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Split one FI-2010 training file into chronological train and validation parts."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1).")
    split_index = int(features.shape[0] * train_ratio)
    if split_index <= 0 or split_index >= features.shape[0]:
        raise ValueError("train_ratio leaves an empty train or validation split.")
    train = (features[:split_index], times[:split_index], labels[:split_index])
    validation = (features[split_index:], times[split_index:], labels[split_index:])
    return train, validation


def save_arrays(
    split_dir: Path,
    stem: str,
    features: np.ndarray,
    times: np.ndarray,
    labels: np.ndarray,
) -> dict[str, str]:
    """Save one split payload as *_features/times/labels.npy files."""
    split_dir.mkdir(parents=True, exist_ok=True)
    feature_path = split_dir / f"{stem}_features.npy"
    time_path = split_dir / f"{stem}_times.npy"
    label_path = split_dir / f"{stem}_labels.npy"
    np.save(feature_path, features.astype(np.float32, copy=False))
    np.save(time_path, times.astype(np.float32, copy=False))
    np.save(label_path, labels.astype(np.int64, copy=False))
    return {"features": str(feature_path), "times": str(time_path), "labels": str(label_path)}


def save_feature_schema(output_dir: Path) -> Path:
    """Write the feature order expected by the generated FI-2010 arrays."""
    target = output_dir / "feature_schema.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ordered_feature_columns": [f"feature_{index:03d}" for index in range(FEATURE_COUNT)]}
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return target


def prepare_fi2010_sequences(
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    horizon: int = 10,
    seq_size: int = 128,
    train_ratio: float = 0.8,
) -> dict[str, Any]:
    """Prepare FI-2010 NoAuction z-score files for the existing LOBDataset."""
    train_data = load_fi2010_text(data_root / TRAIN_RELATIVE_PATH)
    features, times, labels = convert_matrix_to_arrays(train_data, horizon=horizon, seq_size=seq_size)
    train_payload, validation_payload = split_train_validation(
        features,
        times,
        labels,
        train_ratio=train_ratio,
    )
    outputs: dict[str, Any] = {
        "fold_id": FOLD_ID,
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "horizon": int(horizon),
        "seq_size": int(seq_size),
        "train_ratio": float(train_ratio),
        "label_encoding": {
            "source": "FI-2010 raw labels {1, 2, 3}",
            "conversion": "class_id = raw_label - 1",
            "class_semantics": CLASS_SEMANTICS,
            "recommended_project_label_mapping": {-1: 2, 0: 1, 1: 0},
        },
        "splits": {},
    }
    outputs["splits"]["train"] = save_arrays(output_dir / "train", "fi2010_cf7_train", *train_payload)
    outputs["splits"]["validation"] = save_arrays(
        output_dir / "validation",
        "fi2010_cf7_validation",
        *validation_payload,
    )
    outputs["splits"]["test"] = []
    for test_path in TEST_RELATIVE_PATHS:
        data = load_fi2010_text(data_root / test_path)
        test_arrays = convert_matrix_to_arrays(data, horizon=horizon, seq_size=seq_size)
        stem = test_path.stem.lower()
        outputs["splits"]["test"].append(save_arrays(output_dir / "test", stem, *test_arrays))
    outputs["feature_schema"] = str(save_feature_schema(output_dir))
    metadata_path = output_dir / "fi2010_preparation.yaml"
    metadata_path.write_text(yaml.safe_dump(outputs, sort_keys=False), encoding="utf-8")
    outputs["metadata"] = str(metadata_path)
    return outputs


def main() -> None:
    args = parse_args()
    summary = prepare_fi2010_sequences(
        data_root=args.data_root,
        output_dir=args.output_dir,
        horizon=args.horizon,
        seq_size=args.seq_size,
        train_ratio=args.train_ratio,
    )
    print(yaml.safe_dump(summary, sort_keys=False))


if __name__ == "__main__":
    main()
