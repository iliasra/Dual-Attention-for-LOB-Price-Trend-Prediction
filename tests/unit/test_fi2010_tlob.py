from __future__ import annotations

from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts" / "TLOB"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from prepare_fi2010_sequences import (
    convert_matrix_to_arrays,
    prepare_fi2010_sequences,
    split_train_validation,
)


@pytest.fixture()
def artifact_dir(request: pytest.FixtureRequest) -> Path:
    path = Path(__file__).resolve().parent / ".test_artifacts" / request.node.name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def _synthetic_fi2010_matrix(num_rows: int = 256) -> np.ndarray:
    data = np.zeros((149, num_rows), dtype=np.float64)
    data[:144] = np.arange(144 * num_rows, dtype=np.float64).reshape(144, num_rows)
    for offset, row in enumerate(range(144, 149)):
        data[row] = ((np.arange(num_rows) + offset) % 3) + 1
    return data


def test_fi2010_conversion_extracts_features_labels_and_dummy_time() -> None:
    data = _synthetic_fi2010_matrix(256)

    features, times, labels = convert_matrix_to_arrays(data, horizon=20, seq_size=128)

    assert features.shape == (256, 144)
    np.testing.assert_allclose(features[0], data[:144, 0].astype(np.float32))
    np.testing.assert_array_equal(labels, data[145].astype(np.int64) - 1)
    assert times.dtype == np.float32
    assert times[0] == 0.0
    assert times[127] - times[0] == np.float32(1.0)


def test_fi2010_train_validation_split_is_chronological() -> None:
    features = np.arange(10 * 144, dtype=np.float32).reshape(10, 144)
    times = np.arange(10, dtype=np.float32)
    labels = np.arange(10, dtype=np.int64) % 3

    train, validation = split_train_validation(features, times, labels, train_ratio=0.8)

    assert train[0].shape[0] == 8
    assert validation[0].shape[0] == 2
    np.testing.assert_array_equal(train[1], np.arange(8, dtype=np.float32))
    np.testing.assert_array_equal(validation[1], np.asarray([8.0, 9.0], dtype=np.float32))


def test_prepare_fi2010_sequences_writes_expected_npy_files(artifact_dir: Path) -> None:
    data_root = artifact_dir / "F1_2010"
    train_dir = data_root / "NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Training"
    test_dir = data_root / "NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    np.savetxt(train_dir / "Train_Dst_NoAuction_ZScore_CF_7.txt", _synthetic_fi2010_matrix(256))
    for suffix in (7, 8, 9):
        np.savetxt(test_dir / f"Test_Dst_NoAuction_ZScore_CF_{suffix}.txt", _synthetic_fi2010_matrix(160))

    output_dir = artifact_dir / "sequences" / "fi2010_tlob"
    summary = prepare_fi2010_sequences(data_root=data_root, output_dir=output_dir, horizon=10, seq_size=128)

    assert Path(summary["feature_schema"]).exists()
    assert (output_dir / "train" / "fi2010_cf7_train_features.npy").exists()
    assert (output_dir / "validation" / "fi2010_cf7_validation_labels.npy").exists()
    assert len(list((output_dir / "test").glob("*_features.npy"))) == 3
    train_features = np.load(output_dir / "train" / "fi2010_cf7_train_features.npy")
    validation_labels = np.load(output_dir / "validation" / "fi2010_cf7_validation_labels.npy")
    assert train_features.shape == (204, 144)
    assert validation_labels.shape == (52,)
