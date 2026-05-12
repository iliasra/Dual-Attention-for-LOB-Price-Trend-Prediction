from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_training import fold_artifact_paths


def test_fold_artifact_paths_are_scoped_by_fold() -> None:
    paths = fold_artifact_paths(
        sequence_dir=Path("data/sequences"),
        run_log_dir=Path("logs/run_7"),
        run_result_dir=Path("results/run_7"),
        fold_id="fold_003",
    )

    assert paths["sequence_dir"] == Path("data/sequences/fold_003")
    assert paths["log_dir"] == Path("logs/run_7/fold_003")
    assert paths["result_dir"] == Path("results/run_7/fold_003")
