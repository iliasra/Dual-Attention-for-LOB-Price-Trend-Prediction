from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from processing import LobProcessingPipeline


def main() -> None:
    summary = LobProcessingPipeline().run()
    for fold_id, split_summary in summary.items():
        print(fold_id)
        for split, shapes in split_summary.items():
            print(f"  {split}")
            for date, shape in shapes.items():
                print(f"    {date}: {shape}")


if __name__ == "__main__":
    main()
