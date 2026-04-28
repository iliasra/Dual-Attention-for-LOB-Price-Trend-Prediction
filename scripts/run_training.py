from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from E2Epipeline import LobExperimentRunner


def main() -> None:
    summary = LobExperimentRunner().run()
    for split, shapes in summary.items():
        print(split)
        for date, shape in shapes.items():
            print(f"  {date}: {shape}")


if __name__ == "__main__":
    main()
