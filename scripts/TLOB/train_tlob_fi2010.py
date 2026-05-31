from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_fi2010_sequences import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_DIR, FOLD_ID, prepare_fi2010_sequences


DEFAULT_CONFIG = REPO_ROOT / "configs" / "config_TLOB_F1_2010.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare FI-2010 and run TLOB-style training.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--horizon", type=int, choices=[10, 20, 30, 50, 100], default=10)
    parser.add_argument("--seq-size", type=int, default=128)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--run-stem", type=str, default=None)
    parser.add_argument("--skip-prepare", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_prepare:
        prepare_fi2010_sequences(
            data_root=args.data_root,
            output_dir=args.output_dir,
            horizon=args.horizon,
            seq_size=args.seq_size,
            train_ratio=args.train_ratio,
        )

    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_training.py"),
        "--config",
        str(args.config),
        "--fold-id",
        FOLD_ID,
    ]
    if args.run_stem:
        command.extend(["--run-stem", args.run_stem])
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
