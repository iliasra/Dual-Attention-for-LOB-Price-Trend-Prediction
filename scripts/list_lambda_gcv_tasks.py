"""
This script generates the cache tasks from the train dates
of all folds. One task = one output stem + a stream (price/volume).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from configuration import load_config
from processing import LobProcessingPipeline


DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "gcv_cache" / "lambda_gcv_tasks.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List daily GCV cache tasks.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--folds-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def read_fold_ids(path: Path) -> set[str]:
    """Read non-comment fold ids from a folds file."""
    fold_ids: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fold_ids.add(line.split()[0])
    return fold_ids


def write_tasks(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_path = args.output or DEFAULT_OUTPUT_PATH

    if config.preprocessing.kinematic_tokenization.method != "fast":
        write_tasks(output_path, [])
        print(0)
        return

    selected_fold_ids = read_fold_ids(args.folds_file) if args.folds_file else None
    selected_folds = [
        fold
        for fold in config.folds
        if selected_fold_ids is None or fold.id in selected_fold_ids
    ]
    if selected_fold_ids is not None:
        found_fold_ids = {fold.id for fold in selected_folds}
        missing_fold_ids = sorted(selected_fold_ids - found_fold_ids)
        if missing_fold_ids:
            missing = ", ".join(missing_fold_ids)
            raise ValueError(f"Fold id(s) from --folds-file not found in config: {missing}")

    train_dates = sorted({date for fold in selected_folds for date in fold.train_dates})

    pipeline = LobProcessingPipeline(config)
    pairs = pipeline.discover_pairs()

    tasks: list[tuple[str, str, str, str]] = []
    for pair in pairs:
        if pair.date not in train_dates:
            continue
        if config.preprocessing.price_kinematic.enabled:
            tasks.append((pair.symbol, pair.date, "price", pair.output_stem))
        if config.preprocessing.volume_kinematic.enabled:
            tasks.append((pair.symbol, pair.date, "volume", pair.output_stem))

    lines = [
        f"{symbol} {date} {kind} {output_stem}"
        for symbol, date, kind, output_stem in tasks
    ]

    write_tasks(output_path, lines)

    print(len(tasks))


if __name__ == "__main__":
    main()
