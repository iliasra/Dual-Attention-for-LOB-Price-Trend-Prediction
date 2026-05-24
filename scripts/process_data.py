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
from utils import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess LOBSTER data into fold sequence artifacts.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional: Preprocess only the fold with this id. If Not provided, the default behavior" \
        "is to process all the fol+ds sequentially.",
    )
    parser.add_argument("--fold-id", type=str, default=None, help="Preprocess only the fold with this id.")
    parser.add_argument(
        "--fold-index",
        type=int,
        default=None,
        help="Optional: Preprocess only the fold with this id. If Not provided, the default behavior" \
        "is to process all the folds sequentially.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fold_id is not None and args.fold_index is not None:
        raise ValueError("Use either --fold-id or --fold-index, not both.")

    config = load_config(args.config)
    selected_fold_ids: set[str] | None = None
    if args.fold_id is not None:
        selected_fold_ids = {args.fold_id}
    elif args.fold_index is not None:
        if args.fold_index < 1 or args.fold_index > len(config.folds):
            raise ValueError(f"--fold-index must be in [1, {len(config.folds)}], got {args.fold_index}.")
        selected_fold_ids = {config.folds[args.fold_index - 1].id}

    set_global_seed(config.seed)
    print(f"Global seed set to {config.seed}.")
    if selected_fold_ids is not None:
        print(f"Selected fold(s): {', '.join(sorted(selected_fold_ids))}.")
    summary = LobProcessingPipeline(config).run(selected_fold_ids=selected_fold_ids)
    for fold_id, split_summary in summary.items():
        print(fold_id)
        for split, shapes in split_summary.items():
            print(f"  {split}")
            for date, shape in shapes.items():
                print(f"    {date}: {shape}")


if __name__ == "__main__":
    main()
