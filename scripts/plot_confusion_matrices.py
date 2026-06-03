from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from plotting import (
    CONFUSION_SPLITS,
    DEFAULT_CLASS_LABELS,
    confusion_kind_label,
    iter_confusion_matrices,
    load_confusion_yaml,
    normalized_epoch_name,
    plot_confusion_matrix,
    selected_confusion_kinds,
    slug,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for confusion matrix plotting."""
    parser = argparse.ArgumentParser(
        description="Render training confusion_matrices.yaml files as heatmap images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "yaml_path",
        type=Path,
        help="Path to a confusion_matrices.yaml file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where plots are written. Defaults to <yaml_dir>/confusion_plots.",
    )
    parser.add_argument("--fold", default=None, help="Fold id to plot, e.g. fold_006. Defaults to all folds.")
    parser.add_argument(
        "--epoch",
        default=None,
        help="Epoch to plot, e.g. 2 or epoch_2. Defaults to all epochs.",
    )
    parser.add_argument(
        "--split",
        choices=(*CONFUSION_SPLITS, "all"),
        default="all",
        help="Dataset split to plot.",
    )
    parser.add_argument(
        "--kind",
        choices=("raw", "normalized", "both"),
        default="both",
        help="Matrix type to plot.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=list(DEFAULT_CLASS_LABELS),
        help="Class labels in matrix order.",
    )
    parser.add_argument("--format", default="png", help="Output image format supported by matplotlib.")
    parser.add_argument("--dpi", type=int, default=160, help="Output image resolution.")
    parser.add_argument("--cmap", default="Blues", help="Matplotlib colormap name.")
    return parser.parse_args()


def main() -> None:
    """Render selected confusion matrices as image files."""
    args = parse_args()
    payload = load_confusion_yaml(args.yaml_path)
    output_dir = args.output_dir or args.yaml_path.parent / "confusion_plots"
    epoch_filter = normalized_epoch_name(args.epoch)
    kinds = selected_confusion_kinds(args.kind)

    written: list[Path] = []
    for fold_id, epoch_name, split, kind, matrix in iter_confusion_matrices(
        payload,
        fold_filter=args.fold,
        epoch_filter=epoch_filter,
        split_filter=args.split,
        kinds=kinds,
    ):
        plot_kind = confusion_kind_label(kind)
        if fold_id == "files":
            title = f"{epoch_name} {plot_kind}"
            filename = f"{slug(epoch_name)}_{plot_kind}.{args.format}"
        else:
            title = f"{fold_id} {epoch_name} {split} {plot_kind}"
            filename = f"{slug(fold_id)}_{slug(epoch_name)}_{split}_{plot_kind}.{args.format}"
        output_path = output_dir / filename
        plot_confusion_matrix(
            matrix,
            labels=list(args.labels),
            title=title,
            kind=kind,
            cmap=args.cmap,
            output_path=output_path,
            dpi=args.dpi,
        )
        written.append(output_path)

    if not written:
        raise SystemExit("No confusion matrices matched the requested filters.")

    print(f"Wrote {len(written)} plot(s) to {output_dir}:")
    for path in written:
        print(f"- {path}")


if __name__ == "__main__":
    main()
