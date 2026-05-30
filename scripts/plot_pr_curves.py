from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from plotting import load_pr_curve_csv, plot_pr_curve, slug


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for PR curve plotting."""
    parser = argparse.ArgumentParser(
        description="Render saved precision-recall curve CSV files as plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_path", type=Path, help="PR curve CSV file or directory containing CSV files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where plots are written. Defaults to <input_dir>/pr_plots.",
    )
    parser.add_argument("--pattern", default="*.csv", help="Glob pattern used when input_path is a directory.")
    parser.add_argument("--format", default="png", help="Output image format supported by matplotlib.")
    parser.add_argument("--dpi", type=int, default=160, help="Output image resolution.")
    return parser.parse_args()


def selected_csv_paths(input_path: Path, pattern: str) -> list[Path]:
    """Return PR CSV paths selected by a file or directory input."""
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob(pattern))
    raise FileNotFoundError(f"PR curve input path not found: {input_path}")


def default_output_dir(input_path: Path) -> Path:
    """Return the default output directory for PR plots."""
    return (input_path.parent if input_path.is_file() else input_path) / "pr_plots"


def main() -> None:
    """Render selected PR curve CSV files as image files."""
    args = parse_args()
    csv_paths = selected_csv_paths(args.input_path, args.pattern)
    if not csv_paths:
        raise SystemExit("No PR curve CSV files matched the requested input.")
    output_dir = args.output_dir or default_output_dir(args.input_path)

    written: list[Path] = []
    for csv_path in csv_paths:
        frame = load_pr_curve_csv(csv_path)
        output_path = output_dir / f"{slug(csv_path.stem)}.{args.format}"
        plot_pr_curve(
            frame,
            title=csv_path.stem,
            output_path=output_path,
            dpi=args.dpi,
        )
        written.append(output_path)

    print(f"Wrote {len(written)} PR plot(s) to {output_dir}:")
    for path in written:
        print(f"- {path}")


if __name__ == "__main__":
    main()
