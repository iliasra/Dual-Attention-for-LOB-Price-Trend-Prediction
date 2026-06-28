from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - depends on optional plotting env.
    raise SystemExit(
        "matplotlib is required for plotting. "
        "Install it in the active environment, for example with `conda install matplotlib`."
    ) from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from plotting import slug


LOSS_COLUMNS = (
    ("train_loss", "Train loss"),
    ("val_loss", "Validation loss"),
)
F1_COLUMNS = (
    ("val_macro_f1", "Validation macro F1"),
    ("validation_f1", "Validation F1"),
    ("validation_macro_f1", "Validation macro F1"),
    ("val_f1", "Validation F1"),
)
X_CANDIDATES = ("global_step", "validation_index", "epoch")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training metrics plotting."""
    parser = argparse.ArgumentParser(
        description="Render train/validation loss and validation F1 curves from metrics CSV files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_path",
        type=Path,
        nargs="?",
        default=Path("output"),
        help="metrics CSV file or directory searched recursively for metrics CSV files.",
    )
    parser.add_argument(
        "--pattern",
        default="metrics*.csv",
        help="Glob pattern used when input_path is a directory.",
    )
    parser.add_argument(
        "--output-subdir",
        default="metric_curves",
        help="Subdirectory created next to each metrics CSV.",
    )
    parser.add_argument("--format", default="png", help="Output image format supported by matplotlib.")
    parser.add_argument("--dpi", type=int, default=160, help="Output image resolution.")
    return parser.parse_args()


def selected_csv_paths(input_path: Path, pattern: str) -> list[Path]:
    """Return metrics CSV paths selected by a file or recursive directory input."""
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return prefer_canonical_metrics(sorted(input_path.rglob(pattern)))
    raise FileNotFoundError(f"Metrics input path not found: {input_path}")


def prefer_canonical_metrics(paths: list[Path]) -> list[Path]:
    """Prefer metrics.csv over backup or legacy metrics filenames in the same directory."""
    by_parent: dict[Path, list[Path]] = {}
    for path in paths:
        by_parent.setdefault(path.parent, []).append(path)

    selected: list[Path] = []
    for parent in sorted(by_parent):
        parent_paths = sorted(by_parent[parent])
        canonical = [path for path in parent_paths if path.name == "metrics.csv"]
        selected.extend(canonical or parent_paths)
    return selected


def numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric series for a column, coercing blanks and invalid values to NaN."""
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def available_series(frame: pd.DataFrame, columns: tuple[tuple[str, str], ...]) -> list[tuple[str, str, pd.Series]]:
    """Return configured columns that exist and contain at least one numeric value."""
    series: list[tuple[str, str, pd.Series]] = []
    for column, label in columns:
        if column not in frame.columns:
            continue
        values = numeric_column(frame, column)
        if values.notna().any():
            series.append((column, label, values))
    return series


def x_values(frame: pd.DataFrame) -> tuple[str, pd.Series]:
    """Choose a stable x-axis from step, validation index, epoch, or row number."""
    for column in X_CANDIDATES:
        if column not in frame.columns:
            continue
        values = numeric_column(frame, column)
        if values.notna().sum() >= 2 and values.nunique(dropna=True) >= 2:
            return column, values
    return "row", pd.Series(np.arange(1, len(frame) + 1), index=frame.index)


def set_axis_labels(ax: plt.Axes, *, x_label: str, y_label: str, title: str) -> None:
    """Apply shared axis styling."""
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()


def positive_x_mask(x: pd.Series) -> pd.Series:
    """Return rows with a positive x value for log-scaled x-axes."""
    return x.notna() & x.gt(0.0)


def plot_losses(
    x_label: str,
    x: pd.Series,
    losses: list[tuple[str, str, pd.Series]],
    *,
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    """Render train and validation loss curves."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for _column, label, values in losses:
        ax.plot(x, values, linewidth=2.0, marker="o", markersize=3.0, label=label)
    set_axis_labels(ax, x_label=x_label, y_label="Loss", title=title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_losses_loglog(
    x_label: str,
    x: pd.Series,
    losses: list[tuple[str, str, pd.Series]],
    *,
    title: str,
    output_path: Path,
    dpi: int,
) -> bool:
    """Render train and validation loss curves with log-scaled x and y axes."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    plotted = False
    for _column, label, values in losses:
        mask = positive_x_mask(x) & values.notna() & values.gt(0.0)
        if not mask.any():
            continue
        ax.plot(x[mask], values[mask], linewidth=2.0, marker="o", markersize=3.0, label=label)
        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.set_xscale("log")
    ax.set_yscale("log")
    set_axis_labels(ax, x_label=x_label, y_label="Loss", title=title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return True


def plot_f1(
    x_label: str,
    x: pd.Series,
    f1_series: list[tuple[str, str, pd.Series]],
    *,
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    """Render validation F1 curves."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for _column, label, values in f1_series:
        ax.plot(x, values, linewidth=2.0, marker="o", markersize=3.0, label=label)
    ax.set_ylim(-0.02, 1.02)
    set_axis_labels(ax, x_label=x_label, y_label="F1", title=title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_f1_logx(
    x_label: str,
    x: pd.Series,
    f1_series: list[tuple[str, str, pd.Series]],
    *,
    title: str,
    output_path: Path,
    dpi: int,
) -> bool:
    """Render validation F1 curves with a log-scaled x-axis and linear y-axis."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    plotted = False
    for _column, label, values in f1_series:
        mask = positive_x_mask(x) & values.notna()
        if not mask.any():
            continue
        ax.plot(x[mask], values[mask], linewidth=2.0, marker="o", markersize=3.0, label=label)
        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    set_axis_labels(ax, x_label=x_label, y_label="F1", title=title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return True


def plot_overview(
    x_label: str,
    x: pd.Series,
    losses: list[tuple[str, str, pd.Series]],
    f1_series: list[tuple[str, str, pd.Series]],
    *,
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    """Render a compact overview with loss and validation F1 panels."""
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.2), sharex=True, constrained_layout=True)

    for _column, label, values in losses:
        axes[0].plot(x, values, linewidth=2.0, marker="o", markersize=3.0, label=label)
    axes[0].set_title("Loss")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    for _column, label, values in f1_series:
        axes[1].plot(x, values, linewidth=2.0, marker="o", markersize=3.0, label=label)
    axes[1].set_title("Validation F1")
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel("F1")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_metrics_csv(csv_path: Path, *, output_subdir: str, image_format: str, dpi: int) -> list[Path]:
    """Render all configured metric plots for one CSV and return written paths."""
    frame = pd.read_csv(csv_path)
    losses = available_series(frame, LOSS_COLUMNS)
    f1_series = available_series(frame, F1_COLUMNS)
    f1_series = f1_series[:1]
    if not losses and not f1_series:
        return []

    x_label, x = x_values(frame)
    output_dir = csv_path.parent / output_subdir
    prefix = slug(csv_path.stem) or "metrics"
    title = str(csv_path.parent.relative_to(REPO_ROOT)) if csv_path.is_relative_to(REPO_ROOT) else csv_path.parent.name

    written: list[Path] = []
    if losses and f1_series:
        output_path = output_dir / f"{prefix}_overview.{image_format}"
        plot_overview(x_label, x, losses, f1_series, title=title, output_path=output_path, dpi=dpi)
        written.append(output_path)
    if losses:
        output_path = output_dir / f"{prefix}_loss.{image_format}"
        plot_losses(x_label, x, losses, title=f"{title} loss", output_path=output_path, dpi=dpi)
        written.append(output_path)
        output_path = output_dir / f"{prefix}_loss_loglog.{image_format}"
        if plot_losses_loglog(
            x_label,
            x,
            losses,
            title=f"{title} loss log-log",
            output_path=output_path,
            dpi=dpi,
        ):
            written.append(output_path)
    if f1_series:
        output_path = output_dir / f"{prefix}_validation_f1.{image_format}"
        plot_f1(x_label, x, f1_series, title=f"{title} validation F1", output_path=output_path, dpi=dpi)
        written.append(output_path)
        output_path = output_dir / f"{prefix}_validation_f1_logx.{image_format}"
        if plot_f1_logx(
            x_label,
            x,
            f1_series,
            title=f"{title} validation F1 log-x",
            output_path=output_path,
            dpi=dpi,
        ):
            written.append(output_path)

    return written


def main() -> None:
    """Render selected metrics CSV files as image files."""
    args = parse_args()
    csv_paths = selected_csv_paths(args.input_path, args.pattern)
    if not csv_paths:
        raise SystemExit("No metrics CSV files matched the requested input.")

    written: list[Path] = []
    skipped: list[Path] = []
    for csv_path in csv_paths:
        paths = plot_metrics_csv(
            csv_path,
            output_subdir=args.output_subdir,
            image_format=args.format,
            dpi=args.dpi,
        )
        if paths:
            written.extend(paths)
        else:
            skipped.append(csv_path)

    print(f"Wrote {len(written)} plot(s) from {len(csv_paths) - len(skipped)} metrics CSV file(s).")
    for path in written:
        print(f"- {path}")
    if skipped:
        print(f"Skipped {len(skipped)} CSV file(s) without train/validation loss or validation F1:")
        for path in skipped:
            print(f"- {path}")


if __name__ == "__main__":
    main()
