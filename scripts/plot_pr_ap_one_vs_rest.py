from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLASSES = ("down", "neutral", "up")
COLORS = {"down": "#2563eb", "neutral": "#6b7280", "up": "#dc2626"}
MAX_POINTS = 6000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one-vs-rest precision-recall curves from saved probability CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-name", required=True, help="Run label used in plot titles.")
    parser.add_argument(
        "--probability",
        action="append",
        required=True,
        metavar="SPLIT=PATH",
        help="Probability CSV to plot, e.g. validation=path/to/probabilities.csv. May be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for PNGs and AP summary CSV.")
    parser.add_argument("--dpi", type=int, default=170, help="Output image DPI.")
    return parser.parse_args()


def parse_probability_specs(specs: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--probability must be SPLIT=PATH, got: {spec!r}")
        split, raw_path = spec.split("=", 1)
        split = split.strip()
        if not split:
            raise ValueError(f"Empty split in --probability spec: {spec!r}")
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        parsed.append((split, path))
    return parsed


def precision_recall_curve_and_ap(scores: np.ndarray, positives: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    scores = np.asarray(scores, dtype=np.float64)
    positives = np.asarray(positives, dtype=bool)
    n_samples = int(scores.size)
    n_positive = int(positives.sum())
    if n_samples == 0 or n_positive == 0:
        return np.array([0.0, 1.0]), np.array([1.0, 0.0]), 0.0

    order = np.argsort(-scores, kind="mergesort")
    sorted_positive = positives[order]
    true_positives = np.cumsum(sorted_positive, dtype=np.float64)
    ranks = np.arange(1, n_samples + 1, dtype=np.float64)
    precision = true_positives / ranks
    recall = true_positives / float(n_positive)
    average_precision = float(precision[sorted_positive].sum() / float(n_positive))

    if n_samples > MAX_POINTS:
        sampled = np.linspace(0, n_samples - 1, MAX_POINTS).astype(np.int64)
        head = np.arange(min(n_samples, 1000), dtype=np.int64)
        indices = np.unique(np.concatenate([head, sampled, np.array([n_samples - 1], dtype=np.int64)]))
    else:
        indices = np.arange(n_samples, dtype=np.int64)

    recall_plot = np.concatenate([[0.0], recall[indices]])
    precision_plot = np.concatenate([[1.0], precision[indices]])
    return recall_plot, precision_plot, average_precision


def plot_probability_file(
    *,
    run_name: str,
    split: str,
    csv_path: Path,
    output_dir: Path,
    dpi: int,
) -> list[dict[str, object]]:
    frame = pd.read_csv(csv_path)
    required_columns = {"true_label", *(f"p_{label}" for label in CLASSES)}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required column(s): {sorted(missing)}")

    true_labels = frame["true_label"].astype(str).to_numpy()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    rows: list[dict[str, object]] = []

    for class_name in CLASSES:
        scores = frame[f"p_{class_name}"].to_numpy(dtype=np.float64)
        positives = true_labels == class_name
        recall, precision, average_precision = precision_recall_curve_and_ap(scores, positives)
        prevalence = float(np.mean(positives)) if positives.size else 0.0
        ax.step(
            recall,
            precision,
            where="post",
            linewidth=1.8,
            color=COLORS[class_name],
            label=f"{class_name} AP={average_precision:.4f} prev={prevalence:.3%}",
        )
        rows.append(
            {
                "run": run_name,
                "split": split,
                "class": class_name,
                "pr_ap": average_precision,
                "prevalence": prevalence,
                "num_samples": int(len(frame)),
                "num_positive": int(positives.sum()),
                "source_csv": str(csv_path),
            }
        )

    ax.set_title(f"{run_name} - {split} one-vs-rest PR curves")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.savefig(output_dir / f"{split}_pr_ap_one_vs_rest.png", dpi=dpi)
    plt.close(fig)
    return rows


def main() -> None:
    args = parse_args()
    probability_specs = parse_probability_specs(args.probability)
    all_rows: list[dict[str, object]] = []
    for split, csv_path in probability_specs:
        all_rows.extend(
            plot_probability_file(
                run_name=args.run_name,
                split=split,
                csv_path=csv_path,
                output_dir=args.output_dir,
                dpi=args.dpi,
            )
        )
    summary_path = args.output_dir / "pr_ap_summary.csv"
    pd.DataFrame(all_rows).to_csv(summary_path, index=False)
    print(f"Wrote {len(probability_specs)} plot(s) and summary to {args.output_dir}")


if __name__ == "__main__":
    main()
