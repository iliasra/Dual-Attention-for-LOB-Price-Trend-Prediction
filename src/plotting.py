from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - depends on optional plotting env.
    raise SystemExit(
        "matplotlib is required for plotting. "
        "Install it in the active environment, for example with `conda install matplotlib`."
    ) from exc


CONFUSION_SPLITS = ("train", "validation", "test")
CONFUSION_KINDS = ("raw", "normalized_by_true_class")
DEFAULT_CLASS_LABELS = ("down", "neutral", "up")
PR_CURVE_COLUMNS = ("threshold", "precision", "recall", "f1")


def slug(value: str) -> str:
    """Return a filesystem-safe slug."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def load_confusion_yaml(path: Path) -> dict[str, Any]:
    """Load a training confusion_matrices.yaml file."""
    if not path.exists():
        raise FileNotFoundError(f"Confusion matrix YAML not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict) or "folds" not in payload:
        raise ValueError(f"Invalid confusion matrix YAML: missing top-level 'folds' in {path}")
    return payload


def normalized_epoch_name(epoch: str | None) -> str | None:
    """Normalize numeric epoch filters to the YAML epoch key format."""
    if epoch is None:
        return None
    stripped = str(epoch).strip()
    return f"epoch_{stripped}" if stripped.isdigit() else stripped


def selected_confusion_kinds(kind: str) -> tuple[str, ...]:
    """Return the confusion matrix YAML keys selected by a CLI kind."""
    if kind == "raw":
        return ("raw",)
    if kind == "normalized":
        return ("normalized_by_true_class",)
    return CONFUSION_KINDS


def matrix_labels(matrix: np.ndarray, requested_labels: list[str]) -> list[str]:
    """Return labels matching the matrix dimension."""
    if matrix.shape[0] == len(requested_labels):
        return requested_labels
    return [f"class_{index}" for index in range(matrix.shape[0])]


def iter_confusion_matrices(
    payload: dict[str, Any],
    *,
    fold_filter: str | None,
    epoch_filter: str | None,
    split_filter: str,
    kinds: tuple[str, ...],
) -> Iterable[tuple[str, str, str, str, np.ndarray]]:
    """Yield confusion matrices matching fold, epoch, split, and kind filters."""
    folds = payload.get("folds", {})
    if not isinstance(folds, dict):
        raise ValueError("Invalid confusion matrix YAML: 'folds' must be a mapping.")

    for fold_id, fold_payload in folds.items():
        if fold_filter is not None and fold_id != fold_filter:
            continue
        if not isinstance(fold_payload, dict):
            continue
        for epoch_name, epoch_payload in fold_payload.items():
            if epoch_filter is not None and epoch_name != epoch_filter:
                continue
            if not isinstance(epoch_payload, dict):
                continue
            for split in CONFUSION_SPLITS:
                if split_filter != "all" and split != split_filter:
                    continue
                split_payload = epoch_payload.get(split)
                if not isinstance(split_payload, dict):
                    continue
                for kind in kinds:
                    matrix = split_payload.get(kind)
                    if matrix is None:
                        continue
                    yield fold_id, epoch_name, split, kind, np.asarray(matrix, dtype=float)


def confusion_kind_label(kind: str) -> str:
    """Return a readable confusion matrix kind label."""
    return "normalized" if kind == "normalized_by_true_class" else kind


def matrix_to_frame(matrix: np.ndarray, labels: list[str]) -> pd.DataFrame:
    """Wrap a confusion matrix in a labeled DataFrame."""
    index = [f"true_{label}" for label in labels]
    columns = [f"pred_{label}" for label in labels]
    return pd.DataFrame(matrix, index=index, columns=columns)


def annotation_text(value: float, *, kind: str) -> str:
    """Format a confusion cell annotation."""
    if kind == "raw":
        return f"{int(round(value)):,}"
    return f"{value:.1%}"


def plot_confusion_matrix(
    matrix: np.ndarray,
    *,
    labels: list[str],
    title: str,
    kind: str,
    cmap: str,
    output_path: Path,
    dpi: int,
) -> None:
    """Render one confusion matrix heatmap to an image file."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square 2D matrix, got shape {matrix.shape}.")

    labels = matrix_labels(matrix, labels)
    frame = matrix_to_frame(matrix, labels)
    values = frame.to_numpy(dtype=float)
    vmax = float(np.nanmax(values)) if values.size else 0.0

    fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    image = ax.imshow(values, cmap=cmap, vmin=0.0)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Count" if kind == "raw" else "Share within true class")

    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)

    threshold = vmax / 2.0 if vmax > 0 else 0.0
    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            value = values[row_index, col_index]
            color = "white" if value > threshold else "black"
            ax.text(
                col_index,
                row_index,
                annotation_text(value, kind=kind),
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def load_pr_curve_csv(path: Path) -> pd.DataFrame:
    """Load and validate a PR curve CSV."""
    frame = pd.read_csv(path)
    missing = set(PR_CURVE_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"PR curve CSV is missing columns: {sorted(missing)}")
    return frame


def plot_pr_curve(
    frame: pd.DataFrame,
    *,
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    """Render a precision-recall curve to an image file."""
    missing = set(PR_CURVE_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"PR curve frame is missing columns: {sorted(missing)}")

    fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    ax.plot(frame["recall"], frame["precision"], linewidth=2.0)
    if not frame.empty:
        best_index = int(np.nan_to_num(frame["f1"].to_numpy(dtype=float), nan=-np.inf).argmax())
        best_row = frame.iloc[best_index]
        ax.scatter([best_row["recall"]], [best_row["precision"]], s=36, zorder=3)
        ax.annotate(
            f"thr={best_row['threshold']:.3g}\nF1={best_row['f1']:.3f}",
            xy=(best_row["recall"], best_row["precision"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
