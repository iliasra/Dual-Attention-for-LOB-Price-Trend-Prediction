from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from common_executable_evaluator import load_support_audits, summarize_support_censoring


LABEL_COLUMNS = {"broad": "broad_trend_label", "exec": "exec_trend_label"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the CPU L0-L5 label audit from common-support preprocessing artifacts."
    )
    parser.add_argument(
        "--audit",
        action="append",
        required=True,
        metavar="ID=PATH",
        help="Horizon/run ID and directory containing *_support_audit.npz; repeat as needed.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-acf-lag", type=int, default=500)
    parser.add_argument(
        "--realization-config",
        type=Path,
        default=None,
        help="Optional config passed to analyze_label_realization.py so L3 is included in this orchestration.",
    )
    parser.add_argument("--raw-dir", type=Path, default=None, help="Optional L3 raw-data override.")
    parser.add_argument(
        "--realization-splits",
        nargs="+",
        choices=("validation", "test"),
        default=("validation", "test"),
    )
    return parser.parse_args()


def parse_audits(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        run_id, separator, raw_path = value.partition("=")
        if not separator or not run_id.strip() or not raw_path.strip():
            raise ValueError("Each --audit must use ID=PATH.")
        if run_id in result:
            raise ValueError(f"Duplicate audit ID: {run_id}.")
        result[run_id] = Path(raw_path)
    return result


def effective_sample_size(values: np.ndarray, max_lag: int) -> tuple[float, float]:
    x = np.asarray(values, dtype=np.float64)
    n = len(x)
    if n < 3 or float(np.var(x)) <= 1e-15:
        return float(n), 0.0
    centered = x - x.mean()
    denominator = float(np.dot(centered, centered))
    positive_sum = 0.0
    used_lag = 0
    for lag in range(1, min(max_lag, n - 1) + 1):
        rho = float(np.dot(centered[:-lag], centered[lag:]) / denominator)
        if not np.isfinite(rho) or rho <= 0.0:
            break
        positive_sum += rho
        used_lag = lag
    ess = n / max(1.0 + 2.0 * positive_sum, 1.0)
    return float(ess), float(used_lag)


def label_temporal_tables(frame: pd.DataFrame, *, max_acf_lag: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cluster_rows: list[dict[str, object]] = []
    ess_rows: list[dict[str, object]] = []
    for label_family, column in LABEL_COLUMNS.items():
        if column not in frame.columns:
            continue
        for date, day in frame.sort_values(["date", "raw_event_index"]).groupby("date", sort=True):
            labels = pd.to_numeric(day[column], errors="coerce").to_numpy(float)
            times = pd.to_numeric(day["decision_time"], errors="coerce").to_numpy(float)
            valid = np.isfinite(labels) & np.isfinite(times) & np.isin(labels, [-1.0, 0.0, 1.0])
            labels, times = labels[valid].astype(np.int8), times[valid]
            if not len(labels):
                continue
            starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
            ends = np.r_[starts[1:], len(labels)]
            for cluster_index, (start, end) in enumerate(zip(starts, ends), start=1):
                duration = max(float(times[end - 1] - times[start]), 0.0)
                cluster_rows.append(
                    {
                        "label_family": label_family,
                        "date": str(date),
                        "cluster_id": cluster_index,
                        "class": int(labels[start]),
                        "events": int(end - start),
                        "elapsed_seconds": duration,
                        "mean_seconds_between_events": duration / (end - start - 1) if end - start > 1 else 0.0,
                    }
                )
            for class_value in (-1, 0, 1):
                indicator = (labels == class_value).astype(float)
                ess, positive_acf_lags = effective_sample_size(indicator, max_acf_lag)
                class_times = times[labels == class_value]
                ess_rows.append(
                    {
                        "label_family": label_family,
                        "date": str(date),
                        "class": class_value,
                        "events": int(indicator.sum()),
                        "event_rate": float(indicator.mean()),
                        "mean_seconds_between_class_events": (
                            float(np.mean(np.diff(class_times))) if len(class_times) > 1 else np.nan
                        ),
                        "effective_sample_size": ess,
                        "positive_acf_lags": positive_acf_lags,
                    }
                )
    return pd.DataFrame(cluster_rows), pd.DataFrame(ess_rows)


def horizon_summary(frame: pd.DataFrame, audit_id: str) -> dict[str, object]:
    valid = frame["exec_valid"].astype(bool) & frame["feature_history_valid"].astype(bool)
    long = pd.to_numeric(frame.loc[valid, "long_net_return_ticks"], errors="coerce").to_numpy(float)
    short = pd.to_numeric(frame.loc[valid, "short_net_return_ticks"], errors="coerce").to_numpy(float)
    finite = np.isfinite(long) & np.isfinite(short)
    long, short = long[finite], short[finite]
    oracle = np.maximum(0.0, np.maximum(long, short))
    durations: list[float] = []
    if {"entry_index", "exit_index", "decision_time"} <= set(frame.columns):
        for _date, day in frame.groupby("date", sort=False):
            time_by_index = dict(zip(day["raw_event_index"].astype(int), day["decision_time"].astype(float)))
            for entry, exit_ in zip(day["entry_index"], day["exit_index"]):
                if not np.isfinite(entry) or not np.isfinite(exit_):
                    continue
                if int(entry) in time_by_index and int(exit_) in time_by_index:
                    durations.append(max(time_by_index[int(exit_)] - time_by_index[int(entry)], 0.0))
    return {
        "audit_id": audit_id,
        "rows": int(len(frame)),
        "exec_supported_rows": int(len(long)),
        "long_profitable_rate": float(np.mean(long > 0.0)) if len(long) else np.nan,
        "short_profitable_rate": float(np.mean(short > 0.0)) if len(short) else np.nan,
        "any_profitable_rate": float(np.mean(oracle > 0.0)) if len(oracle) else np.nan,
        "oracle_mean_pnl_ticks": float(np.mean(oracle)) if len(oracle) else np.nan,
        "mean_realized_horizon_seconds": float(np.mean(durations)) if durations else np.nan,
        "median_realized_horizon_seconds": float(np.median(durations)) if durations else np.nan,
    }


def label_semantics_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "broad_trend_label",
        "exec_trend_label",
        "broad_valid",
        "exec_valid",
        "feature_history_valid",
        "long_net_return_ticks",
        "short_net_return_ticks",
    }
    if not required <= set(frame.columns):
        return pd.DataFrame(), pd.DataFrame()
    mask = (
        frame["broad_valid"].astype(bool)
        & frame["exec_valid"].astype(bool)
        & frame["feature_history_valid"].astype(bool)
    )
    common = frame.loc[mask].copy()
    common["broad"] = pd.to_numeric(common["broad_trend_label"], errors="coerce")
    common["exec"] = pd.to_numeric(common["exec_trend_label"], errors="coerce")
    common = common[common["broad"].isin([-1, 0, 1]) & common["exec"].isin([-1, 0, 1])]
    confusion = pd.crosstab(common["broad"], common["exec"], dropna=False).rename_axis(
        index="broad_class", columns="exec_class"
    )
    rows = []
    for broad_class, group in common.groupby("broad", sort=True):
        long = group["long_net_return_ticks"].to_numpy(float)
        short = group["short_net_return_ticks"].to_numpy(float)
        chosen = long if int(broad_class) == 1 else short if int(broad_class) == -1 else np.zeros(len(group))
        rows.append(
            {
                "broad_class": int(broad_class),
                "rows": int(len(group)),
                "broad_direction_profitable_rate": float(np.mean(chosen > 0.0)),
                "broad_direction_mean_pnl_ticks": float(np.mean(chosen)),
                "any_executable_action_profitable_rate": float(
                    np.mean(np.maximum(long, short) > 0.0)
                ),
            }
        )
    return confusion.reset_index(), pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.max_acf_lag <= 0:
        raise ValueError("max_acf_lag must be positive.")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    manifest: dict[str, object] = {"schema_version": 1, "audits": {}}
    for audit_id, source in parse_audits(args.audit).items():
        frame = load_support_audits([source])
        audit_dir = output / audit_id
        audit_dir.mkdir(parents=True, exist_ok=True)
        censoring = summarize_support_censoring(frame)
        clusters, ess = label_temporal_tables(frame, max_acf_lag=args.max_acf_lag)
        confusion, semantics = label_semantics_tables(frame)
        censoring.to_csv(audit_dir / "L0_support_and_hourly_censoring.csv", index=False)
        clusters.to_csv(audit_dir / "L4_label_clusters.csv", index=False)
        ess.to_csv(audit_dir / "L4_temporal_dependence_ess.csv", index=False)
        confusion.to_csv(audit_dir / "L5_broad_exec_confusion.csv", index=False)
        semantics.to_csv(audit_dir / "L5_broad_exec_economic_semantics.csv", index=False)
        summary = horizon_summary(frame, audit_id)
        summaries.append(summary)
        (audit_dir / "L2_horizon_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        manifest["audits"][audit_id] = {
            "source": str(source),
            "rows": int(len(frame)),
            "outputs": [
                "L0_support_and_hourly_censoring.csv",
                "L2_horizon_summary.json",
                "L4_label_clusters.csv",
                "L4_temporal_dependence_ess.csv",
                "L5_broad_exec_confusion.csv",
                "L5_broad_exec_economic_semantics.csv",
            ],
        }
    definitions = {
        "L0": "support inventory and censoring by day/hour",
        "L1": {
            "BROAD": "Method-C smoothed direction; -1=down, 0=neutral, 1=up",
            "EXEC_CLS": "best executable action class reconstructed from long/short realized values",
            "EXEC_AV": "continuous net executable values [V_long, V_short] in ticks",
            "causality": "all features and endpoint masks use information available at or before decision time; outcomes are targets only",
        },
        "L2": "horizon prevalence, oracle value and realized wall-clock duration",
        "L3": "ex-ante versus ex-post Method-C realization (optional subprocess below)",
        "L4": "hourly concentration, clusters, elapsed time, spacing and ESS",
        "L5": "BROAD versus EXEC semantic confusion and executable profitability",
    }
    (output / "L1_label_definitions.yaml").write_text(
        yaml.safe_dump(definitions, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    pd.DataFrame(summaries).to_csv(output / "L2_horizon_comparison.csv", index=False)
    if args.realization_config is not None:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "analyze_label_realization.py"),
            "--config",
            str(args.realization_config),
            "--output-dir",
            str(output / "L3_exante_expost"),
            "--splits",
            *args.realization_splits,
        ]
        if args.raw_dir is not None:
            command.extend(["--raw-dir", str(args.raw_dir)])
        subprocess.run(command, check=True)
        manifest["L3_exante_expost"] = {"command": command, "status": "completed"}
    (output / "audit_manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(f"Wrote label audits for {len(summaries)} inputs to {output}.")


if __name__ == "__main__":
    main()
