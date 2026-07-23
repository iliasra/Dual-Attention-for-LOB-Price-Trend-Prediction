from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml


KEY_COLUMNS = ["date", "raw_event_index"]
ECONOMIC_COLUMNS = ["realized_long", "realized_short", "entry_index", "exit_index"]
DEFAULT_BUDGETS = (0.001, 0.0025, 0.005, 0.01, 0.02)


def load_support_audits(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Load full-day censor masks emitted by common-support preprocessing."""
    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            files.extend(sorted(path.rglob("*_support_audit.npz")))
        elif path.exists():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    if not files:
        raise ValueError("No *_support_audit.npz files were found.")
    frames = []
    for path in files:
        with np.load(path, allow_pickle=False) as payload:
            required = {"date", "raw_event_index", "broad_valid", "exec_valid", "feature_history_valid"}
            missing = sorted(required - set(payload.files))
            if missing:
                raise ValueError(f"Support audit {path} is missing {missing}.")
            frames.append(pd.DataFrame({key: np.asarray(payload[key]) for key in payload.files}))
    result = pd.concat(frames, ignore_index=True)
    result["date"] = result["date"].astype(str)
    if result.duplicated(KEY_COLUMNS).any():
        raise ValueError("Support audit inputs contain duplicate (date, raw_event_index) keys.")
    return result


def summarize_support_censoring(audit: pd.DataFrame) -> pd.DataFrame:
    """Summarize B, E, I and censor reasons by day and 15-minute bin."""
    _require_columns(
        audit,
        [*KEY_COLUMNS, "broad_valid", "exec_valid", "feature_history_valid"],
        "preprocessing support audit",
    )
    frame = audit.copy()
    frame["broad_valid"] = frame["broad_valid"].astype(bool)
    frame["exec_valid"] = frame["exec_valid"].astype(bool)
    frame["feature_history_valid"] = frame["feature_history_valid"].astype(bool)
    frame["intersection_valid"] = frame["broad_valid"] & frame["exec_valid"] & frame["feature_history_valid"]
    frame["broad_only"] = frame["broad_valid"] & ~frame["exec_valid"]
    frame["exec_only"] = frame["exec_valid"] & ~frame["broad_valid"]
    if "decision_time" in frame.columns:
        frame["time_bin_15m"] = (np.floor(frame["decision_time"].astype(float) / 900.0) * 900.0).astype(int)
    else:
        frame["time_bin_15m"] = -1
    rows = []
    for (date, time_bin), group in frame.groupby(["date", "time_bin_15m"], sort=True):
        broad_rows = int(group["broad_valid"].sum())
        exec_rows = int(group["exec_valid"].sum())
        common_rows = int(group["intersection_valid"].sum())
        row = {
                "date": date,
                "time_bin_15m": time_bin,
                "rows": len(group),
                "broad_rows": broad_rows,
                "exec_rows": exec_rows,
                "intersection_rows": common_rows,
                "broad_only_rows": int(group["broad_only"].sum()),
                "exec_only_rows": int(group["exec_only"].sum()),
                "broad_censored_rows": int((~group["broad_valid"]).sum()),
                "exec_censored_rows": int((~group["exec_valid"]).sum()),
                "feature_history_censored_rows": int((~group["feature_history_valid"]).sum()),
                "broad_retention": common_rows / broad_rows if broad_rows else np.nan,
                "exec_retention": common_rows / exec_rows if exec_rows else np.nan,
            }
        broad_labels = group.get("broad_trend_label")
        exec_labels = group.get("exec_trend_label")
        realized_long = group.get("long_net_return_ticks")
        realized_short = group.get("short_net_return_ticks")
        support_masks = {
            "intersection": group["intersection_valid"].to_numpy(bool),
            "broad_only": group["broad_only"].to_numpy(bool),
            "exec_only": group["exec_only"].to_numpy(bool),
        }
        for support_name, mask in support_masks.items():
            if broad_labels is not None:
                values = pd.to_numeric(broad_labels[mask], errors="coerce").dropna().to_numpy(float)
                row[f"{support_name}_broad_up_rate"] = float(np.mean(values == 1)) if len(values) else np.nan
                row[f"{support_name}_broad_down_rate"] = float(np.mean(values == -1)) if len(values) else np.nan
            if exec_labels is not None:
                values = pd.to_numeric(exec_labels[mask], errors="coerce").dropna().to_numpy(float)
                row[f"{support_name}_exec_up_rate"] = float(np.mean(values == 1)) if len(values) else np.nan
                row[f"{support_name}_exec_down_rate"] = float(np.mean(values == -1)) if len(values) else np.nan
            if realized_long is not None and realized_short is not None:
                long = pd.to_numeric(realized_long[mask], errors="coerce").to_numpy(float)
                short = pd.to_numeric(realized_short[mask], errors="coerce").to_numpy(float)
                finite = np.isfinite(long) & np.isfinite(short)
                oracle = np.maximum(0.0, np.maximum(long[finite], short[finite]))
                row[f"{support_name}_profitable_rate"] = float(np.mean(oracle > 0.0)) if len(oracle) else np.nan
                row[f"{support_name}_oracle_mean_pnl_ticks"] = float(np.mean(oracle)) if len(oracle) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


@dataclass(frozen=True, slots=True)
class CommonEvaluationResult:
    support_audit: pd.DataFrame
    native_metrics: pd.DataFrame
    label_confusion: pd.DataFrame
    label_semantics: pd.DataFrame
    daily_metrics: pd.DataFrame
    aggregate_metrics: pd.DataFrame
    paired_differences: pd.DataFrame
    trades: pd.DataFrame


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}.")


def _validate_keys(frame: pd.DataFrame, context: str) -> None:
    _require_columns(frame, KEY_COLUMNS, context)
    if frame[KEY_COLUMNS].isna().any().any():
        raise ValueError(f"{context} contains missing sample keys.")
    duplicated = frame.duplicated(KEY_COLUMNS, keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, KEY_COLUMNS].head(5).to_dict("records")
        raise ValueError(f"{context} contains duplicate sample keys, for example {examples}.")


def _canonical_frame(
    frame: pd.DataFrame,
    *,
    model: str,
    score_long: np.ndarray,
    score_short: np.ndarray,
    require_economic: bool = True,
) -> pd.DataFrame:
    required = [*KEY_COLUMNS, *ECONOMIC_COLUMNS] if require_economic else KEY_COLUMNS
    _require_columns(frame, required, model)
    _validate_keys(frame, model)
    result = frame.copy()
    result["date"] = result["date"].astype(str)
    result["raw_event_index"] = result["raw_event_index"].astype(np.int64)
    if require_economic:
        result["entry_index"] = result["entry_index"].astype(np.int64)
        result["exit_index"] = result["exit_index"].astype(np.int64)
        result["realized_long"] = result["realized_long"].astype(float)
        result["realized_short"] = result["realized_short"].astype(float)
    result["score_long"] = np.asarray(score_long, dtype=float)
    result["score_short"] = np.asarray(score_short, dtype=float)
    result["model"] = model
    numeric_columns = ["score_long", "score_short", *ECONOMIC_COLUMNS] if require_economic else ["score_long", "score_short"]
    numeric = result[numeric_columns].select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{model} contains non-finite scores, outcomes, or intervals.")
    if require_economic and (result["entry_index"] > result["exit_index"]).any():
        raise ValueError(f"{model} contains entry indices after exit indices.")
    return result


def load_classification_predictions(
    path: str | Path,
    *,
    model: str,
    up_column: str = "p_up",
    down_column: str = "p_down",
    require_economic: bool = True,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    _require_columns(frame, [up_column, down_column], f"classification predictions {path}")
    return _canonical_frame(
        frame,
        model=model,
        score_long=frame[up_column].to_numpy(dtype=float),
        score_short=frame[down_column].to_numpy(dtype=float),
        require_economic=require_economic,
    )


def load_action_value_predictions(path: str | Path, *, model: str = "EXEC_AV") -> pd.DataFrame:
    with np.load(path, allow_pickle=False) as payload:
        required = {"predictions", "targets", "date", "raw_event_index", "entry_index", "exit_index"}
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"Action-value predictions {path} are missing {missing}.")
        predictions = np.asarray(payload["predictions"], dtype=float)
        targets = np.asarray(payload["targets"], dtype=float)
        if predictions.ndim != 2 or predictions.shape[1] < 2 or targets.shape != (len(predictions), 2):
            raise ValueError("Action-value predictions/targets must have shapes [N,>=2] and [N,2].")
        frame = pd.DataFrame(
            {
                "date": np.asarray(payload["date"]).astype(str),
                "raw_event_index": np.asarray(payload["raw_event_index"], dtype=np.int64),
                "entry_index": np.asarray(payload["entry_index"], dtype=np.int64),
                "exit_index": np.asarray(payload["exit_index"], dtype=np.int64),
                "realized_long": np.asarray(payload["realized_long"], dtype=float)
                if "realized_long" in payload.files
                else targets[:, 0],
                "realized_short": np.asarray(payload["realized_short"], dtype=float)
                if "realized_short" in payload.files
                else targets[:, 1],
            }
        )
        for column in ("broad_label", "exec_label", "decision_time"):
            if column in payload.files:
                frame[column] = np.asarray(payload[column])
    return _canonical_frame(
        frame,
        model=model,
        score_long=predictions[:, 0],
        score_short=predictions[:, 1],
    )


def _key_index(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(frame[KEY_COLUMNS])


def _assert_exec_alignment(exec_cls: pd.DataFrame, exec_av: pd.DataFrame) -> None:
    cls_keys = _key_index(exec_cls)
    av_keys = _key_index(exec_av)
    if len(cls_keys) != len(av_keys) or len(cls_keys.difference(av_keys)) or len(av_keys.difference(cls_keys)):
        missing_cls = av_keys.difference(cls_keys)
        missing_av = cls_keys.difference(av_keys)
        raise ValueError(
            "EXEC_CLS and EXEC_AV must have identical sample keys; "
            f"missing_from_cls={len(missing_cls)}, missing_from_av={len(missing_av)}."
        )
    aligned_av = exec_av.set_index(KEY_COLUMNS).loc[cls_keys]
    for column in ECONOMIC_COLUMNS:
        left = exec_cls[column].to_numpy(dtype=float)
        right = aligned_av[column].to_numpy(dtype=float)
        if not np.allclose(left, right, rtol=0.0, atol=1e-7):
            raise ValueError(f"EXEC_CLS and EXEC_AV disagree on {column!r}.")


def _subset_by_keys(frame: pd.DataFrame, keys: pd.MultiIndex) -> pd.DataFrame:
    indexed = frame.set_index(KEY_COLUMNS, drop=False)
    return indexed.loc[keys].reset_index(drop=True)


def _support_audit(broad: pd.DataFrame, exec_frame: pd.DataFrame) -> pd.DataFrame:
    broad_keys = _key_index(broad)
    exec_keys = _key_index(exec_frame)
    common_keys = broad_keys.intersection(exec_keys, sort=False)
    rows = []
    dates = sorted(set(broad["date"]) | set(exec_frame["date"]))
    for date in dates:
        b = set(map(tuple, broad.loc[broad["date"] == date, KEY_COLUMNS].to_numpy()))
        e = set(map(tuple, exec_frame.loc[exec_frame["date"] == date, KEY_COLUMNS].to_numpy()))
        intersection = b & e
        rows.append(
            {
                "date": date,
                "broad_rows": len(b),
                "exec_rows": len(e),
                "intersection_rows": len(intersection),
                "broad_only_rows": len(b - e),
                "exec_only_rows": len(e - b),
                "broad_retention": len(intersection) / len(b) if b else np.nan,
                "exec_retention": len(intersection) / len(e) if e else np.nan,
            }
        )
    result = pd.DataFrame(rows)
    result.attrs["common_rows"] = len(common_keys)
    return result


def _normalize_label(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"1", "1.0", "up", "class_up"}:
        return "up"
    if text in {"-1", "-1.0", "down", "class_down"}:
        return "down"
    if text in {"0", "0.0", "neutral", "class_neutral"}:
        return "neutral"
    return text


def _exec_labels(frame: pd.DataFrame) -> np.ndarray:
    long = frame["realized_long"].to_numpy(dtype=float)
    short = frame["realized_short"].to_numpy(dtype=float)
    labels = np.full(len(frame), "neutral", dtype="<U7")
    labels[(long > 0.0) & (long > short)] = "up"
    labels[(short > 0.0) & (short > long)] = "down"
    return labels


def _assert_exec_class_contract(frame: pd.DataFrame) -> None:
    """Fail closed unless EXEC_CLS labels are exactly reconstructible from action values."""
    source = "true_label" if "true_label" in frame.columns else "exec_label" if "exec_label" in frame.columns else None
    if source is None:
        return
    reported = np.asarray([_normalize_label(value) for value in frame[source]])
    reconstructed = _exec_labels(frame)
    mismatch = reported != reconstructed
    if mismatch.any():
        examples = frame.loc[mismatch, KEY_COLUMNS].head(5).to_dict("records")
        raise ValueError(
            "EXEC_CLS labels are not exactly reconstructible from realized long/short values; "
            f"mismatches={int(mismatch.sum())}, examples={examples}."
        )


def _native_metrics(frame: pd.DataFrame) -> dict[str, object]:
    row: dict[str, object] = {"model": str(frame["model"].iloc[0]), "rows": len(frame)}
    if {"realized_long", "realized_short"}.issubset(frame.columns):
        for side in ("long", "short"):
            realized = frame[f"realized_{side}"].to_numpy(dtype=float)
            score = frame[f"score_{side}"].to_numpy(dtype=float)
            row[f"mae_{side}"] = float(np.mean(np.abs(score - realized))) if len(frame) else 0.0
            row[f"bias_{side}"] = float(np.mean(score - realized)) if len(frame) else 0.0
            row[f"ic_{side}"] = _safe_ic(score, realized)
            row[f"profitable_ap_{side}"] = _safe_ap(realized > 0.0, score)
            if len(frame) >= 2 and float(np.std(score)) > 0.0:
                slope, intercept = np.polyfit(score, realized, deg=1)
                row[f"calibration_slope_{side}"] = float(slope)
                row[f"calibration_intercept_{side}"] = float(intercept)
            else:
                row[f"calibration_slope_{side}"] = 0.0
                row[f"calibration_intercept_{side}"] = float(np.mean(realized)) if len(frame) else 0.0
    if "true_label" not in frame.columns:
        return row
    true = np.asarray([_normalize_label(value) for value in frame["true_label"]])
    if "pred_label" in frame.columns:
        pred = np.asarray([_normalize_label(value) for value in frame["pred_label"]])
        row["macro_f1"] = _macro_f1(true, pred, ("down", "neutral", "up"))
        row["directional_macro_f1"] = _macro_f1(true, pred, ("down", "up"))
    row["pr_ap_up"] = _safe_ap(true == "up", frame["score_long"].to_numpy(dtype=float))
    row["pr_ap_down"] = _safe_ap(true == "down", frame["score_short"].to_numpy(dtype=float))
    return row


def _safe_ap(target: np.ndarray, score: np.ndarray) -> float:
    target = np.asarray(target, dtype=bool)
    if target.size == 0 or not target.any():
        return 0.0
    scores = np.asarray(score, dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    ordered_target = target[order].astype(np.int64)
    ordered_score = scores[order]
    # Average precision is the step-function area under the PR curve. Tied
    # scores are evaluated as one threshold, so stable input order cannot create
    # artificial skill for constant/no-skill baselines.
    threshold_ends = np.r_[np.flatnonzero(ordered_score[1:] != ordered_score[:-1]), len(scores) - 1]
    cumulative_tp = np.cumsum(ordered_target)
    tp = cumulative_tp[threshold_ends].astype(float)
    selected = threshold_ends.astype(float) + 1.0
    recall = tp / float(target.sum())
    precision = tp / selected
    previous_recall = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - previous_recall) * precision))


def _macro_f1(true: np.ndarray, predicted: np.ndarray, labels: Iterable[str]) -> float:
    values = []
    for label in labels:
        true_positive = int(np.sum((true == label) & (predicted == label)))
        false_positive = int(np.sum((true != label) & (predicted == label)))
        false_negative = int(np.sum((true == label) & (predicted != label)))
        denominator = 2 * true_positive + false_positive + false_negative
        values.append((2.0 * true_positive / denominator) if denominator else 0.0)
    return float(np.mean(values)) if values else 0.0


def _safe_ic(score: np.ndarray, target: np.ndarray) -> float:
    if len(score) < 2:
        return 0.0
    ranked_score = pd.Series(score).rank(method="average")
    ranked_target = pd.Series(target).rank(method="average")
    if ranked_score.nunique() < 2 or ranked_target.nunique() < 2:
        return 0.0
    value = ranked_score.corr(ranked_target)
    return 0.0 if pd.isna(value) else float(value)


def _tie_value(date: str, raw_event_index: int, side: str, seed: int) -> int:
    payload = f"{seed}|{date}|{raw_event_index}|{side}".encode("utf-8")
    return int.from_bytes(blake2b(payload, digest_size=8).digest(), "little", signed=False)


def _overlaps(entry: int, exit_: int, accepted: list[tuple[int, int]]) -> bool:
    return any(entry <= accepted_exit and exit_ >= accepted_entry for accepted_entry, accepted_exit in accepted)


def select_daily_fixed_budget(
    frame: pd.DataFrame,
    *,
    budget: float,
    seed: int,
    support: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Select score-prioritized, side-exclusive trades with one global position."""
    if not 0.0 < budget <= 0.5:
        raise ValueError("budget must be in (0, 0.5].")
    if frame["date"].nunique() != 1 or frame["model"].nunique() != 1:
        raise ValueError("select_daily_fixed_budget expects one model and one day.")
    n = len(frame)
    quota = int(np.ceil(budget * n))
    date = str(frame["date"].iloc[0])
    model = str(frame["model"].iloc[0])
    candidates = []
    for side, score_column in (("long", "score_long"), ("short", "score_short")):
        scores = frame[score_column].to_numpy(dtype=float)
        ranks = pd.Series(scores).rank(method="min", ascending=False).to_numpy(dtype=float)
        for position, (_, row) in enumerate(frame.iterrows()):
            candidates.append(
                {
                    "priority": ranks[position] / max(n, 1),
                    "tie": _tie_value(date, int(row["raw_event_index"]), side, seed),
                    "side": side,
                    "row": row,
                }
            )
    candidates.sort(key=lambda item: (item["priority"], item["tie"]))

    accepted_intervals: list[tuple[int, int]] = []
    used_samples: set[int] = set()
    side_counts = {"long": 0, "short": 0}
    records = []
    for candidate in candidates:
        side = str(candidate["side"])
        row = candidate["row"]
        raw_index = int(row["raw_event_index"])
        entry = int(row["entry_index"])
        exit_ = int(row["exit_index"])
        if side_counts[side] >= quota or raw_index in used_samples or _overlaps(entry, exit_, accepted_intervals):
            continue
        pnl = float(row[f"realized_{side}"])
        records.append(
            {
                "support": support,
                "model": model,
                "date": date,
                "budget": float(budget),
                "raw_event_index": raw_index,
                "entry_index": entry,
                "exit_index": exit_,
                "side": side,
                "score": float(row[f"score_{side}"]),
                "realized_pnl": pnl,
                "profitable": bool(pnl > 0.0),
            }
        )
        side_counts[side] += 1
        used_samples.add(raw_index)
        accepted_intervals.append((entry, exit_))
        if side_counts["long"] >= quota and side_counts["short"] >= quota:
            break

    trades = pd.DataFrame(records)
    pnl = trades["realized_pnl"].to_numpy(dtype=float) if len(trades) else np.asarray([], dtype=float)
    metrics: dict[str, object] = {
        "support": support,
        "model": model,
        "date": date,
        "budget": float(budget),
        "eligible_rows": n,
        "requested_per_side": quota,
        "requested_trades": 2 * quota,
        "executed_trades": len(trades),
        "long_trades": side_counts["long"],
        "short_trades": side_counts["short"],
        "requested_coverage": (2 * quota / n) if n else 0.0,
        "executed_coverage": (len(trades) / n) if n else 0.0,
        "win_rate": float(np.mean(pnl > 0.0)) if len(pnl) else 0.0,
        "total_pnl_ticks": float(pnl.sum()),
        "mean_pnl_per_trade_ticks": float(pnl.mean()) if len(pnl) else 0.0,
        "pnl_per_endpoint_ticks": float(pnl.sum() / n) if n else 0.0,
        "profitable_long_ap": _safe_ap(
            frame["realized_long"].to_numpy(dtype=float) > 0.0,
            frame["score_long"].to_numpy(dtype=float),
        ),
        "profitable_short_ap": _safe_ap(
            frame["realized_short"].to_numpy(dtype=float) > 0.0,
            frame["score_short"].to_numpy(dtype=float),
        ),
        "long_ic": _safe_ic(
            frame["score_long"].to_numpy(dtype=float), frame["realized_long"].to_numpy(dtype=float)
        ),
        "short_ic": _safe_ic(
            frame["score_short"].to_numpy(dtype=float), frame["realized_short"].to_numpy(dtype=float)
        ),
    }
    metrics["mean_ic"] = (float(metrics["long_ic"]) + float(metrics["short_ic"])) / 2.0
    return trades, metrics


def _aggregate_daily(daily: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    numeric = [
        "executed_coverage",
        "win_rate",
        "total_pnl_ticks",
        "mean_pnl_per_trade_ticks",
        "pnl_per_endpoint_ticks",
        "profitable_long_ap",
        "profitable_short_ap",
        "long_ic",
        "short_ic",
        "mean_ic",
    ]
    rows = []
    for keys, group in daily.groupby(["support", "model", "budget"], sort=True):
        row = {"support": keys[0], "model": keys[1], "budget": keys[2], "days": len(group)}
        for column in numeric:
            row[f"daily_mean_{column}"] = float(group[column].mean())
            row[f"daily_median_{column}"] = float(group[column].median())
        row["positive_day_fraction"] = float((group["total_pnl_ticks"] > 0.0).mean())
        row["pooled_total_pnl_ticks"] = float(group["total_pnl_ticks"].sum())
        row["total_executed_trades"] = int(group["executed_trades"].sum())
        pnl_low, pnl_high = _bootstrap_mean(group["total_pnl_ticks"].to_numpy(), seed=seed)
        win_low, win_high = _bootstrap_mean(group["win_rate"].to_numpy(), seed=seed + 1)
        row["daily_mean_total_pnl_ticks_bootstrap_low_95"] = pnl_low
        row["daily_mean_total_pnl_ticks_bootstrap_high_95"] = pnl_high
        row["daily_mean_win_rate_bootstrap_low_95"] = win_low
        row["daily_mean_win_rate_bootstrap_high_95"] = win_high
        rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap_mean(values: np.ndarray, *, seed: int, draws: int = 2000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _paired_differences(daily: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    rows = []
    for (support, budget), group in daily.groupby(["support", "budget"]):
        models = sorted(group["model"].unique())
        for left, right in combinations(models, 2):
            merged = group.loc[group["model"] == right].merge(
                group.loc[group["model"] == left], on=["support", "date", "budget"], suffixes=("_right", "_left")
            )
            if merged.empty:
                continue
            pnl_delta = merged["total_pnl_ticks_right"] - merged["total_pnl_ticks_left"]
            win_delta = merged["win_rate_right"] - merged["win_rate_left"]
            low, high = _bootstrap_mean(pnl_delta.to_numpy(), seed=seed)
            rows.append(
                {
                    "support": support,
                    "budget": budget,
                    "comparison": f"{right}-{left}",
                    "paired_days": len(merged),
                    "mean_daily_pnl_delta_ticks": float(pnl_delta.mean()),
                    "median_daily_pnl_delta_ticks": float(pnl_delta.median()),
                    "mean_win_rate_delta": float(win_delta.mean()),
                    "pnl_delta_bootstrap_low_95": low,
                    "pnl_delta_bootstrap_high_95": high,
                }
            )
    return pd.DataFrame(rows)


def evaluate_common_models(
    broad: pd.DataFrame,
    exec_cls: pd.DataFrame,
    exec_av: pd.DataFrame,
    *,
    budgets: Iterable[float] = DEFAULT_BUDGETS,
    seed: int = 42,
) -> CommonEvaluationResult:
    _assert_exec_alignment(exec_cls, exec_av)
    _assert_exec_class_contract(exec_cls)
    support_audit = _support_audit(broad, exec_cls)
    common_keys = _key_index(broad).intersection(_key_index(exec_cls), sort=False)
    if len(common_keys) == 0:
        raise ValueError("BROAD and EXEC have no exact common (date, raw_event_index) endpoints.")

    broad_common = _subset_by_keys(broad, common_keys)
    cls_common = _subset_by_keys(exec_cls, common_keys)
    av_common = _subset_by_keys(exec_av, common_keys)
    for column in ECONOMIC_COLUMNS:
        reference = cls_common[column].to_numpy(dtype=float)
        if column in broad_common and not np.allclose(
            broad_common[column].to_numpy(dtype=float), reference, rtol=0.0, atol=1e-7
        ):
            raise ValueError(f"BROAD and EXEC disagree on {column!r} over their exact intersection.")
        broad_common[column] = reference

    native_metrics = pd.DataFrame([_native_metrics(broad), _native_metrics(exec_cls), _native_metrics(exec_av)])
    broad_true_source = "true_label" if "true_label" in broad_common.columns else "broad_label"
    _require_columns(broad_common, [broad_true_source], "BROAD predictions")
    broad_labels = np.asarray([_normalize_label(value) for value in broad_common[broad_true_source]])
    exec_labels = _exec_labels(cls_common)
    semantic_base = broad_common.loc[:, [*KEY_COLUMNS, "realized_long", "realized_short"]].copy()
    semantic_base["broad_label"] = broad_labels
    semantic_base["exec_label"] = exec_labels
    if "decision_time" in broad_common.columns:
        semantic_base["time_bin_15m"] = (
            np.floor(broad_common["decision_time"].to_numpy(dtype=float) / 900.0) * 900.0
        ).astype(int)
    else:
        semantic_base["time_bin_15m"] = -1
    semantic_groups: list[tuple[str, str, int, pd.DataFrame]] = [
        ("global", "all", -1, semantic_base)
    ]
    semantic_groups.extend(
        ("day", str(date), -1, group) for date, group in semantic_base.groupby("date", sort=True)
    )
    semantic_groups.extend(
        ("day_time", str(date), int(time_bin), group)
        for (date, time_bin), group in semantic_base.groupby(["date", "time_bin_15m"], sort=True)
    )
    confusion_rows = []
    for scope, date, time_bin, semantic_group in semantic_groups:
        counts = semantic_group.groupby(["broad_label", "exec_label"], dropna=False).size()
        for broad_label in ("down", "neutral", "up"):
            for exec_label in ("down", "neutral", "up"):
                confusion_rows.append(
                    {
                        "scope": scope,
                        "date": date,
                        "time_bin_15m": time_bin,
                        "broad_label": broad_label,
                        "exec_label": exec_label,
                        "count": int(counts.get((broad_label, exec_label), 0)),
                    }
                )
    label_confusion = pd.DataFrame(confusion_rows)
    semantics_rows = []
    for scope, date, time_bin, semantic_group in semantic_groups:
        best_exec = np.maximum(
            semantic_group["realized_long"].to_numpy(float),
            semantic_group["realized_short"].to_numpy(float),
        )
        for label, side in (("down", "short"), ("neutral", None), ("up", "long")):
            mask = semantic_group["broad_label"].to_numpy() == label
            pnl = (
                np.zeros(mask.sum(), dtype=float)
                if side is None
                else semantic_group.loc[mask, f"realized_{side}"].to_numpy(float)
            )
            semantics_rows.append(
                {
                    "scope": scope,
                    "date": date,
                    "time_bin_15m": time_bin,
                    "broad_label": label,
                    "rows": int(mask.sum()),
                    "exec_profitable_probability": float(np.mean(best_exec[mask] > 0.0)) if mask.any() else 0.0,
                    "directional_win_rate": float(np.mean(pnl > 0.0)) if len(pnl) and side is not None else 0.0,
                    "directional_mean_pnl_ticks": float(pnl.mean()) if len(pnl) and side is not None else 0.0,
                    "label_agreement_rate": float(
                        np.mean(semantic_group.loc[mask, "exec_label"].to_numpy() == label)
                    ) if mask.any() else 0.0,
                    "directional_but_not_profitable_rows": int(np.sum(pnl <= 0.0))
                    if side is not None
                    else 0,
                    "profitable_opportunity_missed_by_broad": int(np.sum(best_exec[mask] > 0.0))
                    if label == "neutral"
                    else 0,
                }
            )
    label_semantics = pd.DataFrame(semantics_rows)

    supports = {
        "triple_common": [broad_common, cls_common, av_common],
        "exec_full": [exec_cls, exec_av],
    }
    daily_rows = []
    trade_frames = []
    for support_name, model_frames in supports.items():
        for model_frame in model_frames:
            for _, day in model_frame.groupby("date", sort=True):
                for budget in budgets:
                    trades, metrics = select_daily_fixed_budget(
                        day.reset_index(drop=True), budget=float(budget), seed=seed, support=support_name
                    )
                    daily_rows.append(metrics)
                    if not trades.empty:
                        trade_frames.append(trades)
    daily = pd.DataFrame(daily_rows)
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    aggregate = _aggregate_daily(daily, seed=seed)
    paired = _paired_differences(daily, seed=seed)
    return CommonEvaluationResult(
        support_audit=support_audit,
        native_metrics=native_metrics,
        label_confusion=label_confusion,
        label_semantics=label_semantics,
        daily_metrics=daily,
        aggregate_metrics=aggregate,
        paired_differences=paired,
        trades=trades,
    )


def save_common_evaluation(result: CommonEvaluationResult, output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "support_audit": output / "support_audit_by_day.csv",
        "native_metrics": output / "native_task_metrics.csv",
        "label_confusion": output / "broad_exec_label_confusion.csv",
        "label_semantics": output / "broad_exec_label_semantics.csv",
        "daily_metrics": output / "common_policy_by_day.csv",
        "aggregate_metrics": output / "common_policy_summary.csv",
        "paired_differences": output / "paired_daily_differences.csv",
        "trades": output / "common_policy_trades.csv",
        "budget_pnl_plot": output / "budget_vs_daily_pnl.png",
        "budget_win_rate_plot": output / "budget_vs_win_rate.png",
        "summary": output / "common_evaluation_summary.yaml",
    }
    result.support_audit.to_csv(artifacts["support_audit"], index=False)
    result.native_metrics.to_csv(artifacts["native_metrics"], index=False)
    result.label_confusion.to_csv(artifacts["label_confusion"], index=False)
    result.label_semantics.to_csv(artifacts["label_semantics"], index=False)
    result.daily_metrics.to_csv(artifacts["daily_metrics"], index=False)
    result.aggregate_metrics.to_csv(artifacts["aggregate_metrics"], index=False)
    result.paired_differences.to_csv(artifacts["paired_differences"], index=False)
    result.trades.to_csv(artifacts["trades"], index=False)
    try:
        import matplotlib.pyplot as plt

        triple = result.aggregate_metrics.loc[result.aggregate_metrics["support"] == "triple_common"]
        for metric, artifact_key, ylabel in (
            ("daily_mean_total_pnl_ticks", "budget_pnl_plot", "Equal-day mean total PnL (ticks)"),
            ("daily_mean_win_rate", "budget_win_rate_plot", "Equal-day mean win rate"),
        ):
            figure, axis = plt.subplots(figsize=(7.2, 4.5))
            for model, group in triple.groupby("model", sort=True):
                ordered = group.sort_values("budget")
                axis.plot(100.0 * ordered["budget"], ordered[metric], marker="o", label=model)
            axis.set_xlabel("Daily budget per side (%)")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
            axis.legend()
            figure.tight_layout()
            figure.savefig(artifacts[artifact_key], dpi=160)
            plt.close(figure)
    except (ImportError, RuntimeError):
        artifacts.pop("budget_pnl_plot", None)
        artifacts.pop("budget_win_rate_plot", None)
    summary = {
        "schema_version": 1,
        "sample_key": KEY_COLUMNS,
        "position_limit": 1,
        "interval_convention": "inclusive",
        "budget_scope": "per_day_per_side",
        "triple_common_rows": int(result.support_audit["intersection_rows"].sum()),
        "days": int(len(result.support_audit)),
        "artifacts": {key: path.name for key, path in artifacts.items() if key != "summary"},
    }
    artifacts["summary"].write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    return artifacts
