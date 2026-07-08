from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "output" / "DATA_20260708_110249" / "fold2324"
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "LOBSTER"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "label"

CLASS_TO_ID = {"down": 0, "neutral": 1, "up": 2}
ID_TO_CLASS = {value: key for key, value in CLASS_TO_ID.items()}
POSITION_BY_LABEL = {"down": -1, "neutral": 0, "up": 1}
SPLITS = ("validation", "test")


@dataclass(slots=True)
class LabelComponents:
    frame: pd.DataFrame
    components: pd.DataFrame
    valid_raw_indices: np.ndarray
    valid_exante_labels: np.ndarray
    valid_postex_labels: np.ndarray
    valid_labels: np.ndarray
    valid_label_strings: np.ndarray


@dataclass(slots=True)
class MarketDay:
    split: str
    date: str
    frame: pd.DataFrame
    components: pd.DataFrame
    valid_raw_indices: np.ndarray
    valid_exante_labels: np.ndarray
    valid_postex_labels: np.ndarray
    valid_labels: np.ndarray
    sample_label_positions: np.ndarray
    sample_true_labels: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze whether ex-ante smoothing/adaptive labels line up with post-ex realized "
            "thresholds and realized PnL."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--thresholds", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--symbol", type=str, default="INTC")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument(
        "--max-rows-per-split",
        type=int,
        default=None,
        help="Optional smoke-test cap after split/date alignment.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plot generation.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return payload


def resolve_config_path(run_dir: Path, config_path: Path | None) -> Path:
    if config_path is not None:
        return config_path.resolve()
    path = run_dir / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Could not find config.yaml in {run_dir}")
    return path.resolve()


def split_dates(config: dict[str, Any], fold_id: str | None) -> dict[str, list[str]]:
    folds = config.get("folds") or []
    selected: dict[str, Any] | None = None
    if isinstance(folds, list):
        for fold in folds:
            if not isinstance(fold, dict):
                continue
            if fold_id is not None and str(fold.get("id")) == fold_id:
                selected = fold
                break
        if selected is None and folds and isinstance(folds[0], dict):
            selected = folds[0]

    source = selected if selected is not None else config.get("dataset_splits", {})
    if not isinstance(source, dict):
        raise ValueError("Could not resolve validation/test dates from config.")
    return {
        "validation": [str(value) for value in source.get("validation_dates", [])],
        "test": [str(value) for value in source.get("test_dates", [])],
    }


def find_probability_file(run_dir: Path, split: str) -> Path:
    prob_dir = run_dir / "probabilities"
    candidates = sorted(prob_dir.glob(f"{split}*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No {split} probability CSV found in {prob_dir}")
    best = [path for path in candidates if "best_epoch" in path.name]
    return best[-1] if best else candidates[-1]


def lobster_pair(raw_dir: Path, symbol: str, date: str) -> tuple[Path, Path]:
    messages = sorted(raw_dir.glob(f"{symbol}_{date}_*_message_*.csv"))
    candidates: list[tuple[int, int, int, Path, Path]] = []
    for message_path in messages:
        orderbook_path = Path(str(message_path).replace("_message_", "_orderbook_"))
        if not orderbook_path.exists():
            continue
        parts = message_path.name.split("_")
        start = int(parts[2]) if len(parts) > 3 and parts[2].isdigit() else 0
        end = int(parts[3]) if len(parts) > 4 and parts[3].isdigit() else 0
        candidates.append((-(end - start), start, end, message_path, orderbook_path))
    if not candidates:
        raise FileNotFoundError(f"No LOBSTER message/orderbook pair found for {symbol} {date} in {raw_dir}")
    candidates.sort()
    _, _, _, message_path, orderbook_path = candidates[0]
    return message_path, orderbook_path


def session_bounds(config: dict[str, Any]) -> tuple[float, float]:
    temporal = ((config.get("preprocessing") or {}).get("temporal_features") or {})
    open_seconds = float(temporal.get("market_open_seconds", 34200.0))
    close_seconds = float(temporal.get("market_close_seconds", 57600.0))
    start_offset = float(temporal.get("start_offset_minutes", 0.0)) * 60.0
    end_offset = float(temporal.get("end_offset_minutes", 0.0)) * 60.0
    return open_seconds + start_offset, close_seconds - end_offset


def read_market_frame(raw_dir: Path, symbol: str, date: str, config: dict[str, Any]) -> pd.DataFrame:
    message_path, orderbook_path = lobster_pair(raw_dir, symbol, date)
    start_time, end_time = session_bounds(config)

    times = pd.read_csv(message_path, header=None, usecols=[0], names=["time"], dtype=np.float64)
    book = pd.read_csv(
        orderbook_path,
        header=None,
        usecols=[0, 2],
        names=["ask_price_1", "bid_price_1"],
        dtype=np.float64,
    )
    frame = pd.concat([times, book], axis=1)
    frame = frame.loc[
        frame["time"].between(start_time, end_time, inclusive="both")
        & np.isfinite(frame["ask_price_1"])
        & np.isfinite(frame["bid_price_1"])
        & (frame["ask_price_1"] < 1_000_000_000)
        & (frame["bid_price_1"] > 0)
    ].copy()
    frame = frame.sort_values("time", kind="mergesort", ignore_index=True)
    frame["mid_price"] = (frame["ask_price_1"] + frame["bid_price_1"]) / 2.0
    frame["spread"] = frame["ask_price_1"] - frame["bid_price_1"]
    return frame


def future_rolling_std(values: pd.Series, window: int, min_periods: int) -> pd.Series:
    return values.iloc[::-1].rolling(window=window, min_periods=min_periods).std(ddof=0).iloc[::-1]


def configured_label_timing(config: dict[str, Any]) -> str:
    smoothing = (((config.get("preprocessing") or {}).get("labels") or {}).get("smoothing") or {})
    adaptive = smoothing.get("adaptive_threshold") or {}
    timing = str(adaptive.get("label_timing", "ex_ante")).strip().lower()
    if timing not in {"ex_ante", "ex_post"}:
        raise ValueError("adaptive_threshold.label_timing must be 'ex_ante' or 'ex_post'.")
    return timing


def label_components(frame: pd.DataFrame, config: dict[str, Any]) -> LabelComponents:
    labels_config = ((config.get("preprocessing") or {}).get("labels") or {})
    smoothing = labels_config.get("smoothing") or {}
    if str(labels_config.get("strategy", "smoothing")).lower() != "smoothing":
        raise ValueError("This analysis supports preprocessing.labels.strategy=smoothing only.")
    if str(smoothing.get("method", "C")).upper() != "C":
        raise ValueError("This analysis supports smoothing method C only.")

    k = int(smoothing.get("k", 0))
    h = int(smoothing.get("h", 100))
    if k < 0 or h <= 0 or k >= h:
        raise ValueError("Smoothing method C requires k >= 0, h > 0, and k < h.")

    mid = frame["mid_price"].astype(float)
    spread = frame["spread"].astype(float)
    w_minus = mid.rolling(window=k + 1).mean()
    w_plus = mid.rolling(window=k + 1).mean().shift(-h)
    forward_return = (w_plus - w_minus) / w_minus

    adaptive = smoothing.get("adaptive_threshold") or {}
    if adaptive.get("enabled", False):
        exit_window = int(adaptive.get("exit_spread_window", 100))
        exit_min_periods = min(exit_window, max(10, exit_window // 10))
        estimated_exit_spread = spread.rolling(window=exit_window, min_periods=exit_min_periods).median()
        fees_bps = float(adaptive.get("round_trip_fees_bps", 0.0))
        fee_price = mid * fees_bps / 10000.0
        fee_return = fee_price / w_minus
        exante_cost_floor = (((spread + estimated_exit_spread) / 2.0) + fee_price) / w_minus

        volatility_lambda = float(adaptive.get("volatility_lambda", 0.0))
        if volatility_lambda == 0.0:
            past_vol_threshold = pd.Series(0.0, index=frame.index)
        else:
            volatility_window = int(adaptive.get("volatility_window", 256))
            volatility_min_periods = min(volatility_window, max(32, volatility_window // 10))
            past_h_returns = w_minus.pct_change(periods=h)
            past_sigma = past_h_returns.rolling(
                window=volatility_window,
                min_periods=volatility_min_periods,
            ).std(ddof=0)
            past_vol_threshold = volatility_lambda * past_sigma
        threshold_exante = np.maximum(
            exante_cost_floor.to_numpy(dtype=np.float64),
            past_vol_threshold.to_numpy(dtype=np.float64),
        )
    else:
        threshold_value = smoothing.get("threshold")
        if threshold_value is None or isinstance(threshold_value, str):
            raise ValueError("A numeric smoothing threshold is required when adaptive_threshold.enabled=false.")
        estimated_exit_spread = pd.Series(np.nan, index=frame.index)
        fee_price = pd.Series(0.0, index=frame.index)
        fee_return = pd.Series(0.0, index=frame.index)
        exante_cost_floor = pd.Series(float(threshold_value), index=frame.index)
        past_vol_threshold = pd.Series(0.0, index=frame.index)
        threshold_exante = np.full(len(frame), float(threshold_value), dtype=np.float64)

    realized_exit_spread = spread.shift(-h)
    realized_cost_floor = (((spread + realized_exit_spread) / 2.0) + fee_price) / w_minus
    one_step_returns = mid.pct_change().shift(-1)
    realized_vol_min_periods = min(h, max(2, h // 10))
    future_step_sigma = future_rolling_std(one_step_returns, window=h, min_periods=realized_vol_min_periods)
    volatility_lambda = float(adaptive.get("volatility_lambda", 0.0)) if adaptive.get("enabled", False) else 0.0
    realized_vol_threshold = volatility_lambda * future_step_sigma * math.sqrt(float(h))
    threshold_postex = np.maximum(
        realized_cost_floor.to_numpy(dtype=np.float64),
        realized_vol_threshold.to_numpy(dtype=np.float64),
    )

    pct = forward_return.to_numpy(dtype=np.float64)
    valid_exante = np.isfinite(pct) & np.isfinite(threshold_exante)
    valid_postex = np.isfinite(pct) & np.isfinite(threshold_postex)
    labels_exante = np.full(len(frame), CLASS_TO_ID["neutral"], dtype=np.int8)
    labels_exante[pct > threshold_exante] = CLASS_TO_ID["up"]
    labels_exante[pct < -threshold_exante] = CLASS_TO_ID["down"]
    labels_postex = np.full(len(frame), CLASS_TO_ID["neutral"], dtype=np.int8)
    labels_postex[pct > threshold_postex] = CLASS_TO_ID["up"]
    labels_postex[pct < -threshold_postex] = CLASS_TO_ID["down"]

    components = pd.DataFrame(
        {
            "w_minus": w_minus,
            "w_plus": w_plus,
            "forward_return": forward_return,
            "entry_spread": spread,
            "estimated_exit_spread": estimated_exit_spread,
            "realized_exit_spread": realized_exit_spread,
            "fee_price": fee_price,
            "fee_return": fee_return,
            "exante_cost_floor": exante_cost_floor,
            "past_vol_threshold": past_vol_threshold,
            "threshold_exante": threshold_exante,
            "realized_cost_floor": realized_cost_floor,
            "realized_future_step_sigma": future_step_sigma,
            "realized_vol_threshold": realized_vol_threshold,
            "threshold_postex": threshold_postex,
            "label_exante_id": labels_exante,
            "label_postex_id": labels_postex,
            "valid_exante": valid_exante,
            "valid_postex": valid_postex,
        },
        index=frame.index,
    )

    feature_valid = np.isfinite(threshold_exante)
    timing = configured_label_timing(config)
    configured_valid = valid_exante & feature_valid
    configured_labels = labels_exante
    if timing == "ex_post":
        configured_valid = configured_valid & valid_postex
        configured_labels = labels_postex

    valid_raw_indices = np.flatnonzero(configured_valid).astype(np.int64)
    valid_exante_labels = labels_exante[valid_raw_indices]
    valid_postex_labels = labels_postex[valid_raw_indices]
    valid_labels = configured_labels[valid_raw_indices]
    valid_label_strings = np.asarray([ID_TO_CLASS[int(value)] for value in valid_labels], dtype=object)
    return LabelComponents(
        frame=frame,
        components=components,
        valid_raw_indices=valid_raw_indices,
        valid_exante_labels=valid_exante_labels,
        valid_postex_labels=valid_postex_labels,
        valid_labels=valid_labels,
        valid_label_strings=valid_label_strings,
    )


def supervised_label_start(config: dict[str, Any]) -> int:
    preprocessing = config.get("preprocessing") or {}
    data = config.get("data") or {}
    training = config.get("training") or {}
    supervision = training.get("sequence_supervision") or {}
    snapshot_window = int(preprocessing.get("snapshot_window", 100))
    sequence_window = int(data.get("sequence_window", 1))
    mode = str(supervision.get("mode", "last_window")).lower()
    if mode == "token_chunk":
        loss_warmup = int(supervision.get("loss_warmup_tokens") or 0)
        return snapshot_window - 1 + loss_warmup
    return snapshot_window + sequence_window - 2


def build_market_day(raw_dir: Path, symbol: str, split: str, date: str, config: dict[str, Any]) -> MarketDay:
    frame = read_market_frame(raw_dir, symbol, date, config)
    labeled = label_components(frame, config)
    first_label_position = supervised_label_start(config)
    if len(labeled.valid_raw_indices) <= first_label_position:
        raise ValueError(f"{date}: not enough labeled rows for first supervised position {first_label_position}.")
    sample_label_positions = np.arange(first_label_position, len(labeled.valid_raw_indices), dtype=np.int64)
    sample_true_labels = labeled.valid_labels[sample_label_positions]
    return MarketDay(
        split=split,
        date=date,
        frame=frame,
        components=labeled.components,
        valid_raw_indices=labeled.valid_raw_indices,
        valid_exante_labels=labeled.valid_exante_labels,
        valid_postex_labels=labeled.valid_postex_labels,
        valid_labels=labeled.valid_labels,
        sample_label_positions=sample_label_positions,
        sample_true_labels=sample_true_labels,
    )


def label_strings(labels: np.ndarray) -> np.ndarray:
    return np.asarray([ID_TO_CLASS[int(value)] for value in labels], dtype=object)


def infer_day_counts(probabilities: pd.DataFrame, market_days: list[MarketDay]) -> tuple[list[int], dict[str, Any]]:
    reconstructed = [len(day.sample_label_positions) for day in market_days]
    total_reconstructed = int(sum(reconstructed))
    total_probabilities = int(len(probabilities))
    note: dict[str, Any] = {
        "probability_rows": total_probabilities,
        "reconstructed_rows": total_reconstructed,
        "reconstructed_day_counts": dict(zip([day.date for day in market_days], reconstructed)),
        "mode": "exact",
    }
    if total_probabilities == total_reconstructed:
        return reconstructed, note

    if len(market_days) == 2 and "true_label" in probabilities.columns:
        true_labels = probabilities["true_label"].astype(str).to_numpy(dtype=object)
        first = label_strings(market_days[0].sample_true_labels)
        second = label_strings(market_days[1].sample_true_labels)
        base = len(first)
        radius = min(5000, max(abs(total_reconstructed - total_probabilities) * 4, 1000))
        lower = max(0, base - radius)
        upper = min(base + radius, total_probabilities, len(first))
        best_score = -1.0
        best_boundary = min(base, total_probabilities)
        for boundary in range(lower, upper + 1):
            remaining = total_probabilities - boundary
            if remaining < 0 or remaining > len(second):
                continue
            before_window = min(5000, boundary, len(first))
            after_window = min(5000, remaining, len(second))
            score = 0.0
            parts = 0
            if before_window:
                score += float(
                    np.mean(
                        true_labels[boundary - before_window : boundary]
                        == first[boundary - before_window : boundary]
                    )
                )
                parts += 1
            if after_window:
                score += float(np.mean(true_labels[boundary : boundary + after_window] == second[:after_window]))
                parts += 1
            if parts:
                score /= parts
            if score > best_score:
                best_score = score
                best_boundary = boundary
        counts = [int(best_boundary), int(total_probabilities - best_boundary)]
        note.update(
            {
                "mode": "true_label_boundary_fit",
                "adjusted_day_counts": dict(zip([day.date for day in market_days], counts)),
                "boundary_match_score": best_score,
            }
        )
        return counts, note

    counts = []
    remaining = total_probabilities
    for reconstructed_count in reconstructed:
        count = min(reconstructed_count, remaining)
        counts.append(int(count))
        remaining -= count
    note.update(
        {
            "mode": "sequential_truncate",
            "adjusted_day_counts": dict(zip([day.date for day in market_days], counts)),
        }
    )
    return counts, note


def build_split_samples(
    probabilities: pd.DataFrame,
    market_days: list[MarketDay],
    config: dict[str, Any],
    *,
    split: str,
    max_rows: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    smoothing = (((config.get("preprocessing") or {}).get("labels") or {}).get("smoothing") or {})
    horizon = int(smoothing.get("h", 100))
    rows: list[pd.DataFrame] = []
    counts, note = infer_day_counts(probabilities, market_days)
    global_offset = 0

    for day, count in zip(market_days, counts):
        if max_rows is not None:
            remaining_cap = max_rows - global_offset
            if remaining_cap <= 0:
                break
            count = min(count, remaining_cap)

        label_positions = day.sample_label_positions[:count]
        compact_raw_idx = day.valid_raw_indices[label_positions]
        exit_raw_idx = compact_raw_idx + horizon
        valid_exit = exit_raw_idx < len(day.frame)
        if not bool(np.all(valid_exit)):
            note.setdefault("invalid_exit_rows_by_day", {})[day.date] = int(np.sum(~valid_exit))
            label_positions = label_positions[valid_exit]
            compact_raw_idx = compact_raw_idx[valid_exit]
            exit_raw_idx = exit_raw_idx[valid_exit]
            count = len(compact_raw_idx)

        exante_labels = day.valid_exante_labels[label_positions]
        postex_labels = day.valid_postex_labels[label_positions]
        entry = day.frame.iloc[compact_raw_idx]
        exit_ = day.frame.iloc[exit_raw_idx]
        comp = day.components.iloc[compact_raw_idx]
        date = pd.to_datetime(day.date)
        entry_datetime = date + pd.to_timedelta(entry["time"].to_numpy(dtype=np.float64), unit="s")
        exit_datetime = date + pd.to_timedelta(exit_["time"].to_numpy(dtype=np.float64), unit="s")

        rows.append(
            pd.DataFrame(
                {
                    "split": split,
                    "global_sample_index": np.arange(global_offset, global_offset + count, dtype=np.int64),
                    "date": day.date,
                    "local_sample_index": np.arange(count, dtype=np.int64),
                    "label_position": label_positions,
                    "entry_raw_index": compact_raw_idx,
                    "exit_raw_index": exit_raw_idx,
                    "entry_time": entry["time"].to_numpy(dtype=np.float64),
                    "exit_time": exit_["time"].to_numpy(dtype=np.float64),
                    "entry_datetime": entry_datetime,
                    "exit_datetime": exit_datetime,
                    "entry_bid": entry["bid_price_1"].to_numpy(dtype=np.float64),
                    "entry_ask": entry["ask_price_1"].to_numpy(dtype=np.float64),
                    "entry_mid": entry["mid_price"].to_numpy(dtype=np.float64),
                    "exit_bid": exit_["bid_price_1"].to_numpy(dtype=np.float64),
                    "exit_ask": exit_["ask_price_1"].to_numpy(dtype=np.float64),
                    "exit_mid": exit_["mid_price"].to_numpy(dtype=np.float64),
                    "entry_spread": comp["entry_spread"].to_numpy(dtype=np.float64),
                    "estimated_exit_spread": comp["estimated_exit_spread"].to_numpy(dtype=np.float64),
                    "realized_exit_spread": comp["realized_exit_spread"].to_numpy(dtype=np.float64),
                    "fee_price": comp["fee_price"].to_numpy(dtype=np.float64),
                    "fee_return": comp["fee_return"].to_numpy(dtype=np.float64),
                    "exante_cost_floor": comp["exante_cost_floor"].to_numpy(dtype=np.float64),
                    "past_vol_threshold": comp["past_vol_threshold"].to_numpy(dtype=np.float64),
                    "threshold_exante": comp["threshold_exante"].to_numpy(dtype=np.float64),
                    "realized_cost_floor": comp["realized_cost_floor"].to_numpy(dtype=np.float64),
                    "realized_future_step_sigma": comp["realized_future_step_sigma"].to_numpy(dtype=np.float64),
                    "realized_vol_threshold": comp["realized_vol_threshold"].to_numpy(dtype=np.float64),
                    "threshold_postex": comp["threshold_postex"].to_numpy(dtype=np.float64),
                    "forward_return": comp["forward_return"].to_numpy(dtype=np.float64),
                    "exante_label": label_strings(exante_labels),
                    "postex_label": label_strings(postex_labels),
                    "postex_valid": comp["valid_postex"].to_numpy(dtype=bool),
                }
            )
        )
        global_offset += count

    market = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    probs = probabilities.iloc[: len(market)].reset_index(drop=True).copy()
    market = market.reset_index(drop=True)
    if "sample_index" in probs.columns:
        market["probability_sample_index"] = probs["sample_index"].to_numpy(dtype=np.int64)
    for column in ("true_label", "pred_label", "argmax_pred_label", "p_down", "p_neutral", "p_up"):
        if column in probs.columns:
            market[column if column.startswith("p_") else f"model_{column}"] = probs[column].to_numpy()

    if "model_true_label" in market.columns and len(market):
        configured_label_column = "postex_label" if configured_label_timing(config) == "ex_post" else "exante_label"
        note["configured_label_timing"] = configured_label_timing(config)
        note["configured_label_column"] = configured_label_column
        note["true_label_match_rate"] = float(
            np.mean(
                market["model_true_label"].astype(str).to_numpy()
                == market[configured_label_column].astype(str).to_numpy()
            )
        )
    note["final_rows"] = int(len(market))
    return market, note


def add_pnl_columns(samples: pd.DataFrame, label_column: str, *, tick_size: float, fees_bps: float) -> pd.DataFrame:
    result = samples.copy()
    labels = result[label_column].astype(str).str.lower()
    pos = labels.map(POSITION_BY_LABEL).fillna(0).to_numpy(dtype=np.int8)
    active = pos != 0
    entry_mid = result["entry_mid"].to_numpy(dtype=np.float64)
    mid_pnl_price = pos.astype(np.float64) * (result["exit_mid"].to_numpy(dtype=np.float64) - entry_mid)
    cross_pnl_price = np.zeros(len(result), dtype=np.float64)

    long_mask = pos > 0
    short_mask = pos < 0
    cross_pnl_price[long_mask] = (
        result.loc[long_mask, "exit_bid"].to_numpy(dtype=np.float64)
        - result.loc[long_mask, "entry_ask"].to_numpy(dtype=np.float64)
    )
    cross_pnl_price[short_mask] = (
        result.loc[short_mask, "entry_bid"].to_numpy(dtype=np.float64)
        - result.loc[short_mask, "exit_ask"].to_numpy(dtype=np.float64)
    )
    fee_price = entry_mid * fees_bps / 10000.0
    fee_price[~active] = 0.0

    prefix = label_column.removesuffix("_label")
    result[f"{prefix}_position"] = pos
    result[f"{prefix}_mid_pnl_ticks"] = mid_pnl_price / tick_size
    result[f"{prefix}_cross_pnl_ticks"] = cross_pnl_price / tick_size
    result[f"{prefix}_net_cross_pnl_ticks"] = (cross_pnl_price - fee_price) / tick_size
    result[f"{prefix}_is_win_net_cross"] = result[f"{prefix}_net_cross_pnl_ticks"] > 0
    return result


def non_overlapping_subset(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    selected_indices: list[int] = []
    for _date, group in trades.sort_values(["date", "entry_raw_index"], kind="mergesort").groupby("date", sort=False):
        active_exit = -1
        for idx, row in group.iterrows():
            entry_idx = int(row["entry_raw_index"])
            if entry_idx <= active_exit:
                continue
            selected_indices.append(idx)
            active_exit = int(row["exit_raw_index"])
    return trades.loc[selected_indices].sort_values("exit_datetime", kind="mergesort", ignore_index=True)


def pnl_summaries(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    day_rows: list[dict[str, Any]] = []
    for label_column in ("exante_label", "postex_label"):
        prefix = label_column.removesuffix("_label")
        position_col = f"{prefix}_position"
        pnl_cols = [
            f"{prefix}_mid_pnl_ticks",
            f"{prefix}_cross_pnl_ticks",
            f"{prefix}_net_cross_pnl_ticks",
        ]
        for split, split_df in samples.groupby("split", sort=False):
            active = split_df.loc[split_df[position_col] != 0].copy()
            variants = {
                "independent": active,
                "non_overlap": non_overlapping_subset(active),
            }
            for variant_name, trades in variants.items():
                row = {
                    "strategy": prefix,
                    "split": split,
                    "execution": variant_name,
                    "n_samples": int(len(split_df)),
                    "n_trades": int(len(trades)),
                    "trade_rate": float(len(trades) / max(len(split_df), 1)),
                }
                for col in pnl_cols:
                    short_col = col.replace(f"{prefix}_", "")
                    row[f"total_{short_col}"] = float(trades[col].sum()) if len(trades) else 0.0
                    row[f"mean_{short_col}"] = float(trades[col].mean()) if len(trades) else 0.0
                win_col = f"{prefix}_is_win_net_cross"
                row["win_rate_net_cross"] = float(trades[win_col].mean()) if len(trades) else 0.0
                rows.append(row)

                for date, day in trades.groupby("date", sort=False):
                    day_rows.append(
                        {
                            "strategy": prefix,
                            "split": split,
                            "date": date,
                            "execution": variant_name,
                            "n_trades": int(len(day)),
                            "total_net_cross_pnl_ticks": float(day[f"{prefix}_net_cross_pnl_ticks"].sum()),
                            "mean_net_cross_pnl_ticks": float(day[f"{prefix}_net_cross_pnl_ticks"].mean()),
                            "win_rate_net_cross": float(day[f"{prefix}_is_win_net_cross"].mean()),
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(day_rows)


def component_calibration(samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pairs = [
        ("estimated_exit_spread", "realized_exit_spread"),
        ("past_vol_threshold", "realized_vol_threshold"),
        ("threshold_exante", "threshold_postex"),
        ("exante_cost_floor", "realized_cost_floor"),
    ]
    for split, group in samples.groupby("split", sort=False):
        row: dict[str, Any] = {"split": split, "n_samples": int(len(group))}
        for estimate, realized in pairs:
            valid = np.isfinite(group[estimate].to_numpy(dtype=np.float64)) & np.isfinite(
                group[realized].to_numpy(dtype=np.float64)
            )
            est = group.loc[valid, estimate].to_numpy(dtype=np.float64)
            real = group.loc[valid, realized].to_numpy(dtype=np.float64)
            prefix = f"{estimate}_vs_{realized}"
            row[f"{prefix}_n"] = int(len(est))
            row[f"{prefix}_estimate_mean"] = float(np.mean(est)) if len(est) else np.nan
            row[f"{prefix}_realized_mean"] = float(np.mean(real)) if len(real) else np.nan
            row[f"{prefix}_mean_error"] = float(np.mean(est - real)) if len(est) else np.nan
            row[f"{prefix}_mae"] = float(np.mean(np.abs(est - real))) if len(est) else np.nan
            if len(est) > 1 and np.std(est) > 0.0 and np.std(real) > 0.0:
                row[f"{prefix}_corr"] = float(np.corrcoef(est, real)[0, 1])
            else:
                row[f"{prefix}_corr"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def label_coherence(samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, group in samples.groupby("split", sort=False):
        ex = group["exante_label"].astype(str)
        post = group["postex_label"].astype(str)
        ex_dir = ex.isin(["down", "up"])
        post_dir = post.isin(["down", "up"])
        both_dir = ex_dir & post_dir
        row = {
            "split": split,
            "n_samples": int(len(group)),
            "exante_neutral_rate": float((ex == "neutral").mean()),
            "postex_neutral_rate": float((post == "neutral").mean()),
            "label_agreement_rate": float((ex == post).mean()),
            "label_disagreement_rate": float((ex != post).mean()),
            "directional_accuracy_on_exante_directionals": float((ex[ex_dir] == post[ex_dir]).mean())
            if bool(ex_dir.any())
            else np.nan,
            "directional_accuracy_when_both_directional": float((ex[both_dir] == post[both_dir]).mean())
            if bool(both_dir.any())
            else np.nan,
            "exante_directional_rate": float(ex_dir.mean()),
            "postex_directional_rate": float(post_dir.mean()),
        }
        for label in ("down", "up"):
            pred = ex == label
            true = post == label
            tp = int((pred & true).sum())
            fp = int((pred & ~true).sum())
            fn = int((~pred & true).sum())
            row[f"{label}_precision_exante_vs_postex"] = tp / (tp + fp) if tp + fp else np.nan
            row[f"{label}_recall_exante_vs_postex"] = tp / (tp + fn) if tp + fn else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def confusion_table(samples: pd.DataFrame) -> pd.DataFrame:
    table = pd.crosstab(
        [samples["split"], samples["exante_label"]],
        samples["postex_label"],
        rownames=["split", "exante_label"],
        colnames=["postex_label"],
        dropna=False,
    )
    for label in ("down", "neutral", "up"):
        if label not in table.columns:
            table[label] = 0
    return table[["down", "neutral", "up"]].reset_index()


def save_plots(samples: pd.DataFrame, pnl_by_day: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        print(f"Skipping plots because matplotlib could not be imported: {exc}")
        return

    def scatter_sample(df: pd.DataFrame, x: str, y: str, path: Path, title: str) -> None:
        valid = df[[x, y, "split"]].replace([np.inf, -np.inf], np.nan).dropna()
        if valid.empty:
            return
        if len(valid) > 100_000:
            valid = valid.sample(100_000, random_state=42)
        fig, ax = plt.subplots(figsize=(7, 5))
        for split, group in valid.groupby("split", sort=False):
            ax.scatter(group[x], group[y], s=3, alpha=0.18, label=split)
        lo = float(np.nanmin([valid[x].min(), valid[y].min()]))
        hi = float(np.nanmax([valid[x].max(), valid[y].max()]))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1, alpha=0.6)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    scatter_sample(
        samples,
        "estimated_exit_spread",
        "realized_exit_spread",
        output_dir / "spread_calibration.png",
        "Estimated vs realized exit spread",
    )
    scatter_sample(
        samples,
        "past_vol_threshold",
        "realized_vol_threshold",
        output_dir / "vol_calibration.png",
        "Past volatility threshold vs realized volatility threshold",
    )
    scatter_sample(
        samples,
        "threshold_exante",
        "threshold_postex",
        output_dir / "threshold_calibration.png",
        "Ex-ante vs post-ex threshold",
    )

    confusion = confusion_table(samples)
    labels = ["down", "neutral", "up"]
    for split, group in confusion.groupby("split", sort=False):
        matrix = group.set_index("exante_label").reindex(labels).fillna(0.0)[labels].to_numpy(dtype=np.float64)
        row_sums = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)
        fig, ax = plt.subplots(figsize=(5.5, 4.8))
        im = ax.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("postex_label")
        ax.set_ylabel("exante_label")
        ax.set_title(f"Ex-ante vs post-ex labels ({split})")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{normalized[i, j]:.2%}\n{int(matrix[i, j])}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(output_dir / f"label_confusion_heatmap_{split}.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for prefix in ("exante", "postex"):
        col = f"{prefix}_net_cross_pnl_ticks"
        active = samples.loc[samples[f"{prefix}_position"] != 0, col].replace([np.inf, -np.inf], np.nan).dropna()
        if active.empty:
            continue
        ax.hist(active.clip(active.quantile(0.005), active.quantile(0.995)), bins=80, alpha=0.45, label=prefix)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("net_cross_pnl_ticks")
    ax.set_ylabel("trade count")
    ax.set_title("Realized net cross PnL distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "pnl_distributions.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for prefix in ("exante", "postex"):
        active = samples.loc[samples[f"{prefix}_position"] != 0].copy()
        if active.empty:
            continue
        active = active.sort_values("exit_datetime", kind="mergesort")
        active["cum_pnl"] = active[f"{prefix}_net_cross_pnl_ticks"].cumsum()
        if len(active) > 30_000:
            active = active.iloc[:: int(math.ceil(len(active) / 30_000))]
        ax.plot(active["exit_datetime"], active["cum_pnl"], label=prefix, linewidth=1.2)
    ax.set_xlabel("exit_datetime")
    ax.set_ylabel("cumulative net cross pnl ticks")
    ax.set_title("Cumulative realized PnL, independent trades")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "cumulative_pnl_curves.png", dpi=160)
    plt.close(fig)

    if not pnl_by_day.empty:
        pivot = pnl_by_day[pnl_by_day["execution"] == "independent"].pivot_table(
            index="date",
            columns="strategy",
            values="total_net_cross_pnl_ticks",
            aggfunc="sum",
        )
        fig, ax = plt.subplots(figsize=(8, 4.5))
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel("total net cross pnl ticks")
        ax.set_title("Daily realized PnL")
        fig.tight_layout()
        fig.savefig(output_dir / "pnl_by_day.png", dpi=160)
        plt.close(fig)


def write_summary(
    output_dir: Path,
    *,
    run_dir: Path,
    config_path: Path,
    raw_dir: Path,
    threshold_path: Path | None,
    notes: dict[str, dict[str, Any]],
    calibration: pd.DataFrame,
    coherence: pd.DataFrame,
    pnl_by_label: pd.DataFrame,
    sample_path: Path,
) -> None:
    lines: list[str] = [
        "# Ex-Ante / Post-Ex Label Realization Analysis",
        "",
        f"- run_dir: `{run_dir}`",
        f"- config: `{config_path}`",
        f"- raw_lobster: `{raw_dir}`",
        f"- thresholds: `{threshold_path}`" if threshold_path is not None else "- thresholds: not found",
        f"- sample_realization: `{sample_path}`",
        "",
        "## Alignment Checks",
        "",
    ]
    for split, note in notes.items():
        lines.append(f"### {split}")
        for key, value in note.items():
            lines.append(f"- {key}: {value}")
        lines.append("")

    def table_section(title: str, df: pd.DataFrame) -> None:
        lines.append(f"## {title}")
        lines.append("")
        if df.empty:
            lines.append("_No rows._")
        else:
            lines.append(df.to_markdown(index=False, floatfmt=".6g"))
        lines.append("")

    table_section("Component Calibration By Split", calibration)
    table_section("Label Coherence By Split", coherence)
    table_section("PnL By Label", pnl_by_label)
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    config_path = resolve_config_path(run_dir, args.config)
    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_yaml(config_path)
    fold_id = run_dir.name
    dates_by_split = split_dates(config, fold_id)
    threshold_path = (args.thresholds or (run_dir / "directional_thresholds.yaml")).resolve()
    if not threshold_path.exists():
        threshold_path = None

    smoothing = (((config.get("preprocessing") or {}).get("labels") or {}).get("smoothing") or {})
    adaptive = smoothing.get("adaptive_threshold") or {}
    tick_size = float((config.get("data") or {}).get("tick_size", 1.0))
    fees_bps = float(adaptive.get("round_trip_fees_bps", 0.0))

    all_samples: list[pd.DataFrame] = []
    notes: dict[str, dict[str, Any]] = {}
    for split in args.splits:
        prob_path = find_probability_file(run_dir, split)
        probabilities = pd.read_csv(prob_path)
        market_days = [
            build_market_day(raw_dir, args.symbol, split, date, config)
            for date in dates_by_split.get(split, [])
        ]
        samples, note = build_split_samples(
            probabilities,
            market_days,
            config,
            split=split,
            max_rows=args.max_rows_per_split,
        )
        note["probability_file"] = str(prob_path)
        note["dates"] = dates_by_split.get(split, [])
        notes[split] = note
        all_samples.append(samples)

    if not all_samples:
        raise ValueError("No samples were built.")
    samples = pd.concat(all_samples, ignore_index=True)
    samples = add_pnl_columns(samples, "exante_label", tick_size=tick_size, fees_bps=fees_bps)
    samples = add_pnl_columns(samples, "postex_label", tick_size=tick_size, fees_bps=fees_bps)

    calibration = component_calibration(samples)
    coherence = label_coherence(samples)
    confusion = confusion_table(samples)
    pnl_by_label, pnl_by_day = pnl_summaries(samples)

    sample_path = output_dir / "sample_realization.parquet"
    try:
        samples.to_parquet(sample_path, index=False)
    except Exception as exc:
        sample_path = output_dir / "sample_realization.csv.gz"
        print(f"Could not write parquet ({exc}); writing {sample_path.name} instead.")
        samples.to_csv(sample_path, index=False)

    calibration.to_csv(output_dir / "component_calibration_by_split.csv", index=False)
    coherence.to_csv(output_dir / "label_coherence_by_split.csv", index=False)
    confusion.to_csv(output_dir / "exante_vs_postex_confusion.csv", index=False)
    pnl_by_label.to_csv(output_dir / "pnl_by_label.csv", index=False)
    pnl_by_day.to_csv(output_dir / "pnl_by_day.csv", index=False)

    if not args.no_plots:
        save_plots(samples, pnl_by_day, output_dir)

    write_summary(
        output_dir,
        run_dir=run_dir,
        config_path=config_path,
        raw_dir=raw_dir,
        threshold_path=threshold_path,
        notes=notes,
        calibration=calibration,
        coherence=coherence,
        pnl_by_label=pnl_by_label,
        sample_path=sample_path,
    )
    print(f"Wrote label realization analysis to {output_dir}")


if __name__ == "__main__":
    main()
