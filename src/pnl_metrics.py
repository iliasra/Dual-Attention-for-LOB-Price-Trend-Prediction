from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    from configuration import ExperimentConfig
    from horizon import (
        ADAPTIVE_LABEL_FEATURE_COLUMNS,
        calculate_adaptive_method_c_threshold_components,
        smoothing_pct_changes,
    )
except ImportError:  # pragma: no cover
    from .configuration import ExperimentConfig
    from .horizon import (
        ADAPTIVE_LABEL_FEATURE_COLUMNS,
        calculate_adaptive_method_c_threshold_components,
        smoothing_pct_changes,
    )


POSITION_BY_RAW_LABEL = {-1: -1, 0: 0, 1: 1}
SEQUENCE_STEM_PATTERN = re.compile(r"^(?P<symbol>[^_]+)_(?P<date>\d{4}-\d{2}-\d{2})(?:_|$)")

TEST_PNL_METRIC_KEYS = (
    "test_pnl_valid_samples",
    "test_pnl_invalid_count",
    "test_pnl_true_label_match_rate",
    "test_pnl_net_cross_ticks_mean",
    "test_pnl_net_cross_ticks_total",
    "test_pnl_cross_ticks_mean",
    "test_pnl_mid_ticks_mean",
    "test_pnl_trade_rate",
    "test_pnl_n_trades",
    "test_pnl_win_rate_net_cross",
    "test_pnl_net_cross_ticks_mean_non_overlap",
    "test_pnl_net_cross_ticks_total_non_overlap",
    "test_pnl_cross_ticks_mean_non_overlap",
    "test_pnl_mid_ticks_mean_non_overlap",
    "test_pnl_trade_rate_non_overlap",
    "test_pnl_n_trades_non_overlap",
    "test_pnl_win_rate_net_cross_non_overlap",
)


@dataclass(slots=True)
class PnlResult:
    status: str
    metrics: dict[str, float | int]
    summary: dict[str, Any]
    by_day: pd.DataFrame
    samples: pd.DataFrame


@dataclass(slots=True)
class SequenceDaySpec:
    symbol: str
    date: str
    y_path: Path
    y_positions: np.ndarray
    y_labels: np.ndarray


def resolve_raw_data_dir(config: ExperimentConfig, raw_dir: Path | str | None = None) -> Path:
    """Resolve the raw LOBSTER directory from an override or config-relative path."""
    if raw_dir is not None:
        return Path(raw_dir).resolve()
    configured = Path(config.data.raw_data_dir)
    return configured if configured.is_absolute() else (config.path.parent / configured).resolve()


def pnl_horizon(config: ExperimentConfig) -> int:
    labels = config.preprocessing.labels
    if labels.strategy != "smoothing":
        raise ValueError("PnL metrics currently require preprocessing.labels.strategy=smoothing.")
    smoothing = labels.smoothing
    if smoothing.method.upper() != "C":
        raise ValueError("PnL metrics currently require smoothing method C.")
    return int(smoothing.h)


def pnl_fees_bps(config: ExperimentConfig) -> tuple[float, str]:
    adaptive = config.preprocessing.labels.smoothing.adaptive_threshold
    if adaptive is None:
        return 0.0, "default_zero"
    return float(adaptive.round_trip_fees_bps), "adaptive_threshold.round_trip_fees_bps"


def class_id_to_position(config: ExperimentConfig) -> dict[int, int]:
    missing = [raw_label for raw_label in (-1, 0, 1) if raw_label not in config.data.label_mapping]
    if missing:
        raise ValueError(f"PnL metrics require label_mapping entries for raw labels -1, 0, 1; missing {missing}.")
    return {
        int(config.data.label_mapping[raw_label]): POSITION_BY_RAW_LABEL[raw_label]
        for raw_label in (-1, 0, 1)
    }


def parse_sequence_stem(path: Path, *, symbol_override: str | None = None) -> tuple[str, str]:
    stem = path.name.removesuffix("_labels.npy").removesuffix("_features.npy").removesuffix("_times.npy")
    match = SEQUENCE_STEM_PATTERN.match(stem)
    if match is None:
        raise ValueError(f"Cannot infer symbol/date from sequence shard name: {path.name}")
    symbol = symbol_override or match.group("symbol")
    return symbol, match.group("date")


def supervised_y_positions(num_rows: int, config: ExperimentConfig) -> np.ndarray:
    if config.training.sequence_supervision.token_chunk_enabled:
        if num_rows < config.data.sequence_window:
            return np.asarray([], dtype=np.int64)
        start = int(config.training.sequence_supervision.loss_warmup_tokens or 0)
    else:
        if num_rows < config.data.sequence_window:
            return np.asarray([], dtype=np.int64)
        start = int(config.data.sequence_window) - 1
    return np.arange(start, num_rows, dtype=np.int64)


def sequence_day_specs(
    y_paths: list[str | Path],
    config: ExperimentConfig,
    *,
    symbol_override: str | None = None,
) -> list[SequenceDaySpec]:
    specs: list[SequenceDaySpec] = []
    for y_path_like in y_paths:
        y_path = Path(y_path_like)
        symbol, date = parse_sequence_stem(y_path, symbol_override=symbol_override)
        y_labels = np.load(y_path, mmap_mode="r")
        positions = supervised_y_positions(len(y_labels), config)
        specs.append(
            SequenceDaySpec(
                symbol=symbol,
                date=date,
                y_path=y_path,
                y_positions=positions,
                y_labels=np.asarray(y_labels, dtype=np.int64),
            )
        )
    return specs


def y_paths_from_dataset(dataset: Any) -> list[Path]:
    y_paths = getattr(dataset, "y_paths", None)
    if y_paths is None:
        raise ValueError("PnL metrics require an evaluation dataset exposing y_paths.")
    return [Path(path) for path in y_paths]


def _session_bounds(config: ExperimentConfig) -> tuple[float, float]:
    temporal = config.preprocessing.temporal_features
    return (
        float(temporal.market_open_seconds) + float(temporal.start_offset_minutes) * 60.0,
        float(temporal.market_close_seconds) - float(temporal.end_offset_minutes) * 60.0,
    )


def _lobster_segment_sort_key(path: Path) -> tuple[int, int, str]:
    parts = path.name.split("_")
    try:
        start = int(parts[2])
        end = int(parts[3])
    except (IndexError, ValueError):
        start = 0
        end = 0
    return start, end, path.name


def read_lobster_l1_frame(raw_dir: Path, symbol: str, date: str, config: ExperimentConfig) -> pd.DataFrame:
    """Read and trim raw LOBSTER L1 bid/ask rows for one symbol/date."""
    message_paths = sorted(
        raw_dir.glob(f"{symbol}_{date}_*_message_*.csv"),
        key=_lobster_segment_sort_key,
    )
    if not message_paths:
        raise FileNotFoundError(f"No LOBSTER message file found for {symbol} {date} in {raw_dir}.")

    frames: list[pd.DataFrame] = []
    for message_path in message_paths:
        orderbook_path = Path(str(message_path).replace("_message_", "_orderbook_"))
        if not orderbook_path.exists():
            continue
        times = pd.read_csv(message_path, header=None, usecols=[0], names=["time"], dtype=np.float64)
        book = pd.read_csv(
            orderbook_path,
            header=None,
            usecols=[0, 2],
            names=["ask_price_1", "bid_price_1"],
            dtype=np.float64,
        )
        frames.append(pd.concat([times, book], axis=1))
    if not frames:
        raise FileNotFoundError(f"No matched LOBSTER message/orderbook pair found for {symbol} {date}.")

    frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    frame = frame.drop_duplicates(subset=["time", "ask_price_1", "bid_price_1"], ignore_index=True)
    start_time, end_time = _session_bounds(config)
    frame = frame.loc[
        frame["time"].between(start_time, end_time, inclusive="both")
        & np.isfinite(frame["ask_price_1"])
        & np.isfinite(frame["bid_price_1"])
        & (frame["ask_price_1"] < 1_000_000_000)
        & (frame["bid_price_1"] > 0)
        & (frame["ask_price_1"] >= frame["bid_price_1"])
    ].copy()
    frame = frame.sort_values("time", kind="mergesort", ignore_index=True)
    frame["mid_price"] = (frame["ask_price_1"] + frame["bid_price_1"]) / 2.0
    return frame


def _valid_label_indices(frame: pd.DataFrame, config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    smoothing = config.preprocessing.labels.smoothing
    midprices = frame["mid_price"].astype(float)
    pct_changes = smoothing_pct_changes(midprices, smoothing)

    threshold = smoothing.threshold
    feature_valid_mask: pd.Series | None = None
    if smoothing.method.upper() == "C" and smoothing.adaptive_threshold is not None and smoothing.adaptive_threshold.enabled:
        components = calculate_adaptive_method_c_threshold_components(
            frame,
            midprices,
            k=smoothing.k,
            h=smoothing.h,
            bid_col=smoothing.bid_column,
            ask_col=smoothing.ask_column,
            config=smoothing.adaptive_threshold,
        )
        threshold = components["threshold"]
        if smoothing.adaptive_threshold.include_exante_features:
            adaptive_features = components.loc[:, list(ADAPTIVE_LABEL_FEATURE_COLUMNS)]
            feature_valid_mask = adaptive_features.notna().all(axis=1) & np.isfinite(adaptive_features).all(axis=1)
    if threshold is None or isinstance(threshold, str):
        raise ValueError("PnL label alignment requires a numeric or adaptive smoothing threshold.")

    valid_mask = pct_changes.notna() & np.isfinite(pct_changes)
    if feature_valid_mask is not None:
        valid_mask = valid_mask & feature_valid_mask
    if isinstance(threshold, pd.Series):
        valid_mask = valid_mask & threshold.notna() & np.isfinite(threshold)
        threshold_values: float | np.ndarray = threshold.to_numpy(dtype=np.float64)
    else:
        threshold_values = float(threshold)

    pct = pct_changes.to_numpy(dtype=np.float64)
    raw_labels = np.full(len(frame), 0, dtype=np.int64)
    raw_labels[pct > threshold_values] = 1
    raw_labels[pct < -threshold_values] = -1
    valid_raw_indices = np.flatnonzero(valid_mask.to_numpy(dtype=bool)).astype(np.int64)
    mapped = np.asarray([int(config.data.label_mapping[int(raw_labels[idx])]) for idx in valid_raw_indices])
    return valid_raw_indices, mapped


def build_pnl_samples(
    *,
    config: ExperimentConfig,
    outputs: dict[str, Any],
    y_paths: list[str | Path],
    raw_dir: Path,
    split: str,
    symbol_override: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if config.preprocessing.sample_clock.enabled:
        raise ValueError("PnL metrics on raw LOBSTER rows are not supported when sample_clock is enabled.")

    predictions = np.asarray(outputs.get("predictions", []), dtype=np.int64).reshape(-1)
    targets = np.asarray(outputs.get("targets", []), dtype=np.int64).reshape(-1)
    if predictions.shape[0] != targets.shape[0]:
        raise ValueError("PnL metrics require prediction and target arrays with matching length.")

    horizon = pnl_horizon(config)
    specs = sequence_day_specs(y_paths, config, symbol_override=symbol_override)
    expected_count = int(sum(len(spec.y_positions) for spec in specs))
    if expected_count != int(predictions.shape[0]):
        raise ValueError(
            f"PnL alignment expected {expected_count} predictions from sequence shards, "
            f"but received {predictions.shape[0]}."
        )

    rows: list[pd.DataFrame] = []
    offset = 0
    invalid_count = 0
    label_matches: list[np.ndarray] = []

    for spec in specs:
        count = len(spec.y_positions)
        if count == 0:
            continue
        day_predictions = predictions[offset : offset + count]
        day_targets = targets[offset : offset + count]
        expected_targets = spec.y_labels[spec.y_positions]
        label_matches.append(day_targets == expected_targets)

        frame = read_lobster_l1_frame(raw_dir, spec.symbol, spec.date, config)
        valid_raw_indices, _mapped_labels = _valid_label_indices(frame, config)
        valid_label_positions = int(config.preprocessing.snapshot_window) - 1 + spec.y_positions
        if len(valid_raw_indices) <= int(valid_label_positions.max()):
            raise ValueError(
                f"{spec.date}: reconstructed label rows do not cover sequence labels "
                f"(need position {int(valid_label_positions.max())}, have {len(valid_raw_indices)})."
            )
        entry_raw_idx = valid_raw_indices[valid_label_positions]
        exit_raw_idx = entry_raw_idx + horizon
        valid_exit = exit_raw_idx < len(frame)
        invalid_count += int(np.sum(~valid_exit))
        if not bool(np.any(valid_exit)):
            offset += count
            continue

        entry_raw_idx = entry_raw_idx[valid_exit]
        exit_raw_idx = exit_raw_idx[valid_exit]
        kept_predictions = day_predictions[valid_exit]
        entry = frame.iloc[entry_raw_idx]
        exit_ = frame.iloc[exit_raw_idx]
        date = pd.to_datetime(spec.date)
        rows.append(
            pd.DataFrame(
                {
                    "split": split,
                    "date": spec.date,
                    "global_sample_index": np.arange(offset, offset + count, dtype=np.int64)[valid_exit],
                    "local_sample_index": spec.y_positions[valid_exit],
                    "entry_raw_index": entry_raw_idx,
                    "exit_raw_index": exit_raw_idx,
                    "entry_time": entry["time"].to_numpy(dtype=np.float64),
                    "exit_time": exit_["time"].to_numpy(dtype=np.float64),
                    "entry_datetime": date + pd.to_timedelta(entry["time"].to_numpy(dtype=np.float64), unit="s"),
                    "exit_datetime": date + pd.to_timedelta(exit_["time"].to_numpy(dtype=np.float64), unit="s"),
                    "entry_bid": entry["bid_price_1"].to_numpy(dtype=np.float64),
                    "entry_ask": entry["ask_price_1"].to_numpy(dtype=np.float64),
                    "entry_mid": entry["mid_price"].to_numpy(dtype=np.float64),
                    "exit_bid": exit_["bid_price_1"].to_numpy(dtype=np.float64),
                    "exit_ask": exit_["ask_price_1"].to_numpy(dtype=np.float64),
                    "exit_mid": exit_["mid_price"].to_numpy(dtype=np.float64),
                    "prediction": kept_predictions,
                }
            )
        )
        offset += count

    samples = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    match_rate = float(np.mean(np.concatenate(label_matches))) if label_matches else 1.0
    metadata = {
        "expected_predictions": expected_count,
        "received_predictions": int(predictions.shape[0]),
        "invalid_exit_count": invalid_count,
        "true_label_match_rate": match_rate,
        "dates": [spec.date for spec in specs],
    }
    return samples, metadata


def add_pnl_columns(
    samples: pd.DataFrame,
    *,
    config: ExperimentConfig,
    fees_bps: float,
) -> pd.DataFrame:
    result = samples.copy()
    if result.empty:
        for column in (
            "position",
            "mid_pnl_ticks",
            "cross_pnl_ticks",
            "net_cross_pnl_ticks",
            "is_win_net_cross",
        ):
            result[column] = []
        return result

    id_to_position = class_id_to_position(config)
    positions = np.asarray([id_to_position.get(int(prediction), 0) for prediction in result["prediction"]], dtype=np.int8)
    active = positions != 0
    entry_mid = result["entry_mid"].to_numpy(dtype=np.float64)
    mid_pnl_price = positions.astype(np.float64) * (result["exit_mid"].to_numpy(dtype=np.float64) - entry_mid)
    cross_pnl_price = np.zeros(len(result), dtype=np.float64)

    long_mask = positions > 0
    short_mask = positions < 0
    cross_pnl_price[long_mask] = (
        result.loc[long_mask, "exit_bid"].to_numpy(dtype=np.float64)
        - result.loc[long_mask, "entry_ask"].to_numpy(dtype=np.float64)
    )
    cross_pnl_price[short_mask] = (
        result.loc[short_mask, "entry_bid"].to_numpy(dtype=np.float64)
        - result.loc[short_mask, "exit_ask"].to_numpy(dtype=np.float64)
    )
    fee_price = entry_mid * float(fees_bps) / 10000.0
    fee_price[~active] = 0.0

    tick_size = float(config.data.tick_size)
    if tick_size <= 0.0:
        raise ValueError("data.tick_size must be > 0 for PnL metrics.")
    result["position"] = positions
    result["mid_pnl_ticks"] = mid_pnl_price / tick_size
    result["cross_pnl_ticks"] = cross_pnl_price / tick_size
    result["net_cross_pnl_ticks"] = (cross_pnl_price - fee_price) / tick_size
    result["is_win_net_cross"] = result["net_cross_pnl_ticks"] > 0
    return result


def non_overlapping_subset(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    selected: list[int] = []
    for _date, group in trades.sort_values(["date", "entry_raw_index"], kind="mergesort").groupby("date", sort=False):
        active_exit = -1
        for idx, row in group.iterrows():
            entry_idx = int(row["entry_raw_index"])
            if entry_idx <= active_exit:
                continue
            selected.append(idx)
            active_exit = int(row["exit_raw_index"])
    return trades.loc[selected].sort_values(["date", "entry_raw_index"], kind="mergesort")


def _execution_summary(samples: pd.DataFrame, *, execution: str) -> dict[str, float | int | str]:
    active = samples.loc[samples["position"] != 0].copy() if "position" in samples else samples.copy()
    trades = non_overlapping_subset(active) if execution == "non_overlap" else active
    n_samples = int(len(samples))
    n_trades = int(len(trades))

    def mean_or_zero(column: str) -> float:
        return float(trades[column].mean()) if n_trades else 0.0

    def sum_or_zero(column: str) -> float:
        return float(trades[column].sum()) if n_trades else 0.0

    return {
        "execution": execution,
        "n_samples": n_samples,
        "n_trades": n_trades,
        "trade_rate": float(n_trades / max(n_samples, 1)),
        "net_cross_ticks_mean": mean_or_zero("net_cross_pnl_ticks"),
        "net_cross_ticks_total": sum_or_zero("net_cross_pnl_ticks"),
        "cross_ticks_mean": mean_or_zero("cross_pnl_ticks"),
        "mid_ticks_mean": mean_or_zero("mid_pnl_ticks"),
        "win_rate_net_cross": float(trades["is_win_net_cross"].mean()) if n_trades else 0.0,
    }


def _by_day(samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if samples.empty:
        return pd.DataFrame(rows)
    for execution in ("independent", "non_overlap"):
        active = samples.loc[samples["position"] != 0].copy()
        trades = non_overlapping_subset(active) if execution == "non_overlap" else active
        for date, day in trades.groupby("date", sort=False):
            rows.append(
                {
                    "date": date,
                    "execution": execution,
                    "n_trades": int(len(day)),
                    "total_net_cross_pnl_ticks": float(day["net_cross_pnl_ticks"].sum()),
                    "mean_net_cross_pnl_ticks": float(day["net_cross_pnl_ticks"].mean()),
                    "win_rate_net_cross": float(day["is_win_net_cross"].mean()),
                }
            )
    return pd.DataFrame(rows)


def flatten_pnl_metrics(
    *,
    prefix: str,
    independent: dict[str, float | int | str],
    non_overlap: dict[str, float | int | str],
    valid_samples: int,
    invalid_count: int,
    true_label_match_rate: float,
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        f"{prefix}_pnl_valid_samples": int(valid_samples),
        f"{prefix}_pnl_invalid_count": int(invalid_count),
        f"{prefix}_pnl_true_label_match_rate": float(true_label_match_rate),
        f"{prefix}_pnl_net_cross_ticks_mean": float(independent["net_cross_ticks_mean"]),
        f"{prefix}_pnl_net_cross_ticks_total": float(independent["net_cross_ticks_total"]),
        f"{prefix}_pnl_cross_ticks_mean": float(independent["cross_ticks_mean"]),
        f"{prefix}_pnl_mid_ticks_mean": float(independent["mid_ticks_mean"]),
        f"{prefix}_pnl_trade_rate": float(independent["trade_rate"]),
        f"{prefix}_pnl_n_trades": int(independent["n_trades"]),
        f"{prefix}_pnl_win_rate_net_cross": float(independent["win_rate_net_cross"]),
        f"{prefix}_pnl_net_cross_ticks_mean_non_overlap": float(non_overlap["net_cross_ticks_mean"]),
        f"{prefix}_pnl_net_cross_ticks_total_non_overlap": float(non_overlap["net_cross_ticks_total"]),
        f"{prefix}_pnl_cross_ticks_mean_non_overlap": float(non_overlap["cross_ticks_mean"]),
        f"{prefix}_pnl_mid_ticks_mean_non_overlap": float(non_overlap["mid_ticks_mean"]),
        f"{prefix}_pnl_trade_rate_non_overlap": float(non_overlap["trade_rate"]),
        f"{prefix}_pnl_n_trades_non_overlap": int(non_overlap["n_trades"]),
        f"{prefix}_pnl_win_rate_net_cross_non_overlap": float(non_overlap["win_rate_net_cross"]),
    }
    return metrics


def compute_pnl_from_prediction_outputs(
    *,
    config: ExperimentConfig,
    outputs: dict[str, Any],
    dataset: Any,
    raw_dir: Path,
    split: str = "test",
    prefix: str = "test",
    symbol_override: str | None = None,
) -> PnlResult:
    fees_bps, fees_source = pnl_fees_bps(config)
    samples, metadata = build_pnl_samples(
        config=config,
        outputs=outputs,
        y_paths=y_paths_from_dataset(dataset),
        raw_dir=Path(raw_dir),
        split=split,
        symbol_override=symbol_override,
    )
    samples = add_pnl_columns(samples, config=config, fees_bps=fees_bps)
    independent = _execution_summary(samples, execution="independent")
    non_overlap = _execution_summary(samples, execution="non_overlap")
    metrics = flatten_pnl_metrics(
        prefix=prefix,
        independent=independent,
        non_overlap=non_overlap,
        valid_samples=len(samples),
        invalid_count=int(metadata["invalid_exit_count"]),
        true_label_match_rate=float(metadata["true_label_match_rate"]),
    )
    summary = {
        "status": "computed",
        "split": split,
        "metric_prefix": prefix,
        "convention": {
            "primary_metric": f"{prefix}_pnl_net_cross_ticks_mean",
            "long": "buy entry_ask_t, sell exit_bid_t_plus_h",
            "short": "sell entry_bid_t, buy exit_ask_t_plus_h",
            "neutral": "no trade",
            "fees": "entry_mid * round_trip_fees_bps / 10000 for active trades",
            "mean_scope": "mean metrics are per executed trade; trade_rate reports trades per valid sample",
        },
        "horizon": pnl_horizon(config),
        "tick_size": float(config.data.tick_size),
        "round_trip_fees_bps": fees_bps,
        "fees_source": fees_source,
        "alignment": metadata,
        "independent": independent,
        "non_overlap": non_overlap,
        "metrics": metrics,
    }
    return PnlResult(
        status="computed",
        metrics=metrics,
        summary=summary,
        by_day=_by_day(samples),
        samples=samples,
    )


def skipped_pnl_result(reason: str) -> PnlResult:
    return PnlResult(
        status="skipped",
        metrics={},
        summary={"status": "skipped", "reason": reason},
        by_day=pd.DataFrame(),
        samples=pd.DataFrame(),
    )


def save_pnl_artifacts(result: PnlResult, *, metrics_path: Path, by_day_path: Path) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(result.summary, handle, sort_keys=False, allow_unicode=True)
    by_day_path.parent.mkdir(parents=True, exist_ok=True)
    result.by_day.to_csv(by_day_path, index=False)
