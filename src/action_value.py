from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

try:
    from monitoring import exclusive_directional_top_k
except ImportError:  # pragma: no cover
    from .monitoring import exclusive_directional_top_k


ACTION_VALUE_METRICS_SCHEMA_VERSION = 2


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return deterministic average ranks, including ties, without scipy."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman_rank_correlation(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute Spearman rank correlation and return zero for degenerate inputs."""
    predicted = np.asarray(predictions, dtype=np.float64).reshape(-1)
    realized = np.asarray(targets, dtype=np.float64).reshape(-1)
    valid = np.isfinite(predicted) & np.isfinite(realized)
    if int(valid.sum()) < 2:
        return 0.0
    predicted_ranks = _average_ranks(predicted[valid])
    realized_ranks = _average_ranks(realized[valid])
    if np.std(predicted_ranks) == 0.0 or np.std(realized_ranks) == 0.0:
        return 0.0
    return float(np.corrcoef(predicted_ranks, realized_ranks)[0, 1])


def binary_average_precision(scores: np.ndarray, positives: np.ndarray) -> float:
    """Return threshold-based average precision for a binary economic event.

    Equal scores are evaluated as one threshold group. This makes the metric
    independent of row order and gives a constant score exactly the event
    prevalence, instead of rewarding whichever labels happen to appear first.
    """
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    positive = np.asarray(positives, dtype=bool).reshape(-1)
    valid = np.isfinite(score)
    score = score[valid]
    positive = positive[valid]
    positive_count = int(positive.sum())
    if positive_count == 0:
        return 0.0
    order = np.argsort(-score, kind="mergesort")
    ranked_score = score[order]
    ranked_positive = positive[order].astype(np.float64)
    cumulative_true_positive = np.cumsum(ranked_positive)

    # One PR point per distinct score threshold, including the final row.
    threshold_ends = np.flatnonzero(
        np.r_[ranked_score[1:] != ranked_score[:-1], True]
    )
    true_positive = cumulative_true_positive[threshold_ends]
    predicted_positive = threshold_ends.astype(np.float64) + 1.0
    precision_at_threshold = true_positive / predicted_positive
    recall_at_threshold = true_positive / float(positive_count)
    recall_increment = np.diff(np.r_[0.0, recall_at_threshold])
    return float(np.sum(recall_increment * precision_at_threshold))


def action_value_quantile_calibration(
    quantile_predictions: np.ndarray,
    targets: np.ndarray,
    quantile_levels: np.ndarray | tuple[float, ...],
) -> dict[str, object]:
    """Measure conditional-quantile coverage on held-out action values."""
    predicted = np.asarray(quantile_predictions, dtype=np.float64)
    realized = np.asarray(targets, dtype=np.float64)
    levels = np.asarray(quantile_levels, dtype=np.float64).reshape(-1)
    if predicted.ndim != 3 or predicted.shape[2] != 2:
        raise ValueError("quantile_predictions must have shape [n, quantiles, 2].")
    if realized.shape != (predicted.shape[0], 2):
        raise ValueError("targets must have shape [n, 2] matching quantile predictions.")
    if predicted.shape[1] != len(levels):
        raise ValueError("quantile_levels must match the quantile prediction axis.")
    if len(levels) == 0 or np.any(levels <= 0.0) or np.any(levels >= 1.0) or np.any(np.diff(levels) <= 0.0):
        raise ValueError("quantile_levels must be strictly increasing values in (0, 1).")
    valid = np.isfinite(realized).all(axis=1) & np.isfinite(predicted).all(axis=(1, 2))
    predicted = predicted[valid]
    realized = realized[valid]
    if len(realized) == 0:
        raise ValueError("Cannot calibrate quantiles without finite validation rows.")

    per_action: dict[str, dict[str, dict[str, float]]] = {}
    absolute_errors: list[float] = []
    pinball_values: list[float] = []
    for action_index, action_name in enumerate(("long", "short")):
        action_rows: dict[str, dict[str, float]] = {}
        for quantile_index, level in enumerate(levels):
            forecast = predicted[:, quantile_index, action_index]
            outcome = realized[:, action_index]
            empirical_cdf = float(np.mean(outcome <= forecast))
            calibration_error = empirical_cdf - float(level)
            residual = outcome - forecast
            pinball = float(np.mean(np.maximum(level * residual, (level - 1.0) * residual)))
            action_rows[f"{level:g}"] = {
                "nominal_cdf": float(level),
                "empirical_cdf": empirical_cdf,
                "calibration_error": calibration_error,
                "absolute_calibration_error": abs(calibration_error),
                "pinball_loss": pinball,
            }
            absolute_errors.append(abs(calibration_error))
            pinball_values.append(pinball)
        per_action[action_name] = action_rows

    adjacent_crossing = predicted[:, :-1, :] > predicted[:, 1:, :]
    any_crossing = adjacent_crossing.any(axis=(1, 2)) if adjacent_crossing.size else np.zeros(len(predicted), dtype=bool)
    lower = predicted[:, 0, :]
    upper = predicted[:, -1, :]
    inside = (realized >= lower) & (realized <= upper)
    expected_interval_coverage = float(levels[-1] - levels[0])
    empirical_interval_coverage = float(np.mean(inside))
    return {
        "n": int(len(realized)),
        "quantile_levels": levels.tolist(),
        "per_action": per_action,
        "mean_absolute_calibration_error": float(np.mean(absolute_errors)),
        "max_absolute_calibration_error": float(np.max(absolute_errors)),
        "mean_pinball_loss": float(np.mean(pinball_values)),
        "adjacent_crossing_rate": float(np.mean(adjacent_crossing)) if adjacent_crossing.size else 0.0,
        "row_crossing_rate": float(np.mean(any_crossing)),
        "central_interval": {
            "lower_quantile": float(levels[0]),
            "upper_quantile": float(levels[-1]),
            "nominal_coverage": expected_interval_coverage,
            "empirical_coverage": empirical_interval_coverage,
            "coverage_error": empirical_interval_coverage - expected_interval_coverage,
            "mean_width_ticks": float(np.mean(upper - lower)),
        },
    }


@dataclass(frozen=True, slots=True)
class ActionValueMetrics:
    """Ranking and executable-PnL metrics for long/short value regression.

    ``oracle_mean_pnl_ticks`` is the economic oracle averaged over every row:
    it may choose long, short, or no-trade (zero PnL). The zeros from no-trade
    rows remain in the denominator. ``forced_oracle_mean_pnl_ticks`` is the
    legacy diagnostic that must choose either long or short on every row.
    """

    n: int
    mae_long_ticks: float
    mae_short_ticks: float
    rank_ic_long: float
    rank_ic_short: float
    rank_ic_mean: float
    decision_count: int
    decision_rate: float
    mean_pnl_ticks: float
    total_pnl_ticks: float
    win_rate: float
    fixed_rate: float | None
    fixed_rate_count: int
    fixed_rate_actual_rate: float
    fixed_rate_mean_pnl_ticks: float
    fixed_rate_total_pnl_ticks: float
    fixed_rate_win_rate: float
    fixed_rate_overlap_resolved_count: int
    oracle_mean_pnl_ticks: float
    forced_oracle_mean_pnl_ticks: float = 0.0

    def to_dict(self) -> dict[str, int | float | None]:
        """Serialize metrics while making pre-schema-v2 objects unambiguous.

        Old checkpoints may unpickle an object without the newly added forced
        oracle slot. In that case their ``oracle_mean_pnl_ticks`` was actually
        the forced oracle. Preserve it under the explicit legacy name and mark
        the unavailable economic oracle as null instead of silently relabeling
        the historical value.
        """
        payload = {
            metric_field.name: getattr(self, metric_field.name)
            for metric_field in fields(self)
            if hasattr(self, metric_field.name)
        }
        if not hasattr(self, "forced_oracle_mean_pnl_ticks"):
            payload["forced_oracle_mean_pnl_ticks"] = float(self.oracle_mean_pnl_ticks)
            payload["oracle_mean_pnl_ticks"] = None
        return payload


def _pnl_summary(values: np.ndarray) -> tuple[int, float, float, float]:
    pnl = np.asarray(values, dtype=np.float64).reshape(-1)
    if pnl.size == 0:
        return 0, 0.0, 0.0, 0.0
    return int(pnl.size), float(pnl.mean()), float(pnl.sum()), float(np.mean(pnl > 0.0))


def action_value_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    decision_threshold_ticks: float = 0.0,
    fixed_rate: float | None = None,
) -> ActionValueMetrics:
    """Evaluate value predictions as both rankings and executable actions.

    Columns are ordered ``[long_net_return_ticks, short_net_return_ticks]``.
    Every row can produce at most one action. Fixed-rate selection uses separate
    long/short capacities but resolves overlaps before measuring PnL.
    """
    predicted = np.asarray(predictions, dtype=np.float64)
    realized = np.asarray(targets, dtype=np.float64)
    if predicted.ndim != 2 or predicted.shape[1] != 2:
        raise ValueError("predictions must have shape [n, 2] ordered as [long, short].")
    if realized.shape != predicted.shape:
        raise ValueError("targets must have the same [n, 2] shape as predictions.")
    if decision_threshold_ticks < 0.0:
        raise ValueError("decision_threshold_ticks must be >= 0.")
    valid = np.isfinite(predicted).all(axis=1) & np.isfinite(realized).all(axis=1)
    predicted = predicted[valid]
    realized = realized[valid]
    n_samples = int(predicted.shape[0])
    if n_samples == 0:
        return ActionValueMetrics(
            n=0,
            mae_long_ticks=0.0,
            mae_short_ticks=0.0,
            rank_ic_long=0.0,
            rank_ic_short=0.0,
            rank_ic_mean=0.0,
            decision_count=0,
            decision_rate=0.0,
            mean_pnl_ticks=0.0,
            total_pnl_ticks=0.0,
            win_rate=0.0,
            fixed_rate=fixed_rate,
            fixed_rate_count=0,
            fixed_rate_actual_rate=0.0,
            fixed_rate_mean_pnl_ticks=0.0,
            fixed_rate_total_pnl_ticks=0.0,
            fixed_rate_win_rate=0.0,
            fixed_rate_overlap_resolved_count=0,
            oracle_mean_pnl_ticks=0.0,
            forced_oracle_mean_pnl_ticks=0.0,
        )

    rank_long = spearman_rank_correlation(predicted[:, 0], realized[:, 0])
    rank_short = spearman_rank_correlation(predicted[:, 1], realized[:, 1])
    chosen_side = np.argmax(predicted, axis=1)
    chosen_score = predicted[np.arange(n_samples), chosen_side]
    decision_mask = chosen_score > float(decision_threshold_ticks)
    policy_pnl = realized[np.arange(n_samples), chosen_side][decision_mask]
    decision_count, mean_pnl, total_pnl, win_rate = _pnl_summary(policy_pnl)

    fixed_count = 0
    fixed_mean = fixed_total = fixed_win = 0.0
    fixed_actual_rate = 0.0
    overlap_count = 0
    if fixed_rate is not None:
        if not 0.0 < float(fixed_rate) <= 1.0:
            raise ValueError("fixed_rate must be in (0, 1] or null.")
        k = max(1, int(np.ceil(float(fixed_rate) * n_samples)))
        long_indices, short_indices, overlap_count = exclusive_directional_top_k(
            predicted[:, 0],
            predicted[:, 1],
            k_per_side=k,
        )
        fixed_pnl = np.concatenate([realized[long_indices, 0], realized[short_indices, 1]])
        fixed_count, fixed_mean, fixed_total, fixed_win = _pnl_summary(fixed_pnl)
        fixed_actual_rate = float(fixed_count / n_samples)

    forced_oracle_pnl = np.maximum(realized[:, 0], realized[:, 1])
    economic_oracle_pnl = np.maximum(forced_oracle_pnl, 0.0)
    return ActionValueMetrics(
        n=n_samples,
        mae_long_ticks=float(np.mean(np.abs(predicted[:, 0] - realized[:, 0]))),
        mae_short_ticks=float(np.mean(np.abs(predicted[:, 1] - realized[:, 1]))),
        rank_ic_long=rank_long,
        rank_ic_short=rank_short,
        rank_ic_mean=float((rank_long + rank_short) / 2.0),
        decision_count=decision_count,
        decision_rate=float(decision_count / n_samples),
        mean_pnl_ticks=mean_pnl,
        total_pnl_ticks=total_pnl,
        win_rate=win_rate,
        fixed_rate=fixed_rate,
        fixed_rate_count=fixed_count,
        fixed_rate_actual_rate=fixed_actual_rate,
        fixed_rate_mean_pnl_ticks=fixed_mean,
        fixed_rate_total_pnl_ticks=fixed_total,
        fixed_rate_win_rate=fixed_win,
        fixed_rate_overlap_resolved_count=overlap_count,
        oracle_mean_pnl_ticks=float(economic_oracle_pnl.mean()),
        forced_oracle_mean_pnl_ticks=float(forced_oracle_pnl.mean()),
    )


def action_value_coverage_curve(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    fixed_rates: tuple[float, ...] = (0.001, 0.0025, 0.005, 0.01, 0.02),
) -> list[dict[str, int | float | None]]:
    """Return ranking-to-PnL points over pre-specified disjoint coverages."""
    rows: list[dict[str, int | float | None]] = []
    for fixed_rate in fixed_rates:
        metrics = action_value_metrics(predictions, targets, fixed_rate=float(fixed_rate))
        rows.append(
            {
                "fixed_rate_per_side": float(fixed_rate),
                "actual_total_rate": metrics.fixed_rate_actual_rate,
                "trade_count": metrics.fixed_rate_count,
                "mean_pnl_ticks": metrics.fixed_rate_mean_pnl_ticks,
                "total_pnl_ticks": metrics.fixed_rate_total_pnl_ticks,
                "win_rate": metrics.fixed_rate_win_rate,
                "overlap_resolved_count": metrics.fixed_rate_overlap_resolved_count,
                "rank_ic_mean": metrics.rank_ic_mean,
            }
        )
    return rows


def action_value_policy_frontier(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    coverages: tuple[float, ...] = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0),
) -> list[dict[str, int | float]]:
    """Join ranking, profitable-opportunity PR and PnL along one policy frontier.

    For each row the predicted best action is fixed first, so a row can never be
    both long and short. Rows are then ranked by that action's predicted net
    value. At each total coverage, ``profitable_precision`` is the trade win
    rate and ``profitable_recall`` is the fraction of oracle-profitable rows
    captured with the correct profitable side. ``policy_ap`` summarizes the
    same score/outcome ranking over all rows and is therefore constant across
    frontier points.
    """
    predicted = np.asarray(predictions, dtype=np.float64)
    realized = np.asarray(targets, dtype=np.float64)
    if predicted.ndim != 2 or predicted.shape[1] != 2 or realized.shape != predicted.shape:
        raise ValueError("predictions and targets must both have shape [n, 2] ordered [long, short].")
    valid = np.isfinite(predicted).all(axis=1) & np.isfinite(realized).all(axis=1)
    predicted = predicted[valid]
    realized = realized[valid]
    n_samples = int(len(predicted))
    if n_samples == 0:
        return []
    normalized_coverages = tuple(float(value) for value in coverages)
    if any(not 0.0 < value <= 1.0 for value in normalized_coverages):
        raise ValueError("coverages must contain values in (0, 1].")

    chosen_side = np.argmax(predicted, axis=1)
    row = np.arange(n_samples)
    chosen_score = predicted[row, chosen_side]
    chosen_pnl = realized[row, chosen_side]
    order = np.argsort(-chosen_score, kind="mergesort")
    oracle_profitable_count = int(np.sum(np.max(realized, axis=1) > 0.0))
    policy_ap = binary_average_precision(chosen_score, chosen_pnl > 0.0)
    ap_long = binary_average_precision(predicted[:, 0], realized[:, 0] > 0.0)
    ap_short = binary_average_precision(predicted[:, 1], realized[:, 1] > 0.0)

    rows: list[dict[str, int | float]] = []
    previous_count = -1
    for coverage in normalized_coverages:
        count = min(n_samples, max(1, int(np.ceil(coverage * n_samples))))
        if count == previous_count:
            continue
        previous_count = count
        selected = order[:count]
        selected_score = chosen_score[selected]
        selected_pnl = chosen_pnl[selected]
        profitable_count = int(np.sum(selected_pnl > 0.0))
        rows.append(
            {
                "requested_total_coverage": coverage,
                "actual_total_coverage": float(count / n_samples),
                "trade_count": count,
                "score_threshold_ticks": float(selected_score[-1]),
                "mean_predicted_edge_ticks": float(np.mean(selected_score)),
                "mean_pnl_ticks": float(np.mean(selected_pnl)),
                "total_pnl_ticks": float(np.sum(selected_pnl)),
                "profitable_precision": float(profitable_count / count),
                "profitable_recall": float(
                    profitable_count / oracle_profitable_count if oracle_profitable_count > 0 else 0.0
                ),
                "selected_rank_ic": spearman_rank_correlation(selected_score, selected_pnl),
                "policy_ap": policy_ap,
                "ap_long_profitable": ap_long,
                "ap_short_profitable": ap_short,
            }
        )
    return rows
