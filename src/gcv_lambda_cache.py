from __future__ import annotations

import json
import os
from pathlib import Path
from typing import NamedTuple

import numpy as np
from tqdm import tqdm

try:
    from fast_kinematic_preprocessing import (
        GCVOptimizationResult,
        bspline_smoother_matrix,
        effective_degrees_of_freedom,
        lambda_for_effective_degrees_of_freedom,
        sliding_windows_2d,
    )
except ImportError:  # pragma: no cover
    from .fast_kinematic_preprocessing import (
        GCVOptimizationResult,
        bspline_smoother_matrix,
        effective_degrees_of_freedom,
        lambda_for_effective_degrees_of_freedom,
        sliding_windows_2d,
    )


class DailyGCVCache(NamedTuple):
    candidate_dfs: np.ndarray
    smoothing_lambdas: np.ndarray
    effective_dfs: np.ndarray
    score_sums: np.ndarray
    count: int
    metadata: dict[str, object]


def _token(value: object) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def lambda_gcv_cache_key(
    *,
    window: int,
    n_basis: int,
    max_df: float,
    scale: float,
    n_df_candidates: int,
    stream_signature: str | None = None,
    degree: int = 3,
    ridge: float = 1e-8,
) -> str:
    signature = "" if stream_signature is None else f"_sig{_token(stream_signature)}"
    return (
        f"v1_w{window}"
        f"_nb{n_basis}"
        f"_df{_token(f'{max_df:g}')}"
        f"_scale{_token(f'{scale:g}')}"
        f"_c{n_df_candidates}"
        f"{signature}"
        f"_deg{degree}"
        f"_ridge{ridge:g}"
    )


def daily_lambda_gcv_cache_path(
    cache_dir: str | Path,
    *,
    cache_key: str,
    kind: str,
    output_stem: str,
) -> Path:
    return Path(cache_dir) / cache_key / kind / f"{output_stem}.npz"


def build_gcv_candidate_grid(
    *,
    window: int,
    n_basis: int,
    max_df: float,
    n_df_candidates: int,
    degree: int = 3,
    ridge: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if n_df_candidates <= 0:
        raise ValueError("n_df_candidates must be > 0.")
    if max_df <= 0:
        raise ValueError("max_df must be > 0.")

    df_at_zero = effective_degrees_of_freedom(
        window=window,
        n_basis=n_basis,
        smoothing_lambda=0.0,
        degree=degree,
        ridge=ridge,
    )
    df_cap = min(float(max_df), df_at_zero)

    df_floor = effective_degrees_of_freedom(
        window=window,
        n_basis=n_basis,
        smoothing_lambda=1e12,
        degree=degree,
        ridge=ridge,
    )

    if df_cap <= df_floor:
        candidate_dfs = np.array([df_cap], dtype=np.float64)
    else:
        candidate_dfs = np.linspace(df_floor, df_cap, n_df_candidates, dtype=np.float64)

    smoothing_lambdas: list[float] = []
    smoothers: list[np.ndarray] = []
    effective_dfs: list[float] = []

    for candidate_df in candidate_dfs:
        smoothing_lambda = lambda_for_effective_degrees_of_freedom(
            target_df=float(candidate_df),
            window=window,
            n_basis=n_basis,
            degree=degree,
            ridge=ridge,
        )
        smoother = bspline_smoother_matrix(
            window=window,
            n_basis=n_basis,
            smoothing_lambda=smoothing_lambda,
            degree=degree,
            ridge=ridge,
        )

        smoothing_lambdas.append(float(smoothing_lambda))
        smoothers.append(smoother)
        effective_dfs.append(float(np.trace(smoother)))

    return (
        candidate_dfs,
        np.asarray(smoothing_lambdas, dtype=np.float64),
        np.asarray(effective_dfs, dtype=np.float64),
        np.stack(smoothers, axis=0),
    )


def gcv_score_sums_for_one_day(
    *,
    values: np.ndarray,
    window: int,
    smoothers: np.ndarray,
    effective_dfs: np.ndarray,
    chunk_size: int = 4096,
    centers: np.ndarray | None = None,
    scale: float = 1.0,
    progress_desc: str | None = None,
) -> tuple[np.ndarray, int]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0.")
    if scale <= 0:
        raise ValueError("scale must be > 0.")

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected values with shape [N, F], got {values.shape}.")

    n_rows, n_features = values.shape
    n_windows = n_rows - window + 1
    if n_windows <= 0 or n_features == 0:
        raise ValueError("No train windows available for daily GCV cache.")

    if centers is not None:
        centers = np.asarray(centers, dtype=np.float64)
        if len(centers) < n_windows:
            raise ValueError(f"Expected at least {n_windows} centers, got {len(centers)}.")

    n_candidates = smoothers.shape[0]
    score_sums = np.zeros(n_candidates, dtype=np.float64)
    total_count = 0

    progress = tqdm(
        total=n_windows,
        desc=progress_desc,
        unit="windows",
        mininterval=5,
    ) if progress_desc else None

    try:
        for start in range(0, n_windows, chunk_size):
            stop = min(start + chunk_size, n_windows)
            block = values[start : stop + window - 1]
            windows = sliding_windows_2d(block, window)

            if centers is not None:
                windows = (windows - centers[start:stop, None, None]) / scale

            total_count += (stop - start) * n_features

            for candidate_index in range(n_candidates):
                smoother = smoothers[candidate_index]
                df = float(effective_dfs[candidate_index])
                denominator = (1.0 - df / float(window)) ** 2
                if denominator <= 0:
                    raise ValueError(
                        f"GCV denominator must be positive; got df={df:.6g}."
                    )

                fitted = np.einsum("ij,mjf->mif", smoother, windows, optimize=True)
                rss = np.sum((windows - fitted) ** 2, axis=1)
                scores = (rss / float(window)) / denominator
                score_sums[candidate_index] += float(np.sum(scores))

            if progress is not None:
                progress.update(stop - start)
    finally:
        if progress is not None:
            progress.close()

    return score_sums, int(total_count)


def build_daily_gcv_cache(
    *,
    values: np.ndarray,
    window: int,
    n_basis: int,
    max_df: float,
    n_df_candidates: int,
    degree: int = 3,
    ridge: float = 1e-8,
    chunk_size: int = 4096,
    centers: np.ndarray | None = None,
    scale: float = 1.0,
    metadata: dict[str, object] | None = None,
    progress_desc: str | None = None,
) -> DailyGCVCache:
    candidate_dfs, smoothing_lambdas, effective_dfs, smoothers = build_gcv_candidate_grid(
        window=window,
        n_basis=n_basis,
        max_df=max_df,
        degree=degree,
        ridge=ridge,
        n_df_candidates=n_df_candidates,
    )

    score_sums, count = gcv_score_sums_for_one_day(
        values=values,
        window=window,
        smoothers=smoothers,
        effective_dfs=effective_dfs,
        chunk_size=chunk_size,
        centers=centers,
        scale=scale,
        progress_desc=progress_desc,
    )

    return DailyGCVCache(
        candidate_dfs=candidate_dfs,
        smoothing_lambdas=smoothing_lambdas,
        effective_dfs=effective_dfs,
        score_sums=score_sums,
        count=count,
        metadata=metadata or {},
    )


def write_daily_gcv_cache(path: str | Path, cache: DailyGCVCache) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    tmp = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            candidate_dfs=cache.candidate_dfs,
            smoothing_lambdas=cache.smoothing_lambdas,
            effective_dfs=cache.effective_dfs,
            score_sums=cache.score_sums,
            count=np.asarray(cache.count, dtype=np.int64),
            metadata_json=np.asarray(json.dumps(cache.metadata, sort_keys=True), dtype=np.str_),
        )

    os.replace(tmp, target)
    return target


def load_daily_gcv_cache(path: str | Path) -> DailyGCVCache:
    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        metadata_json = str(data["metadata_json"]) if "metadata_json" in data else "{}"
        return DailyGCVCache(
            candidate_dfs=np.asarray(data["candidate_dfs"], dtype=np.float64),
            smoothing_lambdas=np.asarray(data["smoothing_lambdas"], dtype=np.float64),
            effective_dfs=np.asarray(data["effective_dfs"], dtype=np.float64),
            score_sums=np.asarray(data["score_sums"], dtype=np.float64),
            count=int(data["count"]),
            metadata=json.loads(metadata_json),
        )


def aggregate_daily_gcv_caches(caches: list[DailyGCVCache]) -> tuple[GCVOptimizationResult, int]:
    if not caches:
        raise ValueError("Cannot aggregate an empty list of daily GCV caches.")

    reference = caches[0]

    total_score_sums = np.zeros_like(reference.score_sums, dtype=np.float64)
    total_count = 0

    for cache in caches:
        if not np.allclose(cache.candidate_dfs, reference.candidate_dfs, rtol=0.0, atol=1e-12):
            raise ValueError("Incompatible GCV caches: candidate_dfs differ.")
        if not np.allclose(cache.smoothing_lambdas, reference.smoothing_lambdas, rtol=0.0, atol=1e-12):
            raise ValueError("Incompatible GCV caches: smoothing_lambdas differ.")
        if not np.allclose(cache.effective_dfs, reference.effective_dfs, rtol=0.0, atol=1e-12):
            raise ValueError("Incompatible GCV caches: effective_dfs differ.")

        total_score_sums += cache.score_sums
        total_count += int(cache.count)

    if total_count <= 0:
        raise ValueError("Cannot aggregate GCV caches with zero total count.")

    mean_scores = total_score_sums / float(total_count)
    best_index = int(np.argmin(mean_scores))

    return (
        GCVOptimizationResult(
            smoothing_lambda=float(reference.smoothing_lambdas[best_index]),
            effective_df=float(reference.effective_dfs[best_index]),
            gcv_score=float(mean_scores[best_index]),
        ),
        total_count,
    )
