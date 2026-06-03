from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from configuration import load_config
from gcv_lambda_cache import (
    build_daily_gcv_cache,
    daily_lambda_gcv_cache_path,
    lambda_gcv_cache_key,
    write_daily_gcv_cache,
)
from processing import LobProcessingPipeline
from run_logging import format_duration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one daily GCV lambda cache.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument("--kind", type=str, choices=("price", "volume"), required=True)
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "data" / "gcv_cache")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if config.preprocessing.kinematic_tokenization.method != "fast":
        raise ValueError("GCV lambda cache is only relevant for fast kinematic tokenization.")

    pipeline = LobProcessingPipeline(config)
    pairs = pipeline.discover_pairs()

    matches = [
        pair
        for pair in pairs
        if pair.symbol == args.symbol and pair.date == args.date
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one pair for symbol={args.symbol}, date={args.date}; "
            f"found {len(matches)}."
        )

    pair = matches[0]
    day = pipeline.prepare_pair_for_lambda_cache(pair)

    preprocessing = config.preprocessing
    stream_config = (
        preprocessing.price_kinematic
        if args.kind == "price"
        else preprocessing.volume_kinematic
    )
    if not stream_config.enabled:
        print(f"Stream {args.kind} is disabled; nothing to cache.")
        return

    values_by_day, centers_by_day, scale = pipeline._stream_values_for_lambda_optimization(
        [day],
        kind=args.kind,
    )

    if not values_by_day:
        raise ValueError(f"No values found for {pair.output_stem} / {args.kind}.")

    values = values_by_day[0]
    centers = centers_by_day[0] if centers_by_day is not None else None
    if len(values) < preprocessing.snapshot_window:
        raise ValueError(
            f"{pair.output_stem} has {len(values)} rows, "
            f"less than snapshot_window={preprocessing.snapshot_window}."
        )

    fast_config = stream_config.fast
    chunk_size = min(preprocessing.kinematic_tokenization.chunk_size, 4096)
    n_df_candidates = preprocessing.kinematic_tokenization.n_df_candidates

    cache_key = lambda_gcv_cache_key(
        window=preprocessing.snapshot_window,
        n_basis=fast_config.n_basis,
        max_df=fast_config.df,
        scale=scale,
        n_df_candidates=n_df_candidates,
        stream_signature=pipeline._lambda_cache_stream_signature(kind=args.kind),
    )
    cache_path = daily_lambda_gcv_cache_path(
        args.cache_dir,
        cache_key=cache_key,
        kind=args.kind,
        output_stem=pair.output_stem,
    )

    if cache_path.exists() and not args.force:
        print(f"Cache already exists, skipping: {cache_path}")
        return

    print(
        "Building GCV lambda cache: "
        f"symbol={args.symbol} date={args.date} kind={args.kind} -> {cache_path}"
    )
    build_start = perf_counter()
    cache = build_daily_gcv_cache(
        values=values,
        window=preprocessing.snapshot_window,
        n_basis=fast_config.n_basis,
        max_df=fast_config.df,
        chunk_size=chunk_size,
        n_df_candidates=n_df_candidates,
        centers=centers,
        scale=scale,
        metadata={
            "symbol": pair.symbol,
            "date": pair.date,
            "output_stem": pair.output_stem,
            "kind": args.kind,
            "window": preprocessing.snapshot_window,
            "n_basis": fast_config.n_basis,
            "max_df": fast_config.df,
            "scale": scale,
            "stream_signature": pipeline._lambda_cache_stream_signature(kind=args.kind),
            "chunk_size": chunk_size,
            "n_df_candidates": n_df_candidates,
        },
        progress_desc=f"GCV cache {pair.output_stem} {args.kind}",
    )
    build_duration_seconds = perf_counter() - build_start
    cache = cache._replace(
        metadata={
            **cache.metadata,
            "build_seconds": round(build_duration_seconds, 6),
            "build_duration": format_duration(build_duration_seconds),
        }
    )

    write_daily_gcv_cache(cache_path, cache)
    print(
        "Saved GCV lambda cache: "
        f"symbol={args.symbol} date={args.date} kind={args.kind} -> {cache_path} "
        f"({format_duration(build_duration_seconds)})."
    )


if __name__ == "__main__":
    main()
