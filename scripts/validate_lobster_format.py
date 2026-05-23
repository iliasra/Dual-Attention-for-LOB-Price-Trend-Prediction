"""
Utility script for validating LOBSTER CSV files in a directory before running
the preprocessing pipeline.

The script discovers message/orderbook CSV pairs using the *_message_*.csv and
*_orderbook_*.csv filename patterns, reports unmatched files, and validates each
pair with the project YAML configuration. For each pair, it checks that message
and orderbook files can be read, that raw message files match the column schema
defined in configs/lobster_column_schema.yaml or have one extra trailing column
that is dropped, that raw orderbook files have a valid per-level layout according
to the same schema, and that message/orderbook files have the same number of rows.
It also checks that all orderbook files for the same symbol have the same number
of levels.

Within the configured trading window [market_open_seconds, market_close_seconds],
it checks that at least one row exists and verifies dummy price values
(9999999999 / -9999999999) and NaN values after removing complete ghost levels
from a validation copy. It does not check files outside the LOBSTER naming
patterns.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from configuration import ExperimentConfig, load_config
from kinematic_preprocessing import handle_abnormal_prices
from lobster_io import read_lobster_message_csv, read_lobster_orderbook_csv


@dataclass(slots=True)
class LobsterPair:
    message_path: Path
    orderbook_path: Path

    @property
    def symbol(self) -> str:
        return self.message_path.name.split("_", 1)[0]

    @property
    def label(self) -> str:
        return self.message_path.name.removesuffix(".csv").replace("_message_", " / orderbook ")


@dataclass(slots=True)
class PairValidationResult:
    notes: list[str]
    orderbook_levels: int


def resolve_config_path(config: ExperimentConfig, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (config.path.parent / candidate).resolve()


def discover_pairs(data_dir: Path) -> tuple[list[LobsterPair], list[Path]]:
    pairs: list[LobsterPair] = []
    unmatched: list[Path] = []
    orderbook_files = {path.name: path for path in data_dir.glob("*_orderbook_*.csv")}
    matched_orderbooks: set[Path] = set()

    for message_path in sorted(data_dir.glob("*_message_*.csv")):
        orderbook_path = orderbook_files.get(message_path.name.replace("_message_", "_orderbook_"))
        if orderbook_path is None:
            unmatched.append(message_path)
            continue
        pairs.append(LobsterPair(message_path=message_path, orderbook_path=orderbook_path))
        matched_orderbooks.add(orderbook_path)

    for orderbook_path in sorted(orderbook_files.values()):
        if orderbook_path not in matched_orderbooks:
            unmatched.append(orderbook_path)

    return pairs, unmatched


def _orderbook_level_count(column_count: int) -> int:
    if column_count % 4 != 0:
        raise ValueError(f"orderbook has {column_count} columns; expected a multiple of 4")
    return column_count // 4


def _raise_for_nan_values(frame_name: str, columns_with_nan: list[str]) -> None:
    if columns_with_nan:
        raise ValueError(
            f"{frame_name} contains NaN values inside the configured trading window "
            f"in columns: {columns_with_nan}"
        )


def validate_pair(pair: LobsterPair, config: ExperimentConfig) -> PairValidationResult:
    notes: list[str] = []
    message_result = read_lobster_message_csv(
        pair.message_path,
        time_column=config.data.time_column,
        size_column=config.preprocessing.message.size_column,
        price_column=config.preprocessing.message.price_column,
        order_id_column=config.preprocessing.message.order_id_column,
        categorical_value_map=config.preprocessing.message.categorical_value_map,
    )
    orderbook_result = read_lobster_orderbook_csv(pair.orderbook_path)
    message_df = message_result.dataframe
    orderbook_df = orderbook_result.dataframe

    if len(message_df) != len(orderbook_df):
        raise ValueError(
            f"row mismatch: message has {len(message_df)} rows, "
            f"orderbook has {len(orderbook_df)} rows"
        )
    orderbook_levels = _orderbook_level_count(orderbook_df.shape[1])

    times = message_df[config.data.time_column]
    trading_mask = times.between(
        config.preprocessing.temporal_features.market_open_seconds,
        config.preprocessing.temporal_features.market_close_seconds,
        inclusive="both",
    )
    trading_rows = int(trading_mask.sum())
    if trading_rows == 0:
        raise ValueError(
            "no rows inside configured trading window "
            f"[{config.preprocessing.temporal_features.market_open_seconds}, "
                f"{config.preprocessing.temporal_features.market_close_seconds}]"
        )

    message_clean = message_df.loc[trading_mask].copy()
    orderbook_clean = orderbook_df.loc[trading_mask].copy()
    before_message_columns = set(message_clean.columns)
    before_orderbook_columns = set(orderbook_clean.columns)
    handle_abnormal_prices([message_clean, orderbook_clean])
    dropped_message_columns = sorted(before_message_columns - set(message_clean.columns))
    dropped_orderbook_columns = sorted(before_orderbook_columns - set(orderbook_clean.columns))

    _raise_for_nan_values(
        "message file",
        [str(column) for column, has_nan in message_clean.isna().any().items() if has_nan],
    )
    _raise_for_nan_values(
        "orderbook file",
        [str(column) for column, has_nan in orderbook_clean.isna().any().items() if has_nan],
    )

    notes.append(f"rows={len(message_df)}, trading_rows={trading_rows}")
    notes.append(f"orderbook_levels={orderbook_levels}")
    if message_result.dropped_trailing_extra_column:
        notes.append("dropped trailing extra message column")
    if not message_result.had_header:
        notes.append("message header inferred")
    if not orderbook_result.had_header:
        notes.append("orderbook header inferred")
    if dropped_message_columns:
        notes.append(f"message ghost columns={dropped_message_columns}")
    if dropped_orderbook_columns:
        notes.append(f"orderbook ghost columns={dropped_orderbook_columns}")
    return PairValidationResult(notes=notes, orderbook_levels=orderbook_levels)


def inconsistent_level_messages(levels_by_symbol: dict[str, dict[int, list[str]]]) -> list[str]:
    messages: list[str] = []
    for symbol, files_by_level in sorted(levels_by_symbol.items()):
        if len(files_by_level) <= 1:
            continue
        detail = ", ".join(
            f"{level} levels: {sorted(file_names)}"
            for level, file_names in sorted(files_by_level.items())
        )
        messages.append(f"symbol {symbol} has inconsistent orderbook levels across files: {detail}")
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate LOBSTER CSV files before preprocessing.")
    parser.add_argument("--config", type=Path, default=None, help="Path to the pipeline YAML config.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory containing LOBSTER CSV files.")
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = args.data_dir or resolve_config_path(config, config.data.raw_data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"LOBSTER directory not found: {data_dir}")

    pairs, unmatched = discover_pairs(data_dir)
    failures: list[str] = []
    levels_by_symbol: dict[str, dict[int, list[str]]] = {}

    print(f"Validating {len(pairs)} LOBSTER pair(s) in {data_dir}")
    for path in unmatched:
        failures.append(f"unmatched file: {path.name}")

    for pair in pairs:
        try:
            result = validate_pair(pair, config)
        except Exception as exc:
            failures.append(f"{pair.message_path.name}: {exc}")
            print(f"FAIL {pair.message_path.name}: {exc}")
            continue
        levels_by_symbol.setdefault(pair.symbol, {}).setdefault(result.orderbook_levels, []).append(
            pair.orderbook_path.name
        )
        print(f"OK   {pair.message_path.name}: " + "; ".join(result.notes))

    failures.extend(inconsistent_level_messages(levels_by_symbol))

    if failures:
        print()
        print(f"Validation failed with {len(failures)} issue(s):")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print()
    print("All LOBSTER files passed validation.")


if __name__ == "__main__":
    main()
