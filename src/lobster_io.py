from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import pandas as pd
import yaml


DEFAULT_COLUMN_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "configs" / "lobster_column_schema.yaml"


@dataclass(slots=True)
class LobsterCsvReadResult:
    dataframe: pd.DataFrame
    had_header: bool
    dropped_trailing_extra_column: bool = False


@dataclass(frozen=True, slots=True)
class LobsterColumnSchema:
    message_columns: list[str]
    orderbook_level_columns: list[str]
    orderbook_level_start: int = 1


@lru_cache(maxsize=None)
def load_lobster_column_schema(path: str | Path | None = None) -> LobsterColumnSchema:
    schema_path = DEFAULT_COLUMN_SCHEMA_PATH if path is None else Path(path)
    with schema_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    message_columns = payload.get("message", {}).get("columns")
    orderbook_payload = payload.get("orderbook", {})
    orderbook_level_columns = orderbook_payload.get("per_level_columns")
    if not isinstance(message_columns, list) or not all(isinstance(column, str) for column in message_columns):
        raise ValueError(f"Invalid LOBSTER column schema at {schema_path}: message.columns must be a string list.")
    if not isinstance(orderbook_level_columns, list) or not all(
        isinstance(column, str) for column in orderbook_level_columns
    ):
        raise ValueError(
            f"Invalid LOBSTER column schema at {schema_path}: orderbook.per_level_columns must be a string list."
        )

    return LobsterColumnSchema(
        message_columns=message_columns,
        orderbook_level_columns=orderbook_level_columns,
        orderbook_level_start=int(orderbook_payload.get("level_start", 1)),
    )


def _column_set(df: pd.DataFrame) -> set[str]:
    return {str(column) for column in df.columns}


def _message_column_names(
    *,
    time_column: str,
    size_column: str,
    price_column: str,
    order_id_column: str,
    categorical_value_map: Mapping[str, object],
    column_schema: LobsterColumnSchema | None = None,
) -> list[str]:
    schema = column_schema or load_lobster_column_schema()
    categorical_columns = list(categorical_value_map)
    type_column = "type" if "type" in categorical_columns or not categorical_columns else categorical_columns[0]
    direction_column = (
        "direction"
        if "direction" in categorical_columns or len(categorical_columns) < 2
        else categorical_columns[-1]
    )
    configured_names = {
        "time": time_column,
        "type": type_column,
        "order_id": order_id_column,
        "size": size_column,
        "price": price_column,
        "direction": direction_column,
    }
    return [configured_names.get(column, column) for column in schema.message_columns]


def read_lobster_message_csv(
    path: str | Path,
    *,
    time_column: str,
    size_column: str,
    price_column: str,
    order_id_column: str,
    categorical_value_map: Mapping[str, object],
    column_schema: LobsterColumnSchema | None = None,
) -> LobsterCsvReadResult:
    column_names = _message_column_names(
        time_column=time_column,
        size_column=size_column,
        price_column=price_column,
        order_id_column=order_id_column,
        categorical_value_map=categorical_value_map,
        column_schema=column_schema,
    )
    expected_columns = set(column_names)

    header_preview = pd.read_csv(path, nrows=0)
    had_header = expected_columns <= _column_set(header_preview)
    dataframe = pd.read_csv(path, low_memory=False) if had_header else pd.read_csv(path, header=None, low_memory=False)
    dropped_extra_column = False

    if had_header:
        extra_columns = [column for column in dataframe.columns if str(column) not in expected_columns]
        if len(extra_columns) == 1 and dataframe.columns[-1] == extra_columns[0]:
            dataframe = dataframe.drop(columns=extra_columns)
            dropped_extra_column = True
    else:
        if dataframe.shape[1] == len(column_names) + 1:
            dataframe = dataframe.iloc[:, : len(column_names)].copy()
            dropped_extra_column = True
        elif dataframe.shape[1] != len(column_names):
            raise ValueError(
                f"Message file {path} has {dataframe.shape[1]} columns; expected "
                f"{len(column_names)} or {len(column_names) + 1}."
            )
        dataframe.columns = column_names

    return LobsterCsvReadResult(
        dataframe=dataframe,
        had_header=had_header,
        dropped_trailing_extra_column=dropped_extra_column,
    )


def _orderbook_column_names(
    num_columns: int,
    column_schema: LobsterColumnSchema | None = None,
) -> list[str]:
    schema = column_schema or load_lobster_column_schema()
    level_width = len(schema.orderbook_level_columns)
    if level_width <= 0:
        raise ValueError("Orderbook schema must define at least one per-level column.")
    if num_columns % level_width != 0:
        raise ValueError(f"Orderbook file has {num_columns} columns; expected a multiple of {level_width}.")
    names: list[str] = []
    for level in range(schema.orderbook_level_start, schema.orderbook_level_start + num_columns // level_width):
        names.extend(f"{column}_{level}" for column in schema.orderbook_level_columns)
    return names


def read_lobster_orderbook_csv(
    path: str | Path,
    *,
    column_schema: LobsterColumnSchema | None = None,
) -> LobsterCsvReadResult:
    schema = column_schema or load_lobster_column_schema()
    first_level_columns = {
        f"{column}_{schema.orderbook_level_start}"
        for column in schema.orderbook_level_columns
    }
    header_preview = pd.read_csv(path, nrows=0)
    had_header = first_level_columns <= _column_set(header_preview)
    dataframe = pd.read_csv(path, low_memory=False) if had_header else pd.read_csv(path, header=None, low_memory=False)

    if not had_header:
        dataframe.columns = _orderbook_column_names(dataframe.shape[1], column_schema=schema)

    return LobsterCsvReadResult(dataframe=dataframe, had_header=had_header)
