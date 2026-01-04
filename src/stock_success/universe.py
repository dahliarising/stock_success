"""Utilities for managing ticker universes."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable, List

import pandas as pd


UNIVERSE_DEFAULT_PATH = Path("data/universe_sp500.csv")


def parse_universe(df: pd.DataFrame) -> List[str]:
    """Extract a unique, uppercased ticker list from a DataFrame.

    The first column is used if no obvious ticker column name is found.
    """
    if df.empty:
        return []

    candidate_cols = [c for c in df.columns if c.lower() in {"ticker", "tickers", "symbol", "symbols"}]
    column = candidate_cols[0] if candidate_cols else df.columns[0]
    tickers = (
        df[column]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )
    return tickers


def load_universe_from_csv(path: Path | str) -> List[str]:
    df = pd.read_csv(path)
    return parse_universe(df)


def load_universe_from_bytes(content: bytes) -> List[str]:
    df = pd.read_csv(io.BytesIO(content))
    return parse_universe(df)


def load_default_universe() -> List[str]:
    if not UNIVERSE_DEFAULT_PATH.exists():
        return []
    return load_universe_from_csv(UNIVERSE_DEFAULT_PATH)
