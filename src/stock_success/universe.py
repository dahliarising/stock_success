"""티커 유니버스 로딩 유틸리티."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable, List

import pandas as pd


UNIVERSE_DEFAULT_PATH = Path("data/universe_sample.csv")


def _normalize_tickers(values: Iterable[str]) -> List[str]:
    return (
        pd.Series(values)
        .astype(str)
        .str.strip()
        .str.upper()
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )


def parse_universe(df: pd.DataFrame) -> List[str]:
    """DataFrame에서 티커 컬럼을 파싱한다."""
    if df.empty:
        return []

    candidate_cols = [c for c in df.columns if c.lower() in {"ticker", "tickers", "symbol", "symbols"}]
    column = candidate_cols[0] if candidate_cols else df.columns[0]
    return _normalize_tickers(df[column])


def load_universe_from_csv(path: Path | str) -> List[str]:
    df = pd.read_csv(path)
    return parse_universe(df)


def load_universe_from_bytes(content: bytes) -> List[str]:
    df = pd.read_csv(io.BytesIO(content))
    return parse_universe(df)


def load_default_universe(path: Path | str = UNIVERSE_DEFAULT_PATH) -> List[str]:
    target = Path(path)
    if not target.exists():
        return []
    return load_universe_from_csv(target)
