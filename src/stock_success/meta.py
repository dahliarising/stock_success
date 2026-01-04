"""섹터/산업 메타데이터 캐싱."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd
import yfinance as yf


DEFAULT_CACHE_PATH = Path("data/meta_cache.json")
FUNDAMENTAL_FIELDS = [
    "trailingPE",
    "priceToBook",
    "marketCap",
    "revenueGrowth",
    "profitMargins",
    "debtToEquity",
    "freeCashflow",
    "beta",
    "averageVolume",
    "averageDailyVolume10Day",
]


def load_meta_cache(path: Path | str = DEFAULT_CACHE_PATH) -> Dict[str, dict]:
    cache_path = Path(path)
    if not cache_path.exists():
        return {}
    with cache_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_meta_cache(meta: Dict[str, dict], path: Path | str = DEFAULT_CACHE_PATH) -> None:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _extract_meta(data: dict, ticker: str) -> dict:
    info: dict = {}
    info["sector"] = data.get("sector") or "Unknown"
    info["industry"] = data.get("industry") or "Unknown"
    info["company"] = data.get("shortName") or data.get("longName") or ticker
    for field in FUNDAMENTAL_FIELDS:
        info[field] = data.get(field)
    return info


def fetch_meta_for_tickers(tickers: Iterable[str], cache_path: Path | str = DEFAULT_CACHE_PATH) -> Dict[str, dict]:
    cache = load_meta_cache(cache_path)
    updated = False

    for ticker in tickers:
        if ticker in cache:
            continue
        try:
            data = yf.Ticker(ticker).info or {}
            cache[ticker] = _extract_meta(data, ticker)
        except Exception:
            cache[ticker] = _extract_meta({}, ticker)
        updated = True

    if updated:
        save_meta_cache(cache, cache_path)

    return cache


def meta_dataframe(tickers: Iterable[str], cache_path: Path | str = DEFAULT_CACHE_PATH) -> pd.DataFrame:
    meta = fetch_meta_for_tickers(tickers, cache_path=cache_path)
    df = pd.DataFrame.from_dict(meta, orient="index")
    df.index.name = "ticker"
    df = df.reset_index()
    return df
