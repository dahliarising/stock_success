"""Metadata caching for sector/industry lookups."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import yfinance as yf


DEFAULT_CACHE_PATH = Path("data/meta_cache.json")


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


def fetch_meta_for_tickers(tickers: Iterable[str], cache_path: Path | str = DEFAULT_CACHE_PATH) -> Dict[str, dict]:
    cache = load_meta_cache(cache_path)
    updated = False

    for ticker in tickers:
        if ticker in cache:
            continue
        info = {}
        try:
            data = yf.Ticker(ticker).info or {}
            info["sector"] = data.get("sector") or "Unknown"
            info["industry"] = data.get("industry") or "Unknown"
            info["company"] = data.get("shortName") or data.get("longName") or ticker
        except Exception:
            info = {"sector": "Unknown", "industry": "Unknown", "company": ticker}
        cache[ticker] = info
        updated = True

    if updated:
        save_meta_cache(cache, cache_path)

    return cache
