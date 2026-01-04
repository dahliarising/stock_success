import json
from pathlib import Path

import pandas as pd

from stock_success.meta import fetch_meta_for_tickers, load_meta_cache, save_meta_cache
from stock_success.universe import load_universe_from_bytes, load_universe_from_csv, parse_universe


def test_parse_universe_from_dataframe():
    df = pd.DataFrame({"ticker": ["aapl", " msft ", None, "AMZN"]})
    parsed = parse_universe(df)
    assert parsed == ["AAPL", "MSFT", "AMZN"]


def test_load_universe_from_bytes_reads_first_column():
    content = b"sym\nGOOG\nNVDA\n"
    parsed = load_universe_from_bytes(content)
    assert parsed == ["GOOG", "NVDA"]


def test_meta_cache_roundtrip(tmp_path: Path):
    cache_path = tmp_path / "cache.json"
    data = {"AAPL": {"sector": "Tech", "industry": "Hardware", "company": "Apple"}}
    save_meta_cache(data, cache_path)
    loaded = load_meta_cache(cache_path)
    assert loaded == data


def test_fetch_meta_for_tickers_uses_cache(tmp_path: Path, monkeypatch):
    cache_path = tmp_path / "cache.json"
    save_meta_cache({"MSFT": {"sector": "Tech", "industry": "Software", "company": "MSFT"}}, cache_path)

    class FakeTicker:
        def __init__(self, *_):
            self.info = {"sector": "Industrials", "industry": "Tools", "shortName": "Fake"}

    monkeypatch.setattr("yfinance.Ticker", FakeTicker)

    meta = fetch_meta_for_tickers(["MSFT", "CAT"], cache_path=cache_path)
    assert meta["MSFT"]["sector"] == "Tech"  # from cache
    assert meta["CAT"]["sector"] == "Industrials"
