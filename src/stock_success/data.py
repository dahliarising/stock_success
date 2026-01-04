"""Data loading utilities for U.S. equities."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable, List, Optional

import pandas as pd
import yfinance as yf


def fetch_raw_data(
    tickers: Iterable[str],
    start: str | datetime,
    end: Optional[str | datetime] = None,
    interval: str = "1d",
) -> Dict[str, pd.DataFrame]:
    """Download adjusted OHLCV data for the requested tickers.

    Args:
        tickers: Collection of ticker symbols (e.g., ``["AAPL", "MSFT"]``).
        start: Start date (inclusive).
        end: End date (exclusive). Defaults to today if ``None``.
        interval: Sampling interval supported by yfinance (``"1d"``, ``"1wk"``, etc).

    Returns:
        A mapping of ticker -> DataFrame with OHLCV columns.
    """

    ticker_list = list(tickers)
    if not ticker_list:
        raise ValueError("tickers cannot be empty.")

    # yfinance can return a MultiIndex when requesting multiple tickers. Normalize to a
    # dictionary of single-ticker frames for consistency.
    df = yf.download(
        tickers=ticker_list,
        start=start,
        end=end,
        interval=interval,
        progress=False,
        auto_adjust=True,
        group_by="ticker",
    )

    if isinstance(df.columns, pd.MultiIndex):
        result: Dict[str, pd.DataFrame] = {}
        for ticker in df.columns.get_level_values(0).unique():
            subframe = df[ticker].copy()
            subframe.dropna(how="all", inplace=True)
            result[ticker] = subframe
        return result

    # Single ticker case: return a dictionary with one entry.
    return {ticker_list[0]: df.dropna(how="all").copy()}
