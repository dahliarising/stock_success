"""OHLCV 다운로드 유틸리티."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable

import yfinance as yf


def fetch_ohlcv(tickers: Iterable[str], start: str | datetime, end: str | datetime | None = None, interval: str = "1d") -> Dict[str, object]:
    """yfinance를 통해 조정 OHLCV 데이터를 내려받는다.

    Args:
        tickers: 티커 문자열 이터러블.
        start: 시작 일자(예: ``"2020-01-01"``).
        end: 종료 일자(생략 시 오늘).
        interval: yfinance에서 지원하는 간격.

    Returns:
        티커별 ``pandas.DataFrame`` 딕셔너리.
    """

    ticker_list = list(tickers)
    if not ticker_list:
        raise ValueError("tickers cannot be empty.")

    df = yf.download(
        tickers=ticker_list,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    if hasattr(df, "columns") and getattr(df, "columns", None) is not None and getattr(df, "columns", None).nlevels > 1:
        return {ticker: df[ticker].dropna(how="all").copy() for ticker in df.columns.get_level_values(0).unique()}

    return {ticker_list[0]: df.dropna(how="all").copy()}
