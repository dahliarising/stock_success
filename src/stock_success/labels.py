"""레이블 생성 유틸리티."""

from __future__ import annotations

import pandas as pd


def future_return_30d(prices: pd.DataFrame) -> pd.Series:
    """30거래일 후 수익률을 계산한다.

    누수를 방지하기 위해 미래 종가를 뒤로 이동시켜 사용한다.
    """

    close = prices["Close"]
    return close.shift(-30) / close - 1
