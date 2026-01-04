"""특징 엔지니어링 유틸리티."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict

import numpy as np
import pandas as pd


@dataclass
class FeatureSet:
    features: pd.DataFrame
    target: pd.Series
    feature_columns: List[str]


def _attach_features(df: pd.DataFrame, lookbacks: Iterable[int]) -> Tuple[pd.DataFrame, List[str]]:
    feature_columns: List[str] = []
    for lb in lookbacks:
        df[f"volatility_{lb}"] = df["return"].rolling(lb).std() * np.sqrt(252)
        df[f"momentum_{lb}"] = df["Close"].pct_change(lb)
        df[f"sma_ratio_{lb}"] = df["Close"] / df["Close"].rolling(lb).mean()
        df[f"volume_change_{lb}"] = df["Volume"].pct_change(lb)
        df[f"max_drawdown_{lb}"] = _max_drawdown(df["Close"], window=lb)
        df[f"downside_vol_{lb}"] = df["return"].where(df["return"] < 0).rolling(lb).std() * np.sqrt(252)
        feature_columns.extend(
            [
                f"volatility_{lb}",
                f"momentum_{lb}",
                f"sma_ratio_{lb}",
                f"volume_change_{lb}",
                f"max_drawdown_{lb}",
                f"downside_vol_{lb}",
            ]
        )
    return df, feature_columns


def compute_features(
    prices: pd.DataFrame,
    lookbacks: Iterable[int] = (63, 126, 252),
    forecast_horizon: int = 252,
) -> FeatureSet:
    """OHLCV 시계열에서 학습용 특징과 1년 후 수익률 타깃을 생성한다."""
    df = prices.copy()
    df = df.sort_index()
    df["return"] = df["Close"].pct_change()
    df, feature_columns = _attach_features(df, lookbacks=lookbacks)

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    window = 14
    roll_up = gain.rolling(window).mean()
    roll_down = loss.rolling(window).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    feature_columns.append("rsi_14")

    # MACD
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    feature_columns.extend(["macd", "macd_signal"])

    # ATR
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    feature_columns.append("atr_14")

    return df, feature_columns


def _attach_fundamental_features(df: pd.DataFrame, fundamentals: Dict[str, float]) -> Tuple[pd.DataFrame, List[str]]:
    cols = []
    for key, value in fundamentals.items():
        df[key] = value
        cols.append(key)
    return df, cols


def compute_features(
    prices: pd.DataFrame,
    lookbacks: Iterable[int] = (63, 126, 252),
    forecast_horizon: int = 252,
    feature_set: str = "price",
    fundamentals: Dict[str, float] | None = None,
) -> FeatureSet:
    """OHLCV 시계열에서 학습용 특징과 forward 수익률 타깃을 생성한다."""
    df = prices.copy()
    df = df.sort_index()
    df["return"] = df["Close"].pct_change()
    df, feature_columns = _price_technical_features(df, lookbacks=lookbacks)

    if feature_set in {"fundamentals", "risk"} and fundamentals:
        df, fundamental_cols = _attach_fundamental_features(df, fundamentals)
        feature_columns.extend(fundamental_cols)
    if feature_set == "risk":
        df["dollar_volume_20"] = (df["Close"] * df["Volume"]).rolling(20).mean()
        df["beta_hint"] = fundamentals.get("beta") if fundamentals else None
        feature_columns.extend(["dollar_volume_20", "beta_hint"])

    df["forward_return"] = df["Close"].shift(-forecast_horizon) / df["Close"] - 1
    df = df.dropna(subset=feature_columns + ["forward_return"])

    return FeatureSet(
        features=df[feature_columns].copy(),
        target=df["forward_return"].copy(),
        feature_columns=feature_columns,
    )


def latest_feature_row(
    prices: pd.DataFrame, lookbacks: Iterable[int] = (63, 126, 252)
) -> Tuple[pd.DataFrame, List[str]]:
    """예측용 최신 특징 행을 (타깃 없이) 생성한다."""
    df = prices.copy().sort_index()
    df["return"] = df["Close"].pct_change()
    df, feature_columns = _attach_features(df, lookbacks=lookbacks)
    df = df.dropna(subset=feature_columns)
    return df[feature_columns].tail(1), feature_columns


def _max_drawdown(close: pd.Series, window: int) -> pd.Series:
    rolling_max = close.rolling(window).max()
    drawdown = close / rolling_max - 1.0
    return drawdown
