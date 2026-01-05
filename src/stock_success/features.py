"""OHLCV에서 기술적 특징을 생성한다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class FeatureSet:
    features: pd.DataFrame
    target: pd.Series
    feature_columns: List[str]


def _window_triplet(lookbacks: Iterable[int]) -> Tuple[int, int, int]:
    windows = list(dict.fromkeys(int(x) for x in lookbacks))
    if not windows:
        windows = [5, 10, 20]
    if len(windows) == 1:
        windows.append(windows[0] * 2)
    if len(windows) == 2:
        windows.append(windows[-1] * 2)
    return windows[0], windows[1], windows[2]


def _technical_features(df: pd.DataFrame, lookbacks: Iterable[int]) -> Tuple[pd.DataFrame, List[str]]:
    short, mid, long = _window_triplet(lookbacks)
    feature_columns: List[str] = []

    df["return_1d"] = df["Close"].pct_change()
    df[f"return_{short}d"] = df["Close"].pct_change(short)
    df[f"return_{mid}d"] = df["Close"].pct_change(mid)

    df[f"volatility_{short}d"] = df["Close"].pct_change().rolling(short).std() * np.sqrt(252)
    df[f"volatility_{mid}d"] = df["Close"].pct_change().rolling(mid).std() * np.sqrt(252)

    df[f"sma_ratio_{short}d"] = df["Close"] / df["Close"].rolling(short).mean()
    df[f"sma_ratio_{mid}d"] = df["Close"] / df["Close"].rolling(mid).mean()

    ema_short = df["Close"].ewm(span=short, adjust=False).mean()
    ema_mid = df["Close"].ewm(span=mid, adjust=False).mean()
    df[f"ema_gap_{short}_{mid}"] = ema_short - ema_mid

    df[f"volume_change_{short}d"] = df["Volume"].pct_change(short)
    df[f"volume_change_{mid}d"] = df["Volume"].pct_change(mid)

    feature_columns.extend(
        [
            "return_1d",
            f"return_{short}d",
            f"return_{mid}d",
            f"volatility_{short}d",
            f"volatility_{mid}d",
            f"sma_ratio_{short}d",
            f"sma_ratio_{mid}d",
            f"ema_gap_{short}_{mid}",
            f"volume_change_{short}d",
            f"volume_change_{mid}d",
        ]
    )

    return df, feature_columns


def compute_features(
    prices: pd.DataFrame,
    lookbacks: Iterable[int] = (5, 10, 20),
    forecast_horizon: int = 30,
) -> FeatureSet:
    """OHLCV에서 기술 피처 10개와 forward 수익률 타깃을 생성한다."""

    df = prices.copy().sort_index()
    df, feature_columns = _technical_features(df, lookbacks=lookbacks)

    df["future_return"] = df["Close"].shift(-forecast_horizon) / df["Close"] - 1
    df = df.dropna(subset=feature_columns + ["future_return"])

    return FeatureSet(
        features=df[feature_columns].copy(),
        target=df["future_return"].copy(),
        feature_columns=feature_columns,
    )


def latest_feature_row(prices: pd.DataFrame, lookbacks: Iterable[int] = (5, 10, 20)) -> Tuple[pd.DataFrame, List[str]]:
    """가장 최근 시점의 특징 행을 반환한다."""

    df = prices.copy().sort_index()
    df, feature_columns = _technical_features(df, lookbacks=lookbacks)
    df = df.dropna(subset=feature_columns)
    return df[feature_columns].tail(1), feature_columns
