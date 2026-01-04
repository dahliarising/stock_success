"""특징 엔지니어링 유틸리티."""

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


def _attach_features(df: pd.DataFrame, lookbacks: Iterable[int]) -> Tuple[pd.DataFrame, List[str]]:
    feature_columns: List[str] = []
    for lb in lookbacks:
        df[f"volatility_{lb}"] = df["return"].rolling(lb).std() * np.sqrt(252)
        df[f"momentum_{lb}"] = df["Close"].pct_change(lb)
        df[f"sma_ratio_{lb}"] = df["Close"] / df["Close"].rolling(lb).mean()
        df[f"volume_change_{lb}"] = df["Volume"].pct_change(lb)
        df[f"max_drawdown_{lb}"] = _max_drawdown(df["Close"], window=lb)
        feature_columns.extend(
            [
                f"volatility_{lb}",
                f"momentum_{lb}",
                f"sma_ratio_{lb}",
                f"volume_change_{lb}",
                f"max_drawdown_{lb}",
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
