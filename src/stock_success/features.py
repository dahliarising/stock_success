"""Feature engineering helpers."""

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


def compute_features(
    prices: pd.DataFrame,
    lookbacks: Iterable[int] = (20, 60, 120),
    forecast_horizon: int = 30,
) -> FeatureSet:
    """Create a tabular supervised dataset from raw OHLCV data.

    The target is the forward percentage return over ``forecast_horizon`` days.
    """
    df = prices.copy()
    df = df.sort_index()
    df["return"] = df["Close"].pct_change()

    feature_columns: List[str] = []
    for lb in lookbacks:
        df[f"volatility_{lb}"] = df["return"].rolling(lb).std() * np.sqrt(252)
        df[f"momentum_{lb}"] = df["Close"].pct_change(lb)
        df[f"sma_ratio_{lb}"] = df["Close"] / df["Close"].rolling(lb).mean()
        df[f"max_drawdown_{lb}"] = _max_drawdown(df["Close"], window=lb)
        feature_columns.extend(
            [
                f"volatility_{lb}",
                f"momentum_{lb}",
                f"sma_ratio_{lb}",
                f"max_drawdown_{lb}",
            ]
        )

    df["forward_return"] = df["Close"].shift(-forecast_horizon) / df["Close"] - 1
    df.dropna(inplace=True)

    return FeatureSet(
        features=df[feature_columns].copy(),
        target=df["forward_return"].copy(),
        feature_columns=feature_columns,
    )


def latest_feature_row(prices: pd.DataFrame, lookbacks: Iterable[int]) -> Tuple[pd.DataFrame, List[str]]:
    """Compute the most recent feature row used for scoring/recommendations."""
    features = compute_features(prices, lookbacks=lookbacks)
    return features.features.tail(1), features.feature_columns


def _max_drawdown(close: pd.Series, window: int) -> pd.Series:
    rolling_max = close.rolling(window).max()
    drawdown = close / rolling_max - 1.0
    return drawdown
