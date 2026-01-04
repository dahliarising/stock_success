"""Modeling utilities for recommending promising tickers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from .features import FeatureSet, compute_features, latest_feature_row


@dataclass
class SelectionResult:
    model: object
    cv_score: float
    scored_candidates: pd.DataFrame
    feature_columns: List[str]
    latest_features: Dict[str, pd.Series]
    training_features: pd.DataFrame
    training_target: pd.Series


def train_selector(
    price_history: Dict[str, pd.DataFrame],
    lookbacks: Iterable[int] = (20, 60, 120),
    forecast_horizon: int = 30,
    model: object | None = None,
) -> SelectionResult:
    """Train a regression model that predicts forward returns."""
    frames: List[pd.DataFrame] = []
    feature_columns: List[str] | None = None
    latest_features: Dict[str, pd.Series] = {}
    last_close: Dict[str, float] = {}
    for ticker, prices in price_history.items():
        dataset: FeatureSet = compute_features(prices, lookbacks=lookbacks, forecast_horizon=forecast_horizon)
        feature_columns = dataset.feature_columns
        frame = dataset.features.copy()
        frame["forward_return"] = dataset.target
        frame["ticker"] = ticker
        frames.append(frame)
        latest_features[ticker] = dataset.features.tail(1).iloc[0]
        last_close[ticker] = float(prices["Close"].iloc[-1])

    if not frames or feature_columns is None:
        raise ValueError("No data available to train selector.")

    full = pd.concat(frames, ignore_index=True)
    X = full[feature_columns]
    y = full["forward_return"]

    model = model or RandomForestRegressor(n_estimators=400, random_state=42)
    n_splits = min(5, len(X) - 1)
    if n_splits >= 2:
        cv = TimeSeriesSplit(n_splits=n_splits)
        neg_mse = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error")
        cv_score = float(np.mean(np.sqrt(-neg_mse)))
        model.fit(X, y)
    else:
        model.fit(X, y)
        cv_score = float(np.sqrt(np.mean((model.predict(X) - y) ** 2)))

    scored = []
    for ticker, prices in price_history.items():
        latest_row = latest_features.get(ticker)
        if latest_row is None:
            continue
        predicted_return = model.predict(latest_row.to_frame().T)[0]
        scored.append(
            {
                "ticker": ticker,
                "predicted_forward_return": predicted_return,
                "last_price": prices["Close"].iloc[-1],
            }
        )

    scored_df = pd.DataFrame(scored).sort_values("predicted_forward_return", ascending=False).reset_index(drop=True)
    return SelectionResult(
        model=model,
        cv_score=cv_score,
        scored_candidates=scored_df,
        feature_columns=feature_columns,
        latest_features=latest_features,
        training_features=X,
        training_target=y,
    )
