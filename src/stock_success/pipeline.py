"""핵심 예측 파이프라인."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from .data import fetch_raw_data
from .features import FeatureSet, compute_features, latest_feature_row
from .meta import meta_dataframe
from .models import instantiate_model


def fetch_history(tickers: Iterable[str], years_of_history: int = 5) -> Dict[str, pd.DataFrame]:
    end = datetime.today()
    start = end - timedelta(days=365 * years_of_history)
    return fetch_raw_data(tickers, start=start, end=end)


def build_training_data(
    history: Dict[str, pd.DataFrame], lookbacks=(63, 126, 252), forecast_horizon: int = 252
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    frames: List[pd.DataFrame] = []
    targets: List[pd.Series] = []
    feature_columns: List[str] | None = None

    for ticker, prices in history.items():
        dataset: FeatureSet = compute_features(prices, lookbacks=lookbacks, forecast_horizon=forecast_horizon)
        if dataset.features.empty:
            continue
        feature_columns = dataset.feature_columns
        frames.append(dataset.features.assign(ticker=ticker))
        targets.append(dataset.target)

    if not frames or feature_columns is None:
        raise ValueError("학습에 사용할 특징이 없습니다.")

    X = pd.concat(frames, axis=0, ignore_index=True)
    y = pd.concat(targets, axis=0, ignore_index=True)
    return X[feature_columns], y, feature_columns


def predict_one_year_returns(
    history: Dict[str, pd.DataFrame],
    model_name: str = "Ridge",
    lookbacks=(63, 126, 252),
    forecast_horizon: int = 252,
) -> pd.DataFrame:
    X, y, feature_columns = build_training_data(history, lookbacks=lookbacks, forecast_horizon=forecast_horizon)
    model = instantiate_model(model_name)
    model.fit(X, y)

    rows = []
    latest_features: Dict[str, pd.DataFrame] = {}
    for ticker, prices in history.items():
        latest_row, _ = latest_feature_row(prices, lookbacks=lookbacks)
        if latest_row.empty:
            continue
        latest_features[ticker] = latest_row[feature_columns]
        predicted_return = float(model.predict(latest_features[ticker])[0])
        current_price = float(prices["Close"].iloc[-1])
        rows.append(
            {
                "ticker": ticker,
                "current_price": current_price,
                "predicted_return_1y": predicted_return,
                "predicted_price_1y": current_price * (1 + predicted_return),
            }
        )

    meta = meta_dataframe([r["ticker"] for r in rows])
    result = pd.DataFrame(rows)
    merged = result.merge(meta, on="ticker", how="left")
    return merged, model, X, y, latest_features


def run_pipeline(
    tickers: Iterable[str],
    years_of_history: int = 5,
    model_name: str = "Ridge",
    lookbacks=(63, 126, 252),
    forecast_horizon: int = 252,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], object, pd.DataFrame, pd.Series, Dict[str, pd.DataFrame]]:
    history = fetch_history(tickers, years_of_history=years_of_history)
    predictions, model, X, y, latest_features = predict_one_year_returns(
        history,
        model_name=model_name,
        lookbacks=lookbacks,
        forecast_horizon=forecast_horizon,
    )
    return predictions, history, model, X, y, latest_features
