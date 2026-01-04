"""Convenience orchestration helpers for end-to-end usage."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Iterable

import pandas as pd

from .data import fetch_raw_data
from .forecasting import ForecastResult, train_and_forecast
from .selection import SelectionResult, train_selector
from .models import instantiate_model


def run_selection_pipeline(
    tickers: Iterable[str],
    years_of_history: int = 5,
    forecast_horizon: int = 30,
    lookbacks=(20, 60, 120),
    model_name: str | None = None,
) -> SelectionResult:
    """Fetch data and train the recommendation model."""
    end = datetime.today()
    start = end - timedelta(days=365 * years_of_history)
    history: Dict[str, pd.DataFrame] = fetch_raw_data(tickers, start=start, end=end)
    model = instantiate_model(model_name, random_state=42) if model_name else None
    return train_selector(history, lookbacks=lookbacks, forecast_horizon=forecast_horizon, model=model)


def forecast_selected_ticker(
    history: Dict[str, pd.DataFrame],
    ticker: str,
    forecast_days: int = 252,
    window: int = 60,
    epochs: int = 30,
) -> ForecastResult:
    """Train a GRU model for a single ticker and produce a 1-year forecast."""
    if ticker not in history:
        raise KeyError(f"{ticker} not found in provided price history keys: {list(history)}")
    return train_and_forecast(history[ticker], window=window, forecast_days=forecast_days, epochs=epochs)
