"""Convenience exports for the stock_success package."""

from .data import fetch_raw_data
from .forecasting import ForecastResult, train_and_forecast
from .pipeline import forecast_selected_ticker, run_selection_pipeline
from .meta import fetch_meta_for_tickers, load_meta_cache, save_meta_cache
from .universe import load_default_universe, load_universe_from_bytes, load_universe_from_csv, parse_universe
from .models import instantiate_model, model_registry
from .eval import cross_validate_models, hit_rate
from .selection import SelectionResult, train_selector

__all__ = [
    "fetch_raw_data",
    "train_and_forecast",
    "forecast_selected_ticker",
    "run_selection_pipeline",
    "SelectionResult",
    "ForecastResult",
    "train_selector",
    "fetch_meta_for_tickers",
    "load_meta_cache",
    "save_meta_cache",
    "load_default_universe",
    "load_universe_from_bytes",
    "load_universe_from_csv",
    "parse_universe",
    "instantiate_model",
    "model_registry",
    "cross_validate_models",
    "hit_rate",
]
