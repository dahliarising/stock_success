#!/usr/bin/env python3
"""Example entrypoint for recommending and forecasting U.S. equities."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from stock_success.data import fetch_raw_data
from stock_success.forecasting import train_and_forecast
from stock_success.selection import train_selector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", required=True, help="List of tickers to evaluate, e.g., AAPL MSFT AMZN")
    parser.add_argument("--years", type=int, default=5, help="How many years of history to download")
    parser.add_argument("--forecast-ticker", help="Ticker to forecast for one year using a GRU model")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"), help="Directory to save outputs")
    parser.add_argument("--top-k", type=int, default=5, help="How many recommendations to print")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    end = datetime.today()
    start = end - timedelta(days=365 * args.years)
    history = fetch_raw_data(args.tickers, start=start, end=end)

    selection = train_selector(history)
    print(f"Cross-validated RMSE (lower is better): {selection.cv_score:.6f}")
    print("\nTop candidates by predicted 30-day forward return:")
    print(selection.scored_candidates.head(args.top_k))

    selection.scored_candidates.to_csv(args.output_dir / "candidate_scores.csv", index=False)

    if args.forecast_ticker:
        ticker = args.forecast_ticker.upper()
        if ticker not in history:
            raise SystemExit(f"{ticker} was not downloaded. Available: {list(history)}")
        forecast = train_and_forecast(history[ticker])
        forecast_df = pd.DataFrame({"history": forecast.history, "forecast": forecast.forecast})
        forecast_path = args.output_dir / f"{ticker}_forecast.csv"
        forecast_df.to_csv(forecast_path)
        print(f"\nSaved 1-year forecast for {ticker} to {forecast_path}")


if __name__ == "__main__":
    main()
