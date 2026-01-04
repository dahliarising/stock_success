# Stock Success

End-to-end example for downloading U.S. equity data, training a feature-based
ranking model, and generating a 1-year forecast for selected tickers.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the full pipeline with a list of tickers:

```bash
PYTHONPATH=src python scripts/run_pipeline.py --tickers AAPL MSFT GOOG AMZN --years 5 --top-k 3
```

To also produce a one-year GRU-based forecast for a specific ticker, add
`--forecast-ticker`:

```bash
PYTHONPATH=src python scripts/run_pipeline.py --tickers AAPL MSFT GOOG AMZN --forecast-ticker AAPL
```

Outputs are written to the `artifacts/` directory:

- `candidate_scores.csv` lists tickers ranked by predicted 30-day forward return.
- `<TICKER>_forecast.csv` stores the joined historical closes and 1-year forecast.

## Streamlit app

Run the lightweight UI locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src streamlit run app.py
```
