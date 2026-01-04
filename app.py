"""Simple Streamlit UI for stock selection and forecasting."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import streamlit as st

# Ensure local src/ is on the import path when running `streamlit run app.py`
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stock_success import run_selection_pipeline


st.set_page_config(page_title="Stock Success", layout="wide")
st.title("Stock Success Dashboard")
st.markdown(
    "Use historical U.S. equity data to rank tickers by predicted 30-day forward return."
)

with st.sidebar:
    st.header("Inputs")
    tickers_raw = st.text_input("Tickers (comma-separated)", value="AAPL, MSFT, AMZN")
    years = st.number_input("Years of history", min_value=1, max_value=10, value=3, step=1)
    top_k = st.slider("Top K results", min_value=1, max_value=10, value=5)
    run_button = st.button("Run selection", type="primary")


def parse_tickers(raw: str) -> list[str]:
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


if run_button:
    tickers = parse_tickers(tickers_raw)
    if not tickers:
        st.error("Please provide at least one ticker symbol.")
    else:
        with st.spinner("Downloading data and training model..."):
            try:
                result = run_selection_pipeline(tickers=tickers, years_of_history=years)
            except Exception as exc:  # pragma: no cover - UI error path
                st.error(f"Failed to run pipeline: {exc}")
            else:
                st.success(
                    f"Model ready. Cross-validated RMSE (lower is better): {result.cv_score:.6f}"
                )
                st.markdown(f"**Top {top_k} candidates**")
                table = result.scored_candidates.head(top_k)
                table = table.rename(columns={"predicted_forward_return": "pred_return_30d"})
                st.dataframe(table, use_container_width=True, hide_index=True)

                st.caption(
                    "Data up to {:%Y-%m-%d}. Predictions are illustrative only.".format(
                        datetime.today()
                    )
                )
