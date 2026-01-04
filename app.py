"""Simple Streamlit UI for stock selection and forecasting."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, List

import altair as alt
import pandas as pd
import streamlit as st

# Ensure local src/ is on the import path when running `streamlit run app.py`
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stock_success import (
    fetch_meta_for_tickers,
    fetch_raw_data,
    run_selection_pipeline,
)
from stock_success.eval import cross_validate_models
from stock_success.explain import permutation_importance_df, top_feature_contributions
from stock_success.models import model_registry
from stock_success.universe import load_default_universe, load_universe_from_bytes, parse_universe


st.set_page_config(page_title="Stock Success", layout="wide")
st.title("Stock Success Dashboard")
st.markdown(
    "Rank U.S. equities by predicted 30D forward return, grouped by sector, with explainability and downloads."
)


@st.cache_data(show_spinner=False)
def get_default_universe() -> List[str]:
    return load_default_universe()


@st.cache_data(show_spinner=False)
def load_prices(tickers: List[str], years: int) -> Dict[str, pd.DataFrame]:
    end = datetime.today()
    start = end - pd.Timedelta(days=365 * years)
    return fetch_raw_data(tickers, start=start, end=end)


@st.cache_data(show_spinner=False)
def load_meta(tickers: List[str]) -> Dict[str, dict]:
    return fetch_meta_for_tickers(tickers)


with st.sidebar:
    st.header("Universe")
    default_universe = get_default_universe()
    uploaded = st.file_uploader("Upload universe CSV (tickers column)", type=["csv"])
    if uploaded:
        universe = load_universe_from_bytes(uploaded.getvalue())
    else:
        universe = default_universe
    st.caption(f"Universe size: {len(universe)} tickers")

    st.header("Modeling")
    years = st.number_input("Years of history", min_value=1, max_value=10, value=3, step=1)
    model_names = list(model_registry().keys())
    selected_model = st.selectbox("Model for predictions", model_names, index=model_names.index("RandomForest"))
    top_k = st.slider("Top K per sector", min_value=5, max_value=20, value=10)
    run_button = st.button("Run ranking", type="primary")


def merge_meta(scored: pd.DataFrame, meta: Dict[str, dict]) -> pd.DataFrame:
    meta_df = (
        pd.DataFrame.from_dict(meta, orient="index")
        .rename_axis("ticker")
        .reset_index()
    )
    merged = scored.merge(meta_df, on="ticker", how="left")
    merged["sector"] = merged["sector"].fillna("Unknown")
    merged["industry"] = merged["industry"].fillna("Unknown")
    return merged


def render_rank_table(df: pd.DataFrame, sector_filter: str, top_k: int) -> pd.DataFrame:
    if sector_filter != "ALL":
        df = df[df["sector"] == sector_filter]
    df = df.sort_values("predicted_forward_return", ascending=False).head(top_k).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    st.subheader(f"Top {top_k} in sector: {sector_filter}")
    st.dataframe(
        df[
            [
                "rank",
                "ticker",
                "company",
                "sector",
                "industry",
                "predicted_forward_return",
                "last_price",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )
    return df


if run_button:
    if not universe:
        st.error("Universe is empty. Provide a CSV or ensure the default file exists.")
    else:
        with st.spinner("Downloading data, training models, and ranking..."):
            prices = load_prices(universe, years)
            meta = load_meta(list(prices))

            selection = run_selection_pipeline(
                tickers=list(prices),
                years_of_history=years,
                model_name=selected_model,
            )
            scored = selection.scored_candidates.rename(columns={"predicted_forward_return": "pred_return_30d"})
            scored = merge_meta(scored, meta)
            scored = scored.rename(columns={"pred_return_30d": "predicted_forward_return"})

            models_cv = cross_validate_models(model_registry(), selection.training_features, selection.training_target)

        sectors = ["ALL"] + sorted(scored["sector"].dropna().unique().tolist())
        sector_filter = st.selectbox("Sector filter", sectors, index=0)
        top_table = render_rank_table(scored, sector_filter, top_k=top_k)

        with st.expander("Model performance board"):
            st.dataframe(models_cv, hide_index=True, use_container_width=True)
            chart = (
                alt.Chart(models_cv)
                .mark_bar()
                .encode(x="model", y="rmse", color="model")
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)

        if not top_table.empty:
            selected_ticker = st.selectbox("Select ticker for detail", top_table["ticker"])
            prices_df = prices.get(selected_ticker)
            if prices_df is not None:
                recent = prices_df.tail(252).reset_index()
                line = (
                    alt.Chart(recent)
                    .mark_line()
                    .encode(x="Date:T", y="Close:Q")
                    .properties(height=300)
                )
                st.altair_chart(line, use_container_width=True)

            feature_row = selection.latest_features.get(selected_ticker)
            if feature_row is not None:
                st.subheader("Key features")
                st.dataframe(feature_row.to_frame("value"))

                importance = permutation_importance_df(
                    selection.model, selection.training_features, selection.training_target
                )
                contrib = top_feature_contributions(feature_row, importance)
                st.subheader("Top contributing features")
                st.dataframe(contrib, hide_index=True)
                bar = (
                    alt.Chart(contrib)
                    .mark_bar()
                    .encode(x="feature", y="importance_mean")
                    .properties(height=300)
                )
                st.altair_chart(bar, use_container_width=True)

        st.subheader("Downloads")
        if sector_filter == "ALL":
            filtered_prices = prices
            filtered_predictions = scored
        else:
            sector_tickers = scored[scored["sector"] == sector_filter]["ticker"].tolist()
            filtered_prices = {t: prices[t] for t in sector_tickers if t in prices}
            filtered_predictions = scored[scored["sector"] == sector_filter]

        if filtered_prices:
            combined_prices = pd.concat(filtered_prices, names=["ticker", "date"])
            st.download_button(
                "Download OHLCV CSV",
                data=combined_prices.to_csv().encode(),
                file_name="ohlcv.csv",
                mime="text/csv",
            )

        feature_table = pd.DataFrame(selection.latest_features).T
        st.download_button(
            "Download feature table CSV",
            data=feature_table.to_csv().encode(),
            file_name="features.csv",
            mime="text/csv",
        )

        st.download_button(
            "Download predictions CSV",
            data=filtered_predictions.to_csv(index=False).encode(),
            file_name="predictions.csv",
            mime="text/csv",
        )
