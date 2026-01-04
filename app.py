from __future__ import annotations

from typing import List

import altair as alt
import pandas as pd
import streamlit as st

from stock_success.eval import cross_validate_models
from stock_success.explain import permutation_importance_df, top_feature_contributions
from stock_success.meta import meta_dataframe
from stock_success.models import model_registry
from stock_success.pipeline import run_pipeline
from stock_success.selection import filter_and_rank
from stock_success.universe import load_universe

st.set_page_config(page_title="Stock Success Dashboard", layout="wide")
st.title("현재 주가와 모델이 본 1년 후 가격")
st.caption("현재가, 예상 1년 수익률, 예상 1년 가격을 중심으로 설계")


@st.cache_data
def cached_run_pipeline(
    upload_bytes: bytes | None,
    years_of_history: int,
    model_name: str,
    feature_set: str,
    horizon_days: int,
) -> tuple[pd.DataFrame, dict, object, pd.DataFrame, pd.Series, dict]:
    tickers = load_universe(upload_bytes)
    return run_pipeline(
        tickers=tickers,
        years_of_history=years_of_history,
        model_name=model_name,
        forecast_horizon=horizon_days,
        feature_set=feature_set,
    )


with st.sidebar:
    st.header("입력")
    universe_choice = st.selectbox("유니버스", ["S&P500 샘플", "사용자 CSV 업로드"])
    upload = None
    if universe_choice == "사용자 CSV 업로드":
        upload = st.file_uploader("CSV 업로드", type=["csv"])
    model_name = st.selectbox("모델", list(model_registry().keys()))
    feature_set = st.selectbox("Feature Set", ["price", "fundamentals", "risk"], format_func=lambda x: {
        "price": "Price/Technical",
        "fundamentals": "+Fundamentals",
        "risk": "+Risk/Liquidity",
    }[x])
    horizon = st.selectbox("Horizon", [252, 30], format_func=lambda h: "1Y" if h == 252 else "30D")
    years_of_history = st.slider("데이터 히스토리(년)", 2, 10, 5)
    run_button = st.button("예측 실행", type="primary")


if not run_button:
    st.info("사이드바에서 설정 후 실행하세요.")
    st.stop()

if universe_choice == "사용자 CSV 업로드" and not upload:
    st.error("CSV를 업로드하세요.")
    st.stop()

with st.spinner("데이터 로딩 및 예측 중..."):
    predictions, history, model, train_X, train_y, latest_features = cached_run_pipeline(
        upload.read() if upload else None,
        years_of_history=years_of_history,
        model_name=model_name,
        feature_set=feature_set,
        horizon_days=horizon,
    )

meta = meta_dataframe(predictions["ticker"].tolist())
sectors = ["ALL"] + sorted(meta["sector"].fillna("Unknown").unique())
selected_sector = st.sidebar.selectbox("Sector", sectors)
filtered_industries = meta[meta["sector"] == selected_sector]["industry"].fillna("Unknown").unique()
industries = ["ALL"] + sorted(filtered_industries)
selected_industry = st.sidebar.selectbox("Industry", industries)

filtered = filter_and_rank(predictions, sector=selected_sector, industry=selected_industry, top_k=10)

ranking_tab, ticker_tab, explain_tab, perf_tab, download_tab = st.tabs(
    ["Ranking", "Ticker Detail", "Explain", "Model Performance", "Downloads"]
)

with ranking_tab:
    st.subheader("산업군 Top10")
    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
        column_config={
            "current_price": st.column_config.NumberColumn("현재가", format="$%.2f"),
            "predicted_price_1y": st.column_config.NumberColumn("예상 1년 가격", format="$%.2f"),
            "predicted_return_1y": st.column_config.NumberColumn("예상 1년 수익률", format="%.2f%%"),
            "predicted_return_std": st.column_config.NumberColumn("불확실성", format="%.4f"),
        },
    )

with ticker_tab:
    ticker = st.selectbox("티커 선택", filtered["ticker"].tolist())
    row = predictions[predictions["ticker"] == ticker].iloc[0]
    feature_row = latest_features.get(ticker)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("현재 주가", f"${row['current_price']:.2f}")
    col2.metric("예상 1년 수익률", f"{row['predicted_return_1y']*100:.2f}%")
    col3.metric("예상 1년 가격", f"${row['predicted_price_1y']:.2f}")
    col4.metric("예상 불확실성", f"{row['predicted_return_std']:.4f}")

    price_hist = history[ticker].reset_index().rename(columns={"Date": "date"})
    line = (
        alt.Chart(price_hist)
        .mark_line()
        .encode(x="date:T", y=alt.Y("Close:Q", title="가격"), tooltip=["date:T", "Close:Q"])
        .properties(height=300)
    )
    forecast_point = pd.DataFrame({"date": [price_hist["date"].max()], "pred": [row["predicted_price_1y"]]})
    point = alt.Chart(forecast_point).mark_point(color="red", size=80).encode(x="date:T", y="pred:Q")
    st.altair_chart(line + point, use_container_width=True)

    if feature_row is not None:
        st.dataframe(feature_row.T, use_container_width=True)

with explain_tab:
    st.subheader("Feature 중요도 및 선정 근거")
    importance_df = permutation_importance_df(model, train_X, train_y, n_repeats=3)
    st.bar_chart(importance_df.set_index("feature")["importance_mean"])
    ticker_choice = st.selectbox("티커 선택(선정 근거)", filtered["ticker"].tolist())
    feature_row = latest_features.get(ticker_choice)
    if feature_row is not None:
        contrib = top_feature_contributions(feature_row.iloc[0], importance_df, top_n=10)
        st.dataframe(contrib, use_container_width=True)

with perf_tab:
    st.subheader("모델 성능 비교")
    eval_models = {name: mdl for name, mdl in model_registry().items()}
    perf = cross_validate_models(eval_models, train_X, train_y, n_splits=3)
    st.dataframe(perf.sort_values("rmse"), use_container_width=True, hide_index=True)

with download_tab:
    st.subheader("다운로드")
    selected = st.selectbox("다운로드 티커", filtered["ticker"].tolist())
    price_csv = history[selected].reset_index().to_csv(index=False).encode("utf-8")
    st.download_button("Raw OHLCV CSV", price_csv, file_name=f"{selected}_ohlcv.csv", mime="text/csv")
    feature_row = latest_features.get(selected)
    if feature_row is not None:
        st.download_button(
            "Feature Table CSV",
            feature_row.reset_index(drop=True).to_csv(index=False).encode("utf-8"),
            file_name=f"{selected}_features.csv",
            mime="text/csv",
        )
    st.download_button(
        "Predictions CSV",
        predictions.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
    )

st.markdown("---")
st.subheader("Glossary")
st.markdown(
    """
- **RMSE**: 제곱근 평균제곱오차, 예측 오차의 규모를 나타냄
- **MAE**: 평균절대오차, 예측 오차의 절대값 평균
- **IC (Spearman)**: 예측과 실제 순위 간 상관계수, 랭킹 품질 지표
- **Directional Accuracy**: 방향성 적중률(상승/하락 일치 비율)
- **RSI**: 상대강도지수, 과매수/과매도 판단 지표
- **MACD**: 장단기 지수이동평균 차이로 추세 전환을 포착하는 지표
- **ATR**: Average True Range, 변동성 측정
- **PE/ PB**: 주가수익비/주가순자산비, 밸류에이션 지표
- **EPS**: 주당순이익, 수익성 지표
- **FCF**: Free Cash Flow, 현금 창출력 지표
- **D/E**: 부채비율, 재무 레버리지
- **Sharpe**: 위험 대비 초과수익률 지표
    """
)
