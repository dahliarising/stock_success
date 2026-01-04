from __future__ import annotations

from io import BytesIO
from typing import List

import altair as alt
import pandas as pd
import streamlit as st

from stock_success.meta import meta_dataframe
from stock_success.models import model_registry
from stock_success.pipeline import run_pipeline
from stock_success.selection import filter_and_rank
from stock_success.universe import load_default_universe, load_universe_from_bytes
from stock_success.explain import permutation_importance_df, top_feature_contributions
from stock_success.eval import cross_validate_models
from stock_success.features import compute_features

st.set_page_config(page_title="Stock Success Dashboard", layout="wide")
st.title("현재 주가와 모델이 본 1년 후 가격")
st.caption("현재가, 예상 1년 수익률, 예상 1년 가격을 중심으로 설계")

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

st.set_page_config(page_title="Stock Success", layout="wide")
st.title("이 주식의 현재 가격과 1년 후 예상 가격은?")
st.caption("현재가, 예상 1년 수익률, 예상 1년 가격을 중심으로 설계된 대시보드")


@st.cache_data
def load_universe() -> List[str]:
    return load_default_universe()


with st.sidebar:
    st.header("분석 설정")
    universe_choice = st.radio("유니버스", ["S&P500 샘플", "사용자 CSV 업로드"], index=0)
    upload = None
    if universe_choice == "사용자 CSV 업로드":
        upload = st.file_uploader("티커 CSV 업로드", type=["csv"])
    model_name = st.selectbox("모델", options=list(model_registry().keys()), index=0)
    years_of_history = st.slider("히스토리(년)", 2, 10, 5)
    sector_filter = st.text_input("Sector 필터 (ALL 입력 시 전체)", value="ALL")
    industry_filter = st.text_input("Industry 필터 (ALL 입력 시 전체)", value="ALL")
    run_button = st.button("예측 실행", type="primary")


if not run_button:
    st.info("사이드바에서 설정 후 '예측 실행' 버튼을 누르세요.")
    st.stop()

if universe_choice == "사용자 CSV 업로드":
    if not upload:
        st.error("CSV를 업로드하세요.")
        st.stop()
    try:
        tickers = load_universe_from_bytes(upload.read())
    except Exception as exc:  # pragma: no cover - UI safeguard
        st.error(f"CSV 파싱 실패: {exc}")
        st.stop()
else:
    tickers = load_universe()

if not tickers:
    st.error("티커 리스트가 비어 있습니다.")
    st.stop()

with st.spinner("데이터 수집 및 예측 중..."):
    try:
        predictions, history, model, train_X, train_y, latest_features = run_pipeline(
            tickers,
            years_of_history=years_of_history,
            model_name=model_name,
        )
    except Exception as exc:  # pragma: no cover - UI safeguard
        st.error(f"파이프라인 실패: {exc}")
        st.stop()

meta_df = meta_dataframe(predictions["ticker"].tolist())
sectors = sorted([s for s in meta_df["sector"].dropna().unique() if s])
industries = sorted([i for i in meta_df["industry"].dropna().unique() if i])

filtered = filter_and_rank(
    predictions,
    sector=sector_filter if sector_filter else "ALL",
    industry=industry_filter if industry_filter else "ALL",
    top_k=10,
)

st.subheader("산업군 Top10")
st.dataframe(
    filtered,
    use_container_width=True,
    hide_index=True,
    column_config={
        "current_price": st.column_config.NumberColumn("현재가", format="$%.2f"),
        "predicted_price_1y": st.column_config.NumberColumn("예상 1년 가격", format="$%.2f"),
        "predicted_return_1y": st.column_config.NumberColumn("예상 1년 수익률", format="%.2f%%", help="모델 예측 1년 후 수익률"),
    },
)

if filtered.empty:
    st.warning("필터 결과가 없습니다.")
    st.stop()

selected_ticker = st.selectbox("티커 선택", options=filtered["ticker"].tolist())
current_row = predictions[predictions["ticker"] == selected_ticker].iloc[0]
feature_row = latest_features.get(selected_ticker)

col1, col2, col3 = st.columns(3)
col1.metric("현재 주가", f"${current_row['current_price']:.2f}")
col2.metric("예상 1년 수익률", f"{current_row['predicted_return_1y']*100:.2f}%")
col3.metric("예상 1년 가격", f"${current_row['predicted_price_1y']:.2f}")

price_history = history[selected_ticker].reset_index().rename(columns={"Date": "date"})
price_chart = (
    alt.Chart(price_history)
    .mark_line()
    .encode(x="date:T", y=alt.Y("Close:Q", title="가격"), tooltip=["date:T", "Close:Q"])
    .properties(height=300)
)
forecast_point = pd.DataFrame(
    {
        "date": [price_history["date"].max()],
        "predicted": [current_row["predicted_price_1y"]],
    }
)
pred_point_chart = alt.Chart(forecast_point).mark_point(color="red", size=80).encode(x="date:T", y="predicted:Q")

st.altair_chart(price_chart + pred_point_chart, use_container_width=True)

with st.expander("다운로드"):
    raw_csv = price_history.to_csv(index=False).encode("utf-8")
    st.download_button("Raw OHLCV CSV", data=raw_csv, file_name=f"{selected_ticker}_ohlcv.csv", mime="text/csv")

    if feature_row is not None:
        feature_csv = feature_row.reset_index(drop=True).to_csv(index=False).encode("utf-8")
        st.download_button("Feature CSV", data=feature_csv, file_name=f"{selected_ticker}_features.csv", mime="text/csv")

    pred_csv = predictions.to_csv(index=False).encode("utf-8")
    st.download_button("Prediction CSV", data=pred_csv, file_name="predictions.csv", mime="text/csv")


st.subheader("선정 근거")
tabs = st.tabs(["Feature 테이블", "Feature 중요도", "모델 성능"])

with tabs[0]:
    if feature_row is not None:
        st.dataframe(feature_row.T, use_container_width=True)
    else:
        st.info("특징을 계산할 수 없습니다.")

with tabs[1]:
    importance_df = permutation_importance_df(model, train_X, train_y, n_repeats=5)
    contrib = top_feature_contributions(feature_row.iloc[0], importance_df, top_n=10) if feature_row is not None else None
    st.bar_chart(importance_df.set_index("feature")["importance_mean"])
    if contrib is not None:
        st.dataframe(contrib, use_container_width=True)

with tabs[2]:
    eval_models = {name: mdl for name, mdl in model_registry().items() if name in ["Ridge", "RandomForest", "MLP"]}
    perf = cross_validate_models(eval_models, train_X, train_y, n_splits=3)
    st.dataframe(perf.sort_values("rmse"), use_container_width=True, hide_index=True)


st.caption("모든 예측은 교육용 예시입니다. 투자 결정은 본인 책임입니다.")
