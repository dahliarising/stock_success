# Stock Success Dashboard

현재가와 모델이 예측한 1년 후 가격을 중심으로 종목을 선별하는 Streamlit 대시보드.

## 실행 방법
```bash
python -m pip install -r requirements.txt
PYTHONPATH=src python -m streamlit run app.py
```

## 데이터 소스
- 가격/거래량: `yfinance.download`
- 메타데이터(섹터/산업/회사명) 및 펀더멘털: `yfinance.Ticker.info`

## 주요 화면
- 사이드바: 유니버스/섹터/산업/모델/피처셋/호라이즌 선택, CSV 업로드
- 탭: Ranking, Ticker Detail, Explain, Model Performance, Downloads
- KPI: 현재가, 예상 1년 수익률, 예상 1년 가격, 불확실성

## 캐싱
- `st.cache_data`를 활용해 데이터 다운로드와 예측 결과를 캐싱
- `data/meta_cache.json`에 메타/펀더멘털 캐시 저장

## 테스트
```bash
pytest -q
```

## Glossary
- RMSE: 제곱근 평균제곱오차
- MAE: 평균절대오차
- IC (Spearman): 예측·실제 순위 상관
- RSI: 상대강도지수
- MACD: 이동평균 수렴·확산 지표
- ATR: 평균 진폭
- PE/PB: 밸류에이션 지표
- EPS: 주당순이익
- FCF: 잉여현금흐름
- D/E: 부채비율
- Sharpe: 위험 대비 성과

## Inspiration
- [Streamlit Finance Dashboard Examples](https://github.com/streamlit/streamlit-example)
- [Quantitative Finance Apps](https://github.com/plotly/dash-stock-tickers)
