# Stock Success

yfinance를 이용해 간단한 유니버스와 OHLCV 데이터를 다루는 헬퍼 모듈.

## 설치 및 실행 예시
```bash
python -m pip install -r requirements.txt

# 예시: 샘플 유니버스 로딩 후 OHLCV 다운로드
PYTHONPATH=src python - <<'PY'
from stock_success import fetch_ohlcv, load_default_universe

universe = load_default_universe()
print("Loaded", len(universe), "tickers")
print(universe[:5])

history = fetch_ohlcv(universe[:3], start="2024-01-01")
for t, frame in history.items():
    print(f"\n{t} sample:")
    print(frame.head())
PY
```

## 포함된 파일
- `data/universe_sample.csv`: `ticker` 단일 컬럼, 약 30개 티커 샘플.
- `src/stock_success/data.py`: `fetch_ohlcv` 함수로 yfinance에서 조정 OHLCV 다운로드.
- `src/stock_success/universe.py`: CSV/바이트에서 티커 유니버스를 파싱하는 유틸리티.

## 참고
- 기본 유니버스 경로: `data/universe_sample.csv`
