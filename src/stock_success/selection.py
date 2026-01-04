"""산업군 필터링 및 Top10 선택 로직."""

from __future__ import annotations

import pandas as pd


def filter_and_rank(
    predictions: pd.DataFrame, sector: str = "ALL", industry: str = "ALL", top_k: int = 10
) -> pd.DataFrame:
    df = predictions.copy()
    if sector and sector != "ALL":
        df = df[df["sector"] == sector]
    if industry and industry != "ALL":
        df = df[df["industry"] == industry]

    df = df.sort_values("predicted_return_1y", ascending=False).head(top_k)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df[
        [
            "rank",
            "ticker",
            "company",
            "sector",
            "industry",
            "current_price",
            "predicted_return_1y",
            "predicted_price_1y",
            "predicted_return_std",
        ]
    ]
