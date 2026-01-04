"""Explainability utilities for trained models."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def permutation_importance_df(model, X: pd.DataFrame, y: pd.Series, n_repeats: int = 10, random_state: int = 42) -> pd.DataFrame:
    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state)
    data = {
        "feature": X.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }
    df = pd.DataFrame(data).sort_values("importance_mean", ascending=False)
    return df


def top_feature_contributions(
    feature_row: pd.Series, importance: pd.DataFrame, top_n: int = 5
) -> pd.DataFrame:
    merged = importance.merge(feature_row.rename("value"), left_on="feature", right_index=True, how="left")
    merged["abs_contribution"] = merged["importance_mean"].abs() * merged["value"].abs()
    merged = merged.sort_values("abs_contribution", ascending=False).head(top_n)
    return merged[["feature", "value", "importance_mean", "importance_std", "abs_contribution"]]
