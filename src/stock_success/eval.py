"""Evaluation helpers."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


def hit_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def cross_validate_models(models: Dict[str, object], X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> pd.DataFrame:
    tscv = TimeSeriesSplit(n_splits=min(n_splits, len(X) - 1))
    records = []
    for name, model in models.items():
        fold_preds = []
        fold_truth = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            fold_preds.append(preds)
            fold_truth.append(y_test.to_numpy())
        if not fold_preds:
            continue
        preds_concat = np.concatenate(fold_preds)
        truth_concat = np.concatenate(fold_truth)
        rmse = float(np.sqrt(mean_squared_error(truth_concat, preds_concat)))
        mae = float(mean_absolute_error(truth_concat, preds_concat))
        hr = hit_rate(truth_concat, preds_concat)
        ic, _ = spearmanr(truth_concat, preds_concat)
        records.append({"model": name, "rmse": rmse, "mae": mae, "hit_rate": hr, "spearman_ic": ic})
    return pd.DataFrame(records)
