"""예측 모델 레지스트리."""

from __future__ import annotations

from typing import Dict

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor


def model_registry(random_state: int = 42) -> Dict[str, object]:
    return {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1,
        ),
        "MLP": MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=400, random_state=random_state),
    }


def instantiate_model(name: str, random_state: int = 42):
    registry = model_registry(random_state=random_state)
    if name not in registry:
        raise KeyError(f"Unknown model name: {name}")
    return registry[name]
