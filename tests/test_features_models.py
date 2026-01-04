import numpy as np
import pandas as pd

from stock_success.features import compute_features
from stock_success.models import instantiate_model


def _dummy_prices(n=120):
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": np.linspace(100, 120, n),
            "High": np.linspace(101, 121, n),
            "Low": np.linspace(99, 119, n),
            "Close": np.linspace(100, 130, n),
            "Volume": np.random.randint(1_000_000, 2_000_000, n),
        },
        index=dates,
    )


def test_compute_features_no_future_leak():
    prices = _dummy_prices()
    dataset = compute_features(prices, lookbacks=(10,), forecast_horizon=5)
    assert (dataset.features.index < prices.index.max()).all()
    assert dataset.target.index.equals(dataset.features.index)


def test_model_predict_shape():
    prices = _dummy_prices(150)
    dataset = compute_features(prices, lookbacks=(5,), forecast_horizon=5)
    model = instantiate_model("Ridge")
    model.fit(dataset.features, dataset.target)
    preds = model.predict(dataset.features.head(10))
    assert preds.shape == (10,)
