import pandas as pd

from stock_success.features import compute_features


def test_compute_features_returns_forward_target():
    dates = pd.date_range("2020-01-01", periods=200, freq="B")
    prices = pd.DataFrame(
        {
            "Open": range(200),
            "High": range(1, 201),
            "Low": range(0, 200),
            "Close": range(1, 201),
            "Volume": [1_000_000] * 200,
        },
        index=dates,
    )

    dataset = compute_features(prices, lookbacks=(20,), forecast_horizon=5)

    assert not dataset.features.empty
    assert "forward_return" not in dataset.features.columns
    assert (dataset.target.index == dataset.features.index).all()
    # Should have fewer rows than original due to lookback/shift
    assert len(dataset.features) < len(prices)

