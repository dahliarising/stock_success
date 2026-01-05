import pytest

from stock_success.data import fetch_ohlcv


def test_fetch_ohlcv_empty_input_raises():
    with pytest.raises(ValueError):
        fetch_ohlcv([], start="2020-01-01", end="2020-02-01")
