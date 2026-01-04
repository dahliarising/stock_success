import pytest

from stock_success.data import fetch_raw_data


def test_fetch_raw_data_empty_input_raises():
    with pytest.raises(ValueError):
        fetch_raw_data([], start="2020-01-01", end="2020-02-01")

