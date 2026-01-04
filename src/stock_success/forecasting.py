"""Deep-learning forecaster for long-horizon price projections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


class PriceDataset(Dataset):
    def __init__(self, series: np.ndarray, window: int):
        self.series = series.astype(np.float32)
        self.window = window

    def __len__(self) -> int:
        return len(self.series) - self.window

    def __getitem__(self, idx: int):
        window = self.series[idx : idx + self.window]
        target = self.series[idx + self.window]
        return window, target


class GRUForecaster(nn.Module):
    def __init__(self, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, 1))

    def forward(self, x):
        output, _ = self.gru(x)
        last = output[:, -1, :]
        return self.head(last)


@dataclass
class ForecastResult:
    history: pd.Series
    forecast: pd.Series
    model: GRUForecaster


def train_and_forecast(
    prices: pd.DataFrame,
    window: int = 60,
    forecast_days: int = 252,
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    hidden_size: int = 64,
    device: Optional[str] = None,
) -> ForecastResult:
    """Train a GRU forecaster on closing prices and predict the next ``forecast_days`` values."""
    close = prices["Close"].to_numpy()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PriceDataset(close, window=window)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = GRUForecaster(hidden_size=hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    model.train()
    for _ in tqdm(range(epochs), desc="Training forecaster"):
        for batch_x, batch_y in loader:
            batch_x = batch_x.unsqueeze(-1).to(device)
            batch_y = batch_y.unsqueeze(-1).to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    history = close.tolist()
    forecast_values: List[float] = []
    window_series = history[-window:]
    for _ in tqdm(range(forecast_days), desc="Forecasting horizon"):
        input_window = torch.tensor(window_series[-window:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        with torch.no_grad():
            next_price = model(input_window).item()
        forecast_values.append(next_price)
        window_series.append(next_price)

    forecast_index = pd.date_range(prices.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq="B")
    return ForecastResult(
        history=pd.Series(history, index=prices.index),
        forecast=pd.Series(forecast_values, index=forecast_index),
        model=model,
    )
