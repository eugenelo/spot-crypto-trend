"""Data and utilities for testing."""
from os.path import dirname, join
from typing import Optional

from pathlib import Path
import pandas as pd


def _read_file(filename):
    return pd.read_csv(
        filename,
        index_col=0,
        parse_dates=True,
        infer_datetime_format=True,
    )


def symbol_to_df(symbol: str, sector: str) -> Optional[pd.DataFrame]:
    path = Path("data", sector, f"{symbol}.csv")
    if not path.exists():
        print(f"Data for {symbol} does not exist at {path}!")
        return None
    return _read_file(str(path))


GM = _read_file("data/GM.csv")
"""DataFrame of daily NASDAQ:GM (General Motors) stock price data from 2010 to 2013."""

FORD = _read_file("data/F.csv")
"""DataFrame of daily NASDAQ:F (Ford) stock price data from 1972 to 2023."""

AAPL = _read_file("data/AAPL.csv")
"""DataFrame of daily NASDAQ:AAPL (Apple) stock price data from 2013 to 2023."""
