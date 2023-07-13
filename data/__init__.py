"""Data and utilities for testing."""
import pandas as pd


def _read_file(filename):
    from os.path import dirname, join

    return pd.read_csv(
        join(dirname(__file__), filename),
        index_col=0,
        parse_dates=True,
        infer_datetime_format=True,
    )


GM = _read_file("GM.csv")
"""DataFrame of daily NASDAQ:GM (General Motors) stock price data from 2010 to 2013."""

FORD = _read_file("F.csv")
"""DataFrame of daily NASDAQ:F (Ford) stock price data from 1972 to 2023."""

AAPL = _read_file("AAPL.csv")
"""DataFrame of daily NASDAQ:AAPL (Apple) stock price data from 2013 to 2023."""
