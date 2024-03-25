import pandas as pd
import plotly.express as px
import numpy as np

from signal_generation.common import (
    sort_dataframe,
    volatility_ema,
    bins,
)
from signal_generation.constants import PRICE_COLUMN


def create_rohrbach_signals(
    df_ohlc: pd.DataFrame, periods_per_day: int = 1
) -> pd.DataFrame:
    """Generate trend signals from Rohrbach 2017 paper.

    Args:
        df_ohlc (pd.DataFrame): OHLC dataframe
        periods_per_day (int): Frequency of OHLC data in periods per day (e.g. hourly data == 24, daily data == 1).

    Returns:
        pd.DataFrame: Dataframe with trading signals.
    """
    df_signals = sort_dataframe(df_ohlc)

    # Below calculations reference Rohrbach et. al. (2017)
    # Calculate x_k
    for k, pair in enumerate([(8, 24), (16, 48), (32, 96)]):
        s, l = pair[0], pair[1]
        df_signals[f"x_{k}"] = df_signals[f"{s}d_ema"] - df_signals[f"{l}d_ema"]
    # Calculate realized 3-month normal volatility on prices
    for lookback_days in [91]:
        periods = lookback_days * periods_per_day
        df_signals[f"price_{lookback_days}d_vol"] = volatility_ema(
            df_signals, column=PRICE_COLUMN, periods=periods
        )
    # Calculate y_k (normalize)
    lookback_days = 91
    for k in [0, 1, 2]:
        df_signals[f"y_{k}"] = (
            df_signals[f"x_{k}"] / df_signals[f"price_{lookback_days}d_vol"]
        )
    # Calculate annualized volatility on y_k
    lookback_days = 365
    periods = lookback_days * periods_per_day
    for k in [0, 1, 2]:
        df_signals[f"y_{k}_{lookback_days}d_sd"] = volatility_ema(
            df_signals, column=f"y_{k}", periods=periods
        )
    # Calculate z_k (normalize) and u_k (response)
    lookback_days = 365
    u_k_denom = np.sqrt(2) * np.exp(-0.5)
    for k in [0, 1, 2]:
        df_signals[f"z_{k}"] = (
            df_signals[f"y_{k}"] / df_signals[f"y_{k}_{lookback_days}d_sd"]
        )
        # x * exp(-x^2 / 4)
        df_signals[f"u_{k}"] = (
            df_signals[f"z_{k}"]
            * np.exp(-np.square(df_signals[f"z_{k}"]) / 4)
            / u_k_denom
        )
        # Sigmoid
        df_signals[f"u_{k}_sigmoid"] = 2 / (1 + np.exp(-df_signals[f"z_{k}"])) - 1
    # Calculate weighted signal
    df_signals["trend_signal"] = (
        df_signals["u_0"] / 3 + df_signals["u_1"] / 3 + df_signals["u_2"] / 3
    )
    df_signals["trend_signal_sigmoid"] = (
        df_signals["u_0_sigmoid"] / 3
        + df_signals["u_1_sigmoid"] / 3
        + df_signals["u_2_sigmoid"] / 3
    )
    # Calculate deciles
    try:
        df_signals["trend_decile"] = bins(
            df_signals, column="trend_signal", num_bins=10
        )
        df_signals["trend_sigmoid_decile"] = bins(
            df_signals, column="trend_signal_sigmoid", num_bins=10
        )
    except Exception:
        df_signals["trend_decile"] = np.nan
        df_signals["trend_sigmoid_decile"] = np.nan

    return df_signals
