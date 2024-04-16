import numpy as np
import pandas as pd

from core.constants import PRICE_COL_SIGNAL_GEN
from signal_generation.common import bins, sort_dataframe, volatility_ema


def create_rohrbach_signals(
    df_ohlc: pd.DataFrame, periods_per_day: int = 1
) -> pd.DataFrame:
    """Generate trend signals from Rohrbach 2017 paper.

    Args:
        df_ohlc (pd.DataFrame): OHLC dataframe
        periods_per_day (int): Frequency of OHLC data in periods per day
                               (e.g. hourly data == 24, daily data == 1).

    Returns:
        pd.DataFrame: Dataframe with trading signals.
    """
    df_signals = sort_dataframe(df_ohlc)

    # Below calculations reference Rohrbach et. al. (2017)
    # Calculate x_k
    ema_window_pairs = [(4, 12), (8, 24), (16, 48), (32, 96), (64, 192)]
    for k, pair in enumerate(ema_window_pairs):
        short, long = pair[0], pair[1]
        df_signals[f"x_{k}"] = df_signals[f"{short}d_ema"] - df_signals[f"{long}d_ema"]

    # Calculate realized 3-month normal volatility on prices
    lookback_days = 91
    periods = lookback_days * periods_per_day
    df_signals[f"price_{lookback_days}d_vol"] = volatility_ema(
        df_signals, column=PRICE_COL_SIGNAL_GEN, periods=periods
    )
    # Calculate y_k (normalize)
    for k in range(len(ema_window_pairs)):
        df_signals[f"y_{k}"] = (
            df_signals[f"x_{k}"] / df_signals[f"price_{lookback_days}d_vol"]
        )

    # Calculate annualized volatility on y_k
    lookback_days = 365
    periods = lookback_days * periods_per_day
    for k in range(len(ema_window_pairs)):
        df_signals[f"y_{k}_{lookback_days}d_sd"] = volatility_ema(
            df_signals, column=f"y_{k}", periods=periods
        )
    # Calculate z_k (normalize) and u_k (response)
    u_k_denom = np.sqrt(2) * np.exp(-0.5)
    for k in range(len(ema_window_pairs)):
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
    df_signals["rohrbach_exponential"] = 0
    df_signals["rohrbach_sigmoid"] = 0
    weight = 1 / len(ema_window_pairs)
    for k in range(len(ema_window_pairs)):
        df_signals["rohrbach_exponential"] += df_signals[f"u_{k}"] * weight
        df_signals["rohrbach_sigmoid"] += df_signals[f"u_{k}_sigmoid"] * weight

    # Calculate deciles
    try:
        df_signals["trend_decile"] = bins(
            df_signals, column="rohrbach_exponential", num_bins=10
        )
        df_signals["trend_sigmoid_decile"] = bins(
            df_signals, column="rohrbach_sigmoid", num_bins=10
        )
    except Exception:
        df_signals["trend_decile"] = np.nan
        df_signals["trend_sigmoid_decile"] = np.nan

    return df_signals
