import pandas as pd
import plotly.express as px
import numpy as np


def create_analysis_signals(
    df_ohlc: pd.DataFrame, periods_per_day: int = 1
) -> pd.DataFrame:
    """Generate analysis signals.

    Args:
        df_ohlc (pd.DataFrame): OHLC dataframe
        periods_per_day (int): Frequency of OHLC data in periods per day (e.g. hourly data == 24, daily data == 1).

    Returns:
        pd.DataFrame: Dataframe with analysis signals.
    """
    df_signals = create_trading_signals(df_ohlc)

    # Calculate future returns
    for lookahead_days in [1, 5, 6, 7, 10]:
        colname = f"next_{lookahead_days}d_returns"
        periods = lookahead_days * periods_per_day
        df_signals[colname] = (
            df_signals.groupby("ticker")["close"]
            .pct_change(periods=periods)
            .shift(-periods)
        )
        colname = f"next_{lookahead_days}d_log_returns"
        df_signals[colname] = (
            df_signals.groupby("ticker")["log_returns"]
            .rolling(periods)
            .sum()
            .reset_index(0, drop=True)
            .shift(-periods)
        )

    # Add cols, remove NAs
    df_signals["month"] = pd.DatetimeIndex(df_signals["timestamp"]).month
    df_signals["year"] = pd.DatetimeIndex(df_signals["timestamp"]).year
    df_signals = df_signals.dropna()

    return df_signals


def create_trading_signals(
    df_ohlc: pd.DataFrame, periods_per_day: int = 1
) -> pd.DataFrame:
    """Generate trading signals.

    Args:
        df_ohlc (pd.DataFrame): OHLC dataframe
        periods_per_day (int): Frequency of OHLC data in periods per day (e.g. hourly data == 24, daily data == 1).

    Returns:
        pd.DataFrame: Dataframe with trading signals.
    """
    df_signals = df_ohlc.copy().sort_values(by=["ticker", "timestamp"], ascending=True)

    # Calculate returns
    df_signals["returns"] = df_signals.groupby("ticker")["close"].pct_change(periods=1)
    df_signals["log_returns"] = (
        np.log(df_signals["close"]).groupby(df_signals.ticker).diff()
    )

    # # Calculate EMAs.
    # df_signals["50d_ema"] = (
    #     df_signals.groupby("ticker")["close"]
    #     .ewm(com=50 * periods_per_day)
    #     .mean()
    #     .reset_index(0, drop=True)
    # )
    # df_signals["5d_ema"] = (
    #     df_signals.groupby("ticker")["close"]
    #     .ewm(com=5 * periods_per_day)
    #     .mean()
    #     .reset_index(0, drop=True)
    # )

    # Calculate volume features
    for lookback_days in [1, 30]:
        colname = f"{lookback_days}d_dollar_volume"
        df_signals[colname] = (
            df_signals.groupby("ticker")["dollar_volume"]
            .rolling(lookback_days * periods_per_day)
            .sum()
            .reset_index(0, drop=True)
        )
    df_signals["1d_volume_above_5M"] = df_signals["1d_dollar_volume"] >= 5e6
    df_signals["30d_num_days_volume_above_5M"] = (
        df_signals.groupby("ticker")["1d_volume_above_5M"]
        .rolling(30 * periods_per_day)
        .apply(lambda x: x[::periods_per_day].sum(), raw=True)
        .reset_index(0, drop=True)
    )
    df_signals["volume_consistent"] = df_signals["30d_num_days_volume_above_5M"] > 15

    # Calculate rolling historical returns
    for lookback_days in [15, 21, 30]:
        # Returns
        ret_colname = f"{lookback_days}d_returns"
        periods = lookback_days * periods_per_day
        df_signals[ret_colname] = df_signals.groupby("ticker")["close"].pct_change(
            periods=periods
        )
        # Log returns
        logret_colname = f"{lookback_days}d_log_returns"
        df_signals[logret_colname] = (
            df_signals.groupby("ticker")["log_returns"]
            .rolling(periods)
            .sum()
            .reset_index(0, drop=True)
        )
        # Quintiles
        quintile_colname = f"{lookback_days}d_log_quintile"
        df_signals[quintile_colname] = pd.qcut(
            df_signals[logret_colname], 5, labels=False
        )

    # Add cols, remove NAs
    df_signals["month"] = pd.DatetimeIndex(df_signals["timestamp"]).month
    df_signals["year"] = pd.DatetimeIndex(df_signals["timestamp"]).year
    df_signals = df_signals.dropna()

    return df_signals
