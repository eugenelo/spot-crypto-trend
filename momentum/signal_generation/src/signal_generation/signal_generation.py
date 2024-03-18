import pandas as pd
import plotly.express as px
import numpy as np

from core.utils import apply_hysteresis


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
    for lookahead_days in [1, 5, 6, 7, 10, 14, 21, 28]:
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
    df_signals["day"] = pd.DatetimeIndex(df_signals["timestamp"]).day
    df_signals["month"] = pd.DatetimeIndex(df_signals["timestamp"]).month
    df_signals["year"] = pd.DatetimeIndex(df_signals["timestamp"]).year

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

    # Calculate EMAs.
    multiplier = 1
    for lookback_days in [8, 16, 32, 24, 48, 96]:
        lookback_days = int(lookback_days * multiplier)
        df_signals[f"{lookback_days}d_ema"] = (
            df_signals.groupby("ticker")["close"]
            .ewm(span=lookback_days * periods_per_day)
            .mean()
            .reset_index(0, drop=True)
        )
    # Below calculations reference Rohrbach et. al. (2017)
    # Calculate x_k
    for k, pair in enumerate([(8, 24), (16, 48), (32, 96)]):
        s, l = int(pair[0] * multiplier), int(pair[1] * multiplier)
        df_signals[f"x_{k}"] = df_signals[f"{s}d_ema"] - df_signals[f"{l}d_ema"]
    # Calculate annualized volatility on prices
    for lookback_days in [30, 91, 182, 365]:
        lookback_days = int(lookback_days * multiplier)
        df_signals[f"close_{lookback_days}d_sd"] = (
            df_signals.groupby("ticker")["returns"]
            .rolling(lookback_days * periods_per_day)
            .std()
            .reset_index(0, drop=True)
        ) * np.sqrt(365 / lookback_days)
    # Calculate y_k (normalize)
    lookback_days = int(91 * multiplier)
    for k in [0, 1, 2]:
        df_signals[f"y_{k}"] = (
            df_signals[f"x_{k}"] / df_signals[f"close_{lookback_days}d_sd"]
        )
    # Calculate annualized volatility on y_k
    for k in [0, 1, 2]:
        for lookback_days in [365]:
            lookback_days = int(lookback_days * multiplier)
            df_signals[f"y_{k}_{lookback_days}d_sd"] = (
                df_signals.groupby("ticker")[f"y_{k}"]
                .rolling(lookback_days * periods_per_day)
                .std()
                .reset_index(0, drop=True)
            ) * np.sqrt(365 / lookback_days)
    # Calculate z_k (normalize) and u_k (response)
    lookback_days = int(365 * multiplier)
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
    df_signals = apply_hysteresis(
        df_signals,
        group_col="ticker",
        value_col="30d_num_days_volume_above_5M",
        output_col="volume_consistent",
        entry_threshold=15,
        exit_threshold=9,
    )

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
            df_signals[logret_colname], 10, labels=False
        )
    try:
        df_signals["trend_quintile"] = pd.qcut(
            df_signals["trend_signal"], 10, labels=False
        )
    except Exception:
        df_signals["trend_quintile"] = np.nan

    # Add cols, remove NAs
    df_signals["day"] = pd.DatetimeIndex(df_signals["timestamp"]).day
    df_signals["month"] = pd.DatetimeIndex(df_signals["timestamp"]).month
    df_signals["year"] = pd.DatetimeIndex(df_signals["timestamp"]).year

    return df_signals
