import numpy as np
import pandas as pd

from core.constants import LOG_RETURNS_COL, PRICE_COL_SIGNAL_GEN, RETURNS_COL
from data.constants import DATETIME_COL
from signal_generation.common import (
    bins,
    ema,
    log_returns,
    returns,
    sort_dataframe,
    volatility_ema,
)


def create_historical_return_signals(
    df_ohlc: pd.DataFrame, periods_per_day: int = 1
) -> pd.DataFrame:
    df = sort_dataframe(df_ohlc)

    # Calculate returns
    df[RETURNS_COL] = returns(df, column=PRICE_COL_SIGNAL_GEN, periods=1)
    df[LOG_RETURNS_COL] = log_returns(df, column=PRICE_COL_SIGNAL_GEN, periods=1)

    # Calculate EMAs.
    for lookback_days in [2, 6] + list(range(4, 192 + 1, 4)):
        periods = lookback_days * periods_per_day
        df[f"{lookback_days}d_ema"] = ema(
            df, column=PRICE_COL_SIGNAL_GEN, periods=periods
        )
    # Calculate annualized volatility on returns
    for lookback_days in [30, 182, 365]:
        periods = lookback_days * periods_per_day
        df[f"returns_{lookback_days}d_vol"] = volatility_ema(
            df, column=RETURNS_COL, periods=periods
        ) * np.sqrt(365 / periods_per_day)

    # Calculate rolling historical returns
    for lookback_days in [7, 15, 21, 30]:
        # Returns
        ret_colname = f"{lookback_days}d_returns"
        periods = lookback_days * periods_per_day
        df[ret_colname] = returns(df, column=PRICE_COL_SIGNAL_GEN, periods=periods)
        # Log returns
        logret_colname = f"{lookback_days}d_log_returns"
        df[logret_colname] = log_returns(
            df, column=PRICE_COL_SIGNAL_GEN, periods=periods
        )
        # Quintiles
        decile_colname = f"{lookback_days}d_returns_decile"
        df[decile_colname] = bins(
            df, column=ret_colname, num_bins=10, duplicates="drop"
        )
        decile_colname = f"{lookback_days}d_log_returns_decile"
        df[decile_colname] = bins(
            df, column=logret_colname, num_bins=10, duplicates="drop"
        )

    # Add helper cols
    df["day"] = pd.DatetimeIndex(df[DATETIME_COL]).day
    df["month"] = pd.DatetimeIndex(df[DATETIME_COL]).month
    df["year"] = pd.DatetimeIndex(df[DATETIME_COL]).year

    return df
