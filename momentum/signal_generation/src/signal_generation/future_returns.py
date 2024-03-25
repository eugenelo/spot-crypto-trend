import pandas as pd
import numpy as np

from signal_generation.common import (
    sort_dataframe,
    future_returns,
    future_log_returns,
)
from signal_generation.constants import PRICE_COLUMN


def create_future_return_signals(
    df_ohlc: pd.DataFrame, periods_per_day: int = 1
) -> pd.DataFrame:
    df = sort_dataframe(df_ohlc)

    # Calculate future returns
    for lookahead_days in [1, 5, 6, 7, 10, 14, 21, 28]:
        periods = lookahead_days * periods_per_day
        # Simple returns
        colname = f"next_{lookahead_days}d_returns"
        df[colname] = future_returns(df, column=PRICE_COLUMN, periods=periods)
        # Log returns
        colname = f"next_{lookahead_days}d_log_returns"
        df[colname] = future_log_returns(df, column=PRICE_COLUMN, periods=periods)

    return df
