import pandas as pd

from core.constants import PRICE_COL_SIGNAL_GEN
from signal_generation.common import future_log_returns, future_returns, sort_dataframe


def create_future_return_signals(
    df_ohlc: pd.DataFrame, periods_per_day: int = 1
) -> pd.DataFrame:
    df = sort_dataframe(df_ohlc)

    # Calculate future returns
    for lookahead_days in [1, 5, 6, 7, 10, 14, 21, 28]:
        periods = lookahead_days * periods_per_day
        # Simple returns
        colname = f"next_{lookahead_days}d_returns"
        df[colname] = future_returns(df, column=PRICE_COL_SIGNAL_GEN, periods=periods)
        # Log returns
        colname = f"next_{lookahead_days}d_log_returns"
        df[colname] = future_log_returns(
            df, column=PRICE_COL_SIGNAL_GEN, periods=periods
        )

    return df
