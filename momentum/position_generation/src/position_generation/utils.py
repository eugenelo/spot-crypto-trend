from enum import Enum
import pandas as pd
from typing import Optional, List


class Direction(Enum):
    LongOnly = "LongOnly"
    ShortOnly = "ShortOnly"
    Both = "Both"


def nonempty_positions(
    df_positions: pd.DataFrame, tickers_to_keep: Optional[List[str]] = None
) -> pd.DataFrame:
    if tickers_to_keep is not None:
        tickers_to_keep_lst = tickers_to_keep.copy()
    else:
        tickers_to_keep_lst = []
    res = df_positions.sort_values(
        by=["timestamp", "rank"], ascending=[False, True]
    ).loc[
        (df_positions["scaled_position"] > 0.0)
        | (df_positions["ticker"].isin(tickers_to_keep_lst))
    ]
    return res[
        [
            "timestamp",
            "ticker",
            "30d_log_returns",
            "rank",
            "volume_consistent",
            "30d_num_days_volume_above_5M",
            "scaled_position",
        ]
    ]


def validate_positions(df: pd.DataFrame, freq: str) -> bool:
    # Ensure that no duplicate rows exist for (ticker, timestamp) combination
    assert not df.duplicated(subset=["ticker", "timestamp"], keep=False).any()

    # Ensure that no gaps exist between dates
    tickers = df["ticker"].unique()
    for ticker in tickers:
        df_ticker = df.loc[df["ticker"] == ticker]
        start_date = df_ticker["timestamp"].min()  # Start of your data
        end_date = df_ticker["timestamp"].max()  # End of your data
        full_date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        missing_dates = full_date_range.difference(df_ticker["timestamp"])
        assert missing_dates.empty

    return True
