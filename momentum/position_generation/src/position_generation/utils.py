from enum import Enum
import pandas as pd
from typing import Optional, List

from data.constants import TIMESTAMP_COL, TICKER_COL
from core.constants import POSITION_COL


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
    res = df_positions.sort_values(by=[TIMESTAMP_COL], ascending=[False]).loc[
        (df_positions[POSITION_COL] > 0.0)
        | (df_positions[TICKER_COL].isin(tickers_to_keep_lst))
    ]
    return res


def validate_positions(df: pd.DataFrame, freq: str) -> bool:
    # Ensure that no duplicate rows exist for (ticker, timestamp) combination
    assert not df.duplicated(subset=[TICKER_COL, TIMESTAMP_COL], keep=False).any()

    # Ensure that no gaps exist between dates
    tickers = df[TICKER_COL].unique()
    for ticker in tickers:
        df_ticker = df.loc[df[TICKER_COL] == ticker]
        start_date = df_ticker[TIMESTAMP_COL].min()  # Start of your data
        end_date = df_ticker[TIMESTAMP_COL].max()  # End of your data
        full_date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        missing_dates = full_date_range.difference(df_ticker[TIMESTAMP_COL])
        assert missing_dates.empty

    return True
