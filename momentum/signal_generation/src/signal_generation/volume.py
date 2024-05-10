from typing import Optional

import pandas as pd

from core.constants import (
    AVG_DOLLAR_VOLUME_COL,
    VOLUME_ABOVE_MIN_COL,
    VOLUME_BELOW_MAX_COL,
    VOLUME_FILTER_COL,
)
from core.utils import apply_hysteresis
from data.constants import DOLLAR_VOLUME_COL, TICKER_COL
from signal_generation.common import ema, rolling_sum


def create_volume_signals(df: pd.DataFrame, periods_per_day: int = 1) -> pd.DataFrame:
    # Calculate volume features
    for lookback_days in [1]:
        periods = lookback_days * periods_per_day
        colname = f"{lookback_days}d_dollar_volume"
        df[colname] = rolling_sum(df, column=DOLLAR_VOLUME_COL, periods=periods)
    for lookback_days in [30]:
        periods = lookback_days * periods_per_day
        df[f"avg_1d_dollar_volume_over_{lookback_days}d"] = ema(
            df, column="1d_dollar_volume", periods=periods
        )

    return df


def create_volume_filter_mask(
    df, min_daily_volume: Optional[float], max_daily_volume: Optional[float]
) -> pd.DataFrame:
    # Initialize all rows to passing unfiltered
    df[VOLUME_ABOVE_MIN_COL] = True
    df[VOLUME_BELOW_MAX_COL] = True

    if min_daily_volume is not None:
        df = apply_hysteresis(
            df,
            group_col=TICKER_COL,
            value_col=AVG_DOLLAR_VOLUME_COL,
            output_col=VOLUME_ABOVE_MIN_COL,
            entry_threshold=min_daily_volume,
            exit_threshold=0.75 * min_daily_volume,
        )
    if max_daily_volume is not None:
        # below_max hysteresis needs to be flipped to be accurate
        df = apply_hysteresis(
            df,
            group_col=TICKER_COL,
            value_col=AVG_DOLLAR_VOLUME_COL,
            output_col=VOLUME_BELOW_MAX_COL,
            entry_threshold=max_daily_volume,
            exit_threshold=0.75 * max_daily_volume,
        )
        df[VOLUME_BELOW_MAX_COL] = ~df[VOLUME_BELOW_MAX_COL]

    # Create filtered column (True --> row should be filtered out)
    df[VOLUME_FILTER_COL] = ~((df[VOLUME_ABOVE_MIN_COL]) & (df[VOLUME_BELOW_MAX_COL]))

    return df
