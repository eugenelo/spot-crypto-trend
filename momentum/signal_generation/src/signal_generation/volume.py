import pandas as pd
import numpy as np
from typing import Optional

from signal_generation.common import (
    ema,
    rolling_sum,
)
from core.utils import apply_hysteresis


def create_volume_signals(df: pd.DataFrame, periods_per_day: int = 1) -> pd.DataFrame:
    df = df.copy()

    # Calculate volume features
    for lookback_days in [1]:
        periods = lookback_days * periods_per_day
        colname = f"{lookback_days}d_dollar_volume"
        df[colname] = rolling_sum(df, column="dollar_volume", periods=periods)
    for lookback_days in [30]:
        periods = lookback_days * periods_per_day
        df[f"avg_1d_dollar_volume_over_{lookback_days}d"] = ema(
            df, column="1d_dollar_volume", periods=periods
        )

    return df


def create_volume_filter_mask(
    df, min_daily_volume: Optional[float], max_daily_volume: Optional[float]
) -> pd.Series:
    df_tmp = df.copy()

    # Initialize all rows to passing unfiltered
    volume_above_min = "volume_above_min"
    df_tmp[volume_above_min] = True
    volume_below_max = "volume_below_max"
    df_tmp[volume_below_max] = True

    dollar_volume_col = "1d_dollar_volume"
    if min_daily_volume is not None:
        df_tmp = apply_hysteresis(
            df_tmp,
            group_col="ticker",
            value_col=dollar_volume_col,
            output_col=volume_above_min,
            entry_threshold=min_daily_volume,
            exit_threshold=0.75 * min_daily_volume,
        )
    # Create negative dollar volume column to apply hysteresis to max
    negative_dollar_volume_col = "negative_1d_dollar_volume"
    df_tmp[negative_dollar_volume_col] = -df_tmp[dollar_volume_col]
    if max_daily_volume is not None:
        df_tmp = apply_hysteresis(
            df_tmp,
            group_col="ticker",
            value_col=negative_dollar_volume_col,
            output_col=volume_below_max,
            entry_threshold=-max_daily_volume,
            exit_threshold=-1.25 * max_daily_volume,
        )
    # Create filtered column (True --> row should be filtered out)
    return (~df_tmp[volume_above_min]) | (~df_tmp[volume_below_max])
