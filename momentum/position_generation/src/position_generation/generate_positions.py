import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List

from position_generation.utils import Direction
from position_generation.constants import (
    VOL_SHORT_COL,
    VOL_LONG_COL,
    VOL_FORECAST_COL,
    VOL_TARGET_COL,
    VOL_SCALING_COL,
    ABS_SIGNAL_AVG_COL,
    RANK_COL,
    SCALED_SIGNAL_COL,
    NUM_UNIQUE_ASSETS_COL,
    NUM_LONG_ASSETS_COL,
    NUM_SHORT_ASSETS_COL,
    NUM_KEPT_ASSETS_COL,
    MAX_ABS_POSITION_SIZE_COL,
)
from core.constants import (
    TIMESTAMP_COL,
    TICKER_COL,
    POSITION_COL,
)
from signal_generation.common import cross_sectional_abs_ema
from signal_generation.volume import create_volume_filter_mask


def generate_positions(
    df: pd.DataFrame,
    signal: str,
    periods_per_day: int,
    direction: Direction,
    vol_target: Optional[float],
    cross_sectional_percentage: Optional[float],
    cross_sectional_equal_weight: Optional[bool],
    min_daily_volume: Optional[float],
    max_daily_volume: Optional[float],
) -> pd.DataFrame:
    # Ensure that no duplicate rows exist for (ticker, timestamp) combination
    assert not df.duplicated(subset=[TICKER_COL, TIMESTAMP_COL], keep=False).any()
    df = df.copy()

    df[VOL_FORECAST_COL] = generate_volatility_forecast(df)

    # Construct trade signal
    scaled_signal_col = SCALED_SIGNAL_COL.format(signal=signal)
    df[scaled_signal_col] = df[signal]
    # Don't take positions in assets where we can't estimate volatility
    df.loc[df[VOL_FORECAST_COL].isna(), scaled_signal_col] = np.nan

    # Apply volume filter
    df = create_volume_filter_mask(
        df, min_daily_volume=min_daily_volume, max_daily_volume=max_daily_volume
    )
    df.loc[df["filter_volume"], scaled_signal_col] = np.nan

    # Generate signal ranks
    rank_col = RANK_COL.format(signal=scaled_signal_col)
    df[rank_col] = generate_signal_rank(df, signal=scaled_signal_col)

    # Get number of investable assets in universe per timestamp. Assets which are filtered out
    # from consideration are denoted by np.nan values.
    df[NUM_UNIQUE_ASSETS_COL] = generate_num_unique_assets(df, signal=scaled_signal_col)

    # Scale signal s.t. the absolute average is equal to 1 (to achieve risk target on average)
    # TODO(@eugene.lo): This should probably come after all of the filters, somehow.
    df = scale_signal_avg_to_1(
        df, signal=scaled_signal_col, periods_per_day=periods_per_day
    )

    if cross_sectional_percentage is not None:
        if direction == Direction.Both:
            assert (
                cross_sectional_percentage <= 0.5
            ), "cross_sectional_percentage can't be greater than 0.5 when direction is Both!"
        # Apply cross-sectional filter
        df = apply_cross_sectional_filter(
            df,
            signal=scaled_signal_col,
            rank_col=rank_col,
            cross_sectional_percentage=cross_sectional_percentage,
            cross_sectional_equal_weight=cross_sectional_equal_weight or False,
        )
    else:
        # No cross-sectional filter, could theoretically get long/short our entire universe
        df[NUM_LONG_ASSETS_COL] = df[NUM_UNIQUE_ASSETS_COL]
        df[NUM_SHORT_ASSETS_COL] = df[NUM_UNIQUE_ASSETS_COL]
    df[NUM_KEPT_ASSETS_COL] = df[NUM_LONG_ASSETS_COL] + df[NUM_SHORT_ASSETS_COL]
    df[NUM_KEPT_ASSETS_COL] = df[[NUM_KEPT_ASSETS_COL, NUM_UNIQUE_ASSETS_COL]].max(
        axis=1
    )

    # Apply direction constraints
    df = apply_direction_constraint(df, signal=scaled_signal_col, direction=direction)

    # Position sizing
    if vol_target is not None:
        # Volatility targeting
        df[VOL_TARGET_COL] = vol_target / df[NUM_KEPT_ASSETS_COL]
        df[VOL_SCALING_COL] = df[VOL_TARGET_COL] / df[VOL_FORECAST_COL]
        df[POSITION_COL] = df[scaled_signal_col] * df[VOL_SCALING_COL]
    else:
        # Scale in proportion to signal and number of assets in universe.
        df[POSITION_COL] = df[scaled_signal_col] / df[NUM_KEPT_ASSETS_COL]

    # Leverage constraints + Cap wild sizing from very low volatility estimates
    df = cap_position_size(df, direction=direction, leverage=1)

    # Lag positions by one day, avoid cheating w/ future information
    df[POSITION_COL] = df[POSITION_COL].shift(1)

    return df


def generate_volatility_forecast(df: pd.DataFrame) -> pd.Series:
    df_tmp = df.copy()

    # Generate volatility forecast as a blend of short-term and long-term volatility
    df_tmp.loc[
        (~df_tmp[VOL_LONG_COL].isna()) & (~df_tmp[VOL_SHORT_COL].isna()),
        VOL_FORECAST_COL,
    ] = (
        0.3 * df_tmp[VOL_LONG_COL] + 0.7 * df_tmp[VOL_SHORT_COL]
    )
    # Use short-term estimate if long-term is not available
    df_tmp.loc[
        (df_tmp[VOL_LONG_COL].isna()) & (~df_tmp[VOL_SHORT_COL].isna()),
        VOL_FORECAST_COL,
    ] = df_tmp[VOL_SHORT_COL]
    df_tmp.loc[
        (df_tmp[VOL_LONG_COL].isna()) & (df_tmp[VOL_SHORT_COL].isna()), VOL_FORECAST_COL
    ] = np.nan

    return df_tmp[VOL_FORECAST_COL]


def generate_signal_rank(df: pd.DataFrame, signal: str) -> pd.Series:
    return (
        df.loc[~df[signal].isna()]
        .groupby([TIMESTAMP_COL])[signal]
        .rank(method="dense", ascending=False)
        .astype(int)
        .fillna(np.inf)
    )


def generate_num_unique_assets(df: pd.DataFrame, signal: str) -> pd.Series:
    return (
        df.loc[~df[signal].isna()]
        .groupby(TIMESTAMP_COL)[TICKER_COL]
        .transform(pd.Series.nunique)
    )


def apply_cross_sectional_filter(
    df: pd.DataFrame,
    signal: str,
    rank_col: str,
    cross_sectional_percentage: float,
    cross_sectional_equal_weight: bool,
) -> pd.DataFrame:
    # Keep only top/bottom cross-sectional %
    df[NUM_LONG_ASSETS_COL] = np.round(
        cross_sectional_percentage * df[NUM_UNIQUE_ASSETS_COL]
    )
    df[NUM_SHORT_ASSETS_COL] = np.round(
        cross_sectional_percentage * df[NUM_UNIQUE_ASSETS_COL]
    )
    isna_mask = df[rank_col].isna()

    # Long top bins
    long_mask = (~isna_mask) & (df[rank_col] <= df[NUM_LONG_ASSETS_COL])

    # Short bottom bins
    short_mask = (~isna_mask) & (
        df.loc[df[rank_col] != np.inf][rank_col].max() - df[rank_col]
        < df[NUM_SHORT_ASSETS_COL]
    )

    # Apply equal weight to included bins
    if cross_sectional_equal_weight:
        df.loc[long_mask, signal] = 1.0
        df.loc[short_mask, signal] = -1.0

    # Exclude middle bins
    excluded_mask = (~isna_mask) & (~long_mask) & (~short_mask)
    df.loc[excluded_mask, signal] = 0.0
    return df


def apply_direction_constraint(
    df: pd.DataFrame, direction: Direction, signal: str
) -> pd.DataFrame:
    if direction == Direction.LongOnly:
        df[NUM_SHORT_ASSETS_COL] = 0
        df.loc[df[signal] < 0, signal] = 0.0
    elif direction == Direction.ShortOnly:
        df[NUM_LONG_ASSETS_COL] = 0
        df.loc[df[signal] > 0, signal] = 0.0
    return df


def scale_signal_avg_to_1(
    df: pd.DataFrame, signal: str, periods_per_day: int
) -> pd.DataFrame:
    abs_signal_avg_col = ABS_SIGNAL_AVG_COL.format(signal=signal)
    df[abs_signal_avg_col] = cross_sectional_abs_ema(
        df, column=signal, days=180, periods_per_day=periods_per_day
    )
    df[signal] /= df[abs_signal_avg_col]
    return df


def cap_position_size(df: pd.DataFrame, direction: Direction, leverage: float):
    if direction == Direction.LongOnly:
        df[MAX_ABS_POSITION_SIZE_COL] = leverage / df[NUM_LONG_ASSETS_COL]
    elif direction == Direction.ShortOnly:
        df[MAX_ABS_POSITION_SIZE_COL] = leverage / df[NUM_SHORT_ASSETS_COL]
    else:
        df[MAX_ABS_POSITION_SIZE_COL] = leverage / df[NUM_KEPT_ASSETS_COL]
    df[POSITION_COL] = np.clip(
        df[POSITION_COL],
        -df[MAX_ABS_POSITION_SIZE_COL],
        df[MAX_ABS_POSITION_SIZE_COL],
    )
    return df
