from functools import partial
from typing import Callable, Optional

import numpy as np
import pandas as pd
import static_frame as sf

from core.constants import PAST_7D_RETURNS_COL, POSITION_COL, VOLUME_FILTER_COL
from data.constants import TICKER_COL, TIMESTAMP_COL
from position_generation.constants import (
    ABS_SIGNAL_AVG_COL,
    DM_30D_EMA_COL,
    DM_COL,
    FDM_30D_EMA_COL,
    FDM_COL,
    IDM_30D_EMA_COL,
    IDM_COL,
    MAX_ABS_POSITION_SIZE_COL,
    NUM_KEPT_ASSETS_COL,
    NUM_LONG_ASSETS_COL,
    NUM_OPEN_LONG_POSITIONS_COL,
    NUM_OPEN_POSITIONS_COL,
    NUM_OPEN_SHORT_POSITIONS_COL,
    NUM_SHORT_ASSETS_COL,
    NUM_UNIQUE_ASSETS_COL,
    POSITION_SCALING_FACTOR_COL,
    RANK_COL,
    SCALED_SIGNAL_COL,
    VOL_FORECAST_COL,
    VOL_LONG_COL,
    VOL_SHORT_COL,
    VOL_TARGET_COL,
)
from position_generation.diversification_multipliers import (
    compute_fdm,
    compute_idm,
    vol_target_scaling_dm,
)
from position_generation.utils import Direction
from signal_generation.common import cross_sectional_abs_ema, sort_dataframe
from signal_generation.volume import create_volume_filter_mask


def get_generate_positions_fn(
    params: dict, periods_per_day: int, lag_positions: bool
) -> Callable:
    if params["signal"].startswith("rohrbach"):
        signal = params["signal"]
        direction = Direction(params["direction"])
        volatility_target = params.get("volatility_target", None)
        cross_sectional_percentage = params.get("cross_sectional_percentage", None)
        cross_sectional_equal_weight = params.get("cross_sectional_equal_weight", False)
        min_daily_volume = params.get("min_daily_volume", None)
        max_daily_volume = params.get("max_daily_volume", None)
        leverage = params.get("leverage", 1.0)

        generate_positions_fn = partial(
            generate_positions,
            signal=signal,
            periods_per_day=periods_per_day,
            direction=direction,
            volatility_target=volatility_target,
            cross_sectional_percentage=cross_sectional_percentage,
            cross_sectional_equal_weight=cross_sectional_equal_weight,
            min_daily_volume=min_daily_volume,
            max_daily_volume=max_daily_volume,
            leverage=leverage,
            lag_positions=lag_positions,
        )
    else:
        raise ValueError(
            f"Unsupported 'generate_positions' argument: {params['generate_positions']}"
        )
    return generate_positions_fn


def generate_positions(
    df: pd.DataFrame,
    signal: str,
    periods_per_day: int,
    direction: Direction,
    volatility_target: Optional[float],
    cross_sectional_percentage: Optional[float],
    cross_sectional_equal_weight: Optional[bool],
    min_daily_volume: Optional[float],
    max_daily_volume: Optional[float],
    leverage: float,
    lag_positions: bool,
) -> pd.DataFrame:
    # Ensure that no duplicate rows exist for (ticker, timestamp) combination
    assert not df.duplicated(subset=[TICKER_COL, TIMESTAMP_COL], keep=False).any()
    df = sort_dataframe(df.copy())

    df[VOL_FORECAST_COL] = generate_volatility_forecast(df)

    # Construct trade signal
    df[SCALED_SIGNAL_COL] = df[signal]

    # Scale each signal s.t. the absolute average is equal to 1 to achieve risk
    # target on average. This comes before any filters are applied.
    df = scale_signal_avg_to_1(
        df, signal=SCALED_SIGNAL_COL, periods_per_day=periods_per_day
    )

    # Don't take positions in assets where we can't estimate volatility
    df.loc[df[VOL_FORECAST_COL].isna(), SCALED_SIGNAL_COL] = np.nan

    # Apply volume filter
    df = create_volume_filter_mask(
        df, min_daily_volume=min_daily_volume, max_daily_volume=max_daily_volume
    )
    df.loc[df[VOLUME_FILTER_COL], SCALED_SIGNAL_COL] = np.nan

    # Generate signal ranks
    rank_col = RANK_COL.format(signal=SCALED_SIGNAL_COL)
    df[rank_col] = generate_signal_rank(df, signal=SCALED_SIGNAL_COL)

    # Get number of investable assets in universe per timestamp. Assets which
    # are filtered out from consideration are denoted by np.nan values.
    df[NUM_UNIQUE_ASSETS_COL] = get_num_unique_assets(df)

    # Compute diversification multipliers. This comes after all filters have
    # been applied. Use static frames to hash results for identical inputs.
    idm_cols = [TIMESTAMP_COL, TICKER_COL, SCALED_SIGNAL_COL]
    idm_df = sf.FrameHE.from_pandas(df[idm_cols])
    returns_matrix = sf.FrameHE.from_pandas(
        pd.pivot_table(
            df,
            index=TIMESTAMP_COL,
            columns=TICKER_COL,
            values=PAST_7D_RETURNS_COL,
            dropna=False,
        ).reset_index()
    )
    full_date_range = sf.SeriesHE(
        df.sort_values(TIMESTAMP_COL, ascending=True)[TIMESTAMP_COL]
        .unique()
        .to_pydatetime()
    )
    idm_ser = compute_idm(
        idm_df, feature_mat=returns_matrix, date_range=full_date_range
    )
    idm_30d_ema = idm_ser.ewm(span=30, adjust=True, ignore_na=False).mean()
    fdm_feature_cols = [f"u_{k}" for k in range(5)]
    fdm_df = sf.FrameHE.from_pandas(df[idm_cols + fdm_feature_cols])
    fdm_ser = compute_fdm(
        fdm_df, feature_columns=tuple(fdm_feature_cols), date_range=full_date_range
    )
    fdm_30d_ema = fdm_ser.ewm(span=30, adjust=True, ignore_na=False).mean()
    dm_combined_ser = idm_ser * fdm_ser
    dm_30d_ema = dm_combined_ser.ewm(span=30, adjust=True, ignore_na=False).mean()
    dm_df = (
        pd.DataFrame.from_dict(
            {
                IDM_COL: idm_ser,
                IDM_30D_EMA_COL: idm_30d_ema,
                FDM_COL: fdm_ser,
                FDM_30D_EMA_COL: fdm_30d_ema,
                DM_COL: dm_combined_ser,
                DM_30D_EMA_COL: dm_30d_ema,
            }
        )
        .reset_index()
        .rename(columns={"index": TIMESTAMP_COL})
    )
    df = df.merge(dm_df, how="left", on=TIMESTAMP_COL)

    # Apply cross-sectional weighting, if applicable
    if cross_sectional_percentage is not None:
        if direction == Direction.Both:
            assert cross_sectional_percentage <= 0.5, (
                "cross_sectional_percentage can't be greater than 0.5 when direction is"
                " Both!"
            )
        df = apply_cross_sectional_filter(
            df,
            signal=SCALED_SIGNAL_COL,
            rank_col=rank_col,
            cross_sectional_percentage=cross_sectional_percentage,
            cross_sectional_equal_weight=cross_sectional_equal_weight or False,
        )
    else:
        # No cross-sectional filter, could theoretically get long/short
        # the entire universe
        df[NUM_LONG_ASSETS_COL] = df[NUM_UNIQUE_ASSETS_COL]
        df[NUM_SHORT_ASSETS_COL] = df[NUM_UNIQUE_ASSETS_COL]
    df[NUM_KEPT_ASSETS_COL] = df[NUM_LONG_ASSETS_COL] + df[NUM_SHORT_ASSETS_COL]
    df[NUM_KEPT_ASSETS_COL] = df[[NUM_KEPT_ASSETS_COL, NUM_UNIQUE_ASSETS_COL]].min(
        axis=1
    )

    # Apply direction constraints
    df = apply_direction_constraint(df, signal=SCALED_SIGNAL_COL, direction=direction)

    # Position sizing
    df = get_num_open_positions(df)
    if volatility_target is not None:
        vol_target_scaling = vol_target_scaling_dm(volatility_target)
        # Volatility targeting
        df[VOL_TARGET_COL] = (
            volatility_target / df[NUM_KEPT_ASSETS_COL] * vol_target_scaling
        )
        # df[VOL_TARGET_COL] = volatility_target / df[NUM_OPEN_POSITIONS_COL]
        df[POSITION_SCALING_FACTOR_COL] = df[VOL_TARGET_COL] / df[VOL_FORECAST_COL]
    else:
        # Scale in proportion to signal and number of assets in universe.
        df[POSITION_SCALING_FACTOR_COL] = 1 / df[NUM_OPEN_POSITIONS_COL]
    df[POSITION_SCALING_FACTOR_COL] *= df[DM_30D_EMA_COL]
    df[POSITION_COL] = df[SCALED_SIGNAL_COL] * df[POSITION_SCALING_FACTOR_COL]

    # Leverage constraints + Cap wild sizing from very low volatility estimates
    df = cap_position_size(df, direction=direction, leverage=leverage)

    if lag_positions:
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
    # Disqualify ticker if short-term estimate is not available
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


def get_num_unique_assets(df: pd.DataFrame) -> pd.Series:
    return (
        df.loc[~df[SCALED_SIGNAL_COL].isna()]
        .groupby(TIMESTAMP_COL)[TICKER_COL]
        .transform(pd.Series.nunique)
    )


def get_num_open_positions(df: pd.DataFrame) -> pd.DataFrame:
    # Log open positions
    df[NUM_OPEN_LONG_POSITIONS_COL] = df.groupby(TIMESTAMP_COL)[
        SCALED_SIGNAL_COL
    ].transform(lambda x: (x > 0).sum())
    df[NUM_OPEN_SHORT_POSITIONS_COL] = df.groupby(TIMESTAMP_COL)[
        SCALED_SIGNAL_COL
    ].transform(lambda x: (x < 0).sum())
    df[NUM_OPEN_POSITIONS_COL] = (
        df[NUM_OPEN_LONG_POSITIONS_COL] + df[NUM_OPEN_SHORT_POSITIONS_COL]
    )
    return df


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


def cap_position_size(
    df: pd.DataFrame, direction: Direction, leverage: float
) -> pd.DataFrame:
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
