from functools import cache

import numpy as np
import pandas as pd
import static_frame as sf
import statsmodels.api as sm
from scipy.interpolate import interp1d

from data.constants import TICKER_COL, TIMESTAMP_COL
from position_generation.constants import (
    FDM_COL,
    IDM_COL,
    IDM_REFRESH_PERIOD,
    SCALED_SIGNAL_COL,
)


@cache
def compute_idm(
    df: sf.FrameHE, feature_mat: sf.FrameHE, date_range: sf.SeriesHE
) -> pd.Series:
    df, feature_mat, date_range = (
        df.to_pandas(),
        feature_mat.to_pandas(),
        date_range.to_pandas(),
    )
    last_stamp_updated = None
    idm_lst = []
    for timestamp in date_range:
        timestamp_mask = df[TIMESTAMP_COL] == timestamp
        valid_signal_mask = ~df[SCALED_SIGNAL_COL].isna()
        tickers = df.loc[(timestamp_mask) & (valid_signal_mask)][TICKER_COL]
        if tickers.shape[0] == 0:
            # No valid positions
            idm_lst.append(np.nan)
            continue

        if (
            last_stamp_updated is None
            or (timestamp - last_stamp_updated) > IDM_REFRESH_PERIOD
        ):
            # Equal weight across all assets (risk parity)
            weights = np.full(tickers.shape, fill_value=1 / tickers.shape[0])
            # Replace negative correlations with 0
            corr = (
                feature_mat.loc[feature_mat[TIMESTAMP_COL] <= timestamp][tickers]
                .corr()
                .fillna(value=0)
                .to_numpy()
            )
            corr = np.clip(corr, 0, 1)
            idm_divisor = np.sqrt(weights.dot(corr).dot(weights.T))
            idm = 1 / idm_divisor if idm_divisor > 0 else np.nan
            last_stamp_updated = timestamp
        else:
            # Reuse previous idm
            idm = idm_lst[-1]
        idm_lst.append(idm)
    idm_ser = pd.Series(idm_lst, index=date_range, name=IDM_COL)
    return idm_ser


@cache
def compute_fdm(
    df: sf.FrameHE, feature_columns: tuple[str], date_range: sf.SeriesHE
) -> pd.Series:
    df, feature_columns, date_range = (
        df.to_pandas(),
        list(feature_columns),
        date_range.to_pandas(),
    )
    # For the FDM, we pool signals for each ticker before computing
    # the correlation matrix between signals (not between tickers)
    last_stamp_updated = None
    fdm_lst = []
    for timestamp in date_range:
        timestamp_mask = df[TIMESTAMP_COL] == timestamp
        valid_signal_mask = ~df[SCALED_SIGNAL_COL].isna()
        tickers = df.loc[(timestamp_mask) & (valid_signal_mask)][TICKER_COL]
        if tickers.shape[0] == 0:
            # No valid positions
            fdm_lst.append(np.nan)
            continue

        if (
            last_stamp_updated is None
            or (timestamp - last_stamp_updated) > IDM_REFRESH_PERIOD
        ):
            # Equal weight across all assets (risk parity)
            weights = np.full(len(feature_columns), fill_value=1 / len(feature_columns))
            # Replace negative correlations with 0
            corr = (
                df.loc[
                    (df[TIMESTAMP_COL] <= timestamp) & (df[TICKER_COL].isin(tickers))
                ][feature_columns]
                .corr()
                .fillna(value=0)
                .to_numpy()
            )
            corr = np.clip(corr, 0, 1)
            fdm_divisor = np.sqrt(weights.dot(corr).dot(weights.T))
            fdm = 1 / fdm_divisor if fdm_divisor > 0 else np.nan
            last_stamp_updated = timestamp
        else:
            # Reuse previous fdm
            fdm = fdm_lst[-1]
        fdm_lst.append(fdm)
    fdm_ser = pd.Series(fdm_lst, index=date_range, name=FDM_COL)
    return fdm_ser


def vol_target_scaling_dm(volatility_target: float):
    f = get_vol_target_scaling_lowess_fn()
    return float(f(volatility_target))


@cache
def get_vol_target_scaling_lowess_fn():
    vol_target_to_realized = {
        75.0: 48.11,
        80.0: 49.93,
        65.0: 44.7,
        70.0: 46.59,
        60.0: 42.72,
        85.0: 51.41,
        90.0: 52.77,
        55.0: 40.59,
        110.0: 57.52,
        105.0: 56.67,
        115.0: 58.4,
        95.0: 54.02,
        100.0: 55.49,
        50.0: 38.31,
        130.0: 59.88,
        45.0: 35.86,
        120.0: 59.2,
        135.0: 60.7,
        125.0: 59.96,
        40.0: 33.19,
        140.0: 61.64,
        35.0: 29.97,
        145.0: 62.05,
        30.0: 26.29,
        150.0: 62.13,
        25.0: 22.22,
        155.0: 62.61,
        20.0: 18.1,
        165.0: 63.3,
        15.0: 13.76,
        160.0: 62.67,
        170.0: 63.17,
        10.0: 9.2,
        5.0: 4.62,
        175.0: 63.3,
        180.0: 64.2,
        185.0: 64.24,
        195.0: 64.88,
        190.0: 64.4,
        200.0: 64.68,
    }
    vol_target_to_scaling = {
        k / 100: k / v
        for (k, v) in vol_target_to_realized.items()
        if (k >= 0.0 and k <= 100.0)
    }
    x, y = zip(*vol_target_to_scaling.items())

    # lowess will return our "smoothed" data with a y value for at every x-value
    lowess = sm.nonparametric.lowess(y, x, frac=0.3)

    # unpack the lowess smoothed points to their values
    lowess_x = list(zip(*lowess))[0]
    lowess_y = list(zip(*lowess))[1]

    # run scipy's interpolation.
    f = interp1d(lowess_x, lowess_y, bounds_error=False)
    return f
