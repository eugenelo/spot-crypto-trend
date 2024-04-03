import numpy as np
import pandas as pd
import pytz
from datetime import timedelta
import re
from typing import Callable


def filter_universe(df: pd.DataFrame, whitelist_fn: Callable):
    df_filtered = df.loc[df["ticker"].apply(whitelist_fn)]
    return df_filtered


def apply_hysteresis(
    df, group_col, value_col, output_col, entry_threshold, exit_threshold
):
    # Mark where value crosses entry and where it crosses exit
    df["above_entry"] = df[value_col] > entry_threshold
    df["below_exit"] = df[value_col] < exit_threshold

    # Determine points where state changes
    with pd.option_context("future.no_silent_downcasting", True):
        df["entry_point"] = df["above_entry"] & (
            ~df.groupby(group_col)["above_entry"].shift(1).fillna(False)
        )
        df["exit_point"] = df["below_exit"] & (
            ~df.groupby(group_col)["below_exit"].shift(1).fillna(False)
        )

        # Initialize hysteresis column, use pd.NA to set dtype = object
        df[output_col] = pd.NA

        # Apply hysteresis logic: set to True at entry points and propagate until an exit point within each group
        df.loc[df["entry_point"], output_col] = True
        df.loc[df["exit_point"], output_col] = False

        # Forward fill within groups to propagate state, then backward fill initial NaNs if any
        df[output_col] = df.groupby(group_col)[output_col].ffill().fillna(False)

        # Drop helper columns
        df.drop(
            ["above_entry", "below_exit", "entry_point", "exit_point"],
            axis=1,
            inplace=True,
        )

    return df


def get_periods_per_day(timestamp_series: pd.Series):
    timestamp_series = timestamp_series.copy()
    timestamp_series = timestamp_series.sort_values(ascending=True)
    timestamp_series.index = pd.RangeIndex(len(timestamp_series.index))
    # Count number of periods between first entry in timestamp series and 1 day after
    start_time = timestamp_series.min()
    end_time = start_time + timedelta(days=1)
    start_idx = timestamp_series.loc[timestamp_series == start_time].index[0]
    end_idx = timestamp_series.loc[timestamp_series == end_time].index[0]
    return end_idx - start_idx
