import numpy as np
import pandas as pd
import pytz
import re
from typing import Callable

from core.constants import (
    in_universe_excl_stablecoins,
    in_shitcoin_trending_universe,
    in_mature_trending_universe,
)


def load_ohlc_to_daily_filtered(
    input_path: str, input_freq: str, tz: pytz.timezone, whitelist_fn: Callable
) -> pd.DataFrame:
    return _load_ohlc_to_dataframe_filtered(
        input_path=input_path,
        input_freq=input_freq,
        tz=tz,
        output_freq="1d",
        whitelist_fn=whitelist_fn,
    )


def load_ohlc_to_hourly_filtered(
    input_path: str, input_freq: str, tz: pytz.timezone, whitelist_fn: Callable
) -> pd.DataFrame:
    return _load_ohlc_to_dataframe_filtered(
        input_path=input_path,
        input_freq=input_freq,
        tz=tz,
        output_freq=input_freq,
        whitelist_fn=whitelist_fn,
    )


def _load_ohlc_to_dataframe_filtered(
    input_path: str,
    input_freq: str,
    tz: pytz.timezone,
    output_freq: str,
    whitelist_fn: Callable,
) -> pd.DataFrame:
    SUPPORTED_OUTPUT_FREQ = ["1h", "1d", input_freq]
    assert (
        output_freq in SUPPORTED_OUTPUT_FREQ
    ), f"Supported Output Frequency: {SUPPORTED_OUTPUT_FREQ}"

    # Load OHLC data from csv, always in UTC
    df = load_ohlc_csv(input_path)
    hourly_freq_pattern = re.compile(r"\d{1,2}h")
    input_freq_is_hourly = re.match(hourly_freq_pattern, input_freq)

    # Handle timezone
    if tz.zone != "UTC":
        if input_freq_is_hourly:
            # Relocalize to input timezone
            df.index = df.index.tz_convert(tz)
        else:
            print("Can't relocalize daily data! Ignoring input tz!")

    if input_freq == output_freq:
        # No resampling required
        pass
    elif re.match(hourly_freq_pattern, input_freq) and output_freq == "1d":
        # Resample hourly to daily
        df = resample_ohlc_hour_to_day(df)
    else:
        raise ValueError(
            f"Unsupported data frequency pair! input_freq={input_freq}, output_freq={output_freq}"
        )

    df = df.reset_index()
    df.sort_values(by=["ticker", "timestamp"], ascending=True, inplace=True)
    # Fill dates with no data using the previous day. Volume will still be 0.
    df.ffill(inplace=True)

    # Filter blacklisted symbol pairs
    df = filter_universe(df=df, whitelist_fn=whitelist_fn)

    # Validate data, expects timestamp to be a column and not the index
    if not validate_data(df=df, freq=output_freq):
        raise ValueError("Invalid data!")
    return df


def filter_universe(df: pd.DataFrame, whitelist_fn: Callable):
    df_filtered = df.loc[df["ticker"].apply(whitelist_fn)]
    return df_filtered


def load_ohlc_csv(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path, parse_dates=["timestamp"])[
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "vwap",
            "volume",
            "dollar_volume",
            "ticker",
        ]
    ]
    df.index = pd.to_datetime(df.pop("timestamp"), utc=True, format="mixed")
    return df


def resample_ohlc_hour_to_day(df_hourly: pd.DataFrame) -> pd.DataFrame:
    # Convert hourly to daily OHLC
    df_daily = (
        df_hourly.groupby("ticker")
        .resample("D")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "dollar_volume": "sum",
            }
        )
        .reset_index()
    )
    df_daily["timestamp"] = pd.to_datetime(df_daily["timestamp"])
    df_daily = df_daily.sort_values(by=["ticker", "timestamp"])
    return df_daily


def validate_data(df: pd.DataFrame, freq: str) -> bool:
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


def apply_hysteresis(
    df, group_col, value_col, output_col, entry_threshold, exit_threshold
):
    # Mark where value crosses entry and where it crosses exit
    df["above_entry"] = df[value_col] > entry_threshold
    df["below_exit"] = df[value_col] < exit_threshold

    # Determine points where state changes
    df["entry_point"] = df["above_entry"] & (~df["above_entry"].shift(1).fillna(False))
    df["exit_point"] = df["below_exit"] & (~df["below_exit"].shift(1).fillna(False))

    # Ensure group changes reset entry/exit points
    df["group_change"] = df[group_col] != df[group_col].shift(1)
    df["entry_point"] |= df["group_change"]
    df["exit_point"] &= ~df["group_change"]

    # Initialize hysteresis column
    df[output_col] = np.nan

    # Apply hysteresis logic: set to True at entry points and propagate until an exit point within each group
    df.loc[df["entry_point"], output_col] = True
    df.loc[df["exit_point"], output_col] = False

    # Forward fill within groups to propagate state, then backward fill initial NaNs if any
    df[output_col] = df.groupby(group_col)[output_col].ffill().bfill()

    # Drop helper columns
    df.drop(
        ["above_entry", "below_exit", "entry_point", "exit_point", "group_change"],
        axis=1,
        inplace=True,
    )

    return df
