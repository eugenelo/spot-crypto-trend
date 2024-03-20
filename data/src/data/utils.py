import pandas as pd
import pytz
import re
from typing import Callable

from data.constants import OHLC_COLUMNS
from core.utils import filter_universe


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


def load_ohlc_csv(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path, parse_dates=["timestamp"])[OHLC_COLUMNS]
    df.index = pd.to_datetime(df.pop("timestamp"), utc=True, format="mixed")
    return df


def validate_data(df: pd.DataFrame, freq: str) -> bool:
    # Ensure that no duplicate rows exist for (ticker, timestamp) combination
    duplicate = df.duplicated(subset=["ticker", "timestamp"], keep=False)
    assert not duplicate.any(), df.loc[duplicate]

    # Ensure that no gaps exist between dates
    tickers = df["ticker"].unique()
    for ticker in tickers:
        df_ticker = df.loc[df["ticker"] == ticker]
        start_date = df_ticker["timestamp"].min()  # Start of your data
        end_date = df_ticker["timestamp"].max()  # End of your data
        full_date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        missing_dates = full_date_range.difference(df_ticker["timestamp"])
        assert missing_dates.empty, f"{ticker}: {missing_dates}"

    return True


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

    resampled_data = False
    if input_freq == output_freq:
        # No resampling required
        pass
    elif re.match(hourly_freq_pattern, input_freq) and output_freq == "1d":
        # Resample hourly to daily
        df = _resample_ohlc_hour_to_day(df)
        resampled_data = True
    else:
        raise ValueError(
            f"Unsupported data frequency pair! input_freq={input_freq}, output_freq={output_freq}"
        )

    df = df.reset_index()
    df = df.sort_values(by=["ticker", "timestamp"], ascending=True)
    if resampled_data:
        # Fill dates with no data using the previous day. Volume will still be 0.
        df = df.ffill()

    # Filter blacklisted symbol pairs
    df = filter_universe(df=df, whitelist_fn=whitelist_fn)

    # Validate data, expects timestamp to be a column and not the index
    if not validate_data(df=df, freq=output_freq):
        raise ValueError("Invalid data!")
    return df


def _resample_ohlc_hour_to_day(df_hourly: pd.DataFrame) -> pd.DataFrame:
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
    df_daily["vwap"] = df_daily["dollar_volume"] / df_daily["volume"]
    df_daily["timestamp"] = pd.to_datetime(df_daily["timestamp"])
    df_daily = df_daily.sort_values(by=["ticker", "timestamp"])
    return df_daily
