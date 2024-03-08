import pandas as pd
import pytz
import re

from constants import blacklisted


def load_ohlc_to_daily_filtered(
    input_path: str, input_freq: str, tz: pytz.timezone
) -> pd.DataFrame:
    df = load_ohlc_csv(input_path)
    freq_pattern = re.compile(r"\d{1,2}h")
    if re.match(freq_pattern, input_freq):
        if tz.zone != "UTC":
            # Relocalize to input timezone before converting to daily data
            df.index = df.index.tz_convert(tz)
        df_daily = resample_ohlc_hour_to_day(df)
    elif input_freq == "1d":
        if tz.zone != "UTC":
            print("Can't relocalize daily data! Ignoring input tz!")
        df_daily = df.reset_index()
    else:
        raise ValueError("Unsupported data frequency")
    df_daily.sort_values(by=["ticker", "timestamp"], ascending=True, inplace=True)
    # Filter blacklisted symbol pairs
    return filter_blacklisted_pairs(df_daily)


def load_ohlc_to_hourly_filtered(
    input_path: str, input_freq: str, tz: pytz.timezone
) -> pd.DataFrame:
    df = load_ohlc_csv(input_path)
    if input_freq == "1h":
        if tz.zone != "UTC":
            # Relocalize to input timezone before converting to daily data
            df.index = df.index.tz_convert(tz)
        df_hourly = df.reset_index()
    else:
        raise ValueError("Unsupported data frequency")
    df_hourly.sort_values(by=["ticker", "timestamp"], ascending=True, inplace=True)
    # Filter blacklisted symbol pairs
    return filter_blacklisted_pairs(df_hourly)


def filter_blacklisted_pairs(df: pd.DataFrame):
    df_filtered = df.loc[
        df["ticker"].apply(lambda pair: not blacklisted(pair))
    ].sort_values(by=["timestamp"])
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
