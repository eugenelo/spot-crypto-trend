import pandas as pd
import polars as pl
import pytz
import re
from typing import Callable, Optional, List

from data.constants import (
    TIMESTAMP_COL,
    OPEN_COL,
    HIGH_COL,
    LOW_COL,
    CLOSE_COL,
    VWAP_COL,
    VOLUME_COL,
    DOLLAR_VOLUME_COL,
    TICKER_COL,
    OHLC_COLUMNS,
    ID_COL,
)


def load_ohlc_to_daily_filtered(
    input_path: str,
    input_freq: str,
    tz: pytz.timezone,
    whitelist_fn: Optional[Callable],
) -> pd.DataFrame:
    return _load_ohlc_to_dataframe_filtered(
        input_path=input_path,
        input_freq=input_freq,
        tz=tz,
        output_freq="1d",
        whitelist_fn=whitelist_fn,
    )


def load_ohlc_to_hourly_filtered(
    input_path: str,
    input_freq: str,
    tz: pytz.timezone,
    whitelist_fn: Optional[Callable],
) -> pd.DataFrame:
    return _load_ohlc_to_dataframe_filtered(
        input_path=input_path,
        input_freq=input_freq,
        tz=tz,
        output_freq=input_freq,
        whitelist_fn=whitelist_fn,
    )


def load_ohlc_csv(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path, parse_dates=[TIMESTAMP_COL])[OHLC_COLUMNS]
    df.index = pd.to_datetime(df.pop(TIMESTAMP_COL), utc=True, format="mixed")
    return df


def valid_ohlc_dataframe(df: pd.DataFrame, freq: str) -> bool:
    # Ensure that no duplicate rows exist for (ticker, timestamp) combination
    duplicate = df.duplicated(subset=[TICKER_COL, TIMESTAMP_COL], keep=False)
    if duplicate.any():
        print(f"Duplicate data: {df.loc[duplicate]}")
        return False

    # Ensure that no gaps exist between dates
    tickers = df[TICKER_COL].unique()
    for ticker in tickers:
        df_ticker = df.loc[df[TICKER_COL] == ticker]
        start_date = df_ticker[TIMESTAMP_COL].min()  # Start of your data
        end_date = df_ticker[TIMESTAMP_COL].max()  # End of your data
        full_date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        missing_dates = full_date_range.difference(df_ticker[TIMESTAMP_COL])
        if not missing_dates.empty:
            print(f"Missing Dates: {ticker} - {missing_dates}")
            return False

    return True


def filter_universe(df: pd.DataFrame, whitelist_fn: Callable) -> pd.DataFrame:
    df_filtered = df.loc[df[TICKER_COL].apply(whitelist_fn)]
    return df_filtered


def _load_ohlc_to_dataframe_filtered(
    input_path: str,
    input_freq: str,
    tz: pytz.timezone,
    output_freq: str,
    whitelist_fn: Optional[Callable],
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
        df = _resample_ohlc_hour_to_day(df)
    else:
        raise ValueError(
            f"Unsupported data frequency pair! input_freq={input_freq}, output_freq={output_freq}"
        )

    df = df.reset_index()
    df = df.sort_values(by=[TICKER_COL, TIMESTAMP_COL], ascending=True)

    # Filter blacklisted symbol pairs
    if whitelist_fn is not None:
        df = filter_universe(df=df, whitelist_fn=whitelist_fn)

    # Validate data, expects timestamp to be a column and not the index
    if not valid_ohlc_dataframe(df=df, freq=output_freq):
        raise ValueError("Invalid data!")
    return df


def _resample_ohlc_hour_to_day(df_hourly: pd.DataFrame) -> pd.DataFrame:
    # Convert hourly to daily OHLC
    df_daily = (
        df_hourly.groupby(TICKER_COL)
        .resample(rule="D", convention="start")
        .agg(
            {
                OPEN_COL: "first",
                HIGH_COL: "max",
                LOW_COL: "min",
                CLOSE_COL: "last",
                VOLUME_COL: "sum",
                DOLLAR_VOLUME_COL: "sum",
            }
        )
        .reset_index()
    )
    df_daily[VWAP_COL] = df_daily[DOLLAR_VOLUME_COL] / df_daily[VOLUME_COL]
    df_daily[TIMESTAMP_COL] = pd.to_datetime(df_daily[TIMESTAMP_COL])
    df_daily = df_daily.sort_values(by=[TICKER_COL, TIMESTAMP_COL])
    return df_daily


def missing_elements(lst: List[int]) -> List[int]:
    start, end = lst[0], lst[-1]
    return sorted(set(range(start, end + 1)).difference(lst))


def valid_tick_df_pandas(df: pd.DataFrame, combined: bool) -> bool:
    # Missing elements in sequence
    missing_ids = missing_elements(df[ID_COL].to_list())
    if len(missing_ids) > 0:
        print(f"Missing IDs: {missing_ids}")
        return False

    # Duplicate IDs
    duplicate = df.duplicated(subset=[ID_COL], keep=False)
    if duplicate.any():
        print(f"Duplicate data: \n{df.loc[duplicate]}")
        return False

    # Number of elements
    if combined:
        num_rows = df.shape[0]
        max_id = df[ID_COL].max()
        if num_rows != max_id:
            print(f"Num Rows Mismatch: num_rows ({num_rows}) != max_id ({max_id})")
            return False

    return True


def valid_tick_df_polars(df: pl.DataFrame, combined: bool) -> bool:
    # Missing elements in sequence
    missing_ids = missing_elements(df[ID_COL].to_list())
    if len(missing_ids) > 0:
        print(f"Missing IDs: {missing_ids}")
        return False

    # Duplicate IDs
    duplicate = df.filter(pl.any_horizontal(pl.col(ID_COL).is_duplicated()))
    if not duplicate.is_empty():
        print(f"Duplicate data: {duplicate}")
        return False

    # Number of elements
    if combined:
        num_rows = df.select(pl.count()).item()
        max_id = df.select(pl.max(ID_COL)).item()
        if num_rows != max_id:
            print(f"Num Rows Mismatch: num_rows ({num_rows}) != max_id ({max_id})")
            return False

    return True


def interpolate_missing_ids(ids: pd.Series) -> pd.Series:
    # Try correcting with both ffill and bfill
    for method in ("ffill", "bfill"):
        missing_ids = ids.isna()
        if method == "ffill":
            missing_cumsum = missing_ids.cumsum()
            interp_offset = missing_cumsum - missing_cumsum.where(
                ~missing_ids
            ).ffill().fillna(0)
            ids = ids.ffill() + interp_offset
        else:
            missing_cumsum = missing_ids[::-1].cumsum()[::-1]
            interp_offset = missing_cumsum - missing_cumsum.where(
                ~missing_ids
            ).bfill().fillna(0)
            ids = ids.bfill() - interp_offset

        try:
            ids = ids.astype(int)
            break
        except pd.errors.IntCastingNaNError:
            continue

    if ids.isna().any():
        # Interpolation failed
        raise RuntimeError("ID interpolation failed")
    return ids
