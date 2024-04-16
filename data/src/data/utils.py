import re
from typing import Callable, List, Optional

import ccxt
import pandas as pd
import polars as pl
import pytz

from data.constants import (
    CLOSE_COL,
    DOLLAR_VOLUME_COL,
    HIGH_COL,
    ID_COL,
    LOW_COL,
    OHLC_COLUMNS,
    OPEN_COL,
    TICKER_COL,
    TIMESTAMP_COL,
    VOLUME_COL,
    VWAP_COL,
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

    # Ensure no NaNs
    return not df.isnull().values.any()


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
        df = df.reset_index()
    elif re.match(hourly_freq_pattern, input_freq) and output_freq == "1d":
        # Resample hourly to daily
        df = _resample_ohlc_hour_to_day(df)
        # Fill in days with no transactions
        df = df.reset_index()
        df = fill_missing_ohlc(df)
    else:
        raise ValueError(
            f"Unsupported data frequency pair! input_freq={input_freq},"
            f" output_freq={output_freq}"
        )
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


def fill_missing_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure data is sorted chronologically
    df = df.sort_values(by=[TICKER_COL, TIMESTAMP_COL], ascending=True)

    # Fill in close first to avoid having to shift
    df[CLOSE_COL] = df.groupby(TICKER_COL)[CLOSE_COL].ffill()
    df[OPEN_COL] = df[OPEN_COL].fillna(df[CLOSE_COL])
    df[HIGH_COL] = df[HIGH_COL].fillna(df[CLOSE_COL])
    df[LOW_COL] = df[LOW_COL].fillna(df[CLOSE_COL])
    df[VOLUME_COL] = df[VOLUME_COL].fillna(0)
    df[DOLLAR_VOLUME_COL] = df[DOLLAR_VOLUME_COL].fillna(0)
    df[VWAP_COL] = df[VWAP_COL].fillna(df[CLOSE_COL])

    return df


def missing_elements(lst: List[int]) -> List[int]:
    start, end = lst[0], lst[-1]
    return sorted(set(range(start, end + 1)).difference(lst))


MIN_ID = {
    "HFT/USD": 3,
    "DEFAULT": 1,
}


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
        ticker = df[TICKER_COL].min()
        min_id = df[ID_COL].min()
        expected_min_id = MIN_ID.get(ticker, MIN_ID["DEFAULT"])
        if min_id != expected_min_id:
            print(
                f"Incorrect min_id, min_id ({min_id}) != expected ({expected_min_id})"
            )
            return False

        num_rows = df.shape[0]
        max_id = df[ID_COL].max()
        expected_num_rows = int(max_id - min_id + 1)
        if num_rows != expected_num_rows:
            print(
                f"Num Rows Mismatch: num_rows ({num_rows}) != expected"
                f" ({expected_num_rows})"
            )
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
        ticker = df.select(pl.min(TICKER_COL)).item()
        min_id = df.select(pl.min(ID_COL)).item()
        expected_min_id = MIN_ID.get(ticker, MIN_ID["DEFAULT"])
        if min_id != expected_min_id:
            print(
                f"Incorrect min_id, min_id ({min_id}) != expected ({expected_min_id})"
            )
            return False

        num_rows = df.select(pl.count()).item()
        max_id = df.select(pl.max(ID_COL)).item()
        expected_num_rows = int(max_id - min_id + 1)
        if num_rows != expected_num_rows:
            print(
                f"Num Rows Mismatch: num_rows ({num_rows}) != expected"
                f" ({expected_num_rows})"
            )
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


def get_unified_symbols(exchange: ccxt.Exchange, tickers: List[str]) -> List[str]:
    # Convert ticker to unified symbol
    exchange.load_markets()
    symbols = []
    for ticker in tickers:
        try:
            market = exchange.markets_by_id[ticker][0]
            ccxt_symbol = market["symbol"]
        except Exception:
            # TODO(@eugene.lo): This only works for $X/USD pairs
            print(f"Failed to find {ticker} on ccxt! Converting manually.")
            ticker = ticker.replace("/", "")
            ccxt_symbol = ticker[:-3] + "/" + ticker[-3:]
        print(f"{ticker} -> {ccxt_symbol}")
        symbols.append(ccxt_symbol)
    return symbols


def get_usd_symbols(exchange: ccxt.Exchange) -> List[str]:
    # Get all available tickers on exchange using ccxt to enable using unified symbols
    tickers = exchange.fetch_tickers()
    symbols = tickers.keys()
    symbols = [
        symbol
        for symbol in symbols
        if (symbol.endswith("USD") and not symbol.endswith("PYUSD"))
    ]
    return symbols
