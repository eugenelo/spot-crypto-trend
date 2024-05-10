import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import ccxt
import numpy as np
import pandas as pd
import pytz
from tqdm.auto import tqdm

from ccxt_custom.kraken import KrakenExchange
from data.constants import (
    DATETIME_COL,
    DOLLAR_VOLUME_COL,
    NUM_RETRY_ATTEMPTS,
    OHLC_COLUMNS,
    TICKER_COL,
    VOLUME_COL,
    VWAP_COL,
)
from data.utils import (
    get_unified_symbols,
    get_usd_symbols,
    load_ohlc_to_daily_filtered,
    load_ohlc_to_hourly_filtered,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch Kraken OHLCV data")
    parser.add_argument(
        "--data_frequency",
        "-f",
        type=str,
        required=True,
        help="Frequency for OHLC data",
    )
    parser.add_argument(
        "--drop_last_row",
        "-d",
        action="store_true",
        help="Drop last row of fetched data (which will be incomplete)",
    )
    parser.add_argument("--ticker", "-t", type=str, help="Specific ticker")
    parser.add_argument(
        "--lookback_days", "-l", type=int, help="Lookback period in days", default=45
    )
    parser.add_argument("--output_path", "-o", type=str, help="Output file path")
    parser.add_argument(
        "--append_output", "-a", help="Append to output file", action="store_true"
    )
    return parser.parse_args()


def fetch_ohlcv_data(
    kraken: ccxt.kraken,
    symbol: str,
    lookback: timedelta,
    timeframe: str,
    drop_last_row: bool,
) -> Optional[pd.DataFrame]:
    # Define the timeframe (30 days)
    since = kraken.parse8601((datetime.now(tz=pytz.UTC) - lookback).isoformat())

    # Fetch OHLCV (Open, High, Low, Close, Volume) data for each symbol
    ohlcv = None
    for _attempt in range(NUM_RETRY_ATTEMPTS):
        try:
            ohlcv = kraken.fetch_ohlcv(symbol, timeframe=timeframe, since=since)
        except ccxt.NetworkError as e:
            print(f"{symbol} failed due to a network error: {str(e)}. Retrying!")
            continue
        except ccxt.ExchangeError as e:
            print(f"{symbol} failed due to exchange error: {str(e)}. Skipping!")
            break
        except Exception as e:
            print(f"{symbol} failed with unexpected error: {str(e)}. Skipping!")
            break
        else:
            break
    if ohlcv is None:
        return None

    # Process OHLCV data into a DataFrame
    df_ohlcv = pd.DataFrame(ohlcv, columns=OHLC_COLUMNS[:-2])
    df_ohlcv[DOLLAR_VOLUME_COL] = df_ohlcv[VWAP_COL] * df_ohlcv[VOLUME_COL]
    df_ohlcv[TICKER_COL] = symbol
    df_ohlcv[DATETIME_COL] = pd.to_datetime(df_ohlcv[DATETIME_COL], utc=True, unit="ms")

    if drop_last_row:
        # Drop the last row which will have incomplete data
        df_ohlcv.drop(df_ohlcv.tail(1).index, inplace=True)

    return df_ohlcv


def main(args):
    if args.data_frequency is None:
        raise ValueError("Data frequency must be specified.")

    # Initialize the Kraken exchange
    kraken = KrakenExchange(
        {
            "enableRateLimit": True,
        }
    )

    # Get ccxt symbols
    if args.ticker is not None:
        symbols = get_unified_symbols(kraken, tickers=[args.ticker])
    else:
        symbols = get_usd_symbols(kraken)
    print(f"{len(symbols)} valid USD pairs")

    # Fetch OHLC for each ticker, combine into a single DataFrame
    df_ohlcv = pd.DataFrame(columns=OHLC_COLUMNS)
    df_ohlcv[DATETIME_COL] = pd.to_datetime(df_ohlcv[DATETIME_COL], utc=True, unit="ms")
    for symbol in tqdm(symbols):
        print(f"Fetching data for {symbol}")
        df_symbol = fetch_ohlcv_data(
            kraken,
            symbol=symbol,
            lookback=timedelta(days=args.lookback_days),
            timeframe=args.data_frequency,
            drop_last_row=args.drop_last_row,
        )
        if df_symbol is not None:
            df_ohlcv = pd.concat([df_ohlcv, df_symbol], ignore_index=True)

    if args.output_path is not None:
        # Write to output path
        if args.append_output:
            t0 = time.time()
            # Load existing output file to filter duplicates and reindex
            if args.data_frequency == "1d":
                df_existing_output = load_ohlc_to_daily_filtered(
                    args.output_path,
                    input_freq=args.data_frequency,
                    tz=pytz.UTC,
                    whitelist_fn=None,
                )
            elif args.data_frequency == "1h":
                df_existing_output = load_ohlc_to_hourly_filtered(
                    args.output_path,
                    input_freq=args.data_frequency,
                    tz=pytz.UTC,
                    whitelist_fn=None,
                )
            assert df_ohlcv.columns.equals(df_existing_output.columns)
            t1 = time.time()
            print(
                f"Loaded existing dataframe at '{args.output_path}' in"
                f" {t1-t0:.2f} seconds"
            )
            df_ohlcv = pd.concat([df_existing_output, df_ohlcv])
            df_ohlcv.drop_duplicates(
                subset=[DATETIME_COL, TICKER_COL],
                keep="last",  # Keep the latest available data
                inplace=True,
            )
            df_ohlcv.sort_values(
                by=[DATETIME_COL, TICKER_COL], ascending=True, inplace=True
            )
            df_ohlcv.index = pd.RangeIndex(len(df_ohlcv.index))
            t2 = time.time()
            print(f"Appended existing and new dataframes in {t2-t1:.2f} seconds")
        t0 = time.time()
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_ohlcv.to_csv(str(output_path), mode="w", index=False)
        t1 = time.time()
        print(
            f"Wrote {df_ohlcv.shape} dataframe to '{output_path}' in"
            f" {t1-t0:.2f} seconds"
        )
    else:
        print(df_ohlcv)

    return 0


if __name__ == "__main__":
    np.set_printoptions(linewidth=1000)
    pd.set_option("display.width", 2000)
    pd.set_option("display.precision", 3)
    pd.set_option("display.float_format", "{:.3f}".format)

    args = parse_args()
    exit(main(args))
