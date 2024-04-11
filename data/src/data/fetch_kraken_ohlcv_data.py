import argparse
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import Optional, List
from pathlib import Path
import pytz
from tqdm.auto import tqdm

from data.constants import (
    TIMESTAMP_COL,
    TICKER_COL,
    VWAP_COL,
    VOLUME_COL,
    DOLLAR_VOLUME_COL,
    OHLC_COLUMNS,
    PRICE_COL,
    NUM_RETRY_ATTEMPTS,
)
from data.utils import (
    load_ohlc_to_daily_filtered,
    load_ohlc_to_hourly_filtered,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch Kraken OHLCV data")
    parser.add_argument(
        "--output_path", "-o", type=str, required=True, help="Output file path"
    )
    parser.add_argument(
        "--output_data_freq",
        "-f",
        type=str,
        help="Output data frequency for OHLC data",
    )
    parser.add_argument(
        "--lookback_days", "-l", type=int, help="Lookback period in days", default=45
    )
    parser.add_argument(
        "--append_output", "-a", help="Append to output file", action="store_true"
    )
    return parser.parse_args()


class KrakenOHLCV(ccxt.kraken):
    def parse_ohlcv(self, ohlcv, market=None) -> list:
        return [
            self.safe_timestamp(ohlcv, 0),  # timestamp
            self.safe_number(ohlcv, 1),  # open
            self.safe_number(ohlcv, 2),  # high
            self.safe_number(ohlcv, 3),  # low
            self.safe_number(ohlcv, 4),  # close
            self.safe_number(ohlcv, 5),  # vwap
            self.safe_number(ohlcv, 6),  # volume
        ]


def get_usd_symbols(kraken: ccxt.kraken) -> List[str]:
    # Get all available tickers on Kraken using ccxt to enable using unified symbols
    tickers = kraken.fetch_tickers()
    symbols = tickers.keys()
    symbols = [
        symbol
        for symbol in symbols
        if (symbol.endswith("USD") and not symbol.endswith("PYUSD"))
    ]
    return symbols


def fetch_ohlcv_data(
    kraken: ccxt.kraken, lookback: timedelta, timeframe: str
) -> pd.DataFrame:
    symbols = get_usd_symbols(kraken)
    print(f"{len(symbols)} valid USD pairs")

    # Define the timeframe (30 days)
    since = kraken.parse8601((datetime.now() - lookback).isoformat())

    # Fetch OHLCV (Open, High, Low, Close, Volume) data for each symbol
    ohlcvs = {}
    for symbol in tqdm(symbols):
        print(f"Fetching data for {symbol}")
        for attempt in range(NUM_RETRY_ATTEMPTS):
            try:
                ohlcvs[symbol] = kraken.fetch_ohlcv(
                    symbol, timeframe=timeframe, since=since
                )
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
    print(f"Fetched data for {len(ohlcvs)} valid USD pairs")

    # Process OHLCV data into a DataFrame
    ohlc_combined = pd.DataFrame(columns=OHLC_COLUMNS)
    ohlc_combined[TIMESTAMP_COL] = pd.to_datetime(
        ohlc_combined[TIMESTAMP_COL], utc=True, unit="ms"
    )
    for symbol, ohlcv in ohlcvs.items():
        df = pd.DataFrame(
            ohlcv,
            columns=OHLC_COLUMNS[:-2],
        )
        df[DOLLAR_VOLUME_COL] = df[VWAP_COL] * df[VOLUME_COL]
        df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True, unit="ms")
        df[TICKER_COL] = symbol
        ohlc_combined = pd.concat([ohlc_combined, df], ignore_index=True)
    return ohlc_combined


def main(args):
    # Initialize the Kraken exchange
    kraken = KrakenOHLCV()

    if args.output_data_freq is None:
        raise ValueError("Output data frequency must be specified.")
    timeframe = args.output_data_freq
    df_ohlc = fetch_ohlcv_data(
        kraken, lookback=timedelta(days=args.lookback_days), timeframe=timeframe
    )

    # Write to output path
    if args.append_output:
        # Load existing output file to filter duplicates and reindex
        if args.output_data_freq == "1d":
            df_existing_output = load_ohlc_to_daily_filtered(
                args.output_path,
                input_freq=args.output_data_freq,
                tz=pytz.UTC,
                whitelist_fn=None,
            )
        elif args.output_data_freq == "1h":
            df_existing_output = load_ohlc_to_hourly_filtered(
                args.output_path,
                input_freq=args.output_data_freq,
                tz=pytz.UTC,
                whitelist_fn=None,
            )
        assert df_ohlc.columns.equals(df_existing_output.columns)
        df_ohlc = pd.concat([df_existing_output, df_ohlc])
        df_ohlc.drop_duplicates(
            subset=[TIMESTAMP_COL, TICKER_COL],
            keep="last",  # Keep the latest available data
            inplace=True,
        )
        df_ohlc.sort_values(
            by=[TIMESTAMP_COL, TICKER_COL], ascending=True, inplace=True
        )
        df_ohlc.index = pd.RangeIndex(len(df_ohlc.index))
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_ohlc.to_csv(str(output_path), mode="w", index=False)
    print(f"Wrote {df_ohlc.shape} dataframe to '{output_path}'")

    return 0


if __name__ == "__main__":
    np.set_printoptions(linewidth=1000)
    pd.set_option("display.width", 2000)
    pd.set_option("display.precision", 3)
    pd.set_option("display.float_format", "{:.3f}".format)

    args = parse_args()
    exit(main(args))
