import argparse
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import Optional
from pathlib import Path
import pytz

import sys

from data.utils import load_ohlc_to_daily_filtered, load_ohlc_to_hourly_filtered
from data.constants import OHLC_COLUMNS
from position_generation.v1 import (
    generate_positions_v1,
)
from position_generation.utils import (
    nonempty_positions,
)
from signal_generation.signal_generation import create_trading_signals
from signal_generation.constants import SignalType
from core.utils import filter_universe
from core.constants import in_universe_excl_stablecoins


def parse_args():
    parser = argparse.ArgumentParser(description="Kraken API wrapper")
    parser.add_argument("mode", help="[data, positions, pnl]")
    parser.add_argument("--input_path", "-i", type=str, help="Input data file path")
    parser.add_argument(
        "--input_data_freq", "-if", type=str, help="Input data frequency"
    )
    parser.add_argument(
        "--output_data_freq",
        "-f",
        type=str,
        help="Output data frequency",
    )
    parser.add_argument(
        "--lookback_days", "-l", type=int, help="Lookback period in days", default=45
    )
    parser.add_argument("--timezone", "-t", type=str, help="Timezone", default="UTC")
    parser.add_argument(
        "--account_size",
        "-s",
        type=float,
        help="(Target) Account size for position sizing or PNL calculation",
        default=np.inf,
    )
    parser.add_argument("--output_path", "-o", type=str, help="Output file path")
    parser.add_argument(
        "--append_output", "-a", help="Append to output file", action="store_true"
    )
    return parser.parse_args()


class MyKraken(ccxt.kraken):
    def parse_ohlcv(self, ohlcv, market=None) -> list:
        #
        #     [
        #         1591475640,
        #         "0.02500",
        #         "0.02500",
        #         "0.02500",
        #         "0.02500",
        #         "0.02500",
        #         "9.12201000",
        #         5
        #     ]
        #
        return [
            self.safe_timestamp(ohlcv, 0),  # timestamp
            self.safe_number(ohlcv, 1),  # open
            self.safe_number(ohlcv, 2),  # high
            self.safe_number(ohlcv, 3),  # low
            self.safe_number(ohlcv, 4),  # close
            self.safe_number(ohlcv, 5),  # vwap
            self.safe_number(ohlcv, 6),  # volume
        ]


def fetch_data(
    kraken: ccxt.kraken, lookback: timedelta, timeframe: str
) -> pd.DataFrame:
    # # Get all available symbols (tickers) on Kraken using Kraken public API
    # resp = requests.get("https://api.kraken.com/0/public/Ticker")
    # tickers = resp.json()["result"]
    # symbols = tickers.keys()
    # Filter out invalid symbols with no volume (delisted?)
    # symbols = [
    #     symbol
    #     for symbol in symbols
    #     if symbol.endswith("USD") and float(tickers[symbol]["v"][0]) > 0.0
    # ]
    # Get all available tickers on Kraken using ccxt to enable using unified symbols
    tickers = kraken.fetch_tickers()
    symbols = tickers.keys()
    symbols = [symbol for symbol in symbols if symbol.endswith("USD")]
    print(f"{len(symbols)} valid USD pairs")

    # Define the timeframe (30 days)
    since = kraken.parse8601((datetime.now() - lookback).isoformat())

    # Fetch OHLCV (Open, High, Low, Close, Volume) data for each symbol
    ohlcvs = {}
    for symbol in symbols:
        print(f"Fetching data for {symbol}")
        try:
            ohlcvs[symbol] = kraken.fetch_ohlcv(
                symbol, timeframe=timeframe, since=since
            )
        except ccxt.NetworkError as e:
            print(symbol, "failed due to a network error:", str(e))
        except ccxt.ExchangeError as e:
            print(symbol, "failed due to exchange error:", str(e))
        except Exception as e:
            print(symbol, "failed with unexpected error:", str(e))
    print(f"Fetched data for {len(ohlcvs)} valid USD pairs")

    # Process OHLCV data into a DataFrame
    ohlc_combined = pd.DataFrame(columns=OHLC_COLUMNS)
    ohlc_combined["timestamp"] = pd.to_datetime(
        ohlc_combined["timestamp"], utc=True, unit="ms"
    )
    for symbol, ohlcv in ohlcvs.items():
        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "vwap", "volume"],
        )
        df["dollar_volume"] = df["vwap"] * df["volume"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, unit="ms")
        df["ticker"] = symbol
        ohlc_combined = pd.concat([ohlc_combined, df], ignore_index=True)
    return ohlc_combined


def convert_balances_to_cash(kraken):
    # Balances to ignore when computing available cash value
    # Units are in currency
    bal_to_ignore = {"BTC": 0.11440919, "USD": 2000}

    # Fetch account balance
    balance = kraken.fetch_balance()["total"]

    # Iterate through each asset in the balance
    assets_to_remove = []
    for asset, amount in balance.items():
        if asset != "USD":
            # Fetch current market price of the asset
            market_price = kraken.fetch_ticker(f"{asset}/USD")["last"]
        else:
            market_price = 1.0

        # Calculate the cash value of the position
        amount_to_ignore = bal_to_ignore.get(asset, 0)
        amount -= amount_to_ignore
        cash_value = amount * market_price
        if np.abs(cash_value) < 0.001:
            # Ignore asset
            assets_to_remove.append(asset)
            continue

        print(
            f"Asset: {asset}, Amount: {amount:.4f}, Market Price: {market_price:.4f}, Cash Value: ${cash_value:.2f}"
        )
        balance[asset] = cash_value

    for asset in assets_to_remove:
        del balance[asset]

    return balance


def get_pnl(kraken: ccxt.kraken, starting_bankroll: float):
    assert np.isfinite(starting_bankroll) and starting_bankroll > 0.0
    balance = convert_balances_to_cash(kraken)
    curr_cash_value = sum(balance.values())
    pnl = (curr_cash_value - starting_bankroll) / starting_bankroll
    print(f"Current Cash Value: ${(curr_cash_value):.2f}")
    print(f"Starting Bankroll: ${starting_bankroll:.2f}")
    print(f"PNL: {(pnl * 100):.2f}%")
    return pnl


def get_positions(kraken: ccxt.kraken, df_signals: pd.DataFrame, account_size: float):
    params = {
        "momentum_factor": "30d_log_returns",
        "num_assets_to_keep": int(1e6),
        "min_signal_threshold": 0.05,
        "max_signal_threshold": 0.15,
        "type": "simple",
    }
    df_positions = generate_positions_v1(df_signals, params)

    # Get current open positions
    balance = convert_balances_to_cash(kraken)
    tickers_to_keep = list(balance.keys())
    tickers_to_keep = [ticker + "/USD" for ticker in tickers_to_keep if ticker != "USD"]
    # Get non-empty + current open positions
    df_nonempty_positions = nonempty_positions(
        df_positions, tickers_to_keep=tickers_to_keep
    )
    df_nonempty_positions = df_nonempty_positions.loc[
        df_nonempty_positions["timestamp"] == df_nonempty_positions["timestamp"].max()
    ]

    # Translate positions to dollar amounts
    curr_cash_value = sum(balance.values())
    print(f"Current Cash Value: ${curr_cash_value:.2f}")
    if account_size == np.inf:
        account_size = curr_cash_value
    print(f"Account Size: ${account_size:.2f}")
    df_nonempty_positions["target_dollar_position"] = (
        df_nonempty_positions["scaled_position"] * account_size
    )

    # Translate dollar positions to trades
    df_balance = pd.DataFrame.from_dict(
        balance, orient="index", columns=["current_dollar_position"]
    ).reset_index()
    df_balance["current_position"] = (
        df_balance["current_dollar_position"] / account_size
    )
    df_balance.rename(columns={"index": "ticker"}, inplace=True)
    df_balance["ticker"] = df_balance["ticker"].astype(str) + "/USD"
    df_balance.loc[df_balance["ticker"] == "USD/USD", "ticker"] = "USD"
    df_nonempty_positions = df_nonempty_positions.merge(
        df_balance, how="outer", on="ticker"
    )
    df_nonempty_positions.fillna(0.0, inplace=True)
    df_nonempty_positions["position_delta"] = (
        df_nonempty_positions["scaled_position"]
        - df_nonempty_positions["current_position"]
    )
    df_nonempty_positions["trade"] = (
        df_nonempty_positions["target_dollar_position"]
        - df_nonempty_positions["current_dollar_position"]
    )

    return df_nonempty_positions


def main(args):
    # Initialize the Kraken exchange
    kraken = MyKraken(
        {
            "apiKey": "EvqEd6Mn/yPHovibTJXKl0UAnoQPvs7yxRIPO/AOj4ifbavMH66M1HYF",
            "secret": "8w9/RnVsau3IKNH0/cYliHr+pqroxAAR0qecaKscYBVyFRaOerUerVOLiGpCLO/aduyTpdaSRU4xgl+4ERQl5w==",
        }
    )

    if args.input_path is not None and args.mode == "data":
        raise ValueError(
            "Specifying input path while also fetching data? What is you doing?"
        )
    elif args.input_path is not None and args.input_data_freq is None:
        raise ValueError("Must specify input data frequency with input file")

    if args.mode == "pnl":
        pnl = get_pnl(kraken, starting_bankroll=args.account_size)
        return 0

    if args.output_data_freq is None:
        raise ValueError(
            "Output data frequency must be specified for modes ['data', 'positions']"
        )

    whitelist_fn = in_universe_excl_stablecoins
    if args.input_path is None:
        timeframe = args.output_data_freq
        df_ohlc = fetch_data(
            kraken, lookback=timedelta(days=args.lookback_days), timeframe=timeframe
        )
        df_ohlc = filter_universe(df=df_ohlc, whitelist_fn=whitelist_fn)
    else:
        # Load data from input file
        print(f"Loading data from {args.input_path}")
        tz = pytz.timezone(args.timezone)
        if args.output_data_freq == "1d":
            df_ohlc = load_ohlc_to_daily_filtered(
                args.input_path,
                input_freq=args.input_data_freq,
                tz=tz,
                whitelist_fn=whitelist_fn,
            )
        elif args.output_data_freq == "1h":
            df_ohlc = load_ohlc_to_hourly_filtered(
                args.input_path,
                input_freq=args.input_data_freq,
                tz=tz,
                whitelist_fn=whitelist_fn,
            )

    if args.mode == "data":
        if args.output_path is not None:
            if args.append_output:
                # Load existing output file to filter duplicates and reindex
                if args.output_data_freq == "1d":
                    df_existing_output = load_ohlc_to_daily_filtered(
                        args.output_path,
                        input_freq=args.output_data_freq,
                        tz=pytz.UTC,
                        whitelist_fn=whitelist_fn,
                    )
                elif args.output_data_freq == "1h":
                    df_existing_output = load_ohlc_to_hourly_filtered(
                        args.output_path,
                        input_freq=args.output_data_freq,
                        tz=pytz.UTC,
                        whitelist_fn=whitelist_fn,
                    )
                assert df_ohlc.columns.equals(df_existing_output.columns)
                df_ohlc = pd.concat([df_ohlc, df_existing_output])
                df_ohlc.sort_values(
                    by=["timestamp", "ticker"], ascending=True, inplace=True
                )
                df_ohlc.drop_duplicates(
                    subset=["timestamp", "ticker"],
                    keep="first",  # Keep the latest available data
                    inplace=True,
                )
                df_ohlc.index = pd.RangeIndex(len(df_ohlc.index))
            output_path = Path(args.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df_ohlc.to_csv(str(output_path), mode="w")
            print(f"Wrote {df_ohlc.shape} dataframe to '{output_path}'")
    elif args.mode == "positions":
        if args.output_data_freq in ("1h"):
            # Special case, generate signals directly from 30 days of hourly data. Ignore timezone.
            df_signals = create_trading_signals(
                df_ohlc, periods_per_day=24, signal_type=SignalType.HistoricalReturns
            )
        else:
            # Standard path
            df_signals = create_trading_signals(
                df_ohlc, periods_per_day=1, signal_type=SignalType.HistoricalReturns
            )
        df_nonempty_positions = get_positions(
            kraken, df_signals, account_size=args.account_size
        )
        position_cols = [
            "timestamp",
            "ticker",
            "30d_num_days_volume_above_5M",
            "volume_consistent",
            "30d_log_returns",
            "scaled_position",
            "current_position",
            "position_delta",
            "target_dollar_position",
            "current_dollar_position",
            "trade",
        ]
        print(
            df_nonempty_positions.sort_values(by="trade", ascending=False)[
                position_cols
            ]
        )
        # Output to file
        if args.output_path is not None:
            output_path = Path(args.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df_nonempty_positions[position_cols].to_csv(
                str(output_path),
                mode="a" if args.append_output else "w",
                header=not args.append_output,
            )
            print(f"Wrote positions to '{output_path}'")
    else:
        raise ValueError("Unsupported mode")

    return 0


if __name__ == "__main__":
    np.set_printoptions(linewidth=1000)
    pd.set_option("display.width", 2000)
    pd.set_option("display.precision", 3)
    pd.set_option("display.float_format", "{:.3f}".format)

    args = parse_args()
    exit(main(args))
