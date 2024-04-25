import argparse
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

from ccxt_custom.kraken import KrakenExchange
from data.constants import TICKER_COL
from data.utils import load_ohlc_csv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Kraken OHLC data native symbols to ccxt symbols"
    )
    parser.add_argument(
        "--input_path", "-i", type=str, required=True, help="Input data file path"
    )
    parser.add_argument(
        "--output_path", "-o", type=str, required=True, help="Input data file path"
    )
    return parser.parse_args()


SYMBOL_SPECIAL_CASES = {
    # BTC symbol from downloaded tick data differs from API
    "XBTUSD": "BTC/USD",
    # Luna / Terra
    "LUNAUSD": "LUNC/USD",
    "LUNA2USD": "LUNA/USD",
    "USTUSD": "USTC/USD",
    # REP v1 and v2 still both trade in separate markets on Kraken
    "REPUSD": "REPV1/USD",
    "REPV2USD": "REP/USD",
}


def convert_tickers(kraken: ccxt.kraken, input_path: str, output_path: str):
    print(f"Available Symbols: {kraken.symbols}")

    df = load_ohlc_csv(input_path)
    tickers = df[TICKER_COL].unique()
    for kraken_symbol in tickers:
        if kraken_symbol in SYMBOL_SPECIAL_CASES:
            ccxt_symbol = SYMBOL_SPECIAL_CASES[kraken_symbol]
        else:
            try:
                market = kraken.markets_by_id[kraken_symbol][0]
                ccxt_symbol = market["symbol"]
            except Exception:
                print(f"Failed to find {kraken_symbol} on ccxt! Converting manually.")
                kraken_symbol = kraken_symbol.replace("/", "")
                ccxt_symbol = kraken_symbol[:-3] + "/" + kraken_symbol[-3:]
        print(f"{kraken_symbol} -> {ccxt_symbol}")
        df.loc[df[TICKER_COL] == kraken_symbol, TICKER_COL] = ccxt_symbol
    # Write output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(output_path), index=False)
    print(f"Wrote output to '{output_path}'")


def main(args):
    # Initialize the Kraken exchange and load markets
    kraken = KrakenExchange(
        {
            "enableRateLimit": True,
        }
    )
    kraken.load_markets()

    convert_tickers(kraken, args.input_path, args.output_path)

    return 0


if __name__ == "__main__":
    np.set_printoptions(linewidth=1000)
    pd.set_option("display.width", 2000)
    pd.set_option("display.precision", 3)
    pd.set_option("display.float_format", "{:.3f}".format)

    args = parse_args()
    exit(main(args))
