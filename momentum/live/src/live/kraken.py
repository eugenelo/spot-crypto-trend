import argparse
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import Optional
from pathlib import Path
import pytz
import yaml

from core.constants import (
    AVG_DOLLAR_VOLUME_COL,
    POSITION_COL,
    in_universe_excl_stablecoins,
)
from core.utils import get_periods_per_day
from data.utils import (
    load_ohlc_to_daily_filtered,
    load_ohlc_to_hourly_filtered,
    filter_universe,
)
from data.constants import (
    TIMESTAMP_COL,
    TICKER_COL,
    VWAP_COL,
    VOLUME_COL,
    DOLLAR_VOLUME_COL,
    OHLC_COLUMNS,
)
from live.constants import (
    CURRENT_POSITION_COL,
    POSITION_DELTA_COL,
    TARGET_DOLLAR_POSITION_COL,
    CURRENT_DOLLAR_POSITION_COL,
    TRADE_COL,
    TRADE_COLUMNS,
)
from signal_generation.signal_generation import create_trading_signals
from signal_generation.constants import get_signal_type
from position_generation.constants import (
    VOL_FORECAST_COL,
    VOL_TARGET_COL,
    SCALED_SIGNAL_COL,
    NUM_OPEN_POSITIONS_COL,
)
from position_generation.position_generation import get_generate_positions_fn
from position_generation.utils import nonempty_positions, validate_positions, Direction


def parse_args():
    parser = argparse.ArgumentParser(description="Kraken API wrapper")
    parser.add_argument("mode", help="[positions, pnl]")
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
    parser.add_argument("--timezone", "-t", type=str, help="Timezone", default="UTC")
    parser.add_argument(
        "--account_size",
        "-s",
        type=float,
        help="(Target) Account size for position sizing or PNL calculation",
        default=np.inf,
    )
    parser.add_argument("--params_path", "-p", type=str, help="Params yaml file path")
    parser.add_argument("--output_path", "-o", type=str, help="Output file path")
    parser.add_argument(
        "--append_output", "-a", help="Append to output file", action="store_true"
    )
    return parser.parse_args()


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


def get_trades(
    kraken: ccxt.kraken,
    df_positions: pd.DataFrame,
    periods_per_day: int,
    account_size: float,
) -> pd.DataFrame:
    # Get current open positions
    balance = convert_balances_to_cash(kraken)
    tickers_to_keep = list(balance.keys())
    tickers_to_keep = [ticker + "/USD" for ticker in tickers_to_keep if ticker != "USD"]
    # Get non-empty + current open positions at latest timestamp
    df_nonempty_positions = nonempty_positions(
        df_positions, tickers_to_keep=tickers_to_keep
    )
    df_trades = df_nonempty_positions.loc[
        df_nonempty_positions[TIMESTAMP_COL]
        == df_nonempty_positions[TIMESTAMP_COL].max()
    ]

    # Translate positions to dollar amounts
    curr_cash_value = sum(balance.values())
    print(f"Current Cash Value: ${curr_cash_value:.2f}")
    if account_size == np.inf:
        account_size = curr_cash_value
    print(f"Account Size: ${account_size:.2f}")
    df_trades[TARGET_DOLLAR_POSITION_COL] = df_trades[POSITION_COL] * account_size

    # Translate dollar positions to trades
    df_balance = pd.DataFrame.from_dict(
        balance, orient="index", columns=[CURRENT_DOLLAR_POSITION_COL]
    ).reset_index()
    df_balance[CURRENT_POSITION_COL] = (
        df_balance[CURRENT_DOLLAR_POSITION_COL] / account_size
    )
    df_balance.rename(columns={"index": TICKER_COL}, inplace=True)
    df_balance[TICKER_COL] = df_balance[TICKER_COL].astype(str) + "/USD"
    df_balance.loc[df_balance[TICKER_COL] == "USD/USD", TICKER_COL] = "USD"
    df_trades = df_trades.merge(df_balance, how="outer", on=TICKER_COL)
    df_trades.fillna(0.0, inplace=True)
    df_trades[POSITION_DELTA_COL] = (
        df_trades[POSITION_COL] - df_trades[CURRENT_POSITION_COL]
    )
    df_trades[TRADE_COL] = (
        df_trades[TARGET_DOLLAR_POSITION_COL] - df_trades[CURRENT_DOLLAR_POSITION_COL]
    )

    return df_trades


def main(args):
    # Initialize the Kraken exchange
    kraken = ccxt.kraken(
        {
            "apiKey": "EvqEd6Mn/yPHovibTJXKl0UAnoQPvs7yxRIPO/AOj4ifbavMH66M1HYF",
            "secret": "8w9/RnVsau3IKNH0/cYliHr+pqroxAAR0qecaKscYBVyFRaOerUerVOLiGpCLO/aduyTpdaSRU4xgl+4ERQl5w==",
        }
    )

    if args.mode == "pnl":
        pnl = get_pnl(kraken, starting_bankroll=args.account_size)
        return 0

    if args.input_path is None or args.input_data_freq is None:
        raise ValueError(
            "Input path and data frequency must be specified for modes ['positions']"
        )
    if args.output_data_freq is None:
        raise ValueError(
            "Output data frequency must be specified for modes ['positions']"
        )

    whitelist_fn = in_universe_excl_stablecoins
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
    else:
        raise ValueError("Unsupported output data frequency!")

    if args.mode == "positions":
        periods_per_day = get_periods_per_day(
            timestamp_series=df_ohlc.loc[
                df_ohlc[TICKER_COL] == df_ohlc[TICKER_COL].unique()[0]
            ][TIMESTAMP_COL]
        )

        # Load position generation params
        if args.params_path is None:
            raise ValueError("Params path must be specified for modes ['positions']")
        params = {}
        with open(args.params_path, "r") as yaml_file:
            params = yaml.safe_load(yaml_file)
        print(f"Loaded params: {params}")
        assert "signal" in params, "Signal should be specified in params!"
        # Lag positions if backtesting
        generate_positions_fn = get_generate_positions_fn(
            params, periods_per_day=periods_per_day, lag_positions=False
        )

        df_signals = create_trading_signals(
            df_ohlc,
            periods_per_day=periods_per_day,
            signal_type=get_signal_type(params),
        )
        df_positions = generate_positions_fn(df_signals)
        df_trades = get_trades(
            kraken,
            df_positions=df_positions,
            periods_per_day=periods_per_day,
            account_size=args.account_size,
        )
        cols_of_interest = [
            TIMESTAMP_COL,
            TICKER_COL,
            AVG_DOLLAR_VOLUME_COL,
            VOL_FORECAST_COL,
            VOL_TARGET_COL,
            SCALED_SIGNAL_COL,
            NUM_OPEN_POSITIONS_COL,
            POSITION_COL,
        ] + TRADE_COLUMNS
        print(df_trades.sort_values(by=TRADE_COL, ascending=False)[cols_of_interest])
        # Output to file
        if args.output_path is not None:
            output_path = Path(args.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df_trades[cols_of_interest].to_csv(
                str(output_path),
                mode="a" if args.append_output else "w",
                header=not args.append_output,
                index=False,
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
