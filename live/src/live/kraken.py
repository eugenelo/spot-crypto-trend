import argparse
from functools import partial
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd
import pytz
import yaml

from core.constants import (
    AVG_DOLLAR_VOLUME_COL,
    POSITION_COL,
    in_universe_excl_stablecoins,
)
from core.utils import get_periods_per_day
from data.constants import DATETIME_COL, TICKER_COL
from data.utils import load_ohlc_to_daily_filtered, load_ohlc_to_hourly_filtered
from live.constants import (
    CURRENT_DOLLAR_POSITION_COL,
    CURRENT_POSITION_COL,
    POSITION_DELTA_COL,
    TARGET_DOLLAR_POSITION_COL,
    TRADE_COL,
    TRADE_COLUMNS,
)
from live.utils import fetch_cash_balances
from position_generation.constants import (
    SCALED_SIGNAL_COL,
    VOL_FORECAST_COL,
    VOL_TARGET_COL,
)
from position_generation.position_generation import get_generate_positions_fn
from position_generation.utils import nonempty_positions
from signal_generation.constants import get_signal_type
from signal_generation.signal_generation import create_trading_signals
from simulation.constants import DEFAULT_REBALANCING_BUFFER


def parse_args():
    parser = argparse.ArgumentParser(description="Kraken API wrapper")
    parser.add_argument("mode", help="[trades]")
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
    return parser.parse_args()


def get_trades(
    kraken: ccxt.kraken,
    df_positions: pd.DataFrame,
    periods_per_day: int,
    account_size: float,
    rebalancing_buffer: float,
) -> pd.DataFrame:
    # Get current open positions
    balance = fetch_cash_balances(kraken, verbose=False)
    tickers_to_keep = list(balance.keys())
    tickers_to_keep = [ticker + "/USD" for ticker in tickers_to_keep if ticker != "USD"]
    # Get non-empty + current open positions at latest timestamp
    df_nonempty_positions = nonempty_positions(
        df_positions, tickers_to_keep=tickers_to_keep
    )
    df_trades = df_nonempty_positions.loc[
        df_nonempty_positions[DATETIME_COL] == df_nonempty_positions[DATETIME_COL].max()
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
    df_trades = df_trades.merge(df_balance, how="outer", on=TICKER_COL).reset_index()
    df_trades[DATETIME_COL] = df_trades[DATETIME_COL].ffill()
    df_trades.fillna(0.0, inplace=True)
    df_trades[POSITION_DELTA_COL] = (
        df_trades[POSITION_COL] - df_trades[CURRENT_POSITION_COL]
    )
    df_trades[TRADE_COL] = (
        df_trades[TARGET_DOLLAR_POSITION_COL] - df_trades[CURRENT_DOLLAR_POSITION_COL]
    )
    # Zero out trades below rebalancing buffer (ignore base currency row)
    no_trade_mask = (np.abs(df_trades[POSITION_DELTA_COL]) < rebalancing_buffer) & (
        df_trades[TICKER_COL] != "USD"
    )
    df_trades.loc[no_trade_mask, TRADE_COL] = 0
    df_trades.drop(df_trades[no_trade_mask].index, inplace=True)

    return df_trades


def main(args):
    # Initialize the Kraken exchange
    kraken = ccxt.kraken(
        {
            "apiKey": "EvqEd6Mn/yPHovibTJXKl0UAnoQPvs7yxRIPO/AOj4ifbavMH66M1HYF",
            "secret": "8w9/RnVsau3IKNH0/cYliHr+pqroxAAR0qecaKscYBVyFRaOerUerVOLiGpCLO/aduyTpdaSRU4xgl+4ERQl5w==",  # noqa: B950
        }
    )

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

    if args.mode == "trades":
        periods_per_day = get_periods_per_day(
            timestamp_series=df_ohlc.loc[
                df_ohlc[TICKER_COL] == df_ohlc[TICKER_COL].unique()[0]
            ][DATETIME_COL]
        )

        # Load position generation params
        if args.params_path is None:
            raise ValueError("Params path must be specified for modes ['trades']")
        params = {}
        with open(args.params_path, "r") as yaml_file:
            params = yaml.safe_load(yaml_file)
        print(f"Loaded params: {params}")
        assert "signal" in params, "Signal should be specified in params!"
        rebalancing_buffer = params.get(
            "rebalancing_buffer", DEFAULT_REBALANCING_BUFFER
        )
        print(f"rebalancing_buffer: {rebalancing_buffer:.4g}")

        # Don't lag positions, not backtesting. Take current day positions to
        # harvest next day returns.
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
            rebalancing_buffer=rebalancing_buffer,
        )
        # Add row containing column totals
        df_trades.loc["total"] = df_trades.sum(numeric_only=True, axis=0)
        cols_of_interest = [
            DATETIME_COL,
            TICKER_COL,
            AVG_DOLLAR_VOLUME_COL,
            VOL_FORECAST_COL,
            VOL_TARGET_COL,
            SCALED_SIGNAL_COL,
            POSITION_COL,
        ] + TRADE_COLUMNS
        print(df_trades.sort_values(by=TRADE_COL, ascending=False)[cols_of_interest])
        # Output to file
        if args.output_path is not None:
            output_path = Path(args.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df_trades[cols_of_interest].to_csv(
                str(output_path),
                mode="w",
                header=True,
                index=False,
            )
            print(
                f"Wrote {df_trades[cols_of_interest].shape} dataframe to"
                f" '{output_path}'"
            )
    else:
        raise ValueError("Unsupported mode")

    return 0


if __name__ == "__main__":
    np.set_printoptions(linewidth=1000)
    pd.set_option("display.width", 2000)
    pd.set_option("display.precision", 4)
    pd.set_option(
        "display.float_format",
        partial(np.format_float_positional, precision=4, trim="0"),
    )

    args = parse_args()
    exit(main(args))
