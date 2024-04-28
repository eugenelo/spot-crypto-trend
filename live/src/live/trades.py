import argparse
import time
from datetime import datetime
from datetime import time as datetime_time
from datetime import timedelta
from functools import partial
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd
import pytz
import yaml

from core.constants import POSITION_COL, in_universe_excl_stablecoins
from core.utils import get_periods_per_day
from data.constants import DATETIME_COL, ID_COL, TICKER_COL
from data.utils import load_ohlc_to_daily_filtered, load_ohlc_to_hourly_filtered
from live.constants import (
    BASE_CURRENCY,
    CURRENT_DOLLAR_POSITION_COL,
    CURRENT_POSITION_COL,
    CURRENT_PRICE_COL,
    POSITION_DELTA_COL,
    TARGET_DOLLAR_POSITION_COL,
    TRADE_AMOUNT_COL,
    TRADE_COLUMNS,
    TRADE_DOLLAR_COL,
)
from live.execution import place_market_order
from live.utils import KrakenExchange, fetch_cash_balances, fetch_mid_price
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
    parser.add_argument(
        "mode", choices=["display", "execute"], help="Mode of operation"
    )
    parser.add_argument(
        "--input_path", "-i", type=str, help="Input OHLC data file path", required=True
    )
    parser.add_argument(
        "--input_data_freq", "-if", type=str, help="Input data frequency", required=True
    )
    parser.add_argument(
        "--output_data_freq",
        "-f",
        type=str,
        help="Output data frequency",
        required=True,
    )
    parser.add_argument("--timezone", "-t", type=str, help="Timezone", default="UTC")
    parser.add_argument(
        "--account_size",
        "-s",
        type=float,
        help="(Target) Account size for position sizing or PNL calculation",
        default=np.inf,
    )
    parser.add_argument(
        "--params_path", "-p", type=str, help="Params yaml file path", required=True
    )
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
    tickers_to_keep = [
        ticker + f"/{BASE_CURRENCY}"
        for ticker in tickers_to_keep
        if ticker != BASE_CURRENCY
    ]
    # Get non-empty + current open positions at latest timestamp
    df_trades = nonempty_positions(df_positions, tickers_to_keep=tickers_to_keep)
    df_trades = df_trades.loc[df_trades[DATETIME_COL] == df_trades[DATETIME_COL].max()]

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
    df_balance[TICKER_COL] = df_balance[TICKER_COL].astype(str) + f"/{BASE_CURRENCY}"
    df_balance.loc[
        df_balance[TICKER_COL] == f"{BASE_CURRENCY}/{BASE_CURRENCY}", TICKER_COL
    ] = BASE_CURRENCY
    df_trades = df_trades.merge(df_balance, how="outer", on=TICKER_COL).reset_index(
        drop=True
    )
    df_trades[DATETIME_COL] = df_trades[DATETIME_COL].ffill()
    df_trades.fillna(0.0, inplace=True)
    df_trades[POSITION_DELTA_COL] = (
        df_trades[POSITION_COL] - df_trades[CURRENT_POSITION_COL]
    )
    df_trades[TRADE_DOLLAR_COL] = (
        df_trades[TARGET_DOLLAR_POSITION_COL] - df_trades[CURRENT_DOLLAR_POSITION_COL]
    )
    # Zero out trades below rebalancing buffer (ignore base currency row)
    no_trade_mask = (np.abs(df_trades[POSITION_DELTA_COL]) < rebalancing_buffer) & (
        df_trades[TICKER_COL] != BASE_CURRENCY
    )
    df_trades.loc[no_trade_mask, TRADE_DOLLAR_COL] = 0
    df_trades.drop(df_trades[no_trade_mask].index, inplace=True)

    return df_trades


def execute_trades(kraken: ccxt.kraken, df_trades: pd.DataFrame):
    df_trades_remaining = df_trades.sort_values(by=TRADE_DOLLAR_COL, ascending=False)[
        [
            TICKER_COL,
            CURRENT_DOLLAR_POSITION_COL,
            TARGET_DOLLAR_POSITION_COL,
            TRADE_DOLLAR_COL,
        ]
    ].copy()
    assert not df_trades_remaining.duplicated(subset=[TICKER_COL], keep=False).any()
    tickers = df_trades_remaining[TICKER_COL].unique()
    tickers_traded = []
    while True:
        # Update current cash balances
        cash_balance = fetch_cash_balances(kraken, verbose=False)
        for ticker in tickers:
            symbol = ticker.split("/")[0]
            curr_dollar_position = cash_balance.get(symbol, 0)
            df_trades_remaining.loc[
                df_trades_remaining[TICKER_COL] == ticker, CURRENT_DOLLAR_POSITION_COL
            ] = curr_dollar_position
            # Also update latest price
            if symbol != BASE_CURRENCY:
                market_price = fetch_mid_price(exchange=kraken, ticker=ticker)
            else:
                market_price = 1.0
            df_trades_remaining.loc[
                df_trades_remaining[TICKER_COL] == ticker, CURRENT_PRICE_COL
            ] = market_price
        df_trades_remaining[TRADE_DOLLAR_COL] = (
            df_trades_remaining[TARGET_DOLLAR_POSITION_COL]
            - df_trades_remaining[CURRENT_DOLLAR_POSITION_COL]
        )
        df_trades_remaining[TRADE_AMOUNT_COL] = (
            df_trades_remaining[TRADE_DOLLAR_COL]
            / df_trades_remaining[CURRENT_PRICE_COL]
        )
        # For sell orders, can't sell more than we own
        amt_balance = kraken.fetch_balance()["total"]
        for ticker in tickers:
            symbol = ticker.split("/")[0]
            curr_amount = amt_balance.get(symbol, 0)
            df_trades_remaining.loc[
                df_trades_remaining[TICKER_COL] == ticker, TRADE_AMOUNT_COL
            ].clip(lower=-curr_amount, inplace=True)

        # Update remaining positions to fill
        df_trades_remaining_subset = df_trades_remaining.loc[
            (~df_trades_remaining[TICKER_COL].isin(tickers_traded))
            & (df_trades_remaining[TICKER_COL] != "USD")
        ]
        print(f"Tickers Traded: {tickers_traded}")
        print(f"Remaining:\n{df_trades_remaining_subset}\n")
        if df_trades_remaining_subset.empty:
            break
        time.sleep(4)
        print("Executing orders")

        # Place orders
        orders_placed = []
        for row in df_trades_remaining_subset.itertuples():
            ticker = getattr(row, TICKER_COL)
            if ticker == "USD":
                continue
            amount = getattr(row, TRADE_AMOUNT_COL)

            try:
                order = place_market_order(
                    exchange=kraken,
                    ticker=ticker,
                    amount=amount,
                )
                if order is not None:
                    orders_placed.append(order)
            except Exception as e:
                print(
                    f"Placing order for {ticker} failed with unexpected error:"
                    f" {str(e)}. Skipping!"
                )
                continue
        print(f"{len(orders_placed)} orders placed, pausing")
        time.sleep(4)
        print("Checking status of orders")

        # Check fill status
        for order in orders_placed:
            order_id = order[ID_COL]
            try:
                updated_order_status = kraken.fetch_order(id=order_id)
            except Exception as e:
                print(
                    f"Fetching order info for order id {order_id} failed with"
                    f" unexpected error: {str(e)}."
                )
                continue
            print(
                "{id}: {side} {symbol} {filled}@{average} {status}".format(
                    **updated_order_status
                )
            )
            if updated_order_status["status"] != "closed":
                # Cancel and retry
                print(f"Cancelling order {order_id}")
                updated_order_status = kraken.cancel_orders(ids=[order_id])
                print(updated_order_status)
            else:
                tickers_traded.append(order["symbol"])


def main(args):
    # Initialize the Kraken exchange
    kraken = KrakenExchange(
        {
            "apiKey": "EvqEd6Mn/yPHovibTJXKl0UAnoQPvs7yxRIPO/AOj4ifbavMH66M1HYF",
            "secret": "8w9/RnVsau3IKNH0/cYliHr+pqroxAAR0qecaKscYBVyFRaOerUerVOLiGpCLO/aduyTpdaSRU4xgl+4ERQl5w==",  # noqa: B950
        }
    )

    whitelist_fn = in_universe_excl_stablecoins
    # Load data from input file
    print(f"Loading OHLC data from {args.input_path}")
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

    # Load position generation params
    params = {}
    with open(args.params_path, "r") as yaml_file:
        params = yaml.safe_load(yaml_file)
    print(f"Loaded params: {params}")
    assert "signal" in params, "Signal should be specified in params!"
    rebalancing_buffer = params.get("rebalancing_buffer", DEFAULT_REBALANCING_BUFFER)
    print(f"rebalancing_buffer: {rebalancing_buffer:.4g}")

    # Generate positions. Don't lag (not backtesting, take current day positions
    # to harvest next day returns).
    periods_per_day = get_periods_per_day(
        timestamp_series=df_ohlc.loc[
            df_ohlc[TICKER_COL] == df_ohlc[TICKER_COL].unique()[0]
        ][DATETIME_COL]
    )
    generate_positions_fn = get_generate_positions_fn(
        params, periods_per_day=periods_per_day, lag_positions=False
    )
    df_signals = create_trading_signals(
        df_ohlc,
        periods_per_day=periods_per_day,
        signal_type=get_signal_type(params),
    )
    df_positions = generate_positions_fn(df_signals)

    # Translate positions to trades
    df_trades = get_trades(
        kraken,
        df_positions=df_positions,
        periods_per_day=periods_per_day,
        account_size=args.account_size,
        rebalancing_buffer=rebalancing_buffer,
    )

    if args.mode == "display":
        cols_of_interest = [
            DATETIME_COL,
            TICKER_COL,
            VOL_FORECAST_COL,
            VOL_TARGET_COL,
            SCALED_SIGNAL_COL,
            POSITION_COL,
        ] + TRADE_COLUMNS

        # Add row containing column totals for printing only
        df_tmp = df_trades.copy()
        df_tmp.loc["total"] = df_tmp.sum(numeric_only=True, axis=0)
        print(
            df_tmp.sort_values(by=TRADE_DOLLAR_COL, ascending=False)[cols_of_interest]
        )

        # Output to file
        if args.output_path is not None:
            output_path = Path(args.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df_trades.sort_values(by=TRADE_DOLLAR_COL, ascending=False)[
                cols_of_interest
            ].to_csv(
                str(output_path),
                mode="w",
                header=True,
                index=False,
            )
            print(
                f"Wrote {df_trades[cols_of_interest].shape} dataframe to"
                f" '{output_path}'"
            )
    elif args.mode == "execute":
        # Check date
        yesterday = datetime.combine(
            pytz.UTC.localize(datetime.utcnow()), datetime_time()
        ).date() - timedelta(days=1)
        df_trades_date = df_trades[DATETIME_COL].max().date()
        assert (
            df_trades_date == yesterday
        ), f"OHLC data is outdated! {df_trades_date} != {yesterday}"

        execute_trades(kraken=kraken, df_trades=df_trades)
        # Display current balances after executing trades
        balance = fetch_cash_balances(kraken, verbose=False)
        df_tmp = df_trades.sort_values(by=TARGET_DOLLAR_POSITION_COL, ascending=False)[
            [
                TICKER_COL,
                CURRENT_DOLLAR_POSITION_COL,
                TARGET_DOLLAR_POSITION_COL,
                TRADE_DOLLAR_COL,
            ]
        ].copy()
        tickers = df_tmp[TICKER_COL].unique()
        for ticker in tickers:
            symbol = ticker.split("/")[0]
            curr_bal = balance.get(symbol, 0)
            df_tmp.loc[df_tmp[TICKER_COL] == ticker, CURRENT_DOLLAR_POSITION_COL] = (
                curr_bal
            )
        df_tmp[TRADE_DOLLAR_COL] = (
            df_tmp[TARGET_DOLLAR_POSITION_COL] - df_tmp[CURRENT_DOLLAR_POSITION_COL]
        )
        print(f"Current:\n{df_tmp}")

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
