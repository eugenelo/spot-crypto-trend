import argparse
import logging
import os
import time
from datetime import datetime
from datetime import time as datetime_time
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import List

import ccxt
import numpy as np
import pandas as pd
import pytz
import yaml
from ccxt.base.types import Order

from core.constants import (
    AVG_DOLLAR_VOLUME_COL,
    POSITION_COL,
    in_universe_excl_stablecoins,
)
from core.utils import get_periods_per_day
from data.constants import DATETIME_COL, TICKER_COL
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
    TRADE_EXECUTION_PAUSE_INTERVAL,
    UPDATE_TRADES_INTERVAL,
)
from live.execution import (
    limit_order_book_side,
    limit_price_stale,
    place_limit_order,
    place_market_order,
)
from live.utils import (
    KrakenExchange,
    balance_to_dataframe,
    fetch_balance,
    fetch_bid_ask_spread,
    get_account_size,
)
from logging_custom.utils import setup_logging
from position_generation.constants import (  # noqa: F401
    ABS_SIGNAL_AVG_COL,
    SCALED_SIGNAL_COL,
    VOL_FORECAST_COL,
    VOL_LONG_COL,
    VOL_SHORT_COL,
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
        "mode", choices=["display", "execute", "cancel_all"], help="Mode of operation"
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
    parser.add_argument(
        "--credentials_path", type=str, help="Credentials yaml file path", required=True
    )
    parser.add_argument("--output_path", "-o", type=str, help="Output file path")
    parser.add_argument(
        "--execution_strategy",
        "-e",
        choices=["market", "limit"],
        type=str,
        help="Execution strategy",
        default="limit",
    )
    parser.add_argument(
        "--validate",
        "-v",
        action="store_true",
        help="Validate execution (won't place real orders)",
    )
    parser.add_argument(
        "--skip_confirm",
        action="store_true",
        help="Skip trade confirmation before executing",
    )
    return parser.parse_args()


def get_trades(
    kraken: ccxt.kraken,
    df_positions: pd.DataFrame,
    account_size: float,
    rebalancing_buffer: float,
) -> pd.DataFrame:
    # Get current open positions
    balance = fetch_balance(kraken)
    tickers_to_keep = list(balance.keys())

    # Get non-empty + current open positions at latest timestamp
    df_trades = nonempty_positions(df_positions, tickers_to_keep=tickers_to_keep)
    df_trades = df_trades.loc[df_trades[DATETIME_COL] == df_trades[DATETIME_COL].max()]

    # Translate positions to dollar amounts
    curr_cash_value = get_account_size(balance)
    account_size = min(account_size, curr_cash_value)
    logger.info(f"Current Cash Value: ${curr_cash_value:.2f}")
    logger.info(f"Account Size: ${account_size:.2f}")
    df_trades[TARGET_DOLLAR_POSITION_COL] = df_trades[POSITION_COL] * account_size

    # Unroll balance into dataframe
    df_balance = balance_to_dataframe(balance)
    df_balance[CURRENT_POSITION_COL] = (
        df_balance[CURRENT_DOLLAR_POSITION_COL] / account_size
    )

    # Merge balance with trades dataframe
    df_trades = df_trades.merge(df_balance, how="outer", on=TICKER_COL).reset_index(
        drop=True
    )
    df_trades[DATETIME_COL] = df_trades[DATETIME_COL].ffill()
    df_trades.fillna(0.0, inplace=True)
    # Update current prices for those tickers with 0 balance
    tickers_to_update = df_trades.loc[df_trades[CURRENT_PRICE_COL] == 0][
        TICKER_COL
    ].unique()
    for ticker in tickers_to_update:
        market_price = fetch_bid_ask_spread(exchange=kraken, ticker=ticker).mid
        df_trades.loc[df_trades[TICKER_COL] == ticker, CURRENT_PRICE_COL] = market_price

    # Translate dollar positions to trades
    df_trades[POSITION_DELTA_COL] = (
        df_trades[POSITION_COL] - df_trades[CURRENT_POSITION_COL]
    )
    df_trades[TRADE_DOLLAR_COL] = (
        df_trades[TARGET_DOLLAR_POSITION_COL] - df_trades[CURRENT_DOLLAR_POSITION_COL]
    )
    df_trades[TRADE_AMOUNT_COL] = (
        df_trades[TRADE_DOLLAR_COL] / df_trades[CURRENT_PRICE_COL]
    )

    # Zero out trades below rebalancing buffer (ignore base currency row)
    no_trade_mask = (np.abs(df_trades[POSITION_DELTA_COL]) < rebalancing_buffer) & (
        df_trades[TICKER_COL] != BASE_CURRENCY
    )
    df_trades.loc[no_trade_mask, TRADE_DOLLAR_COL] = 0
    df_trades.drop(df_trades[no_trade_mask].index, inplace=True)

    return df_trades


def place_orders(
    kraken: ccxt.kraken,
    df_trades: pd.DataFrame,
    execution_strategy: str,
    validate: bool,
) -> List[Order]:
    if df_trades.empty:
        return []

    orders_placed = []
    for row in df_trades.itertuples():
        ticker = getattr(row, TICKER_COL)
        if ticker == BASE_CURRENCY:
            continue
        amount = getattr(row, TRADE_AMOUNT_COL)

        try:
            if execution_strategy == "market":
                order = place_market_order(
                    exchange=kraken,
                    ticker=ticker,
                    amount=amount,
                    validate=validate,
                )
            else:
                order = place_limit_order(
                    exchange=kraken,
                    ticker=ticker,
                    amount=amount,
                    validate=validate,
                )
            if order is not None:
                orders_placed.append(order)
        except Exception as e:
            logger.warning(
                f"Placing order for {ticker} failed with unexpected error:"
                f" {str(e)}. Skipping!"
            )
            continue
    return orders_placed


def handle_open_orders(kraken: ccxt.kraken, open_orders: List[Order]) -> List[str]:
    # Wait up to timeout for orders to be filled, then cancel and retry
    tickers_traded: List[str] = []
    for order in open_orders:
        order_id = order["id"]
        while True:
            try:
                order = kraken.fetch_order(id=order_id)
            except ccxt.NetworkError as e:
                logger.warning(
                    f"Fetching order {order_id} failed due to a network error:"
                    f" {str(e)}. Retrying!"
                )
                continue
            except Exception as e:
                logger.warning(
                    f"Fetching order {order_id} failed with unexpected error: {str(e)}."
                )
                break
            else:
                break
        logger.info("{id}: {side} {symbol} {amount}@{price} - {status}".format(**order))
        order_status = order["status"]
        if order_status == "open":
            # Price staleness check (limit only)
            price_stale = False
            ticker = order["symbol"]
            order_type = order["type"]
            if order_type == "limit":
                order_side = order["side"]
                order_price = order["price"]
                bid_ask_spread = fetch_bid_ask_spread(exchange=kraken, ticker=ticker)
                order_book_side = limit_order_book_side(order_side=order_side)
                market_price = (
                    bid_ask_spread.second_bid
                    if order_book_side == "bids"
                    else bid_ask_spread.second_ask
                )
                price_stale = limit_price_stale(
                    order_price=order_price,
                    market_price=market_price,
                    order_side=order_side,
                )
                logger.info(
                    f"\t order_price: ${order_price:.6f}, market_price:"
                    f" {market_price:.6f}, order_side: {order_side}, price_stale:"
                    f" {price_stale}"
                )
            if price_stale:
                # Cancel order
                logger.info(
                    f"Canceling order {order_id} for {ticker}:"
                    f" price_stale={price_stale}"
                )
                response = kraken.cancel_order(id=order_id)
                logger.info(f"Canceled {int(response['result']['count'])} order(s)")
        elif order_status == "closed":
            # Order has been filled
            tickers_traded.append(order["symbol"])
        else:
            # Order failed
            logger.warning(f"Order {order_id} failed with status {order_status}")
    return tickers_traded


def execute_trades(
    kraken: ccxt.kraken,
    df_positions: pd.DataFrame,
    account_size: float,
    rebalancing_buffer: float,
    execution_strategy: str,
    validate: bool,
    skip_confirm: bool,
):
    df_trades, tickers, last_time_updated_trades = None, None, None

    def update_trades():
        nonlocal df_trades, tickers, last_time_updated_trades
        t0 = time.time()
        df_trades = get_trades(
            kraken,
            df_positions=df_positions,
            account_size=args.account_size,
            rebalancing_buffer=rebalancing_buffer,
        ).sort_values(by=TRADE_DOLLAR_COL, ascending=False)
        assert not df_trades.duplicated(subset=[TICKER_COL], keep=False).any()
        tickers = df_trades[TICKER_COL].unique()
        last_time_updated_trades = datetime.now(tz=pytz.UTC)
        t1 = time.time()
        logger.info(f"Updated trades in {t1-t0:.2f} seconds")

    # Display trades to be executed and ask for user confirmation
    update_trades()
    display_trades(df_trades)
    if not skip_confirm:
        while True:
            proceed = input("Proceed with trades? [y/n]:")
            if proceed.lower() not in ["y", "n"]:
                print("Input must be one of ['Y', 'y', 'N', 'n']")
                continue
            elif proceed.lower() == "n":
                print("Aborting trade execution!")
                return
            else:
                print()
                break

    reupdate_trades = False
    tickers_traded = []
    open_order_ids = []
    while True:
        try:
            # Fetch open orders before updating trades to avoid data race
            # because updating trades takes a very long time
            open_orders = kraken.fetch_open_orders()

            # Update trades to do with current market prices
            now = datetime.now(tz=pytz.UTC)
            reupdate_trades |= (
                last_time_updated_trades is None
                or (now - last_time_updated_trades).seconds > UPDATE_TRADES_INTERVAL
            )
            if reupdate_trades:
                logger.info("Updating trades")
                update_trades()
                # For sell orders, can't sell more than we own
                balance = fetch_balance(kraken)
                for ticker in tickers:
                    if ticker in balance:
                        curr_amount = balance[ticker].amount
                    else:
                        curr_amount = 0
                    df_trades.loc[
                        df_trades[TICKER_COL] == ticker, TRADE_AMOUNT_COL
                    ].clip(lower=-curr_amount, inplace=True)

            # Get tickers to trade this iteration based on the following criteria:
            #   1. Ticker has not already been traded since this function started
            #   2. Ticker does not have an outstanding open order
            tickers_with_open_orders = [order["symbol"] for order in open_orders]
            df_trades_subset = df_trades.loc[
                (~df_trades[TICKER_COL].isin(tickers_traded))
                & (~df_trades[TICKER_COL].isin(tickers_with_open_orders))
                & (df_trades[TICKER_COL] != BASE_CURRENCY)
            ]
            if df_trades_subset.empty and len(tickers_with_open_orders) == 0:
                break
            # Reupdate cash balances if open orders have changed
            new_open_order_ids = sorted([order["id"] for order in open_orders])
            reupdate_trades = open_order_ids != new_open_order_ids
            open_order_ids = new_open_order_ids

            logger.info("\nNew Iter")
            logger.info(f"Tickers Traded: {tickers_traded}")
            logger.info(f"Open Orders: {tickers_with_open_orders}")

            # Place new orders
            if not df_trades_subset.empty:
                logger.info("Remaining:")
                display_trades(df_trades_subset)
                logger.info("Executing orders")
                new_orders = place_orders(
                    kraken=kraken,
                    df_trades=df_trades_subset,
                    execution_strategy=execution_strategy,
                    validate=validate,
                )
                logger.info(f"{len(new_orders)} orders placed, pausing")
                open_orders.extend(new_orders)

            # Handle open orders
            logger.info("Checking status of open orders")
            tickers_traded_this_iter = handle_open_orders(
                kraken=kraken, open_orders=open_orders
            )
            tickers_traded.extend(tickers_traded_this_iter)
            time.sleep(TRADE_EXECUTION_PAUSE_INTERVAL)
        except Exception as e:
            logger.warning(f"Caught exception: {e}. Retrying!")


def cancel_all_orders(kraken: ccxt.kraken) -> None:
    open_orders = kraken.fetch_open_orders()
    num_open_orders = len(open_orders)
    logger.info(f"Canceling {num_open_orders} orders")

    while num_open_orders > 0:
        response = kraken.cancel_all_orders()
        num_open_orders -= int(response["result"]["count"])
        logger.info(f"Remaining num_open_orders={num_open_orders}")


def display_trades(df_trades: pd.DataFrame) -> None:
    cols_of_interest = [
        DATETIME_COL,
        TICKER_COL,
        AVG_DOLLAR_VOLUME_COL,
        # VOL_FORECAST_COL,
        # VOL_TARGET_COL,
        SCALED_SIGNAL_COL,
        POSITION_COL,
    ] + TRADE_COLUMNS
    # cols_of_interest = [
    #     DATETIME_COL,
    #     TICKER_COL,
    #     VOL_SHORT_COL,
    #     VOL_LONG_COL,
    #     VOL_FORECAST_COL,
    #     VOL_TARGET_COL,
    #     "rohrbach_exponential",
    #     ABS_SIGNAL_AVG_COL.format(signal=SCALED_SIGNAL_COL),
    #     SCALED_SIGNAL_COL,
    # ]

    # Add row containing column totals for printing only
    df_tmp = df_trades.copy().sort_values(by=TRADE_DOLLAR_COL, ascending=False)
    df_tmp.loc["total"] = df_tmp.sum(numeric_only=True, axis=0)
    logger.info(f"\n{df_tmp[cols_of_interest]}")


def output_trades(df_trades: pd.DataFrame, output_path: Path) -> None:
    cols_of_interest = [
        DATETIME_COL,
        TICKER_COL,
        AVG_DOLLAR_VOLUME_COL,
        VOL_FORECAST_COL,
        VOL_TARGET_COL,
        SCALED_SIGNAL_COL,
        POSITION_COL,
    ] + TRADE_COLUMNS

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_trades.sort_values(by=TRADE_DOLLAR_COL, ascending=False)[
        cols_of_interest
    ].to_csv(
        str(output_path),
        mode="w",
        header=True,
        index=False,
    )
    logger.info(
        f"Wrote {df_trades[cols_of_interest].shape} dataframe to '{output_path}'"
    )


def display_balance(df_balance: pd.DataFrame) -> None:
    # Add row containing column totals for printing only
    df_tmp = df_balance.copy().sort_values(
        by=CURRENT_DOLLAR_POSITION_COL, ascending=False
    )
    df_tmp.loc["total"] = df_tmp.sum(numeric_only=True, axis=0)
    logger.info(f"Balance:\n{df_tmp}")


def main(args):
    # Initialize the Kraken exchange
    with open(args.credentials_path, "r") as yaml_file:
        credentials = yaml.safe_load(yaml_file)
    kraken = KrakenExchange(
        {
            "apiKey": credentials["apiKey"],
            "secret": credentials["secret"],
            "enableRateLimit": True,
        }
    )
    if args.mode == "cancel_all":
        logger.info("Canceling all open orders!")
        cancel_all_orders(kraken)
        return 0

    # Set timezone
    if args.timezone == "latest":
        # Choose timezone which throws away the least amount of data
        # (assumes data was recently updated)
        curr_utc_hour = datetime.now(tz=pytz.UTC).hour
        if curr_utc_hour > 12:
            timezone_str = f"Etc/GMT-{24-curr_utc_hour}"
        else:
            timezone_str = f"Etc/GMT+{curr_utc_hour}"
        tz = pytz.timezone(timezone_str)
    else:
        tz = pytz.timezone(args.timezone)
    logger.info(f"Timezone: {tz}")

    whitelist_fn = in_universe_excl_stablecoins
    # Load data from input file
    t0 = time.time()
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
    t1 = time.time()
    logger.info(f"Loaded OHLC data from '{args.input_path}' in {t1-t0:.2f} seconds")

    # Load position generation params
    params = {}
    with open(args.params_path, "r") as yaml_file:
        params = yaml.safe_load(yaml_file)
    logger.info(f"Loaded params: {params}")
    assert "signal" in params, "Signal should be specified in params!"
    rebalancing_buffer = params.get("rebalancing_buffer", DEFAULT_REBALANCING_BUFFER)
    logger.info(f"rebalancing_buffer: {rebalancing_buffer:.4g}")

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

    if args.mode == "display":
        df_trades = get_trades(
            kraken,
            df_positions=df_positions,
            account_size=args.account_size,
            rebalancing_buffer=rebalancing_buffer,
        )
        display_trades(df_trades)
        # Output to file
        if args.output_path is not None:
            output_trades(df_trades=df_trades, output_path=Path(args.output_path))

    elif args.mode == "execute":
        # Check date
        yesterday = datetime.combine(
            datetime.now(tz=tz), datetime_time()
        ).date() - timedelta(days=1)
        last_positions_date = df_positions[DATETIME_COL].max().date()
        assert (
            last_positions_date == yesterday
        ), f"OHLC data is outdated! {last_positions_date} != {yesterday}"

        execute_trades(
            kraken=kraken,
            df_positions=df_positions,
            account_size=args.account_size,
            rebalancing_buffer=rebalancing_buffer,
            execution_strategy=args.execution_strategy,
            validate=args.validate,
            skip_confirm=args.skip_confirm,
        )
        # Display current balances after executing trades
        balance = fetch_balance(kraken)
        df_balance = balance_to_dataframe(balance).sort_values(
            by=CURRENT_DOLLAR_POSITION_COL, ascending=False
        )
        display_balance(df_balance)
        df_trades = get_trades(
            kraken,
            df_positions=df_positions,
            account_size=args.account_size,
            rebalancing_buffer=rebalancing_buffer,
        )
        display_trades(df_trades)

    return 0


if __name__ == "__main__":
    np.set_printoptions(linewidth=4000)
    pd.set_option("display.width", 4000)
    pd.set_option("display.precision", 4)
    pd.set_option(
        "display.float_format",
        partial(np.format_float_positional, precision=4, trim="0"),
    )

    # Configure logging
    log_config_path = Path(__file__).parent / Path(
        "../../../logging_custom/logging_config/live_config.yaml"
    )
    setup_logging(config_path=log_config_path)
    logger = logging.getLogger(__name__)
    # Google Cloud Logging
    if os.environ.get("USE_STACKDRIVER") == "true":
        import google.cloud.logging

        client = google.cloud.logging.Client()
        client.setup_logging(log_level=logging.INFO)

    args = parse_args()
    exit(main(args))
