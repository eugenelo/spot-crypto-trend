import argparse
from datetime import datetime
from datetime import time as datetime_time
from datetime import timedelta
from functools import partial
from typing import Dict, NamedTuple

import ccxt
import numpy as np
import pandas as pd
import plotly.express as px
import pytz
import yaml

from data.constants import DATETIME_COL, ORDER_SIDE_COL, TICKER_COL
from data.utils import load_ohlc_to_daily_filtered
from live.constants import (
    AMOUNT_COL,
    BASE_CURRENCY,
    COST_COL,
    CURRENCY_COL,
    FEE_COL,
    FEES_COL,
    ID_COL,
    LEDGER_DIRECTION_COL,
    PNL_DATA_FETCH_START_DATE,
)
from live.utils import (
    KrakenExchange,
    fetch_balance,
    fetch_deposits,
    fetch_my_trades,
    get_account_size,
    portfolio_value,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Print/plot strategy PNL")
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument(
        "--input_path", "-i", type=str, help="Input data file path", required=True
    )
    parser.add_argument(
        "--data_freq", "-f", type=str, help="Input data frequency", required=True
    )
    parser.add_argument(
        "--start_date",
        "-s",
        type=str,
        help="Start date inclusive",
        default="1970-01-01",
    )
    parser.add_argument(
        "--credentials_path", type=str, help="Credentials yaml file path", required=True
    )
    return parser.parse_args()


def get_current_pnl(
    kraken: ccxt.kraken, start_date: datetime, df_ohlc: pd.DataFrame, verbose: bool
) -> float:
    # Compute starting bankroll from account size on start date
    account_size = get_historical_account_size(
        kraken,
        start_date=start_date,
        df_ohlc=df_ohlc,
    )
    print(f"Account Size: {account_size}")
    starting_bankroll = account_size.iloc[0]
    assert np.isfinite(starting_bankroll) and starting_bankroll > 0.0
    # Compute PNL
    balance = fetch_balance(kraken)
    if verbose:
        for ticker, balance_entry in balance.items():
            amount = balance_entry.amount
            market_price = balance_entry.mid_price
            cash_value = balance_entry.base_currency
            print(
                f"Ticker: {ticker}, Amount: {amount:.4f}, Market Price:"
                f" ${market_price:.2f}, Cash Value: ${cash_value:.2f}"
            )

    curr_cash_value = get_account_size(balance)
    pnl = (curr_cash_value - starting_bankroll) / starting_bankroll
    if verbose:
        print(f"Current Cash Value: ${(curr_cash_value):.2f}")
        print(f"Starting Bankroll: ${starting_bankroll:.2f}")
        print(f"PNL: {(pnl * 100):.2f}%")
    return pnl


def update_positions_with_trade(positions: Dict[str, float], trade: NamedTuple) -> None:
    # Increment/Decrement target asset
    ticker = getattr(trade, TICKER_COL)
    side = getattr(trade, ORDER_SIDE_COL)
    amount = getattr(trade, AMOUNT_COL)
    cost = getattr(trade, COST_COL)
    fees = getattr(trade, FEES_COL)
    assert not any([x is None for x in [ticker, side, amount, cost, fees]])
    if side == "sell":
        amount = -amount
        cost = -cost
    positions[ticker] = positions.get(ticker, 0) + amount
    # Increment/Decrement base currency (USD)
    assert BASE_CURRENCY in positions.keys()
    positions[BASE_CURRENCY] -= cost
    # Account for costs
    for fee_dict in fees:
        assert fee_dict[CURRENCY_COL] == BASE_CURRENCY
        positions[BASE_CURRENCY] -= fee_dict[COST_COL]


def update_positions_with_deposit(
    positions: Dict[str, float], deposit: NamedTuple
) -> None:
    # Increment/Decrement currency
    currency = getattr(deposit, CURRENCY_COL)
    direction = getattr(deposit, LEDGER_DIRECTION_COL)
    amount = getattr(deposit, AMOUNT_COL)
    fee = getattr(deposit, FEE_COL)
    assert not any([x is None for x in [currency, direction, amount, fee]])
    if direction == "out":
        amount = -amount
    positions[currency] = positions.get(currency, 0) + amount
    # Account for costs
    positions[fee[CURRENCY_COL]] -= fee[COST_COL]


def get_historical_trades(kraken: ccxt.kraken, start_date: datetime) -> pd.DataFrame:
    df_trades = fetch_my_trades(kraken=kraken, start_date=start_date)
    # TODO(@eugene.lo): Use txid to filter trades for different strategies...
    trade_ids_to_ignore = ["T4OYDZ-A2UXV-SRLODL"]
    df_trades = df_trades.loc[~df_trades[ID_COL].isin(trade_ids_to_ignore)]
    return df_trades


def get_historical_deposits(kraken: ccxt.kraken, start_date: datetime) -> pd.DataFrame:
    df_deposits = fetch_deposits(kraken=kraken, start_date=start_date)
    ledger_ids_to_ignore = [
        "LMPK6N-QHRNK-TI5KNR",
        "LU6MZ7-UNXJZ-4IJBM7",
        "LF32RU-XUIVS-OXECF7",
    ]
    df_deposits = df_deposits.loc[~df_deposits[ID_COL].isin(ledger_ids_to_ignore)]
    return df_deposits


def get_historical_account_size(
    kraken: ccxt.kraken,
    start_date: datetime,
    df_ohlc: pd.DataFrame,
) -> pd.Series:
    # Fetch all historical trades
    df_trades = get_historical_trades(
        kraken=kraken, start_date=PNL_DATA_FETCH_START_DATE
    )

    # Get historical deposits
    df_deposits = get_historical_deposits(
        kraken=kraken, start_date=PNL_DATA_FETCH_START_DATE
    )

    # Set start date to first deposit date at the earliest
    min_deposit_date = df_deposits[DATETIME_COL].min()

    # Reconstruct daily positions from trade ledger &
    # daily account size from positions + daily prices
    DT_1DAY = timedelta(days=1)
    idx_start_date = datetime.combine(min_deposit_date, datetime_time())
    idx_end_date = datetime.combine(datetime.now(tz=pytz.UTC), datetime_time())
    idx = pd.date_range(idx_start_date, idx_end_date, freq="1D", tz=pytz.UTC)
    account_size = pd.Series(index=idx)
    # `positions` maps from Ticker -> Amount (units in asset, NOT base currency)
    positions = {BASE_CURRENCY: 0}
    for i, date in enumerate(idx):
        # Update deposits
        df_deposits_for_date = df_deposits.loc[
            (df_deposits[DATETIME_COL] >= date)
            & (df_deposits[DATETIME_COL] < date + DT_1DAY)
        ]
        if not df_deposits_for_date.empty:
            for row in df_deposits_for_date.itertuples():
                update_positions_with_deposit(positions=positions, deposit=row)

        # Update positions with trades which occurred on date
        df_trades_for_date = df_trades.loc[
            (df_trades[DATETIME_COL] >= date)
            & (df_trades[DATETIME_COL] < date + DT_1DAY)
        ]
        if not df_trades_for_date.empty:
            for row in df_trades_for_date.itertuples():
                update_positions_with_trade(positions=positions, trade=row)

        # Sanity check no negative positions
        amount_EPS = 1e-3
        for ticker, amount in positions.items():
            if ticker != BASE_CURRENCY:
                assert amount >= -amount_EPS, (date, ticker, amount)

        # Get prices on date
        df_prices = df_ohlc.loc[
            (df_ohlc[DATETIME_COL] >= date) & (df_ohlc[DATETIME_COL] < date + DT_1DAY)
        ]
        if df_prices.empty:
            assert i == len(idx) - 1, date
            break

        account_size.iloc[i] = portfolio_value(positions=positions, df_prices=df_prices)

    if np.isnan(account_size.iloc[-1]):
        # Populate last day's account size with current prices
        account_size.iloc[-1] = get_account_size(fetch_balance(kraken))
    return account_size.loc[account_size.index >= start_date]


def get_historical_pnl(
    kraken: ccxt.kraken, start_date: datetime, account_size: pd.Series
) -> pd.Series:
    # Get historical deposits
    df_deposits = get_historical_deposits(
        kraken=kraken, start_date=PNL_DATA_FETCH_START_DATE
    )
    base_currency_deposits = df_deposits.loc[df_deposits[CURRENCY_COL] == BASE_CURRENCY]

    # When computing log returns, account for deposits which occurred on that day
    pnl = np.log(account_size / account_size.shift(periods=1))
    DT_1DAY = timedelta(days=1)
    for i, date in enumerate(pnl.index):
        if i == 0:
            # Skip first date, nan
            continue
        # Get today's deposits
        df_deposits_for_date = base_currency_deposits.loc[
            (base_currency_deposits[DATETIME_COL] > date)
            & (base_currency_deposits[DATETIME_COL] <= date + DT_1DAY)
        ]
        if not df_deposits_for_date.empty:
            assert i > 0
            account_size_for_day = account_size.iloc[i]
            for row in df_deposits_for_date.itertuples():
                account_size_for_day -= getattr(row, AMOUNT_COL, 0)
            if account_size_for_day == 0:
                pnl.iloc[i] = 0
            else:
                pnl.iloc[i] = np.log(account_size_for_day / account_size.iloc[i - 1])

    return pnl.dropna()


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
    start_date = pytz.UTC.localize(
        datetime.strptime(args.start_date.replace("/", "-"), "%Y-%m-%d")
    )
    df_ohlc = load_ohlc_to_daily_filtered(
        args.input_path,
        input_freq=args.data_freq,
        tz=pytz.UTC,
        whitelist_fn=None,
    )

    if args.skip_plots:
        get_current_pnl(kraken, start_date=start_date, df_ohlc=df_ohlc, verbose=True)
    else:
        # Plot account size over time
        account_size = get_historical_account_size(
            kraken,
            start_date=start_date,
            df_ohlc=df_ohlc,
        )
        fig = px.line(account_size, title="Account Size")
        fig.show()

        # Convert account sizes to log returns
        pnl = get_historical_pnl(
            kraken, start_date=start_date, account_size=account_size
        )
        fig = px.bar(pnl, title="Log Returns")
        fig.show()
        # Plot cumulative log returns
        fig = px.line(pnl.cumsum(), title="Cumulative Log Returns")
        fig.show()
        # Print realized volatility
        realized_volatility = np.sqrt(np.sum(pnl**2))
        print(f"Realized Volatility: {100 * realized_volatility:.2f}%")

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
