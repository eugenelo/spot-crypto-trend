from datetime import datetime
from typing import Dict

import ccxt
import numpy as np
import pandas as pd

from ccxt_custom.kraken import KrakenExchange
from data.constants import CLOSE_COL, DATETIME_COL, TICKER_COL, TIMESTAMP_COL
from live.constants import (
    BASE_CURRENCY,
    HISTORICAL_TRADE_COLUMNS,
    ID_COL,
    LEDGER_COLUMNS,
)


def portfolio_value(positions: Dict[str, float], df_prices: pd.DataFrame) -> float:
    value = 0.0
    for ticker, amount in positions.items():
        if ticker == BASE_CURRENCY:
            value += amount
        else:
            # Translate ticker to base currency with day's close price
            close = df_prices.loc[df_prices[TICKER_COL] == ticker, CLOSE_COL]
            assert close.size == 1
            close = close.iloc[0]
            value += amount * close
    return value


def fetch_cash_balances(exchange: ccxt.Exchange, verbose: bool) -> Dict[str, float]:
    # Balances to ignore when computing available cash value
    # Units are in currency
    bal_to_ignore = {"BTC": 0.11440919, "USD": 12000}

    # Fetch account balance
    balance = exchange.fetch_balance()["total"]

    # Iterate through each asset in the balance
    assets_to_remove = []
    for asset, amount in balance.items():
        if asset != "USD":
            # Fetch current market price of the asset
            market_price = exchange.fetch_ticker(f"{asset}/USD")["last"]
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

        if verbose:
            print(
                f"Asset: {asset}, Amount: {amount:.4f}, Market Price:"
                f" ${market_price:.2f}, Cash Value: ${cash_value:.2f}"
            )
        balance[asset] = cash_value

    for asset in assets_to_remove:
        del balance[asset]

    return balance


def fetch_my_trades(kraken: KrakenExchange, start_date: datetime) -> pd.DataFrame:
    all_trades = []
    trades = kraken.fetch_my_trades(end=datetime.utcnow().timestamp())
    while len(trades) > 0:
        all_trades.extend(trades[::-1])
        last_trade = trades[0]
        end = last_trade[ID_COL]
        trades = kraken.fetch_my_trades(end=end)

        if (
            len(trades) == 0
            or trades[0][ID_COL] == end
            or trades[-1][TIMESTAMP_COL] < start_date.timestamp()
        ):
            break
    # Return dataframe sorted by timestamp with corrected column names
    df_trades = pd.DataFrame(all_trades[::-1])
    df_trades = df_trades.rename(
        columns={"symbol": TICKER_COL},
    ).sort_values(by=TIMESTAMP_COL, ascending=True)
    df_trades[DATETIME_COL] = pd.to_datetime(df_trades[DATETIME_COL], utc=True)
    # Filter trades which occurred before start date
    return df_trades.loc[df_trades[DATETIME_COL] > start_date][HISTORICAL_TRADE_COLUMNS]


def fetch_deposits(kraken: KrakenExchange, start_date: datetime) -> pd.DataFrame:
    ledger = []
    params = {"type": "deposit"}
    transactions = kraken.fetch_ledger(end=datetime.utcnow().timestamp(), params=params)
    while len(transactions) > 0:
        ledger.extend(transactions[::-1])
        last_tx = transactions[0]
        end = last_tx[ID_COL]
        transactions = kraken.fetch_ledger(end=end, params=params)

        if (
            len(transactions) == 0
            or transactions[0][ID_COL] == end
            or transactions[-1][TIMESTAMP_COL] < start_date.timestamp()
        ):
            break
    # Return dataframe sorted by timestamp with corrected column names
    df_tx = pd.DataFrame(ledger[::-1])
    df_tx = df_tx.rename(
        columns={"symbol": TICKER_COL},
    ).sort_values(by=TIMESTAMP_COL, ascending=True)
    df_tx[DATETIME_COL] = pd.to_datetime(df_tx[DATETIME_COL], utc=True)
    # Filter trades which occurred before start date
    return df_tx.loc[df_tx[DATETIME_COL] > start_date][LEDGER_COLUMNS]
