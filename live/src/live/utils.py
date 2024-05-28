from dataclasses import dataclass
from datetime import datetime
from typing import Dict

import ccxt
import numpy as np
import pandas as pd
import pytz
from ccxt.base.types import OrderBook

from ccxt_custom.kraken import KrakenExchange
from data.constants import CLOSE_COL, DATETIME_COL, TICKER_COL, TIMESTAMP_COL
from live.constants import (
    BASE_CURRENCY,
    CURRENT_AMOUNT_COL,
    CURRENT_DOLLAR_POSITION_COL,
    CURRENT_PRICE_COL,
    HISTORICAL_TRADE_COLUMNS,
    ID_COL,
    LEDGER_COLUMNS,
    NUM_RETRY_ATTEMPTS,
)


def portfolio_value(positions: Dict[str, float], df_prices: pd.DataFrame) -> float:
    value = 0.0
    for ticker, amount in positions.items():
        if ticker == "LSKUSD":
            # Kraken delisted LSK, so symbol conversion doesn't work. Headache
            ticker = "LSK/USD"
        if ticker == BASE_CURRENCY:
            value += amount
        else:
            # Translate ticker to base currency with day's close price
            close = df_prices.loc[df_prices[TICKER_COL] == ticker, CLOSE_COL]
            if ticker == "LSK/USD" and close.size < 1:
                # Kraken delisted LSK, treat as $0 once delisted.
                close = 0
            else:
                assert close.size == 1, f"{ticker}, {df_prices}"
                close = close.iloc[0]
            value += amount * close
    return value


@dataclass
class BalanceEntry:
    ticker: str
    amount: float
    base_currency: float
    mid_price: float


def fetch_balance(exchange: ccxt.Exchange) -> Dict[str, BalanceEntry]:
    """Fetch balance in both asset units and base currency units.

    Base currency unit translation is doing using mid prices.

    Args:
        exchange (ccxt.Exchange): Exchange interface

    Returns:
        Dict[str, BalanceEntry]: Dict from ticker -> BalanceEntry
    """
    # Balances to ignore when computing available cash value
    # Units are in currency
    bal_to_ignore = {"BTC": 0.11440919 - 0.00065952, BASE_CURRENCY: 12000}

    # Fetch account balance
    balance = None
    for _attempt in range(NUM_RETRY_ATTEMPTS):
        try:
            balance = exchange.fetch_balance()["total"]
        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
            logger.warning(
                f"Fetching balance failed due to a network error: {str(e)}. Retrying!"
            )
            continue
        else:
            break
    if balance is None:
        raise RuntimeError("Failed to fetch balance")

    # Mark assets to market
    out: Dict[str, BalanceEntry] = {}
    for asset, amount in balance.items():
        if asset == "LSK":
            # 2024/05/25: Kraken delisted LSK from trading pending its migration to LSK2.
            # Attempting to fetch the mid price will throw an exception.
            # For now, just ignore the amount on balance.
            ticker = f"{asset}/{BASE_CURRENCY}"
            market_price = 0
        elif asset != BASE_CURRENCY:
            # Fetch current market price of the asset
            ticker = f"{asset}/{BASE_CURRENCY}"
            market_price = fetch_bid_ask_spread(exchange=exchange, ticker=ticker).mid
        else:
            ticker = asset
            market_price = 1.0

        # Calculate the cash value of the position
        amount_to_ignore = bal_to_ignore.get(asset, 0)
        amount -= amount_to_ignore
        cash_value = amount * market_price
        if np.abs(cash_value) < 0.001:
            # Ignore asset
            continue

        out[ticker] = BalanceEntry(
            ticker=ticker,
            amount=amount,
            base_currency=cash_value,
            mid_price=market_price,
        )

    return out


def balance_to_dataframe(balance: Dict[str, BalanceEntry]) -> pd.DataFrame:
    """Convert balance dictionary to DataFrame

    Args:
        balance (Dict[str, BalanceEntry]): Balance from `fetch_balance()`

    Returns:
        pd.DataFrame: Balance DataFrame
    """
    balance_unrolled = {
        TICKER_COL: [],
        CURRENT_DOLLAR_POSITION_COL: [],
        CURRENT_AMOUNT_COL: [],
        CURRENT_PRICE_COL: [],
    }
    for ticker, balance_entry in balance.items():
        balance_unrolled[TICKER_COL].append(ticker)
        balance_unrolled[CURRENT_DOLLAR_POSITION_COL].append(
            balance_entry.base_currency
        )
        balance_unrolled[CURRENT_AMOUNT_COL].append(balance_entry.amount)
        balance_unrolled[CURRENT_PRICE_COL].append(balance_entry.mid_price)
    return pd.DataFrame.from_dict(balance_unrolled)


def get_account_size(balance: Dict[str, BalanceEntry]) -> float:
    """Get size of account in base currency units from balance.

    Args:
        balance (Dict[str, BalanceEntry]): Balance from `fetch_balance()`

    Returns:
        float: Account size base currency units
    """
    return sum([entry.base_currency for entry in balance.values()])


def fetch_my_trades(kraken: KrakenExchange, start_date: datetime) -> pd.DataFrame:
    all_trades = []
    trades = kraken.fetch_my_trades(end=datetime.now(tz=pytz.UTC).timestamp())
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
    # Filter duplicate trade entries (based on ID)
    df_trades.drop_duplicates(subset=[ID_COL], inplace=True)
    # Filter trades which occurred before start date
    return df_trades.loc[df_trades[DATETIME_COL] > start_date][HISTORICAL_TRADE_COLUMNS]


def fetch_deposits(kraken: KrakenExchange, start_date: datetime) -> pd.DataFrame:
    ledger = []
    params = {"type": "deposit"}
    transactions = kraken.fetch_ledger(
        end=datetime.now(tz=pytz.UTC).timestamp(), params=params
    )
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
    # Filter duplicate ledger entries (based on ID)
    df_tx.drop_duplicates(subset=[ID_COL], inplace=True)
    # Filter entries which occurred before start date
    return df_tx.loc[df_tx[DATETIME_COL] > start_date][LEDGER_COLUMNS]


@dataclass
class BidAskSpread:
    bid: float
    ask: float
    mid: float
    spread: float
    second_bid: float
    second_ask: float


def fetch_bid_ask_spread(exchange: ccxt.Exchange, ticker: str) -> BidAskSpread:
    order_book = fetch_order_book(exchange=exchange, ticker=ticker)
    bid = order_book["bids"][0][0]
    ask = order_book["asks"][0][0]
    mid = (bid + ask) / 2
    spread = bid - ask
    second_bid = order_book["bids"][1][0]
    second_ask = order_book["asks"][1][0]
    return BidAskSpread(
        bid=bid,
        ask=ask,
        mid=mid,
        spread=spread,
        second_bid=second_bid,
        second_ask=second_ask,
    )


def fetch_order_book(exchange: ccxt.Exchange, ticker: str) -> OrderBook:
    """Fetch order book for ticker

    Args:
        exchange (ccxt.Exchange): Exchange interface
        ticker (str): Ticker pair to fetch order book for

    Returns:
        OrderBook: Order book for ticker
    """
    order_book = None
    limit = 500
    for _attempt in range(NUM_RETRY_ATTEMPTS):
        try:
            order_book = exchange.fetch_order_book(symbol=ticker, limit=limit)
        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
            logger.warning(
                f"{ticker} failed due to a network error: {str(e)}. Retrying!"
            )
            continue
        else:
            break
    if order_book is None:
        raise RuntimeError(f"Failed to fetch order book for {ticker}")
    assert order_book["symbol"] == ticker
    return order_book
