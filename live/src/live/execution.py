import logging
from typing import Optional

import ccxt
import numpy as np
from ccxt.base.types import Order, OrderBook, OrderSide, OrderType

from live.constants import (
    LIMIT_ORDER_TIMEOUT_TIME,
    MARKET_ORDER_TIMEOUT_TIME,
    MAX_ACCEPTABLE_SLIPPAGE,
    MAX_SINGLE_TRADE_SIZE,
)
from live.utils import fetch_order_book

logger = logging.getLogger(__name__)


def get_best_price(order_book: OrderBook, side: str) -> float:
    """Get 'best' price (bid/ask) from order book

    Args:
        order_book (OrderBook): Order book
        side (str): Side of order book ("bids", "asks")

    Returns:
        _type_: Best price
    """
    return order_book[side][0][0]


def get_mid_price(order_book: OrderBook) -> float:
    """Get mid price from order book

    Args:
        order_book (OrderBook): Order book

    Returns:
        float: Mid price [base currency]
    """
    bid = get_best_price(order_book, side="bids")
    ask = get_best_price(order_book, side="asks")
    mid = (bid + ask) / 2
    return mid


def market_order_book_side(order_side: OrderSide) -> str:
    """Get side of order book to look at based on order side
    (for a market order).

    Args:
        order_side (OrderSide): "buy" or "sell"

    Returns:
        str: Order book side
    """
    return "asks" if order_side == "buy" else "bids"


def limit_order_book_side(order_side: OrderSide) -> str:
    """Get side of order book to look at based on order side
    (for a limit order).

    Args:
        order_side (OrderSide): "buy" or "sell"

    Returns:
        str: Order book side
    """
    return "bids" if order_side == "buy" else "asks"


def get_order_side(amount: float) -> OrderSide:
    """Get order side of order based on sign of amount

    Args:
        amount (float): Order amount

    Returns:
        OrderSide: Order side
    """
    return "buy" if amount > 0 else "sell"


def estimate_slippage(order_book: OrderBook, amount: float) -> float:
    """Estimate slippage of an order of ticker for amount.

    Assume we take from the order book for the full size of our order. Compute the
    proportional difference between the average price of our order vs the best price.

    Args:
        order_book (OrderBook): Order book for ticker
        amount (float): Amount of target asset to buy/sell [units of asset]

    Returns:
        float: Estimated slippage [percentage]
    """
    if amount == 0:
        return 0
    order_side = get_order_side(amount)
    book = order_book[market_order_book_side(order_side=order_side)]

    # Compute slippage if order placed for full size
    remaining_abs_amt = abs(amount)  # [units of asset]
    entries_hit = []
    for entry in book:
        if remaining_abs_amt <= 0:
            break
        price, volume, timestamp = entry
        traded_volume = min(remaining_abs_amt, volume)  # [units of asset]
        entries_hit.append([price, traded_volume])
        remaining_abs_amt -= traded_volume
    if remaining_abs_amt > 0:
        # Can't fill entire order from first 250 entries in book,
        # assume slippage is large
        return np.inf

    price, volume = zip(*entries_hit)
    traded_volume = sum(volume)
    vwap = sum([price[i] * volume[i] for i in range(len(price))]) / traded_volume
    best_market_price = book[0][0]
    slippage = abs(best_market_price - vwap) / best_market_price
    return slippage


def place_market_order(
    exchange: ccxt.Exchange, ticker: str, amount: float, validate: bool = False
) -> Optional[Order]:
    """Place market order on exchange for ${amount} [units of asset] of ticker

    Args:
        exchange (ccxt.Exchange): Exchange interface
        ticker (str): Ticker pair to place order for
        amount (float): Amount of target asset to buy [units of asset]

    Returns:
        Optional[Order]: Order or None if no order was placed
    """  # noqa: B950
    if amount == 0:
        return None

    order_book = fetch_order_book(exchange=exchange, ticker=ticker)
    return _place_order(
        exchange=exchange,
        ticker=ticker,
        amount=amount,
        order_book=order_book,
        order_type="market",
        validate=validate,
    )


def place_limit_order(
    exchange: ccxt.Exchange, ticker: str, amount: float, validate: bool = False
) -> Optional[Order]:
    """Place limit order on exchange for ${amount} [units of asset] of ticker
    at mid price

    Args:
        exchange (ccxt.Exchange): Exchange interface
        ticker (str): Ticker pair to place order for
        amount (float): Amount of target asset to buy [units of asset]

    Returns:
        Optional[Order]: Order or None if no order was placed
    """  # noqa: B950
    if amount == 0:
        return None

    order_book = fetch_order_book(exchange=exchange, ticker=ticker)
    return _place_order(
        exchange=exchange,
        ticker=ticker,
        amount=amount,
        order_book=order_book,
        order_type="limit",
        validate=validate,
    )


def _place_order(
    exchange: ccxt.Exchange,
    ticker: str,
    amount: float,
    order_book: OrderBook,
    order_type: OrderType,
    validate: bool = False,
) -> Optional[Order]:
    """Place order on exchange for ${amount} [units of asset] of ticker

    Args:
        exchange (ccxt.Exchange): Exchange interface
        ticker (str): Ticker pair to place order for
        amount (float): Amount of target asset to buy [units of asset]
        order_book (OrderBook): Order book for ticker
        order_type (OrderType): Order type

    Returns:
        Optional[Order]: Order or None if no order was placed
    """  # noqa: B950
    if amount == 0:
        return None
    order_side = get_order_side(amount=amount)

    slippage = estimate_slippage(order_book=order_book, amount=amount)
    best_market_price = get_best_price(
        order_book=order_book, side=market_order_book_side(order_side=order_side)
    )
    avg_market_price = abs(slippage * best_market_price - best_market_price)
    limit_order_price = get_best_price(
        order_book=order_book, side=limit_order_book_side(order_side=order_side)
    )
    execution_price = avg_market_price if order_type == "market" else limit_order_price
    abs_amount = abs(amount)
    dollar_volume = abs_amount * execution_price
    # fmt: off
    bid = get_best_price(order_book, side="bids")
    ask = get_best_price(order_book, side="asks")
    mid = get_mid_price(order_book)
    logger.info(
        f"Ticker: {ticker}\n"
        f"\t order_type: {order_type}, {order_side} {abs_amount:.4f}@{execution_price:.6f} (dollar_volume: ${dollar_volume:.4f})\n"  # noqa: B950
        f"\t best_market_price: ${best_market_price:.6f}, avg_market_price: ${avg_market_price:.6f}, slippage: {slippage:.4f}\n"  # noqa: B950
        f"\t bid: ${bid:.6f}, ask: ${ask:.6f}, mid: ${mid:.6f}, limit_order_price: ${limit_order_price:.6f}\n"  # noqa: B950
    )
    # fmt: on

    # Sanity check
    if dollar_volume > MAX_SINGLE_TRADE_SIZE:
        logger.warning(
            "This trade exceeds the maximum allowed (dollar) size, skipping!!!"
        )
        return None

    if slippage < MAX_ACCEPTABLE_SLIPPAGE:
        # Place order
        existing_orders = exchange.fetch_open_orders(symbol=ticker)
        assert len(existing_orders) == 0, "Open orders exist for {ticker}!"
        expiretm = (
            MARKET_ORDER_TIMEOUT_TIME
            if order_type == "market"
            else LIMIT_ORDER_TIMEOUT_TIME
        )

        return exchange.create_order(
            symbol=ticker,
            type=order_type,
            side=order_side,
            amount=abs_amount,
            price=execution_price,
            params={
                "postOnly": order_type == "limit",  # Cannot be true for market orders
                "timeinforce": "GTD",
                "expiretm": f"+{expiretm}",
                "validate": validate,
            },
        )
    else:
        # Try again later when more liquidity is available.
        logger.warning(
            f"Slippage for {ticker} seems high ({slippage:.4f}), maybe try again later?"
        )
        return None


def limit_price_stale(
    order_price: float, market_price: float, order_side: OrderSide
) -> bool:
    """Determine whether limit order price is stale given current market price

    Args:
        order_price (float): Limit order price
        market_price (float): Market price (e.g. mid price)
        order_side (OrderSide): "buy" or "sell"

    Returns:
        bool: Whether limit order price is stale (and order should be canceled)
    """
    directional_slippage = (order_price - market_price) / market_price
    if order_side == "buy":
        # Don't want to overpay by too much for asset (order price > market price)
        return directional_slippage > MAX_ACCEPTABLE_SLIPPAGE
    else:
        # Don't want to undersell by too much for asset (order price < market price)
        return directional_slippage < -MAX_ACCEPTABLE_SLIPPAGE
