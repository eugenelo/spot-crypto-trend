import ccxt

from live.constants import MAX_ACCEPTABLE_SLIPPAGE, MAX_SINGLE_TRADE_SIZE


def place_market_order(exchange: ccxt.Exchange, ticker: str, amount: float):
    """Place market order on exchange for ${amount} (units of asset, NOT BASE_CURRENCY) of ticker

    Args:
        exchange (ccxt.Exchange): Exchange interface
        ticker (str): Ticker pair to place order for
        amount (float): Amount of target asset to buy [units of asset, NOT BASE_CURRENCY]
    """  # noqa: B950
    if amount == 0:
        return
    limit = 500
    order_book = exchange.fetch_order_book(symbol=ticker, limit=limit)
    assert order_book["symbol"] == ticker
    if amount > 0:
        # Buying, look at asks
        side = "asks"
        order_side = "buy"
    else:
        # Selling, look at bids
        side = "bids"
        order_side = "sell"
    book = order_book[side]

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
    price, volume = zip(*entries_hit)
    traded_volume = sum(volume)
    vwap = sum([price[i] * volume[i] for i in range(len(price))]) / traded_volume
    best_price = book[0][0]
    slippage = abs(best_price - vwap) / best_price
    # Can't sell more than we own
    if amount < 0:
        traded_volume = min(traded_volume, -amount)

    bid = order_book["bids"][0][0]
    ask = order_book["asks"][0][0]
    mid = (bid + ask) / 2
    print(
        f"Ticker: {ticker}, Best Price: ${best_price:.4f}, vwap: ${vwap:.4f}, slippage:"
        f" {slippage:.4f}, volume: {traded_volume:.4f}, dollar_volume:"
        f" ${traded_volume * vwap:.4f}, bid: ${bid:.4f}, ask: ${ask:.4f}, mid:"
        f" ${mid:.4f}"
    )

    # Sanity check
    dollar_volume = traded_volume * vwap
    if dollar_volume > MAX_SINGLE_TRADE_SIZE:
        print("This trade is very large!!! Are you sure???")
        return None

    if slippage < MAX_ACCEPTABLE_SLIPPAGE:
        # Blast a market order
        existing_orders = exchange.fetch_open_orders(symbol=ticker)
        assert len(existing_orders) == 0, "Open orders exist for {ticker}!"
        return exchange.create_order(
            symbol=ticker,
            type="market",
            side=order_side,
            amount=traded_volume,
            # params={"validate": True},
        )
    else:
        # Concerning. Try again later when volume is higher.
        print(f"Slippage seems high ({slippage:.4f}), maybe try again later?")
        return None
