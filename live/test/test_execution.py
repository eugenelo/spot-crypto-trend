import unittest
from datetime import datetime
from typing import Dict, List, Optional

import ccxt
import numpy as np
import pytz
from ccxt.base.types import Order, OrderSide, OrderType

from live.constants import (
    LIMIT_ORDER_TIMEOUT_TIME,
    MARKET_ORDER_TIMEOUT_TIME,
    MAX_ACCEPTABLE_SLIPPAGE,
)
from live.execution import (
    _place_order,
    estimate_slippage,
    get_best_price,
    get_mid_price,
    get_order_side,
    limit_order_book_side,
    limit_price_stale,
    market_order_book_side,
)


class MockExchange(ccxt.Exchange):
    def __init__(self):
        self.id: int = 0
        self.open_orders: List[Order] = []

    def create_order(
        self,
        symbol: str,
        type: OrderType,
        side: OrderSide,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict] = None,
    ) -> Optional[Order]:
        self.id += 1
        order = Order(
            {
                "id": self.id,
                "timestamp": datetime.now(tz=pytz.UTC).timestamp(),
                "status": "open",
                "symbol": symbol,
                "type": type,
                "side": side,
                "price": price,
                "amount": amount,
                "filled": 0,
                "postOnly": params.get("postOnly", False),
            }
        )
        if params is not None:
            order.update(params)
        self.open_orders.append(order)
        return order

    def fill_order(self, order_id: int) -> Optional[Order]:
        # Find order by id
        found_order = False
        for _idx, order in enumerate(self.open_orders):
            if order["id"] == order_id:
                found_order = True
                break
        if not found_order:
            return None
        # Fill order, remove from open orders
        order = self.open_orders.pop(_idx)
        order["status"] = "filled"
        order["filled"] = order["amount"]
        return order

    def fetch_open_orders(self, symbol: str) -> List[Order]:
        return self.open_orders


class TestExecution(unittest.TestCase):
    def setUp(self):
        # Order book entry = (price, volume, timestamp)
        self.ticker = "BTC/USD"
        self.order_book = {
            "bids": [(9, 100, 0), (8.5, 100, 0), (1, 100, 0)],
            "asks": [(11, 100, 0), (15.5, 100, 0), (20, 100, 0)],
            "symbol": self.ticker,
        }

    def test_estimate_slippage(self):
        # Entire order can be filled at the best price, no slippage
        amount = 100
        expected_slippage = 0
        self.assertAlmostEqual(
            expected_slippage,
            estimate_slippage(order_book=self.order_book, amount=amount),
        )
        self.assertAlmostEqual(
            expected_slippage,
            estimate_slippage(order_book=self.order_book, amount=-amount),
        )

        # Combination of a few prices
        amount = 150
        expected_slippage_buy = abs(11 - (11 * 100 + 50 * 15.5) / 150) / 11
        self.assertAlmostEqual(
            expected_slippage_buy,
            estimate_slippage(order_book=self.order_book, amount=amount),
        )
        expected_slippage_sell = abs(9 - (100 * 9 + 50 * 8.5) / 150) / 9
        self.assertAlmostEqual(
            expected_slippage_sell,
            estimate_slippage(order_book=self.order_book, amount=-amount),
        )

        # Can't fill entire order, infinite slippage
        amount = 4000
        expected_slippage = np.inf
        self.assertAlmostEqual(
            expected_slippage,
            estimate_slippage(order_book=self.order_book, amount=amount),
        )
        self.assertAlmostEqual(
            expected_slippage,
            estimate_slippage(order_book=self.order_book, amount=-amount),
        )

    def test_get_best_price(self):
        self.assertEqual(9, get_best_price(order_book=self.order_book, side="bids"))
        self.assertEqual(11, get_best_price(order_book=self.order_book, side="asks"))

    def test_get_mid_price(self):
        self.assertAlmostEqual(10.0, get_mid_price(order_book=self.order_book))

    def test_get_order_side(self):
        self.assertEqual("buy", get_order_side(amount=+100))
        self.assertEqual("sell", get_order_side(amount=-100))

    def test_market_order_book_side(self):
        # Buying --> look at asks
        self.assertEqual("asks", market_order_book_side(order_side="buy"))
        # Selling --> look at bids
        self.assertEqual("bids", market_order_book_side(order_side="sell"))

    def test_limit_order_book_side(self):
        # Buying --> look at bids
        self.assertEqual("bids", limit_order_book_side(order_side="buy"))
        # Selling --> look at asks
        self.assertEqual("asks", limit_order_book_side(order_side="sell"))

    def test_limit_price_stale(self):
        market_price = 10

        # Limit buy lower than best ask
        order_price = 5
        order_side = "buy"
        self.assertFalse(
            limit_price_stale(
                order_price=order_price,
                market_price=market_price,
                order_side=order_side,
            )
        )

        # Limit sell higher than best bid
        order_price = 150
        order_side = "sell"
        self.assertFalse(
            limit_price_stale(
                order_price=order_price,
                market_price=market_price,
                order_side=order_side,
            )
        )

        # Stale ask
        order_price = market_price * (1 + MAX_ACCEPTABLE_SLIPPAGE * 2)
        order_side = "buy"
        self.assertTrue(
            limit_price_stale(
                order_price=order_price,
                market_price=market_price,
                order_side=order_side,
            )
        )

        # Stale bid
        order_price = market_price * (1 - MAX_ACCEPTABLE_SLIPPAGE * 2)
        order_side = "sell"
        self.assertTrue(
            limit_price_stale(
                order_price=order_price,
                market_price=market_price,
                order_side=order_side,
            )
        )

    def test_place_order(self):
        exchange = MockExchange()
        best_bid = get_best_price(order_book=self.order_book, side="bids")
        best_ask = get_best_price(order_book=self.order_book, side="asks")

        # Size too large
        order_type = "market"
        amount = +100
        order = _place_order(
            exchange=exchange,
            ticker=self.ticker,
            amount=amount,
            order_book=self.order_book,
            order_type=order_type,
        )
        self.assertIsNone(order)

        # Market order, expect price at bid/ask
        order_type = "market"
        amount = +10
        market_buy = _place_order(
            exchange=exchange,
            ticker=self.ticker,
            amount=amount,
            order_book=self.order_book,
            order_type=order_type,
        )
        self.assertIsNotNone(market_buy)
        self.assertEqual(order_type, market_buy["type"])
        self.assertEqual(get_order_side(amount), market_buy["side"])
        self.assertAlmostEqual(best_ask, market_buy["price"])
        self.assertFalse(market_buy["postOnly"])
        self.assertEqual("GTD", market_buy["timeinforce"])
        self.assertEqual(f"+{MARKET_ORDER_TIMEOUT_TIME}", market_buy["expiretm"])

        exchange.fill_order(market_buy["id"])

        amount = -10
        market_sell = _place_order(
            exchange=exchange,
            ticker=self.ticker,
            amount=amount,
            order_book=self.order_book,
            order_type=order_type,
        )
        self.assertIsNotNone(market_sell)
        self.assertEqual(order_type, market_sell["type"])
        self.assertEqual(get_order_side(amount), market_sell["side"])
        self.assertAlmostEqual(best_bid, market_sell["price"])
        self.assertFalse(market_sell["postOnly"])
        self.assertEqual("GTD", market_sell["timeinforce"])
        self.assertEqual(f"+{MARKET_ORDER_TIMEOUT_TIME}", market_sell["expiretm"])

        exchange.fill_order(market_sell["id"])

        # Limit order, expect price at mid
        order_type = "limit"
        amount = +10
        limit_buy = _place_order(
            exchange=exchange,
            ticker=self.ticker,
            amount=amount,
            order_book=self.order_book,
            order_type=order_type,
        )
        self.assertIsNotNone(limit_buy)
        self.assertEqual(order_type, limit_buy["type"])
        self.assertEqual(get_order_side(amount), limit_buy["side"])
        self.assertAlmostEqual(best_bid, limit_buy["price"])
        self.assertTrue(limit_buy["postOnly"])
        self.assertEqual("GTD", limit_buy["timeinforce"])
        self.assertEqual(f"+{LIMIT_ORDER_TIMEOUT_TIME}", limit_buy["expiretm"])

        exchange.fill_order(limit_buy["id"])

        amount = -10
        limit_sell = _place_order(
            exchange=exchange,
            ticker=self.ticker,
            amount=amount,
            order_book=self.order_book,
            order_type=order_type,
        )
        self.assertIsNotNone(limit_sell)
        self.assertEqual(order_type, limit_sell["type"])
        self.assertEqual(get_order_side(amount), limit_sell["side"])
        self.assertAlmostEqual(best_ask, limit_sell["price"])
        self.assertTrue(limit_sell["postOnly"])
        self.assertEqual("GTD", limit_sell["timeinforce"])
        self.assertEqual(f"+{LIMIT_ORDER_TIMEOUT_TIME}", limit_sell["expiretm"])

        # Can't place order with existing open order
        with self.assertRaises(AssertionError):
            _place_order(
                exchange=exchange,
                ticker=self.ticker,
                amount=amount,
                order_book=self.order_book,
                order_type=order_type,
            )


if __name__ == "__main__":
    unittest.main()
