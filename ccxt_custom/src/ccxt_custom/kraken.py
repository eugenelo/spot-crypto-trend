from typing import Any, Optional

import ccxt


class KrakenExchange(ccxt.kraken):
    """Custom CCXT Kraken interface, supports missing functionality."""

    def parse_ohlcv(self, ohlcv, market=None) -> list:
        return [
            self.safe_timestamp(ohlcv, 0),  # timestamp
            self.safe_number(ohlcv, 1),  # open
            self.safe_number(ohlcv, 2),  # high
            self.safe_number(ohlcv, 3),  # low
            self.safe_number(ohlcv, 4),  # close
            self.safe_number(ohlcv, 5),  # vwap
            self.safe_number(ohlcv, 6),  # volume
        ]

    def fetch_trades(
        self,
        symbol: str,
        since: Optional[int] = None,
        limit: Optional[int] = None,
        params: Optional[dict] = None,
    ):
        """
        get the list of most recent trades for a particular symbol
        :see: https://docs.kraken.com/rest/#tag/Spot-Market-Data/operation/getRecentTrades
        :param str symbol: unified symbol of the market to fetch trades for
        :param int [since]: timestamp in ns of the earliest trade to fetch
        :param int [limit]: the maximum amount of trades to fetch
        :param dict [params]: extra parameters specific to the exchange API endpoint
        :returns Trade[]: a list of `trade structures <https://docs.ccxt.com/#/?id=public-trades>`
        """  # noqa: B950
        self.load_markets()
        market = self.market(symbol)
        id = market["id"]
        request = {
            "pair": id,
        }
        # https://support.kraken.com/hc/en-us/articles/218198197-How-to-pull-all-trade-data-using-the-Kraken-REST-API
        # https://github.com/ccxt/ccxt/issues/5677
        if since is not None:
            # request['since'] = str(since) + "000000"
            request["since"] = str(since)  # expected to be in nanoseconds
        if limit is not None:
            request["count"] = limit

        if params is not None:
            request = self.extend(request, params)
        response = self.publicGetTrades(request)
        #
        #     {
        #         "error": [],
        #         "result": {
        #             "XETHXXBT": [
        #                 ["0.032310","4.28169434",1541390792.763,"s","l",""]
        #             ],
        #             "last": "1541439421200678657"
        #         }
        #     }
        #
        result = response["result"]
        trades = result[id]
        # trades is a sorted array: last(most recent trade) goes last
        length = len(trades)
        if length <= 0:
            return []
        lastTrade = trades[length - 1]
        lastTradeId = self.safe_string(result, "last")
        lastTrade.append(lastTradeId)
        trades[length - 1] = lastTrade
        return self.parse_trades(trades, market, since=None, limit=limit)

    def fetch_my_trades(
        self,
        symbol: Optional[str] = None,
        end: Optional[Any] = None,
        limit: Optional[int] = None,
        params: Optional[dict] = None,
    ):
        """
        fetch all trades made by the user
        :see: https://docs.kraken.com/rest/#tag/Account-Data/operation/getTradeHistory
        :param str symbol: unified market symbol
        :param int [end]: Ending unix timestamp or trade tx ID of results (inclusive)
        :param int [limit]: the maximum number of trades structures to retrieve
        :param dict [params]: extra parameters specific to the exchange API endpoint
        :returns Trade[]: a list of `trade structures <https://docs.ccxt.com/#/?id=trade-structure>`
        """  # noqa: B950
        self.load_markets()
        request = {}
        if end is not None:
            request["end"] = end
        if params is not None:
            request = self.extend(request, params)
        response = self.privatePostTradesHistory(request)
        trades = response["result"]["trades"]
        ids = list(trades.keys())
        for i in range(0, len(ids)):
            trades[ids[i]]["id"] = ids[i]
        market = None
        if symbol is not None:
            market = self.market(symbol)
        return self.parse_trades(trades, market, since=None, limit=limit)

    def fetch_ledger(
        self,
        code: Optional[str] = None,
        end: Optional[Any] = None,
        limit: Optional[int] = None,
        params: Optional[dict] = None,
    ):
        """
        fetch the history of changes, actions done by the user or operations that altered balance of the user
        :see: https://docs.kraken.com/rest/#tag/Account-Data/operation/getLedgers
        :param str code: unified currency code, default is None
        :param int [end]: Ending unix timestamp or ledger ID of results (inclusive)
        :param int [limit]: max number of ledger entrys to return, default is None
        :param dict [params]: extra parameters specific to the exchange API endpoint
        :param int [params.until]: timestamp in ms of the latest ledger entry
        :returns dict: a `ledger structure <https://docs.ccxt.com/#/?id=ledger-structure>`
        """  # noqa: B950
        # https://www.kraken.com/features/api#get-ledgers-info
        self.load_markets()
        request = {}
        currency = None
        if code is not None:
            currency = self.currency(code)
            request["asset"] = currency["id"]
        if end is not None:
            request["end"] = end
        if params is not None:
            request = self.extend(request, params)
        response = self.privatePostLedgers(request)
        result = self.safe_value(response, "result", {})
        ledger = self.safe_value(result, "ledger", {})
        keys = list(ledger.keys())
        items = []
        for i in range(0, len(keys)):
            key = keys[i]
            value = ledger[key]
            value["id"] = key
            items.append(value)
        return self.parse_ledger(items, currency, since=None, limit=limit)
