# ccxt_custom

This project contains custom ccxt hookups to get around limitations of the native [ccxt Kraken API](https://github.com/ccxt/ccxt/blob/master/python/ccxt/kraken.py) (particularly around fetching data).


## Retaining VWAP from fetched OHLCV data

The native API discards the Volume Weighted Average Price (VWAP) when fetching OHLCV data via `fetch_ohlcv()`. We modify the [`parse_ohlcv()`](src/ccxt_custom/kraken.py#L9) ccxt function to retain the VWAP.


## Fetching complete trade history for a Kraken Asset Pair

There are several challenges with fetching trade history (i.e. tick data) for asset pairs from the Kraken API. These issues can be exacerbated for very liquid asset pairs (e.g. `BTC/USD`, `USDT/USD`, etc.), but may also cause problems for less liquid tickers due to bugs in the Kraken API.

To resolve these issues, we modify the [`fetch_trades()`](src/ccxt_custom/kraken.py#L20) ccxt function in several ways:

- The `since` parameter is now expected to be in nanoseconds, not milliseconds.
- `since` is no longer used to filter results in `parse_trades()`. The results should now be natively filtered by the Kraken REST API itself.

Refer to [Challenges of Fetching Kraken Tick Data](../docs/challenges-of-fetching-kraken-tick-data.md) for a detailed writeup.


## Fetching complete personal trade/ledger history for a user

The Kraken APIs for both fetching [user trade history](https://docs.kraken.com/rest/#tag/Account-Data/operation/getTradeHistory) and [user ledger history](https://docs.kraken.com/rest/#tag/Account-Data/operation/getLedgers) share some similarities:

- Both APIs return (at most) 50 results per query, or less if there is insufficient history given the query parameters.
- Officially, both APIs have `start` (exclusive) and `end` (inclusive) parameters for narrowing results. These can be specified as either Unix timestamps (seconds or nanoseconds) or tx/ledger IDs respectively. However, in practice, the `end` parameter is prioritized and thus becomes the only parameter that matters.
  - If both `start` and `end` are null, the API will return the 50 most recent trades/ledger entries for the user by default.
  - If `start` is non-null and `end` is null, the API will still return the 50 most recent entries.
  - If both `start` and `end` are non-null, the API will return the 50 entries preceeding and including `end`.

The corresponding ccxt functions `fetch_my_trades` and `fetch_ledger` share similarities as well:

- Both functions expect `since` (i.e. `start`) to be specified in milliseconds. Similar to the [tick data ccxt API](../docs/challenges-of-fetching-kraken-tick-data.md), this parameter will be converted to a whole number of seconds by flooring before being passed to the Kraken API. As we covered above, `start` is not useful as a query parameter. We could circumvent this and still use the stock ccxt functions by passing key-value pairs for `end` via the `params` dict. These would get passed onto the REST API as is, enabling us to specify either nanosecond Unix stamps or even tx/ledger IDs. However, this solution is clunky and does not disambiguate which parameters actually matter.
- Additionally, both functions use the millisecond value of `since`, if non-null, to filter the Kraken API results from the function output. To disable this feature, the user would need to remember to keep `since=None` (the default value) or use a small enough timestamp. From testing, unexpected results may occur when using `since=0` (valid transactions are filtered out) vs `since=None` or even `since=1`, despite 1ms converting to 0s internally. To make use of this feature, the user would need to be mindful of the units mismatch between `since` [ms] and `end` [s/ns] in `params`.

This minor flexibility is not worth the accompanying confusion. We modify both [`fetch_my_trades()`](src/ccxt_custom/kraken.py#L76) and [`fetch_ledger()`](src/ccxt_custom/kraken.py#L76) to replace `since` with `end` as an optional kwarg which can be specified as either a Unix timestamp in [s/ns] or a tx/ledger ID. To fetch a user's complete history, we make multiple queries in a loop shifting the `end` parameter back until the first transaction has been fetched (see [`live/utils/fetch_my_trades()`](../live/src/live/utils.py#L159)).
