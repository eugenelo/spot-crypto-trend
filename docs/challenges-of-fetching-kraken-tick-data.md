# Challenges of Fetching Kraken Tick Data

There are several challenges with fetching trade history (i.e. tick data) for asset pairs from the Kraken API.


## API Overview

First, an overview of what we're working with:

- The [Kraken REST API for fetching trades](https://docs.kraken.com/rest/#tag/Spot-Market-Data/operation/getRecentTrades) takes as query parameters (1) `pair`: the asset pair, (2) `since`: the query start UNIX timestamp, and (3) `count`: the desired number of results, up to 1000. The result is an array of trade entries of the form `[<price>, <volume>, <time>, <buy/sell>, <market/limit>, <miscellaneous>, <trade_id>]`, plus a `last` string ID field.

  - The query stamp can be specified in either fractional seconds or nanoseconds. For example, both of the following queries are valid:
    ```
    # Timestamp in fractional seconds
    $ curl "https://api.kraken.com/0/public/Trades?pair=XBTUSD&since=1616666442.304882029&count=5"
    {"error":[],"result":{"XXBTZUSD":[["53018.10000","0.04628248",1616666442.2951703,"b","m","",33185849],["53019.50000","0.00168601",1616666442.296356,"b","m","",33185850],["53041.00000","0.00117527",1616666442.2980762,"b","m","",33185851],["53047.40000","0.00087289",1616666442.2997677,"b","m","",33185852],["53048.90000","0.00640000",1616666442.3010457,"b","m","",33185853]],"last":"1616666442301045766"}}

    # Timestamp in nanoseconds
    $ curl "https://api.kraken.com/0/public/Trades?pair=XBTUSD&since=1616666442304882029&count=5"
    {"error":[],"result":{"XXBTZUSD":[["53050.00000","0.00000004",1616666442.304882,"b","m","",33185856],["53051.30000","0.10000000",1616666442.305828,"b","m","",33185857],["53059.10000","0.03292369",1616666442.3068447,"b","m","",33185858],["53059.20000","0.37719475",1616666442.308082,"b","m","",33185859],["53018.50000","0.00809682",1616666442.3903322,"s","l","",33185860]],"last":"1616666442390332264"}}
    ```
    Note that the query results are slightly different even though the timestamps are equivalent. In particular, the nanosecond query contains results that are strictly at or after the query stamp whereas the fractional second query includes some results preceeding the stamp.
  - The Kraken REST API returns a `last` field as the *"ID to be used as `since` when polling for new trade data."* This appears to be exactly the timestamp in nanoseconds of the next trade for the asset pair.
  - Despite supporting nanosecond `since` parameter values, the `time` field for each trade entry result is specified in hundreds of nanoseconds. The only way to get a valid nanosecond trade timestamp is from `last`.
  - A `trade_id` is included for each trade entry. This is a sequential integer which (*usually*) starts from 1 for the first trade per ticker. In other words, the `trade_id` for any particular trade is (*usually*) an inclusive count of the total number of trades that have been executed on the Kraken exchange for that asset pair (see [Data validation](#7-data-validation) for clarification of *"usually"*).
- The [ccxt API](https://github.com/ccxt/ccxt/blob/master/python/ccxt/kraken.py) provides a `fetch_trades()` function for fetching trades. This function takes as input the `since` parameter specified in milliseconds. ccxt will then, under the hood, convert `since` [ms] to a whole number of seconds by flooring, before finally passing this into the Kraken REST API query. Finally, ccxt will parse the (maximum 1000) results from the REST API to discard any trade entries which occur before `since` [ms], and then return the filtered output from the function.

  Consider the following example:
  ```
  $ kraken = ccxt.kraken({"enableRateLimit": True})
  $ trades = kraken.fetch_trades(symbol="BTC/USD", since=1616666442305, limit=5000)  # Intentionally set limit above the REST API limit of 1000
  ```

  Under the hood, this will construct an API request of the form:
  ```
  https://api.kraken.com/0/public/Trades?pair=XBTUSD&since=1616666442&limit=5000
  ```

  Contained in the result will be 1000 trade entries, with the first entry being:
  ```
  ["53018.10000","0.04628248",1616666442.2951703,"b","m","",33185849]
  ```

  The final result from ccxt will be a list of 992 trades, with the first entry being:
  ```
  ["53051.30000","0.10000000",1616666442.305828,"b","m","","33185857"]
  ```
  which is exactly 8 trades from the original first entry.


From this example, and from [ccxt's own documentation on pagination](https://github.com/ccxt/ccxt/wiki/Manual#pagination), it's clear that in order to fetch a complete history of trades, we will need to make multiple calls to the API in a loop, updating the `since` parameter for each call and defining some criteria for loop termination. Some sensible criteria may include checking that the timestamp (`time`) or `trade_id` of the last trade entry in a given batch of results is not a repeat / duplicate / older than the latest data already queried.



## Challenges

Now, the challenges:

### 1. Kraken API rate limits

The Kraken API [rate limits](https://support.kraken.com/hc/en-us/articles/206548367-What-are-the-API-rate-limits-#1) at 1 public endpoint call per second $=$ 1000 trades per second.
   - As of 2024/06/08 (unix stamp `1717804800`), there have been ~70.9M trades of `BTC/USD` alone. This indicates at least 19+ hours of continuous querying to fetch the complete Kraken history for `BTC/USD`.
   - To minimize both the chances and the impact of running out of RAM or having to restart the process, you would likely want to batch writes of the data to disk. However, too small a choice of batch size could adversely impact the ability to fix corrupt `trade_id`s in the data. (see [(6)](#6-invalid-trade_id-of-0)).


### 2. Insufficient resolution for `since`

The ccxt interface only supporting millisecond resolution (really, second resolution) for `since` can be problematic when there are more than 1000 executed trades for a ticker within a single millisecond (e.g. [XBTUSD @ 1705792331.777](https://api.kraken.com/0/public/Trades?pair=XBTUSD&since=1705792331.777)).
  ```
  "result":{"XXBTZUSD":[
        ["41773.90000","0.00214117",1705792331.7778208,"s","m","",66432458],
        ["41770.10000","0.16702599",1705792331.7778208,"s","m","",66432459],
        ...,
        ["40100.00000","0.08076991",1705792331.7778342,"s","m","",66433456],
        ["40100.00000","0.20000000",1705792331.7778342,"s","m","",66433457]],
  ```
   - You would never be able to query beyond the first 1000 trades in this millisecond without a higher resolution query parameter. Thus, your data fetching loop would either continue querying the same data indefinitely or terminate prematurely after detecting that no new trades can be fetched.


### 3. `last` missing from query results

Occasionally, the `last` field will be missing from the query results.
   - This seems to be a bug that happens sporadically.
   - This can cause problems if you solely are relying on `last` to advance your query loop.


### 4. Duplicate entries in successive queries

Since `since` is timestamp-based and not id-based, it's easy to accidentally query for duplicates (e.g. to fetch trades [1, 1000], then [1000, 1999]).
   - You would likely want to dededuplicate trade entries based on `trade_id`, `time`, or the entire contents of the entry, both between queries within a single session as well as when updating an existing dataset.


### 5. Non-unique timestamps for trades

The hundred-nanosecond resolution timestamp (`time`) is not necessarily unique for each trade of the same ticker (e.g. [XBTUSD @ 1705792331.7778225](https://api.kraken.com/0/public/Trades?pair=XBTUSD&since=1705792331777822495)).
   ```
   "result":{"XXBTZUSD":[
       ["41450.00000","0.03406341",1705792331.7778225,"s","m","",66432589],
       ["41450.00000","0.00071142",1705792331.7778225,"s","m","",66432590],
       ["41450.00000","0.05277035",1705792331.7778225,"s","m","",66432591],
       ["41450.00000","0.03728435",1705792331.7778225,"s","m","",66432592],...
   ```
   - This occurs when multiple trades for the same ticker are executed within a 100 nanosecond window.
   - Occurences like these invalidate data validation rules which check that each trade has a unique timestamp.
   - The only way to chronologically sort these trades is with `trade_id`. However...


### 6. Invalid `trade_id` of 0

**UPDATE: The Kraken team has reported that this issue is fixed as of 2024/06/13. This section is kept for posterity.**

Occasionally, the returned `trade_id` for a trade entry will be 0 (e.g. [APTUSD @ 1676437015.3481097](https://api.kraken.com/0/public/Trades?pair=APTUSD&since=1676437015348109827)).
   ```
   "result":{"APTUSD":[
        ["14.41500000","59.73063489",1676437015.3481097,"b","l","",0],
        ["14.41860000","6.36173259",1676437015.3481467,"b","l","",0],
        ["14.29880000","4.28290340",1676442424.2578423,"s","l","",0],...
   ```
   - This issue occurs sporadically and may persist for any number of consecutive entries. The issue may resolve itself within 1000 entries, or it may persist for up to tens/hundreds of thousands of entries. As the page size for Kraken tick data is 1000 trades, you could easily query a page of results where every entry has `trade_id` 0 depending on your `since`.
   - This issue can cause:
     - premature termination of the data fetching loop if your criteria for query completion consists solely of checking for duplicate `trade_id`s
     - problems with checking for/discarding duplicates based solely on `trade_id`
     - problems with data validation rules which expect that a set of $N$ unique trades for a given ticker with minimum `trade_id` $=t_0$ will contain exactly one entry with `trade_id` $=t_0 + i$ for each $i \in [0,...,N-1]$, which would normally be a valid assumption
   - Fortunately, it seems that all non-zero `trade_id`s are valid regardless of the presence of invalid `tid=0` entries. For example, if the last valid `tid` was 42 and the next 100,000 trades were invalid with `tid=0`, the 100,001-th valid trade would have a `tid` of $42 + 100000 + 1 = 100043$.
     - This implies we can correct invalid `tid=0` entries by linearly interpolating between two non-zero `trade_id`s, provided that both of the sandwiching non-zero values are known.
     - As alluded to above, this has implications for batch writing. To guarantee that `trade_id`s will be corrected prior to writing to disk, you must be able query an arbitrarily large batch size until the next valid, non-zero `trade_id` is encountered. In practice, a batch size on the order of 100k (~3MB) was sufficient to fix all historical cases. You could also choose to first write the invalid data to disk/DB and then fix the `trade_id`s afterwards, so long as you're careful about maintaining the queried order of the results (you can't rely solely on timestamp due to [(5)](#5-non-unique-timestamps-for-trades)).


## Solutions

Solutions for the above problems have been implemented as follows:

### 1. Write batching / Iterative updates

Trade entries are queried until either [no new data is available](#4-loop-termination-logic) or a maximum batch size has been accumulated (312,500 trades or 10MB by default). Batches of entries are first [deduplicated](#5-handling-of-duplicates), [corrected](#6-correcting-invalid-trade_id), and [checked for validity](#7-data-validation) before being written to disk. Thus, once a trade entry has been written to file, it is assumed to be valid.

New updates can be merged with existing datasets with an appropriately specified configuration. Deduplication and data validation of the complete dataset will be performed before any updates are written to disk.


### 2. Nanosecond resolution for query parameters

The [`fetch_trades()`](../ccxt_custom/src/ccxt_custom/kraken.py#L20) ccxt function is modified as follows:

- The `since` parameter is now to be specified in nanoseconds, not milliseconds. Contrary to native ccxt which converts `since` [ms] to [s], `since` [ns] will be passed to the Kraken API without conversion. This enables us to directly update the query parameter using `last` as intended by the Kraken API spec.
- `since` is no longer used to filter results in `parse_trades()`. [Handling of duplicates](#5-handling-of-duplicates) is done outside of this function.


### 3. Loop advancement - parameter updating

Between successive queries, `since` is updated as follows:

- If the previous query failed due to a network error, retry with the same `since`.
- If the previous query was successful and `last` is available, set `since = last` for the next query.
- If the previous query was successful but `last` is missing, convert the `time` of the most recent trade from [s] to [ns], decrement a small epsilon (10ns), and use this as `since` for the next query.
  - We decrement by eps to avoid missing trades, but this requires [duplicate management](#5-handling-of-duplicates).


### 4. Loop termination logic

Due to batching, there are two query loops: (1) the inner loop for the Kraken API and (2) the outer loop for writing batches of entries to disk.

The inner query loop for new trade entries is terminated if:

- ***(Error)*** The same query exceeds the max number of retries allotted for network errors
- ***(Success)*** A valid response is received with an empty `result` (no trades)
- ***(Success)*** No new trades were fetched with the latest query
  - The `trade_id` of the 1000-th entry from the previous result was non-zero && equals the `trade_id` of the 1000-th entry from the current result
- ***(Success)*** [Maximum batch size](#1-write-batching--iterative-updates) is reached

The outer loop is terminated if:

- ***(Success)*** No new trades were fetched with the latest query
- ***(Error)*** Skipped trades are detected between the earliest trade in the current batch and the latest trade in the existing dataset
  - Gaps in `trade_id` are used as the detection criteria
- ***(Error)*** [`trade_id` interpolation](#6-correcting-invalid-trade_id) fails for the current batch of results
  - A message will prompt the user to retry with a larger maximum batch size; the program won't automatically exceed the limit


### 5. Handling of duplicates

Prior to [`trade_id` interpolation](#6-correcting-invalid-trade_id) (whether required or not), trade entries in the same batch are deduplicated based on the entire contents of the entry. So two entries will be detected as duplicates, and one will be dropped, if each of the `[<price>, <volume>, <time>, <buy/sell>, <market/limit>, <trade_id>]` fields are exactly equal. The assumption is that, though possible, it is very unlikely for two trades of identical size to have been executed at the same price within the same 100ns window and also have corrupted `trade_id`s of 0. And if such cases do occur, they can still be caught with [data validation](#7-data-validation) rules later.

After this initial deduplication, [`trade_id` interpolation](#6-correcting-invalid-trade_id) will be done to fix invalid `trade_id`s of 0. Then we can deduplicate the current batch of trades with the existing dataset based on `trade_id` alone, before finally combining the two into a single dataset and writing to file.


### 6. Correcting invalid `trade_id`

After a batch of entries have been queried, we check for whether entries with `trade_id=0` exist. If any, we try to fix these invalid ids with linear interpolation.

As mentioned above, the `trade_id=0` bug is manageable due to the fact that all non-zero `trade_id`s are valid. We can correct invalid `tid=0` entries by linearly interpolating between two non-zero `trade_id`s. In fact, if we assume that non-zero `trade_id`s are valid and the sequence increment is 1, we only need one non-zero value per batch to fix all of the zeros.

For example,

- `[1,2,0,0,5,6,0,0,9]` would be converted to `[1,2,3,4,5,6,7,8,9]`
- `[1,0,0,0]` would be converted to `[1,2,3,4]`
- `[0,0,0,4]` would be converted to `[1,2,3,4]`
- `[0,2,0,0,0,6,7,0,0]` would be converted to `[1,2,3,4,5,6,7,8,9]`


Correctness relies on the assumption that non-zero `trade_id`s are valid. A case like `[1,2,0,0,9]` would be converted to `[1,2,3,4,9]` (we forward-fill before backfilling) and would indicate that entries `[5,6,7,8]` were missing from the batch. This would need to be caught during [data validation](#7-data-validation) and indicate that the missing data should be re-queried.

If we can't apply this fix due to not having a single non-zero `trade_id` in the batch, an error will be reported and the user will be prompted to retry the process with a larger maximum batch size.


### 7. Data validation

Trade entries / tick data are validated primarily using `trade_id`. This is why it's important that all [invalid `trade_id`s are corrected](#6-correcting-invalid-trade_id) prior to running these checks.

A valid tick dataset is expected to be complete and not contain duplicates:

- With $N$ total trades and initial `trade_id` $=t_0$, we expect our dataset to contain exactly one entry with `trade_id` $=t_0 + i$ for each $i \in [0,...,N-1]$
  - By default, $t_0 = 1$ for most tickers, but there are exceptions to this rule. [`HFTUSD`](https://api.kraken.com/0/public/Trades?pair=HFTUSD&since=0) is one such example with the first available trade having a `trade_id` of 3.
- Given a ticker with min `trade_id` $t_{min}$ and max `trade_id` $t_{max}$, the total number of trades $N = t_{max} - t_{min} + 1$
