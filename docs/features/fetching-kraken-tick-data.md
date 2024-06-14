# Fetching Kraken Tick Data

## Fetching via API

Kraken tick data can be fetched from the API with the `//data:fetch_kraken_tick_data` binary. Refer to [Challenges of Fetching Kraken Tick Data](../challenges-of-fetching-kraken-tick-data.md) for a peek of what goes on under the hood.


### Specifying Query Lookback

The lookback period can be specified as either a number of days in the past (`--lookback_days`) or a Unix timestamp (`--since`). If using `--lookback_days`, the query start stamp will be computed as follows:

- Take the current date in UTC time and subtract `lookback` days to get `target_date`
- Set start time to midnight UTC on `target_date`

For example, if the current time were `2024-06-10T12:00:00+0` and `--lookback_days=30`, the start stamp would be set to `2024-05-11T00:00:00+0 --> 1715385600`.

There's an additional option to automatically fetch data from the latest available timestamp (per ticker) given an existing dataset, see [Automatic Updates](#automatic-updates).


### Single Ticker

`--ticker` exists for specifying individual tickers of interest.

**Example:** To fetch the entire trade history for a single asset (e.g. `BTC/USD`) from Kraken, use
```
bazel run //data:fetch_kraken_tick_data -- --output_dir {output_dir} --ticker BTC/USD --since 0
```

Tickers can be specified as either raw exchange symbols (e.g. [`XBTUSD` for Kraken](https://support.kraken.com/hc/en-us/articles/360001206766-Bitcoin-currency-code-XBT-vs-BTC)) or [unified ccxt market symbols](https://docs.ccxt.com/#/?id=contract-naming-conventions) (e.g. `BTC/USD`).


### Multiple Tickers

If `--ticker` is empty, the binary will fetch data for all `USD` asset pairs by default.

The list of "available asset pairs" is produced using ccxt's `fetch_tickers()` function (and then filtered to retain only `USD` pairs).

**Example:** To fetch the entire trade history for all `/USD` tickers from Kraken, use
```
bazel run //data:fetch_kraken_tick_data -- --output_dir {output_dir} --since 0
```


### Output Format

The data fetched from this binary will be formatted into a table with the structure `[<id>, <timestamp [s]>, <price>, <volume>, <buy/sell>, <market/limit>, <ticker>]`.

If `--output_dir` is not specified, the table will be printed before the program exits. Otherwise, the data will be written to CSV files  **at a monthly cadence**. For example, running
```
bazel run //data:fetch_kraken_tick_data -- --output_dir {output_dir} --ticker W/USD --since 1711929600 --end 1719791999
```
will produce three files in the `WUSD` subdirectory:
```
{output_dir}/WUSD/WUSD_2024-04-01_2024-05-01.csv
{output_dir}/WUSD/WUSD_2024-05-01_2024-06-01.csv
{output_dir}/WUSD/WUSD_2024-06-01_2024-07-01.csv
```

`WUSD_2024-04-01_2024-05-01.csv` will contain all trades which occurred between `2024-04-01T00:00:00+0` inclusive and `2024-05-01T00:00:00+0` exclusive, `WUSD_2024-05-01_2024-06-01.csv` will contain all trades which occurred between `2024-05-01T00:00:00+0` inclusive and `2024-06-01T00:00:00+0` exclusive, and so on.


### Incremental / Automatic Updates

An existing dataset of tick data can be incrementally updated without redundantly requerying the ticker's entire history with the command:
```
bazel run //data:fetch_kraken_tick_data -- --output_dir {output_dir} --from_latest
```

where `--output_dir` should point to a non-empty directory containing the existing dataset.

This requires that the existing dataset is contained in CSV files organized under the subdirectory structure specified in [Output Format](#output-format). If so, then for each ticker,

1. The trade history from the **latest file only** will be loaded into memory to get the timestamp of the latest trade on file
2. This latest trade timestamp is decremented by a small buffer (1ms) to avoid missing trades
3. This decremented timestamp is used as the starting `since` for the ticker's query loop


Once a new batch of trades has been queried, it needs to be validated and merged with the existing dataset before being written to disk. For each monthly file,

1. The existing trades from the monthly file are loaded into a dataframe `df_existing`
2. The trades from the new batch which fall into this month's time window are stored in a dataframe `df_new`
3. `df_existing` and `df_new` are checked for schema consistency and then vertically stacked into `df_combined`
4. `df_combined` is sorted by `id`
5. Duplicates in `df_combined` are detected via `id` and are dropped
6. `df_combined` is written to disk, completely replacing the previous monthly file


## Downloading CSV from Kraken

Kraken provides tick data in the form of downloadable CSV files. These files are updated quarterly and can be downloaded from [Kraken's website](https://support.kraken.com/hc/en-us/articles/360047543791-Downloadable-historical-market-data-time-and-sales-). As of this writing, these CSV files are formatted as `<timestamp [s]>, <price>, <volume>` with no headers.


Note that the symbols used to name these files are Kraken-specific (**not** ccxt unified).
