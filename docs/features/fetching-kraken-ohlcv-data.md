# Fetching Kraken OHLCV Data

## Fetching via API

Kraken tick data can be fetched from the API with the `//data:fetch_kraken_ohlcv_data` binary.


### Specifying Time Frame Interval

The time frame interval is specified via `--data_frequency`. This arg is passed directly as the `timeframe` arg to `ccxt.fetch_ohlcv()`. Valid values for this include: `[1m, 1h, 1d, etc.]`.


### Specifying Query Lookback

The lookback period can be specified as a number of days in the past (`--lookback_days`). The query start stamp (`since` for the [REST API](https://docs.kraken.com/rest/#tag/Spot-Market-Data/operation/getOHLCData)) will be computed by taking the current datetime in UTC and subtracting exactly `lookback` days (or $86400 \times$ `lookback` seconds) to get the target stamp.


### API Limits

The [REST API](https://docs.kraken.com/rest/#tag/Spot-Market-Data/operation/getOHLCData) is limited to returning 720 OHLC data points since the query stamp. The API always returns the most recent data points from the latest time frame. This means that, using this binary,

- Only the most recent 720 days of 1d OHLCV data can be fetched
- Only the most recent 720 hours (30 days) of 1h OHLCV data can be fetched
- Only the most recent 720 minutes (12 hours) of 1m OHLCV data can be fetched


### Single Ticker

`--ticker` exists for specifying individual tickers of interest.

**Example:** To fetch the last 30 days of 1d OHLCV data for `BTC/USD` from Kraken, use
```
bazel run //data:fetch_kraken_ohlcv_data -- --output_path {output_path} --ticker BTC/USD --data_frequency 1d --lookback_days 30
```

Tickers can be specified as either raw exchange symbols (e.g. [`XBTUSD` for Kraken](https://support.kraken.com/hc/en-us/articles/360001206766-Bitcoin-currency-code-XBT-vs-BTC)) or [unified ccxt market symbols](https://docs.ccxt.com/#/?id=contract-naming-conventions) (e.g. `BTC/USD`).


### Multiple Tickers

If `--ticker` is empty, the binary will fetch data for all `USD` asset pairs by default.

The list of "available asset pairs" is produced using ccxt's `fetch_tickers()` function (and then filtered to retain only `USD` pairs).

**Example:** To fetch the last 30 days of 1d OHLCV data for all `/USD` tickers from Kraken, use
```
bazel run //data:fetch_kraken_ohlcv_data -- --output_path {output_path} --data_frequency 1d --lookback_days 30
```


### Output Format

The data fetched from this binary will be formatted into a table with the structure `[<datetime_UTC>, <open>, <high>, <low>, <close>, <vwap>, <volume>, <dollar_volume>, <ticker>]`.

If `--output_path` is not specified, the table will be printed before the program exits. Otherwise, the data will be written to a CSV file at the path specified.


### Incremental Updates

An existing OHLCV CSV file can be incrementally updated by adding the `--append` option to the command, e.g.
```
bazel run //data:fetch_kraken_ohlcv_data -- --output_path {output_path} --data_frequency 1d --lookback_days 30 --append
```

To resolve duplicates and merge updates,

1. The new OHLCV data is stored into a dataframe `df_new`
2. The existing OHLCV data is loaded into a dataframe `df_existing`
3. `df_existing` and `df_new` are checked for schema consistency and then vertically stacked into `df_combined`
4. Duplicates in `df_combined` are detected via (`datetime`, `ticker`). The existing row is dropped and the newly queried row is kept for each duplicate.
   - For complete rows, the two should be equivalent. For incomplete rows, the newly queried row should be more up to date.
5. `df_combined` is sorted by (`datetime`, `ticker`)
6. `df_combined` is written to disk, completely replacing the previous file


### Dropping incomplete rows

At any point in time, the last queried row of OHLCV data will be incomplete. To avoid keeping this incomplete row around in our dataset to be fixed later, we can drop it completely before saving to disk via `--drop_last_row`.

This is useful to avoid unexpected behavior when generating positions for the most recent timeframe.


## Downloading CSV from Kraken

Kraken provides OHLCV data in various time frames in the form of downloadable CSV files. These files are updated quarterly and can be downloaded from [Kraken's website](https://support.kraken.com/hc/en-us/articles/360047124832-Downloadable-historical-OHLCVT-Open-High-Low-Close-Volume-Trades-data).

Note that the symbols used within these files are Kraken-specific (**not** ccxt unified). These symbols can be converted to ccxt unified market symbols via
```
bazel run //data:convert_kraken_symbols -- --input_path {input_ohlcv_filepath} --output_path {converted_ohlcv_filepath}
```


## Deriving from Tick Data

Tick data can be converted into OHLCV data, see [Converting Tick to OHLCV Data](./converting-tick-to-ohlcv-data.md).
