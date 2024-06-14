# Live Trading

The goal of live trading is to translate `target positions [% of portfolio]` into trades of `+/- asset @ price` based on portfolio size, current holdings, and current asset prices, and then to execute these trades on the live exchange.

Live trading is done using the `//live:trades` binary.

```
bazel run //live:trades -- {mode} \
    --input_path {input_OHLCV_CSV_path} \
    --input_data_freq {orig_OHLCV_timeframe} \
    --output_data_freq {target_OHLCV_timeframe} \
    --timezone {desired_timezone} \
    --credentials_path {exchange_API_key_path} \
    --params_path {params_YAML_path} \
    --execution_strategy {execution_strategy}
```

- `mode`: Mode of operation. Valid values `["display", "execute", "cancel_all"]`.
  - `display`: Display / write to file the latest trades to execute.
  - `execute`: Execute the latest trades.
  - `cancel_all`: Cancel all outstanding trades.
- `--input_path`: Path to OHLCV CSV file.
- `--input_data_freq`: Frequency of OHLCV data (e.g. 1h, 1d).
- `--output_data_freq`: Target frequency of OHLCV data. The input data will be resampled to match this target frequency if necessary.
  - A common use case would be to resample hourly data to daily data with respect to a particular timezone, from which daily signals & positions can be generated.
- `--timezone`: Timezone with which to resample data. Defaults to UTC.
  - Trades will be placed based on the positions for the [last complete period in the data](./fetching-kraken-ohlcv-data.md#dropping-incomplete-rows). For example, if the input file contains hourly data up to `2024-06-10T12:00:00+0`, `--output_data_freq=1d`, and `--timezone=UTC`, the positions for `2024-06-10T00:00:00+0` will be used to generate trades (effectively discarding the last 12 hours of the input data).
  - Note that `pytz` is used to parse timezones, which in turn uses the IANA time zone database. This requires [inverting the timezones passed into `--timezone`](https://stackoverflow.com/questions/4008960/pytz-and-etc-gmt-5#:~:text=If%20you%20need%20to%20convert,timezone%20you%20want%20to%20convert.) (e.g. `GMT+5 --> Etc/GMT-5`).
  - `--timezone=latest` can be specified to automatically set the timezone to minimize the amount of data discarded, see [Automatic Timezone](#automatic-timezone)
- `--credentials_path`: Path to the [Kraken API key](../../README.md#generate-kraken-api-key)
- `--params_path`: Path to YAML file containing [signal / position generation parameters](./signal-position-generation.md#parameter-specification)
- `--execution_strategy`: Trade execution strategy. Valid values `["market", "limit", "limit-then-market"]`.
  - See [Execution Strategies](#execution-strategies).
- `--account_size`: (Optional) Target portfolio size, used for position sizing. By default, positions will be sized as a \% of the entire account balance on the exchange.
- `--output_path`: (Optional) Path of CSV file to which trades should be written.
- `--validate`: (Optional) Whether to place orders with the [`validate` flag](https://docs.kraken.com/rest/#tag/Spot-Trading/operation/addOrder) to validate inputs without actually submitting the order.
- `--skip_confirm`: (Optional) Whether to skip user confirmation during trade execution. Useful in automated environments.


## Target Position Generation

Target positions are generated according to [Signal & Position Generation](./signal-position-generation.md). Since this is for the live trading environment, positions should **not** be lagged.


### Output Data Frequency

Although the signal generation internals properly handle differing data frequencies (e.g. 30 rows of daily data == 720 rows of hourly data for a rolling 30d sum), signals generated from hourly vs daily data will be significantly different due to the widespread use of EWMAs. More recent days are weighed more heavily for hourly vs daily data due to the increased number of data points and the exponential decay of older data.

The [trend signals](./signal-position-generation.md#trend-signal-generation) were originally formulated using relationships which span multiple days / weeks, and empirically the signals are more explanatory when generated using daily data vs hourly data. We recommend operating on daily data, i.e. `--output_data_freq=1d`.


### Automatic Timezone

Since we are operating on daily data, our choice of timezone potentially matters. If we wish to execute our trades at a particular time of day (e.g. 12pm PST), we will need our daily data to be aligned to this timezone such that each OHLCV row contains only trades from `12:00:00 PST` from the previous day to `11:59:59 PST` on the current day.

The easiest way to flexibly maintain and update timezone-aligned daily data is to produce it from hourly data via downsampling. This avoids having to maintain a separate copy of daily data for each timezone of interest and retains the ability to switch timezones in the future. When resampling hourly data to daily, the `datetime` for each daily row will be set to midnight in the desired timezone (i.e. each daily row will contain trades from `00:00:00` to `23:59:59` on the current day). Thus, to trade at say `12:00:00 PST`, we would ideally choose to resample w.r.t. the timezone in which `12:00:00 PST = 00:00:00` (for this example, this turns out to be `GMT+5 --> Etc/GMT-5`).

The current time is known whenever the live trading program is initiated. A natural next step would be to automatically select the timezone for which the resampled data would be most aligned to the current time. This turns out to be the timezone in which the current time floors to `00:00:00`. Specifying `--timezone=latest` will perform this calculation to automatically set the timezone for resampling.

However, this makes the assumption that the input data actually contains complete data up to at least `00:00:00` in the auto-chosen timezone. **Thus, one should only use `--timezone=latest` when the input hourly data is updated past the turn of the hour.** This can be ensured by (1) initiating the [OHLCV data update](./fetching-kraken-ohlcv-data.md) after the turn of the hour, and (2) always fetching the latest OHLCV data prior to running live trade execution.


## Trade Generation

Trades are generated as follows:

1. Resample [hourly data to daily](#output-data-frequency) w.r.t. the [latest timezone](#automatic-timezone).
2. Generate the latest daily positions (i.e. the positions for the `T-1` datetime).
3. Query the exchange for current portfolio size and asset holdings.
4. Convert the target positions to target trades based on current portfolio size and asset holdings.
   - Refer to [How orders are filled](./backtesting.md#how-orders-are-filled) for how this conversion is done.
   - Rebalancing buffers are considered at this step.
   - Volume constraints are considered during [trade execution](#trade-execution).

## Trade Execution / Order Management

Trades are executed and managed in a loop as follows:

1. For each trade, formulate an order according to one of the [execution strategies](#execution-strategies).
2. Prior to placing each order, estimate the expected slippage and abort if slippage seems high.
   - Compute how far the best bid/ask will move assuming your order is filled in its entirety from the top of the book.
   - If this exceeds the minimum acceptable threshold, abort the order and try again later.
   - This same heuristic is used for both market orders and limit orders (despite being more applicable to the former).
3. After all orders for this iteration have been placed, manage the open orders.
   - Record closed / filled orders.
   - Cancel / record expired orders.
   - Cancel [stale limit orders](#execution-strategies).
   - Note orders which failed to be placed (network / exchange issues).
4. Re-query current portfolio size and asset holdings from the exchange.
5. [Reformulate target trades](#trade-generation) for tickers which **do not have an outstanding open order** based on updated holdings.
6. Repeat from (1) until there are no more trades to execute.
   - To avoid potentially trading in and out of the same ticker within the same session due to volatile price movements, each ticker will be traded at most once (fully filled, may be split into multiple orders by the exchange). Recording the closed / filled orders is used for this.
   - This information is only kept in the program's local memory. Stopping and restarting the program will start a new session, and every ticker may potentially be traded once again.

### Execution Strategies

- `market`: Use market orders to fully fill every trade. Orders expire after 10 seconds.
- `limit`: Place limit orders at the top of the order book (i.e. at the *best bid* for buys, at the *best ask* for sells) to fill every trade.

  Orders expire after 1 hour. Orders are also canceled if price goes "stale": the market moves s.t. the *order* price is worse than the *mid* price by more than $x$%.

  Canceled & expired orders are replaced with a new limit order using the same best bid/offer pricing strategy. Repeat until all desired trades have been made.
- `limit-then-market`: Place orders according to the `limit` strategy mentioned above. After 2 hours have passed, cancel all outstanding limit orders and use market orders to finish executing any remaining trades.


## Realized PnL

Realized PnL can be measured and plotted.

```
bazel run //live:pnl -- \
    --input_path {input_OHLCV_CSV_path} \
    --data_freq {OHLCV_timeframe} \
    --credentials_path {exchange_API_key_path}
```

- `--input_path`: Path to OHLCV CSV file.
- `--data_freq`: Frequency of OHLCV data (e.g. 1h, 1d).
- `--credentials_path`: Path to the [Kraken API key](../../README.md#generate-kraken-api-key)
- `--start_date`: (Optional) Date from which to compute PnL, inclusive. Deposits & trades made prior to this date will be ignored.
- `--skip_plots`: (Optional) Skip plotting and just print out current PnL.


Daily asset holdings are recreated from deposits and trades made to/from the account on each day. Historical account size is then estimated from asset holdings and historical *close* prices. The current account size is computed from asset holdings and current *mid* prices. Daily log returns are then computed from the historical account size, taking into account deposits (deposits will be subtracted prior to computing returns).

**NOTE: this assumes that all deposits and trades are made w.r.t. a common base currency (USD).**

The current PnL is reported as the cumulative sum of log returns since `start_date`. Daily realized volatility and annualized volatility are also reported.

### Plots

Historical account size (equity curve), daily log returns (bar), and cumulative log returns (line) plots are generated when plotting is enabled.
