# Introduction

This is an open source version of my trend-following system for spot assets on the Kraken Exchange. The pipeline includes:

- [Data Management](../data/)
  - Fetching of Kraken tick and OHLCV data via [ccxt](https://github.com/ccxt/ccxt) and the [Kraken REST API](https://docs.kraken.com/rest/) with [custom ccxt hookups](../ccxt_custom/) to get around limitations of the API
  - Conversion of tick data to OHLCV data with arbitrary bin size (minimum 1 microsecond)
  - Splitting of tick `.csv` files into multiple files based on time cadence (e.g. daily, weekly, monthly, etc.)
  - Combining of OHLCV `.csv` files into one or more consolidated files
- [Data Analysis](../momentum/analysis/)
  - [Example Jupyter notebook](<../momentum/Crypto Momentum.ipynb>) with analysis on cross-sectional and time-series momentum in cryptocurrencies
- [Signal Generation](../momentum/signal_generation/)
  - Signals consist of combining multiple EWMA crossovers of varying window lengths which have been normalized and passed through a non-linear response function (motivated by [Rohrbach et al. 2017](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2949379)), see
  - Validity filters include sufficient data history and minimum daily traded volume
- [Position Generation & Sizing](../momentum/position_generation/)
  - Volatility Targeting
  - Leverage Constraints
  - Maximum Size guardrails per individual asset
- [Simulation / Backtesting](../momentum/simulation/)
  - Backtesting Engine via [vectorbt](https://github.com/polakowo/vectorbt)
  - Volume Constraints (% of available volume)
  - Dynamic Fees (e.g. based on rolling 30d trading volume)
  - Rebalancing Buffers (rebal to edge of buffer)
  - Adjustable Rebalancing Frequency
  - Parameter Optimization
- [Live Trading](../live/)
  - Order Management System
  - Historical PnL Visualization

## Limitations

As I reside in the US and am not an [Eligible Contract Participant (ECP)](https://uscode.house.gov/view.xhtml?req=granuleid:USC-prelim-title7-section1a&num=0&edition=prelim), I am subject to certain trading restrictions, including:

- Inability to trade certain tickers due to geographic restrictions.
  - The full list of restrictions is located in [core](../momentum/core/src/core/constants.py).
  - By default, restricted tickers will be excluded from the investable universe (treated as if they don't exist). Signals and correlations will be computed ignoring restricted tickers.
- Inability to trade on margin and therefore to go short.
  - For the purposes of position generation and backtesting, this is configurable via the [direction key-value param](../momentum/params/optimize_rohrbach.yaml#L4) (see the [Direction Enum](../momentum/position_generation/src/position_generation/utils.py#L10) for valid values).
  - **For live trading, trading on margin is not implemented. Only long-only positions with no leverage are supported.**
