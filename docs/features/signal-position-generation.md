# Signal & Position Generation

The latest positions for a given trend signal can be generated, displayed, and output to file with the command:

```
bazel run //momentum:momentum -- positions \
    --input_path {input_OHLCV_CSV_path} \
    --data_freq {OHLCV_timeframe} \
    --timezone {desired_timezone} \
    --params_path {params_YAML_path} \
    --output_path {output_path}
```

- `--input_path`: Path to OHLCV CSV file.
- `--data_freq`: Frequency of OHLCV data (e.g. 1h, 1d).
- `--timezone`: Timezone with which to resample data. Defaults to UTC.
  - If specified, frequency of OHLCV data must be at least 1h or finer, else non-default timezone will be ignored.
- `--params_path`: Path to YAML file containing signal / position generation parameters, see [Parameter Specification](#parameter-specification).
- `--output_path`: (Optional) path to CSV file where latest positions will be written. If not specified, positions will only be printed.

## Trend Signal Generation

Trend signal generation is primarily motivated by [Rohrbach et al. 2017](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2949379) and [Rob Carver's Advanced Futures Trading Strategies](https://www.systematicmoney.org/advanced-futures).

1. Multiple Exponential Weighted Moving Average (EWMA) Crossovers are created with trend speeds varying from faster (4 days $-$ 12 days) to slower (64d $-$ 192d).
2. Each EWMAC is normalized by the 3-month price volatility to create the intermediary signal $y_k$.
3. Each $y_k$ is the normalized by the 1-year $y_k$ volatility to create $z_k$.
4. Each $z_k$ is passed through one of two response functions to create $u_k$.

$$
\begin{aligned}
\text{Exponential:}\quad f(x) &= \frac{x \cdot e^{-x^2 / 4}}{\sqrt{2} \cdot e^{-1/2}} \\
\text{Scaled Sigmoid:}\quad f(x) &= 2 \cdot \left( \frac{1}{1 + e^{-x}} \right) - 1
\end{aligned}
$$

5. Finally, the $u_k$'s are all averaged together with equal weights to create the trend signal.

The final (raw) trend signal has a range of $[-1, 1]$ regardless of which response function is used. In practice, the exponential response function is chosen as it seems to have higher explanatory power / perform better in the backtest.


## Position Generation & Sizing

Signals are translated into target positions via the following steps.

### Volatility Forecasting

Volatility forecasts are needed to do [volatility targeting](#volatility-targeting). Forecasts are generated for each ticker as a simple blend of short-term and long-term volatility.

$$
\begin{aligned}
vol_{forecast} &= 0.3 \cdot vol_{long} + 0.7 \cdot vol_{short} \\
\text{where }\hspace{0.5em} vol_{long} &= \text{Exponetial Weighted Moving Standard Deviation (EWMSD) of returns over 182 days}  \\
\text{and }\hspace{0.5em} vol_{short} &= \text{EWMSD of returns over 30 days} \\
\end{aligned}
$$

If the long-term volatility cannot be measured due to insufficient data, the short-term measurement will be used directly for the forecast.

$$
\begin{aligned}
vol_{forecast} &= vol_{short}
\end{aligned}
$$

If the short-term measurement is unavailable, the ticker is [disqualifed from trading](#data-history-constraints).


### Data History Constraints

In addition to the implicit "30 days of history rule" required for [forecasting volatility](#volatility-forecasting), there is an explicit requirement of 120 days of history for a ticker to be tradeable. Thus, the actual minimum data history for a ticker to be considered for trading is 120 days.


### Volume Constraints

Minimum and maximum daily volume constraints can be specified as [position generation parameters](#parameter-specification). Tickers which do not meet these constraints will be considered untradeable, and target position will be forced to 0. Note that this implies non-zero positions will be liquidated if a previously valid ticker becomes invalid, and so the constraints need to be robustly specified to avoid unnecessary churn.

An EWMA of the daily traded volume ***in the base currency (USD)*** is computed with a 30 day lookback period. For any given day, the EWMA must be above the daily minimum and below the daily maximum for the ticker to be tradeable that day. Hysteresis of 75% is applied for both constraints. For example, if the minimum is \$10M and the EWMA has gone from \$11M to \$9M, the ticker would still be considered **tradeable** until the EWMA has dipped below $`0.75 \cdot \$10\text{M} = \$7.5\text{M}`$, after which the ticker would be **untradeable** until the EWMA breaches \$10M again. Similarly, if the maximum is \$25M and the EWMA has run up to \$29M, the ticker would be **untradeable** until the EWMA has dipped below $`0.75 \cdot \$25\text{M} = \$18.75\text{M}`$, after which the ticker would be **tradeable** until the EWMA breaches \$25M again.

In practice, we use a minimum daily volume of 10,000 USD and leave maximum daily volume unconstrained.


### Raw Signal Scaling

The raw trend signals are now scaled such that the absolute average of the signal in aggregate is equal to 1.

First, we estimate the absolute signal average by:

1. Compute the absolute values of the trend signal for every ticker on every day
2. Form a new sequence $x_k$ which contains the cross-sectional mean of the trend signal absolute values for each day.
3. Compute the EWMA of $x_k$ with a lookback period of 182 days.
4. Take the latest EWMA value as the absolute signal average estimate (denote by `abs_trend_avg`).

Then, divide each raw trend signal by `abs_trend_avg` to produce `signal_scaled`.


### Cross-Sectional Signal Strength

Signals can now be considered either cross-sectionally or as a time series. With cross-sectional momentum, the *scaled* signals for each day will be ranked from most positive to most negative. A long position is put on for the top $x$% of tickers and a short position for the bottom $x$%. The middle $(100-2x)$% will be flat. $x$ is a [configurable parameter](#parameter-specification).

Equal weighting is another configurable parameter. If equal weighting is specified, the trend signal magnitudes will be discarded. The long tickers (top $x$%) will be treated as having a `signal_scaled` of 1, the short tickers (bottom $x$%) as having -1, and the flat tickers having 0. Otherwise, only the flat tickers will be treated as having a `signal_scaled` of 0, and the short/long tickers will keep their original `signal_scaled` values. Thus any difference in signal magnitude will propagate through to the final position sizes.


### Time-Series Signal Strength

With time-series momentum, no signal ranks are considered. Each `signal_scaled` will be used as is when volatility targeting / equal sizing. Larger magnitudes of `signal_scaled` will translate to larger positions. `signal_scaled` values closer to 0 will translate to smaller positions.


### Volatility Targeting

A volatility target for each asset is generated by taking the target portfolio volatility and dividing by the number of traded assets. For cross-sectional momentum, this is the number of non-flat tickers ($2x$% of the asset universe). For time-series momentum, this is the total number of tradeable tickers in the universe (not including those filtered out by volume / data history constraints).

Once an asset volatility target has been generated, it is divided by the [asset's volatility forecast](#volatility-forecasting) to create the position scaling factor. This factor is then multiplied by the [diversification multipliers](#diversification-multipliers) and the `signal_scaled` strength to generate the asset's target position size as a percentage of the portfolio.


#### (alt.) Equal Weight Sizing

Alternatively, one can opt to forego volatility targeting for equal weight sizing. In this case, the position scaling factor is computed as one over the number of open long & short positions for that day. As before, the position scaling factor is multiplied by the [diversification multipliers](#diversification-multipliers) and the `signal_scaled` strength to generate the asset's target position size as a percentage of the portfolio.


### Diversification Multipliers

We use diversification multipliers to compensate for the effects of diversification on the realized volatility of our portfolio. The ideas and implementation details are drawn primarily from chapters 4 and 9 of [Advanced Futures Trading Strategies](https://www.systematicmoney.org/advanced-futures), and we suggest referencing the book for more details. Roughly, this method involves:

1. Estimating the correlation matrix of returns
   - Our implementation uses daily data instead of the suggested weekly data since we don't have to worry about aligning different market open hours
2. Computing the Instrument Diversification Multiplier (IDM) using the correlations from (1) and equal weights
3. Estimating the correlation matrix of $u_k$ (pooled from all tickers)
4. Computing the Forecast Diversification Multiplier (FDM) using the correlations from (2) and equal weights
5. Computing the Combined DM $= IDM * FDM$
6. Computing the 30-day EWMA of the Combined DM

Finally, a scale factor between ~1.08 to ~1.80 is included as an additional diversification multiplier. This scale factor comes from backtesting the positions generated without this scaling on the training data with portfolio volatility targets varying from 5% to 200%, recording the realized annual volatility in the backtest, and then fitting a lowess function to the `(target, realized)` datapoints. It is essentially a compensating factor for the inability to hit larger volatility targets using the IDM and the FDM alone.

The 30-day Combined DM EWMA from (6) and the LOWESS scale factor are multiplied with the position scaling factor (either from [volatility targeting](#volatility-targeting) or [equal weight sizing](#alt-equal-weight-sizing)) and the `signal_scaled` value to generate the asset's target position size as a percentage of the portfolio.


### Leverage Constraints & Position Size Caps

Lastly, leverage / direction constraints and position size caps are applied. If shorting is not allowed, any short positions are zeroed out **without any rescaling of the long positions**, and vice versa if long positions are disallowed.

A maximum position magnitude of `leverage` / `num_assets` is also enforced, with `leverage` being configurable. Any positions which exceed this in magnitude (short or long) will be clipped to the size cap.


### Position Lagging

If generating positions for [backtesting](./backtesting.md), positions are lagged by 1 day to avoid lookahead bias. The backtesting engine will execute trades using open prices. This resembles a live trading process of producing positions from the previous day's data and executing immediately.


## Parameter Specification

Configurable parameters related to signal / position generation can be specified via YAML file, e.g. [optimize_rohrbach.yaml](../../momentum/params/optimize_rohrbach.yaml).

- `signal`: The trend signal to be used. Valid values `[rohrbach_exponential, rohrbach_sigmoid]`
- `direction`: Allowed position direction. Valid values `[LongOnly, ShortOnly, Both]`
- `volatility_target`: Portfolio volatility target. Values between `[0.05, 1.0]` supported, or `null` to disable.
- `cross_sectional_percentage`: The percentage of the cross-sectional universe to keep, or `null` to disable.
- `cross_sectional_equal_weight`: Whether to apply equal weighting when considering cross-sectional momentum. Valid values `[True, False]`
- `min_daily_volume`: Minimum daily volume constraint in base currency, or `null` to disable.
- `max_daily_volume`: Maximum daily volume constraint in base currency, or `null` to disable.
- `leverage`: Allowable leverage for position sizing. 1.0 for no leverage.
