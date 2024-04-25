from typing import Callable, List, Optional

import pandas as pd
import plotly.express as px

from core.constants import POSITION_COL, PRICE_COL_BACKTEST
from data.constants import DATETIME_COL, TICKER_COL, VOLUME_COL
from position_generation.constants import (
    NUM_LONG_ASSETS_COL,
    NUM_OPEN_LONG_POSITIONS_COL,
    NUM_OPEN_POSITIONS_COL,
    NUM_OPEN_SHORT_POSITIONS_COL,
    NUM_SHORT_ASSETS_COL,
    NUM_UNIQUE_ASSETS_COL,
)
from simulation.constants import DEFAULT_REBALANCING_BUFFER, DEFAULT_VOLUME_MAX_SIZE
from simulation.fees import FeeType, compute_fees
from simulation.stats import (
    display_stats,
    get_trade_volume,
    get_turnover,
    plot_cumulative_returns,
    plot_rolling_returns,
)
from simulation.utils import get_segment_mask
from simulation.vbt import get_log_returns, get_returns, simulate, vbt


def backtest(
    df_backtest: pd.DataFrame,
    periods_per_day: int,
    initial_capital: float,
    leverage: float = 1.0,
    rebalancing_freq: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    with_fees: bool = True,
    volume_max_size: float = DEFAULT_VOLUME_MAX_SIZE,
    rebalancing_buffer: float = DEFAULT_REBALANCING_BUFFER,
    verbose: bool = False,
) -> vbt.Portfolio:
    """
    Backtest a pf_strategy

    Args:
        df_backtest (pd.DataFrame): Should contain timestamp, close, position columns. Assuming 0 slippage, entire order can be filled.
        initial_capital (float): Initial capital in USD.
    """  # noqa: B950
    # Copy dataframe to avoid corrupting original contents
    df_backtest = df_backtest.copy()

    def apply_time_window(df):
        if start_date is not None:
            df = df.loc[df[DATETIME_COL] >= start_date]
        if end_date is not None:
            df = df.loc[df[DATETIME_COL] <= end_date]
        return df

    df_backtest = apply_time_window(df_backtest)

    # Transform data from long to wide
    df_backtest = df_backtest.sort_values([DATETIME_COL, TICKER_COL])
    price = df_backtest[[DATETIME_COL, TICKER_COL, PRICE_COL_BACKTEST]]
    # Prices can't be exactly 0 for purpose of computing stats. If
    # they are equal to 0, it likely means that there was no volume here.
    EPS = 1e-6
    price.loc[price[PRICE_COL_BACKTEST] == 0, PRICE_COL_BACKTEST] = EPS
    price = pd.pivot_table(
        price,
        index=DATETIME_COL,
        columns=TICKER_COL,
        values=PRICE_COL_BACKTEST,
        dropna=False,
        fill_value=EPS,
    )
    volume = df_backtest[[DATETIME_COL, TICKER_COL, VOLUME_COL]]
    volume = pd.pivot_table(
        volume,
        index=DATETIME_COL,
        columns=TICKER_COL,
        values=VOLUME_COL,
        dropna=False,
        fill_value=0.0,
    )
    # Save original price index for computing daily volume later
    orig_price_index = price.index.copy()
    # Create positions for daily rebalancing
    positions = df_backtest[[DATETIME_COL, TICKER_COL, POSITION_COL]]
    positions = pd.pivot_table(
        positions,
        index=DATETIME_COL,
        columns=TICKER_COL,
        values=POSITION_COL,
        dropna=False,
        fill_value=0.0,
    )
    if rebalancing_freq is None:
        segment_mask = None
    else:
        segment_mask = get_segment_mask(periods_per_day, rebalancing_freq)

    # fmt: off
    # Simulate trades without fees to calculate trading volume
    portfolio_no_fees = simulate(
        price=price,
        positions=positions,
        volume=volume,
        volume_max_size=volume_max_size,
        rebalancing_buffer=rebalancing_buffer,
        initial_capital=initial_capital,
        segment_mask=segment_mask,
        direction=vbt.portfolio.enums.Direction.Both,
        fees=0,
        fixed_fees=0,
        slippage=0.005,
        leverage=leverage,
    )
    # fmt: on
    if verbose:
        print("--- Before Fees ---")
        print(portfolio_no_fees.stats())
        print()
    if not with_fees:
        return portfolio_no_fees

    # Calculate 30d rolling trading volume
    trade_volume_per_period = get_trade_volume(portfolio_no_fees)
    # Reindex back to price index, fill in gaps where no trades were placed
    trade_volume_per_period = trade_volume_per_period.reindex(
        orig_price_index, fill_value=0
    )

    rolling_30d_trade_volume = trade_volume_per_period.rolling(
        window=30 * periods_per_day, min_periods=1
    ).sum()
    if verbose:
        # Plot rolling 30d volume
        tmp_for_plot = rolling_30d_trade_volume.reset_index()
        fig = px.line(
            tmp_for_plot,
            x=DATETIME_COL,
            y="Traded Size [$]",
            title="30d Rolling Volume",
        )
        fig.update_layout(
            xaxis_title=DATETIME_COL, yaxis_title="30d Rolling Volume", hovermode="x"
        )
        fig.show()

    # Compute dynamic fees based on rolling trading volume
    dynamic_fees = compute_fees(rolling_30d_trade_volume, fee_type=FeeType.TAKER)
    if verbose:
        # Plot daily fees [%]
        tmp_for_plot = dynamic_fees.reset_index()
        fig = px.line(
            tmp_for_plot, x=DATETIME_COL, y="Fees [%]", title="Daily Fees [%]"
        )
        fig.update_layout(
            xaxis_title=DATETIME_COL, yaxis_title="Daily Fees (%)", hovermode="x"
        )
        fig.show()

    # Now, simulate trades with dynamic fees. It's still not entirely accurate
    # because trading volume will depend on account size (which depends on fees),
    # but close enough for an approximation.
    portfolio_with_fees = simulate(
        price=price,
        positions=positions,
        volume=volume,
        volume_max_size=volume_max_size,
        rebalancing_buffer=rebalancing_buffer,
        initial_capital=initial_capital,
        segment_mask=segment_mask,
        direction=vbt.portfolio.enums.Direction.Both,
        fees=dynamic_fees,
        fixed_fees=0,
        slippage=0.005,
        leverage=leverage,
    )

    if verbose:
        print("--- After Fees ---")
        print(portfolio_with_fees.stats())
        portfolio_with_fees.plot_drawdowns().show()
        portfolio_with_fees.plot_net_exposure().show()
        print()

        # Plot returns
        df_returns = get_returns(portfolio_with_fees).reset_index()
        df_returns.rename(columns={"group": "Returns"}, inplace=True)
        fig = px.bar(df_returns, x=DATETIME_COL, y="Returns", title="Daily Returns [%]")
        fig.update_layout(
            xaxis_title=DATETIME_COL, yaxis_title="Returns [%]", hovermode="x"
        )
        fig.show()

        # Plot log returns
        df_log_returns = get_log_returns(portfolio_with_fees).reset_index()
        df_log_returns.rename(columns={"group": "Returns"}, inplace=True)
        fig = px.bar(
            df_log_returns, x=DATETIME_COL, y="Returns", title="Daily Log Returns [%]"
        )
        fig.update_layout(
            xaxis_title=DATETIME_COL, yaxis_title="Log Returns [%]", hovermode="x"
        )
        fig.show()

        # Plot turnover
        df_turnover = get_turnover(portfolio_with_fees).reset_index()
        turnover_col = "Turnover [%]"
        turnover_30d_ema_col = "Turnover 30d EMA [%]"
        df_turnover[turnover_30d_ema_col] = (
            df_turnover[turnover_col]
            .ewm(span=30 * periods_per_day, adjust=True, ignore_na=False)
            .mean()
        )
        fig = px.line(
            df_turnover,
            x=DATETIME_COL,
            y=[turnover_col, turnover_30d_ema_col],
            title="Turnover",
        )
        fig.show()

        # Plot open positions
        df_tmp = (
            df_backtest.groupby([DATETIME_COL])
            .agg(
                {
                    NUM_LONG_ASSETS_COL: "max",
                    NUM_SHORT_ASSETS_COL: "max",
                    NUM_UNIQUE_ASSETS_COL: "max",
                    NUM_OPEN_LONG_POSITIONS_COL: "max",
                    NUM_OPEN_SHORT_POSITIONS_COL: "max",
                    NUM_OPEN_POSITIONS_COL: "max",
                }
            )
            .reset_index()
        )
        fig = px.line(
            df_tmp,
            x=DATETIME_COL,
            y=[
                NUM_LONG_ASSETS_COL,
                NUM_SHORT_ASSETS_COL,
                NUM_UNIQUE_ASSETS_COL,
                NUM_OPEN_LONG_POSITIONS_COL,
                NUM_OPEN_SHORT_POSITIONS_COL,
                NUM_OPEN_POSITIONS_COL,
            ],
            title="Num Open Positions",
        )
        fig.show()

    return portfolio_with_fees


def backtest_crypto(
    df_analysis: pd.DataFrame,
    generate_positions_fn: Callable,
    generate_benchmark_fn: Callable,
    periods_per_day: int,
    start_date: str,
    end_date: str,
    initial_capital: float,
    leverage: float = 1.0,
    rebalancing_freq: Optional[str] = None,
    volume_max_size: float = DEFAULT_VOLUME_MAX_SIZE,
    rebalancing_buffer: float = DEFAULT_REBALANCING_BUFFER,
    skip_plots: bool = False,
) -> List[vbt.Portfolio]:
    """
    Convenience wrapper for backtesting on crypto dataset

    Args:
        df_analysis (pd.DataFrame): _description_
        generate_positions_fn (Callable): Generates positions for df_analysis. Signature `(pd.DataFrame, dict) -> pd.DataFrame`.
        generate_benchmark_fn (Callable): Generates benchmark positions for df_analysis. Signature `(pd.DataFrame, dict) -> pd.DataFrame`.
        start_date (str): _description_
        end_date (str): _description_
        initial_capital (float): _description_
        skip_plots (bool, optional): _description_. Defaults to False.

    Returns:
        List[vbt.Portfolio]: _description_
    """  # noqa: B950
    # Backtest benchmark
    df_benchmark = generate_benchmark_fn(df_analysis)
    pf_benchmark = backtest(
        df_benchmark,
        periods_per_day=periods_per_day,
        initial_capital=initial_capital,
        leverage=leverage,
        rebalancing_freq=rebalancing_freq,
        start_date=start_date,
        end_date=end_date,
        with_fees=True,
        verbose=False,
    )
    # Backtest portfolio
    df_portfolio = generate_positions_fn(df_analysis)
    pf_portfolio = backtest(
        df_portfolio,
        periods_per_day=periods_per_day,
        initial_capital=initial_capital,
        leverage=leverage,
        rebalancing_freq=rebalancing_freq,
        start_date=start_date,
        end_date=end_date,
        with_fees=True,
        volume_max_size=volume_max_size,
        rebalancing_buffer=rebalancing_buffer,
        verbose=not skip_plots,
    )

    # Compare returns, stats
    pf_names = ["Benchmark", "pf_Strategy"]
    portfolios = [pf_benchmark, pf_portfolio]
    display_stats(portfolios, pf_names)
    if not skip_plots:
        plot_cumulative_returns(portfolios, pf_names)
        plot_rolling_returns(portfolios, pf_names, window=30)

    return portfolios
