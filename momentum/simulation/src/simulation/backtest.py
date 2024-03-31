import pandas as pd
import plotly.express as px
import numpy as np
from typing import List, Callable, Optional
from numba import njit

import vectorbt as vbt
from vectorbt.base.reshape_fns import flex_select_auto_nb, to_2d
from vectorbt.portfolio.enums import (
    SizeType,
    Direction,
    NoOrder,
    OrderStatus,
    OrderSide,
)
from vectorbt.portfolio import nb
from vectorbt.portfolio.enums import (
    SizeType,
    Direction,
    NoOrder,
)

from core.constants import (
    TIMESTAMP_COL,
    TICKER_COL,
    VOLUME_COL,
    PRICE_COL_BACKTEST,
    POSITION_COL,
)
from position_generation.constants import (
    NUM_LONG_ASSETS_COL,
    NUM_SHORT_ASSETS_COL,
    NUM_UNIQUE_ASSETS_COL,
    NUM_OPEN_LONG_POSITIONS_COL,
    NUM_OPEN_SHORT_POSITIONS_COL,
    NUM_OPEN_POSITIONS_COL,
)
from position_generation.utils import nonempty_positions
from position_generation.benchmark import generate_benchmark_btc
from simulation.fees import compute_fees, FeeType
from simulation.stats import (
    get_stats_of_interest,
    get_turnover,
    get_trade_volume,
    get_num_open_positions,
    display_stats,
    plot_cumulative_returns,
    plot_rolling_returns,
)
from simulation.constants import DEFAULT_VOLUME_MAX_SIZE, DEFAULT_REBALANCING_BUFFER
from simulation.utils import get_segment_mask
from core.utils_nb import clip_nb


@njit
def pre_group_func_nb(c):
    order_value_out = np.empty(c.group_len, dtype=np.float_)
    return (order_value_out,)


@njit
def pre_segment_func_nb(c, order_value_out, size, price, size_type, direction):
    """Update last_val_price with today's price for use in `order_func_nb`

    Args:
        c (vbt.portfolio.enums.OrderContext): OrderContext

    Returns:
        tuple: Empty tuple
    """
    for col in range(c.from_col, c.to_col):
        c.last_val_price[col] = nb.get_col_elem_nb(c, col, c.close)
    nb.sort_call_seq_nb(c, size, size_type, direction, order_value_out)
    return ()


@njit
def order_func_nb(
    c,
    size,
    volume,
    size_type,
    direction,
    fees,
    fixed_fees,
    slippage,
    volume_max_size: float,
    rebalancing_buffer: float,
):
    """Execute long-only orders while respecting sizing constraints (% of available volume) and rebalancing buffers.

    Args:
        c (vbt.portfolio.enums.OrderContext): OrderContext
        size (_type_): _description_
        volume (_type_): _description_
        volume_max_size (float, optional): _description_. Defaults to 0.01.
        rebalancing_buffer (float, optional): _description_. Defaults to 0.01.

    Returns:
        vbt.portfolio.enums.Order: Order
    """
    # Select info related to this order
    # flex_select_auto_nb allows us to pass size as single number, 1-dim or 2-dim array
    # If flex_2d is True, 1-dim array will be per column, otherwise per row
    target_now = flex_select_auto_nb(
        np.asarray(size), c.i, c.col, flex_2d=True
    )  # [% of Account Size]
    account_size = c.value_now  # [$]
    # close is always 2-dim array
    price_now = c.close[c.i, c.col]  # [$ / Unit]
    last_position = nb.get_elem_nb(c, c.last_position)  # [Units of Asset]
    size_now = price_now * last_position  # [$]
    pct_now = size_now / account_size  # [% of Account Size]

    # Check if position deviation exceeds buffer percentage.
    position_deviation = np.abs(target_now - pct_now)  # [Percentage of Account Size]
    rebalance = position_deviation > rebalancing_buffer
    if not rebalance:
        return NoOrder
    # Rebalance to the edge of the buffer (not to the target), minimize trading.
    if target_now > pct_now:
        target_now -= rebalancing_buffer
    else:
        target_now += rebalancing_buffer

    # Translate target position into trade amount
    target_size = target_now * account_size  # [$]
    trade_size = target_size - size_now  # [$]
    trade_amnt = trade_size / price_now  # [Units of Asset]

    # Calculate position size based on % of available volume
    # TODO(@eugene.lo): Distinguish between short and long sizing...
    volume_now = flex_select_auto_nb(
        np.asarray(volume), c.i, c.col, flex_2d=True
    )  # [Units of Asset]
    if np.isnan(volume_now):
        volume_now = 0
    max_position_amnt = volume_max_size * volume_now  # [Units of Asset]
    trade_amnt = clip_nb(trade_amnt, -max_position_amnt, max_position_amnt)
    trade_size = trade_amnt * price_now  # [$]
    trade_pct = trade_size / account_size  # [%]

    # Get order size based on size type
    order_size_type = nb.get_elem_nb(c, size_type)
    if order_size_type == SizeType.Amount:
        order_size = trade_amnt
    elif order_size_type == SizeType.Value:
        order_size = trade_size
    elif order_size_type == SizeType.Percent:
        order_size = trade_pct
    else:
        raise ValueError("Unsupported size type!")

    # Submit order
    if order_size == 0:
        return NoOrder
    return nb.order_nb(
        size=order_size,
        price=price_now,
        size_type=order_size_type,
        direction=nb.get_elem_nb(c, direction),
        fees=nb.get_elem_nb(c, fees),
        fixed_fees=nb.get_elem_nb(c, fixed_fees),
        slippage=nb.get_elem_nb(c, slippage),
    )


def simulate(
    price: pd.DataFrame,
    positions: pd.DataFrame,
    volume: pd.DataFrame,
    volume_max_size: float,
    rebalancing_buffer: float,
    initial_capital: float,
    segment_mask: int = None,
    direction: Direction = Direction.LongOnly,
    fees: float = 0.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
) -> vbt.Portfolio:
    size_np = positions.to_numpy()
    volume_np = volume.to_numpy()
    # fmt: off
    pf = vbt.Portfolio.from_order_func(
        price,
        order_func_nb,
        # Args for order_func_nb
        size_np, volume_np, vbt.Rep('size_type'), vbt.Rep('direction'),
        vbt.Rep('fees'), vbt.Rep('fixed_fees'), vbt.Rep('slippage'),
        volume_max_size, rebalancing_buffer,
        # Kwargs
        pre_group_func_nb=pre_group_func_nb,
        pre_segment_func_nb=pre_segment_func_nb,
        pre_segment_args=(
            size_np,
            vbt.Rep('price'),
            vbt.Rep('size_type'),
            vbt.Rep('direction')
        ),
        broadcast_named_args=dict(  # broadcast against each other
            price=price,
            size_type=SizeType.Amount,
            direction=direction,
            fees=fees,
            fixed_fees=fixed_fees,
            slippage=slippage,
        ),
        init_cash=initial_capital,
        cash_sharing=True,
        group_by=True,
        segment_mask=segment_mask,
    )
    # fmt: on
    return pf


def backtest(
    df_backtest: pd.DataFrame,
    periods_per_day: int,
    initial_capital: float,
    rebalancing_freq: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    with_fees: bool = True,
    volume_max_size: float = DEFAULT_VOLUME_MAX_SIZE,
    rebalancing_buffer: float = DEFAULT_REBALANCING_BUFFER,
    verbose: bool = False,
) -> vbt.Portfolio:
    """Backtest a pf_strategy

    Args:
        df_backtest (pd.DataFrame): Should contain timestamp, close, position columns. Assuming 0 slippage, entire order can be filled.
        initial_capital (float): Initial capital in USD.
    """
    # Copy dataframe to avoid corrupting original contents
    df_backtest = df_backtest.copy()

    def apply_time_window(df):
        if start_date is not None:
            df = df.loc[df[TIMESTAMP_COL] >= start_date]
        if end_date is not None:
            df = df.loc[df[TIMESTAMP_COL] <= end_date]
        return df

    df_backtest = apply_time_window(df_backtest)

    # Transform data from long to wide
    df_backtest = df_backtest.sort_values([TIMESTAMP_COL, TICKER_COL])
    price = df_backtest[[TIMESTAMP_COL, TICKER_COL, PRICE_COL_BACKTEST]]
    price = pd.pivot_table(
        price,
        index=TIMESTAMP_COL,
        columns=TICKER_COL,
        values=PRICE_COL_BACKTEST,
        dropna=False,
        fill_value=0.0,
    )
    volume = df_backtest[[TIMESTAMP_COL, TICKER_COL, VOLUME_COL]]
    volume = pd.pivot_table(
        volume,
        index=TIMESTAMP_COL,
        columns=TICKER_COL,
        values=VOLUME_COL,
        dropna=False,
        fill_value=0.0,
    )
    # Save original price index for computing daily volume later
    orig_price_index = price.index.copy()
    # Create positions for daily rebalancing
    positions = df_backtest[[TIMESTAMP_COL, TICKER_COL, POSITION_COL]]
    positions = pd.pivot_table(
        positions,
        index=TIMESTAMP_COL,
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
        direction=Direction.Both,
        fees=0,
        fixed_fees = 0,
        slippage=0.005,
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
            x=TIMESTAMP_COL,
            y="Traded Size [$]",
            title="30d Rolling Volume",
        )
        fig.update_layout(
            xaxis_title=TIMESTAMP_COL, yaxis_title="30d Rolling Volume", hovermode="x"
        )
        fig.show()

    # Compute dynamic fees based on rolling trading volume
    dynamic_fees = compute_fees(rolling_30d_trade_volume, fee_type=FeeType.TAKER)
    if verbose:
        # Plot daily fees [%]
        tmp_for_plot = dynamic_fees.reset_index()
        fig = px.line(
            tmp_for_plot, x=TIMESTAMP_COL, y="Fees [%]", title="Daily Fees [%]"
        )
        fig.update_layout(
            xaxis_title=TIMESTAMP_COL, yaxis_title="Daily Fees (%)", hovermode="x"
        )
        fig.show()

    # Now, simulate trades with dynamic fees. It's still not entirely accurate because trading volume
    # will depend on account size (which depends on fees), but close enough for an approximation.
    portfolio_with_fees = simulate(
        price=price,
        positions=positions,
        volume=volume,
        volume_max_size=volume_max_size,
        rebalancing_buffer=rebalancing_buffer,
        initial_capital=initial_capital,
        segment_mask=segment_mask,
        direction=Direction.Both,
        fees=dynamic_fees,
        fixed_fees=0,
        slippage=0.005,
    )

    if verbose:
        print("--- After Fees ---")
        print(portfolio_with_fees.stats())
        portfolio_with_fees.plot_drawdowns().show()
        portfolio_with_fees.plot_net_exposure().show()
        print()

        # Plot returns
        df_returns = portfolio_with_fees.returns().reset_index()
        df_returns.rename(columns={"group": "Returns"}, inplace=True)
        fig = px.bar(
            df_returns, x=TIMESTAMP_COL, y="Returns", title="Daily Returns [%]"
        )
        fig.update_layout(
            xaxis_title=TIMESTAMP_COL, yaxis_title="Returns [%]", hovermode="x"
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
            x=TIMESTAMP_COL,
            y=[turnover_col, turnover_30d_ema_col],
            title="Turnover",
        )
        fig.show()

        # Plot open positions
        df_tmp = get_num_open_positions(df_backtest)
        df_tmp = (
            get_num_open_positions(df_backtest)
            .groupby([TIMESTAMP_COL])
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
            x=TIMESTAMP_COL,
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
    generate_positions: Callable,
    periods_per_day: int,
    start_date: str,
    end_date: str,
    initial_capital: float,
    rebalancing_freq: Optional[str] = None,
    volume_max_size: float = DEFAULT_VOLUME_MAX_SIZE,
    rebalancing_buffer: float = DEFAULT_REBALANCING_BUFFER,
    generate_benchmark: Callable = generate_benchmark_btc,
    skip_plots: bool = False,
) -> List[vbt.Portfolio]:
    """Convenience wrapper for backtesting on crypto dataset

    Args:
        df_analysis (pd.DataFrame): _description_
        generate_positions (Callable): Generates positions for df_analysis. Signature `(pd.DataFrame, dict) -> pd.DataFrame`.
        generate_benchmark (Callable): Generates benchmark positions for df_analysis. Signature `(pd.DataFrame, dict) -> pd.DataFrame`.
        start_date (str): _description_
        end_date (str): _description_
        initial_capital (float): _description_
        skip_plots (bool, optional): _description_. Defaults to False.

    Returns:
        List[vbt.Portfolio]: _description_
    """
    # Backtest benchmark
    df_benchmark = generate_benchmark(df_analysis)
    pf_benchmark = backtest(
        df_benchmark,
        periods_per_day=periods_per_day,
        initial_capital=initial_capital,
        rebalancing_freq=rebalancing_freq,
        start_date=start_date,
        end_date=end_date,
        with_fees=True,
        verbose=False,
    )
    # Backtest portfolio
    df_portfolio = generate_positions(df_analysis)
    pf_portfolio = backtest(
        df_portfolio,
        periods_per_day=periods_per_day,
        initial_capital=initial_capital,
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