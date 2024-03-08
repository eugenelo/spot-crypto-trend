import vectorbt as vbt
import pandas as pd
import plotly.express as px
import numpy as np
from functools import reduce
from typing import List

from position_generation import (
    generate_benchmark,
    generate_positions,
    nonempty_positions,
    CRYPTO_MOMO_DEFAULT_PARAMS,
)


def kraken_maker_fees(rolling_30d_volume: float):
    # Define your fee structure based on rolling_30d_volume
    if rolling_30d_volume <= 50000:
        return 0.0016
    elif rolling_30d_volume <= 100000:
        return 0.0014
    elif rolling_30d_volume <= 250000:
        return 0.0012
    elif rolling_30d_volume <= 500000:
        return 0.0010
    elif rolling_30d_volume <= 1000000:
        return 0.0008
    elif rolling_30d_volume <= 2500000:
        return 0.0006
    elif rolling_30d_volume <= 5000000:
        return 0.0004
    elif rolling_30d_volume <= 10000000:
        return 0.0002
    else:
        return 0.0


def compute_fees(rolling_30d_volume):
    fees = rolling_30d_volume.apply(lambda x: kraken_maker_fees(x))
    return fees


def get_stats_of_interest(portfolio: vbt.Portfolio, name: str):
    metrics_of_interest = [
        "start",
        "end",
        "start_value",
        "end_value",
        "total_return",
        "max_gross_exposure",
        "total_fees_paid",
        "max_dd",
        "max_dd_duration",
        "total_trades",
        "win_rate",
        "sharpe_ratio",
        "sortino_ratio",
    ]
    stats = portfolio.stats(metrics=metrics_of_interest)
    # Append annualized return and volatility
    tmp = pd.Series(
        {
            "Annualized Return [%]": 100.0 * portfolio.annualized_return(),
            "Annualized Volatility [%]": 100.0 * portfolio.annualized_volatility(),
        }
    )
    stats = pd.concat([stats, tmp]).reset_index()
    stats.rename(columns={stats.columns[-1]: name}, inplace=True)
    return stats


def backtest(
    df_backtest,
    initial_capital,
    rebalancing_freq: str,
    start_date=None,
    end_date=None,
    with_fees=True,
    verbose=False,
) -> vbt.Portfolio:
    """Backtest a strategy

    Args:
        df_backtest (pd.DataFrame): Should contain timestamp, close, position columns. Assuming 0 slippage, entire order can be filled.
        fee_policy (Function): Fee policy to apply to the change in positions every day.
        starting_bankroll (int): Startin bankroll in USD
    """
    # Copy dataframe to avoid corrupting original contents
    df_backtest = df_backtest.copy()

    def apply_time_window(df):
        if start_date is not None:
            df = df.loc[df["timestamp"] >= start_date]
        if end_date is not None:
            df = df.loc[df["timestamp"] <= end_date]
        return df

    df_backtest = apply_time_window(df_backtest)

    # Transform data from long to wide
    df_backtest = df_backtest.sort_values(["timestamp", "ticker"])
    close = df_backtest[["timestamp", "ticker", "close"]]
    close = pd.pivot_table(close, index="timestamp", columns="ticker", values="close")
    # Create positions for daily rebalancing
    positions_daily = df_backtest[["timestamp", "ticker", "scaled_position"]]
    positions_daily = pd.pivot_table(
        positions_daily,
        index="timestamp",
        columns="ticker",
        values="scaled_position",
        fill_value=0.0,
    )
    # Resample to input rebalancing frequency
    positions = positions_daily.resample(rebalancing_freq, origin="start_day").first()
    # Resample price data as well
    orig_close_index = close.index.copy()
    close = close.resample(rebalancing_freq, origin="start_day").first()

    # Simulate trades without fees to calculate trading volume
    portfolio_no_fees = vbt.Portfolio.from_orders(
        close=close,
        size=positions,
        size_type="targetpercent",
        init_cash=initial_capital,
        cash_sharing=True,
        fees=0,
        freq=rebalancing_freq,
    )
    if verbose:
        print("--- Before Fees ---")
        print(portfolio_no_fees.stats())
        portfolio_no_fees.plot_cum_returns().show()
        print()
    if not with_fees:
        return portfolio_no_fees

    # Calculate 30d rolling trading volume
    daily_volumes = portfolio_no_fees.trades.records_readable.groupby(
        portfolio_no_fees.trades.records_readable["Entry Timestamp"]
    )["Size"].sum()
    # Reindex back to daily (not rebalance freq), fill in gaps where no trades were placed
    daily_volumes = daily_volumes.reindex(orig_close_index, fill_value=0)

    rolling_30d_volume = daily_volumes.rolling(window=30, min_periods=1).sum()
    if verbose:
        # Plot rolling 30d volume
        tmp_for_plot = rolling_30d_volume.reset_index()
        fig = px.line(tmp_for_plot, x="timestamp", y="Size", title="30d Rolling Volume")
        fig.update_layout(
            xaxis_title="Timestamp", yaxis_title="30d Rolling Volume", hovermode="x"
        )
        fig.show()

    # Compute dynamic fees based on rolling trading volume
    dynamic_fees = compute_fees(rolling_30d_volume)
    if verbose:
        # Plot daily fees [%]
        tmp_for_plot = dynamic_fees.reset_index()
        fig = px.line(tmp_for_plot, x="timestamp", y="Size", title="Daily Fees [%]")
        fig.update_layout(
            xaxis_title="Timestamp", yaxis_title="Daily Fees (%)", hovermode="x"
        )
        fig.show()
    # Reindex back to rebalance freq
    dynamic_fees = dynamic_fees.reindex(close.index)

    # Now, simulate trades with dynamic fees. It's still not entirely accurate because trading volume will depend on account size (which depends on fees), but close enough for an approximation.
    portfolio_with_fees = vbt.Portfolio.from_orders(
        close=close,
        size=positions,
        size_type="targetpercent",
        init_cash=initial_capital,
        cash_sharing=True,
        fees=dynamic_fees,
        freq=rebalancing_freq,
    )

    if verbose:
        print("--- After Fees ---")
        print(portfolio_with_fees.stats())
        portfolio_with_fees.plot_cum_returns().show()
        portfolio_with_fees.plot_drawdowns().show()
        portfolio_with_fees.plot_net_exposure().show()
        # print(
        #     portfolio_with_fees.entry_trades.records_readable.sort_values(
        #         by="Entry Timestamp"
        #     ).head(10)
        # )
        # # portfolio_with_fees.plot_value().show()
        # # portfolio_with_fees.plot_trades().show()

        # Plot returns
        df_returns = portfolio_with_fees.returns().reset_index()
        df_returns.rename(columns={"group": "Returns"}, inplace=True)
        fig = px.bar(df_returns, x="timestamp", y="Returns", title="Daily Returns [%]")
        fig.update_layout(
            xaxis_title="Timestamp", yaxis_title="Returns [%]", hovermode="x"
        )
        fig.show()
        print()

    return portfolio_with_fees


def backtest_crypto(
    df_analysis: pd.DataFrame,
    start_date,
    end_date,
    initial_capital,
    params: dict,
    skip_plots: bool = False,
) -> List[vbt.Portfolio]:
    # Benchmark is 100% BTC
    df_benchmark = generate_benchmark(df_analysis)
    pf_benchmark = backtest(
        df_benchmark,
        initial_capital=initial_capital,
        rebalancing_freq="1d",
        start_date=start_date,
        end_date=end_date,
        with_fees=True,
        verbose=False,
    )
    df_portfolio = generate_positions(df_analysis, params=params)
    pf_portfolio = backtest(
        df_portfolio,
        initial_capital=initial_capital,
        rebalancing_freq=params["rebalancing_freq"],
        start_date=start_date,
        end_date=end_date,
        with_fees=True,
        verbose=not skip_plots,
    )

    # Compare returns, stats
    pf_names = ["Benchmark", "Strategy"]
    portfolios = [pf_benchmark, pf_portfolio]
    display_stats(portfolios, pf_names)
    if not skip_plots:
        plot_cumulative_returns(portfolios, pf_names)
        plot_rolling_returns(portfolios, pf_names, window=30)

    # Double check max total num positions on at any time
    print()
    print(f"max_total_num_positions: {df_portfolio['total_num_positions'].max()}")
    print(f"num_assets_to_keep: {CRYPTO_MOMO_DEFAULT_PARAMS['num_assets_to_keep']}")

    # # Print positions on last day
    # print(nonempty_positions(df_portfolio, timestamp=end_date))

    return portfolios


def display_stats(portfolios: List[vbt.Portfolio], portfolio_names: List[str]):
    # Display Stats for all portfolios
    stats = []
    for i, pf in enumerate(portfolios):
        pf_stats = get_stats_of_interest(pf, portfolio_names[i])
        stats.append(pf_stats)
    df_stats = reduce(
        lambda left, right: pd.merge(left, right, on=["index"], how="outer"), stats
    ).set_index("index")
    print(df_stats)


def plot_cumulative_returns(
    portfolios: List[vbt.Portfolio], portfolio_names: List[str]
):
    assert len(portfolios) == len(portfolio_names) and len(portfolios) > 0

    # Gather cumulative returns for all portfolios
    cumulative_returns = []
    first_pf_cumulative = None
    for i, pf in enumerate(portfolios):
        df_cumulative = pf.cumulative_returns().reset_index()
        df_cumulative.rename(columns={"group": portfolio_names[i]}, inplace=True)

        # Reindex cumulative returns to be consistent with the first portfolio's index
        if i == 0:
            first_pf_cumulative = df_cumulative.set_index("timestamp")
        else:
            df_cumulative = (
                df_cumulative.set_index("timestamp")
                .reindex(first_pf_cumulative.index)
                .ffill()
                .reset_index()
            )

        cumulative_returns.append(df_cumulative)

    # Plot all cum returns overlayed on one plot
    df_cumulative = reduce(
        lambda left, right: pd.merge(left, right, on=["timestamp"], how="outer"),
        cumulative_returns,
    )
    fig = px.line(
        df_cumulative,
        x="timestamp",
        y=portfolio_names,
        title="Cumulative Returns",
    )
    fig.show()


def plot_rolling_returns(
    portfolios: List[vbt.Portfolio], portfolio_names: List[str], window=30
):
    assert len(portfolios) == len(portfolio_names) and len(portfolios) > 0

    # Gather rolling returns for all portfolios
    returns = []
    first_pf_returns = None
    for i, pf in enumerate(portfolios):
        pf_returns = pf.returns().reset_index()
        pf_returns.rename(columns={"group": portfolio_names[i]}, inplace=True)

        # Reindex returns to be consistent with the first portfolio's index
        if i == 0:
            first_pf_returns = pf_returns.set_index("timestamp")
        else:
            pf_returns = (
                pf_returns.set_index("timestamp")
                .reindex(first_pf_returns.index)
                .ffill()
                .reset_index()
            )

        returns.append(pf_returns)

    # Plot all rolling returns overlayed on one plot
    rolling_returns = [
        x.set_index("timestamp").rolling(window).mean().reset_index() for x in returns
    ]
    df_rolling = reduce(
        lambda left, right: pd.merge(left, right, on=["timestamp"], how="outer"),
        rolling_returns,
    )
    fig = px.line(
        df_rolling,
        x="timestamp",
        y=portfolio_names,
        title=f"Rolling {window}d Returns",
    )
    fig.show()
