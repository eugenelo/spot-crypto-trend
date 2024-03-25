import plotly.express as px
import pandas as pd
from typing import List
from functools import reduce

import vectorbt as vbt


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
            "Avg. Fee per Trade [$]": stats["Total Fees Paid"] / stats["Total Trades"],
        }
    )
    # Add units to some column names
    stats["Start Value [$]"] = stats.pop("Start Value")
    stats["End Value [$]"] = stats.pop("End Value")
    stats["Total Fees Paid [$]"] = stats.pop("Total Fees Paid")
    stats = pd.concat([stats, tmp]).reset_index()
    stats.rename(columns={stats.columns[-1]: name}, inplace=True)
    return stats


def display_stats(portfolios: List[vbt.Portfolio], portfolio_names: List[str]):
    # Display Stats for all portfolios
    stats = []
    for i, pf in enumerate(portfolios):
        pf_stats = get_stats_of_interest(pf, portfolio_names[i])
        stats.append(pf_stats)
    df_stats = reduce(
        lambda left, right: pd.merge(left, right, on=["index"], how="outer"), stats
    ).set_index("index")
    column_order = [
        "Annualized Return [%]",
        "Annualized Volatility [%]",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Max Drawdown [%]",
        "Max Drawdown Duration",
        "Max Gross Exposure [%]",
        "Win Rate [%]",
        "Start",
        "End",
        "Start Value [$]",
        "End Value [$]",
        "Total Return [%]",
        "Total Trades",
        "Total Fees Paid [$]",
        "Avg. Fee per Trade [$]",
    ]
    print(df_stats.reindex(column_order))


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
