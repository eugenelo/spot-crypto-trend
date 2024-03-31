import plotly.express as px
import pandas as pd
from typing import List
from functools import reduce

import vectorbt as vbt

from core.constants import TIMESTAMP_COL, POSITION_COL
from position_generation.constants import (
    NUM_OPEN_LONG_POSITIONS_COL,
    NUM_OPEN_SHORT_POSITIONS_COL,
    NUM_OPEN_POSITIONS_COL,
)


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
    # Append additional stats
    tmp = pd.Series(
        {
            "Annualized Return [%]": 100.0 * portfolio.annualized_return(),
            "Annualized Volatility [%]": 100.0 * portfolio.annualized_volatility(),
            "Avg. Fee per Trade [$]": stats["Total Fees Paid"] / stats["Total Trades"],
            "Avg Daily Turnover [%]": 100.0 * get_turnover(portfolio).mean(),
        }
    )
    # Add units to some column names
    stats["Start Value [$]"] = stats.pop("Start Value")
    stats["End Value [$]"] = stats.pop("End Value")
    stats["Total Fees Paid [$]"] = stats.pop("Total Fees Paid")
    stats = pd.concat([stats, tmp]).reset_index()
    stats.rename(columns={stats.columns[-1]: name}, inplace=True)
    return stats


def get_turnover(pf_portfolio: vbt.Portfolio) -> pd.Series:
    trade_volume = get_trade_volume(pf_portfolio)
    pf_value = pf_portfolio.value()
    trade_volume = trade_volume.reindex(pf_value.index, fill_value=0)
    turnover = (trade_volume / pf_portfolio.value()).rename("Turnover [%]")
    return turnover


def get_trade_volume(pf_portfolio: vbt.Portfolio) -> pd.Series:
    entry_trades = pf_portfolio.entry_trades.records_readable
    entry_trades["Entry Size [$]"] = (
        entry_trades["Size"] * entry_trades["Avg Entry Price"]
    )
    entry_volume = entry_trades[["Entry Timestamp", "Entry Size [$]"]]
    entry_volume = entry_volume.rename(columns={"Entry Timestamp": TIMESTAMP_COL})
    entry_volume = (
        entry_volume.sort_values(by=TIMESTAMP_COL)
        .groupby(TIMESTAMP_COL)
        .agg({"Entry Size [$]": "sum"})
        .reset_index()
    )

    exit_trades = pf_portfolio.exit_trades.records_readable
    exit_trades["Exit Size [$]"] = entry_trades["Size"] * entry_trades["Avg Exit Price"]
    exit_volume = exit_trades[["Exit Timestamp", "Exit Size [$]"]]
    exit_volume = exit_volume.rename(columns={"Exit Timestamp": TIMESTAMP_COL})
    exit_volume = (
        exit_volume.sort_values(by=TIMESTAMP_COL)
        .groupby(TIMESTAMP_COL)
        .agg({"Exit Size [$]": "sum"})
        .reset_index()
    )

    df_volume = entry_volume.merge(exit_volume, how="outer", on=TIMESTAMP_COL).fillna(
        value=0
    )
    df_volume["Traded Size [$]"] = (
        df_volume["Entry Size [$]"] + df_volume["Exit Size [$]"]
    )

    trade_volume = df_volume["Traded Size [$]"]
    trade_volume.index = df_volume[TIMESTAMP_COL]
    return trade_volume


def get_num_open_positions(df: pd.DataFrame) -> pd.DataFrame:
    # Log open positions
    df[NUM_OPEN_LONG_POSITIONS_COL] = df.groupby(TIMESTAMP_COL)[POSITION_COL].transform(
        lambda x: (x > 0).sum()
    )
    df[NUM_OPEN_SHORT_POSITIONS_COL] = df.groupby(TIMESTAMP_COL)[
        POSITION_COL
    ].transform(lambda x: (x < 0).sum())
    df[NUM_OPEN_POSITIONS_COL] = (
        df[NUM_OPEN_LONG_POSITIONS_COL] + df[NUM_OPEN_SHORT_POSITIONS_COL]
    )

    return df


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
            first_pf_cumulative = df_cumulative.set_index(TIMESTAMP_COL)
        else:
            df_cumulative = (
                df_cumulative.set_index(TIMESTAMP_COL)
                .reindex(first_pf_cumulative.index)
                .ffill()
                .reset_index()
            )

        cumulative_returns.append(df_cumulative)

    # Plot all cum returns overlayed on one plot
    df_cumulative = reduce(
        lambda left, right: pd.merge(left, right, on=[TIMESTAMP_COL], how="outer"),
        cumulative_returns,
    )
    fig = px.line(
        df_cumulative,
        x=TIMESTAMP_COL,
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
            first_pf_returns = pf_returns.set_index(TIMESTAMP_COL)
        else:
            pf_returns = (
                pf_returns.set_index(TIMESTAMP_COL)
                .reindex(first_pf_returns.index)
                .ffill()
                .reset_index()
            )

        returns.append(pf_returns)

    # Plot all rolling returns overlayed on one plot
    rolling_returns = [
        x.set_index(TIMESTAMP_COL).rolling(window).mean().reset_index() for x in returns
    ]
    df_rolling = reduce(
        lambda left, right: pd.merge(left, right, on=[TIMESTAMP_COL], how="outer"),
        rolling_returns,
    )
    fig = px.line(
        df_rolling,
        x=TIMESTAMP_COL,
        y=portfolio_names,
        title=f"Rolling {window}d Returns",
    )
    fig.show()
