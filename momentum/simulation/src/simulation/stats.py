from functools import reduce
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

from data.constants import DATETIME_COL
from simulation.vbt import (
    ENTRY_TIMESTAMP_COL,
    EXIT_TIMESTAMP_COL,
    get_annualized_return,
    get_annualized_volatility,
    get_cumulative_returns,
    get_entry_trades,
    get_exit_trades,
    get_returns,
    get_value,
    vbt,
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
            "Annualized Return [%]": 100.0 * get_annualized_return(portfolio),
            "Annualized Volatility [%]": 100.0 * get_annualized_volatility(portfolio),
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
    pf_value = get_value(pf_portfolio)
    trade_volume = trade_volume.reindex(pf_value.index, fill_value=0)
    turnover = (trade_volume / pf_value).rename("Turnover [%]")
    return turnover


def get_trade_volume(pf_portfolio: vbt.Portfolio) -> pd.Series:
    entry_trades = get_entry_trades(pf_portfolio)
    entry_trades["Entry Size [$]"] = (
        entry_trades["Size"] * entry_trades["Avg Entry Price"]
    )
    entry_volume = entry_trades[[ENTRY_TIMESTAMP_COL, "Entry Size [$]"]]
    entry_volume = entry_volume.rename(columns={ENTRY_TIMESTAMP_COL: DATETIME_COL})
    entry_volume = (
        entry_volume.sort_values(by=DATETIME_COL)
        .groupby(DATETIME_COL)
        .agg({"Entry Size [$]": "sum"})
        .reset_index()
    )

    exit_trades = get_exit_trades(pf_portfolio)
    exit_trades["Exit Size [$]"] = entry_trades["Size"] * entry_trades["Avg Exit Price"]
    exit_volume = exit_trades[[EXIT_TIMESTAMP_COL, "Exit Size [$]"]]
    exit_volume = exit_volume.rename(columns={EXIT_TIMESTAMP_COL: DATETIME_COL})
    exit_volume = (
        exit_volume.sort_values(by=DATETIME_COL)
        .groupby(DATETIME_COL)
        .agg({"Exit Size [$]": "sum"})
        .reset_index()
    )

    df_volume = entry_volume.merge(exit_volume, how="outer", on=DATETIME_COL).fillna(
        value=0
    )
    df_volume["Traded Size [$]"] = (
        df_volume["Entry Size [$]"] + df_volume["Exit Size [$]"]
    )

    trade_volume = df_volume["Traded Size [$]"]
    trade_volume.index = df_volume[DATETIME_COL]
    return trade_volume


def aggregate_stats(
    portfolios: List[vbt.Portfolio], portfolio_names: List[str]
) -> pd.DataFrame:
    # Consolidate stats for all portfolios
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
    return df_stats.reindex(column_order)


def plot_cumulative_returns(
    portfolios: List[vbt.Portfolio], portfolio_names: List[str]
):
    assert len(portfolios) == len(portfolio_names) and len(portfolios) > 0

    # Gather cumulative returns for all portfolios
    cumulative_returns = []
    first_pf_cumulative = None
    for i, pf in enumerate(portfolios):
        df_cumulative = get_cumulative_returns(pf).reset_index()
        df_cumulative.rename(columns={"group": portfolio_names[i]}, inplace=True)

        # Reindex cumulative returns to be consistent with the first portfolio's index
        if i == 0:
            first_pf_cumulative = df_cumulative.set_index(DATETIME_COL)
        else:
            df_cumulative = (
                df_cumulative.set_index(DATETIME_COL)
                .reindex(first_pf_cumulative.index)
                .ffill()
                .reset_index()
            )

        cumulative_returns.append(df_cumulative)

    # Plot all cum returns overlayed on one plot
    df_cumulative = reduce(
        lambda left, right: pd.merge(left, right, on=[DATETIME_COL], how="outer"),
        cumulative_returns,
    )
    fig = px.line(
        df_cumulative,
        x=DATETIME_COL,
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
        pf_returns = get_returns(pf).reset_index()
        pf_returns.rename(columns={"group": portfolio_names[i]}, inplace=True)

        # Reindex returns to be consistent with the first portfolio's index
        if i == 0:
            first_pf_returns = pf_returns.set_index(DATETIME_COL)
        else:
            pf_returns = (
                pf_returns.set_index(DATETIME_COL)
                .reindex(first_pf_returns.index)
                .ffill()
                .reset_index()
            )

        returns.append(pf_returns)

    # Plot all rolling returns overlayed on one plot
    rolling_returns = [
        x.set_index(DATETIME_COL).rolling(window).mean().reset_index() for x in returns
    ]
    df_rolling = reduce(
        lambda left, right: pd.merge(left, right, on=[DATETIME_COL], how="outer"),
        rolling_returns,
    )
    fig = px.line(
        df_rolling,
        x=DATETIME_COL,
        y=portfolio_names,
        title=f"Rolling {window}d Returns",
    )
    fig.show()


def _comp(returns: pd.Series):
    """Calculates total compounded returns"""
    return returns.add(1).prod() - 1


def plot_returns_distribution(
    returns: pd.Series,
    benchmark: Optional[pd.Series] = None,
    resampling_freq="ME",
    bins=20,
    fontname="Arial",
    title="Returns",
    kde=True,
    figsize=(10, 6),
    ylabel=True,
    subtitle=True,
    compounded=True,
    savefig=None,
    show=True,
):
    colors = [
        "#FEDD78",
        "#348DC1",
        "#BA516B",
        "#4FA487",
        "#9B59B6",
        "#613F66",
        "#84B082",
        "#DC136C",
        "#559CAD",
        "#4A5899",
    ]

    apply_fnc = _comp if compounded else np.sum
    if benchmark is not None:
        benchmark = (
            benchmark.fillna(0)
            .resample(resampling_freq)
            .apply(apply_fnc)
            .resample(resampling_freq)
            .last()
        )

    returns = (
        returns.fillna(0)
        .resample(resampling_freq)
        .apply(apply_fnc)
        .resample(resampling_freq)
        .last()
    )

    figsize = (0.995 * figsize[0], figsize[1])
    fig, ax = plt.subplots(figsize=figsize)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    fig.suptitle(
        title, y=0.94, fontweight="bold", fontname=fontname, fontsize=14, color="black"
    )

    if subtitle:
        ax.set_title(
            "%s - %s           \n"
            % (
                returns.index.date[:1][0].strftime("%Y"),
                returns.index.date[-1:][0].strftime("%Y"),
            ),
            fontsize=12,
            color="gray",
        )

    fig.set_facecolor("white")
    ax.set_facecolor("white")

    if isinstance(returns, pd.DataFrame) and len(returns.columns) == 1:
        returns = returns[returns.columns[0]]

    pallete = colors[1:2] if benchmark is None else colors[:2]
    alpha = 0.7
    if isinstance(returns, pd.DataFrame):
        pallete = (
            colors[1 : len(returns.columns) + 1]
            if benchmark is None
            else colors[: len(returns.columns) + 1]
        )
        if len(returns.columns) > 1:
            alpha = 0.5

    if benchmark is not None:
        benchmark_df = benchmark.to_frame().rename(mapper=lambda x: "benchmark", axis=1)
        if isinstance(returns, pd.Series):
            returns_df = returns.to_frame().rename(mapper=lambda x: "Returns", axis=1)
            combined_returns = (
                benchmark_df.join(returns_df)
                .stack()
                .reset_index()
                .rename(columns={"level_1": "", 0: "Returns"})
            )
        elif isinstance(returns, pd.DataFrame):
            combined_returns = (
                benchmark_df.join(returns)
                .stack()
                .reset_index()
                .rename(columns={"level_1": "", 0: "Returns"})
            )
        sns.histplot(
            data=combined_returns,
            x="Returns",
            bins=bins,
            alpha=alpha,
            kde=kde,
            stat="density",
            hue="",
            palette=pallete,
            ax=ax,
        )

    else:
        if isinstance(returns, pd.Series):
            combined_returns = returns.copy()
            if kde:
                sns.kdeplot(data=combined_returns, color="black", ax=ax)
            sns.histplot(
                data=combined_returns,
                bins=bins,
                alpha=alpha,
                kde=False,
                stat="density",
                color=colors[1],
                ax=ax,
            )

        elif isinstance(returns, pd.DataFrame):
            combined_returns = (
                returns.stack()
                .reset_index()
                .rename(columns={"level_1": "", 0: "Returns"})
            )
            # sns.kdeplot(data=combined_returns, color='black', ax=ax)
            sns.histplot(
                data=combined_returns,
                x="Returns",
                bins=bins,
                alpha=alpha,
                kde=kde,
                stat="density",
                hue="",
                palette=pallete,
                ax=ax,
            )

    # Why do we need average?
    if isinstance(combined_returns, pd.Series) or len(combined_returns.columns) == 1:
        ax.axvline(
            combined_returns.mean(),
            ls="--",
            lw=1.5,
            zorder=2,
            label="Average",
            color="red",
        )

    # plt.setp(x.get_legend().get_texts(), fontsize=11)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, loc: "{:,}%".format(int(x * 100)))
    )

    # Removed static lines for clarity
    # ax.axhline(0.01, lw=1, color="#000000", zorder=2)
    # ax.axvline(0, lw=1, color="#000000", zorder=2)

    ax.set_xlabel("")
    ax.set_ylabel(
        "Occurrences", fontname=fontname, fontweight="bold", fontsize=12, color="black"
    )
    ax.yaxis.set_label_coords(-0.1, 0.5)

    # fig.autofmt_xdate()

    try:
        plt.subplots_adjust(hspace=0, bottom=0, top=1)
    except Exception:
        pass

    try:
        fig.tight_layout()
    except Exception:
        pass

    if savefig:
        if isinstance(savefig, dict):
            plt.savefig(**savefig)
        else:
            plt.savefig(savefig)

    if show:
        plt.show(block=False)

    plt.close()

    if not show:
        return fig

    return None
