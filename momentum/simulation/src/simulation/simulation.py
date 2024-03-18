import pandas as pd
import numpy as np
import plotly.express as px

from position_generation.v1 import (
    generate_positions_v1,
    CRYPTO_MOMO_DEFAULT_PARAMS,
)


def simulation(df_analysis: pd.DataFrame) -> pd.DataFrame:
    # Generate positions
    df_analysis = generate_positions_v1(df_analysis, CRYPTO_MOMO_DEFAULT_PARAMS)

    # Generate returns
    # Daily rebalancing
    df_analysis["strat_simple_returns_daily"] = (
        df_analysis["scaled_position"] * df_analysis["next_1d_returns"]
    )
    # Weekly rebalancing, sell on Thursday 12am UTC, buy on Friday 12am UTC. Put on position to get next day returns.
    df_analysis["strat_simple_returns_weekly"] = df_analysis.apply(
        lambda x: x["scaled_position"] * x["next_6d_returns"]
        if x["timestamp"].day_name() == "Thursday"
        else 0,
        axis=1,
    )
    # Benchmark 100% long BTC
    df_analysis["benchmark_simple_returns"] = df_analysis.apply(
        lambda x: x["next_1d_returns"] if x["ticker"] == "BTC/USD" else 0, axis=1
    )
    df_analysis["benchmark_log_returns"] = df_analysis.apply(
        lambda x: x["next_1d_log_returns"] if x["ticker"] == "BTC/USD" else 0, axis=1
    )
    df_analysis = df_analysis.dropna()

    # Plot cumulative returns
    df_cumulative = (
        df_analysis.groupby("timestamp")
        .agg(
            {
                "strat_simple_returns_daily": "sum",
                "strat_simple_returns_weekly": "sum",
                "benchmark_simple_returns": "sum",
                "benchmark_log_returns": "sum",
            }
        )
        .reset_index()
        .sort_values(by="timestamp")
    )
    df_cumulative["timestamp"] = pd.to_datetime(df_cumulative["timestamp"])
    df_cumulative["strat_log_returns_daily"] = np.log(
        df_cumulative["strat_simple_returns_daily"] + 1
    )
    df_cumulative["strat_log_returns_weekly"] = np.log(
        df_cumulative["strat_simple_returns_weekly"] + 1
    )
    df_cumulative["cum_strat_log_returns_daily"] = df_cumulative[
        "strat_log_returns_daily"
    ].cumsum()
    df_cumulative["cum_strat_log_returns_weekly"] = df_cumulative[
        "strat_log_returns_weekly"
    ].cumsum()
    # df_cumulative["cum_benchmark_log_returns"] = df_cumulative["benchmark_log_returns"].cumsum()
    df_cumulative["cum_benchmark_log_returns"] = np.log(
        df_cumulative["benchmark_simple_returns"] + 1
    ).cumsum()

    starting_bankroll = 50000
    df_cumulative["cum_strat_returns_daily"] = starting_bankroll * (
        1 + (np.exp(df_cumulative["cum_strat_log_returns_daily"]) - 1)
    )
    df_cumulative["cum_strat_returns_weekly"] = starting_bankroll * (
        1 + (np.exp(df_cumulative["cum_strat_log_returns_weekly"]) - 1)
    )
    df_cumulative["cum_benchmark_returns"] = starting_bankroll * (
        1 + (np.exp(df_cumulative["cum_benchmark_log_returns"]) - 1)
    )

    fig = px.line(
        df_cumulative,
        x="timestamp",
        y=[
            "cum_strat_log_returns_daily",
            "cum_strat_log_returns_weekly",
            "cum_benchmark_log_returns",
        ],
        title="Cum Log Returns",
    )
    fig.show()
    fig = px.line(
        df_cumulative,
        x="timestamp",
        y=[
            "cum_strat_returns_daily",
            "cum_strat_returns_weekly",
            "cum_benchmark_returns",
        ],
        title="Cum Returns",
    )
    fig.show()

    return df_analysis
