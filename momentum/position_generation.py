import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List


CRYPTO_MOMO_DEFAULT_PARAMS = {
    "momentum_factor": "30d_log_returns",
    "num_assets_to_keep": int(1e6),
    "min_signal_threshold": 0.05,
    "max_signal_threshold": 0.15,
    "type": "simple",
    "rebalancing_freq": "1d",
}
position_types = [
    "simple",
    "quintile",
    # "crossover"
]


def generate_positions(df: pd.DataFrame, params: dict):
    momentum_factor = params["momentum_factor"]
    quintile_factor = momentum_factor.split("_")[0] + "_log_quintile"
    num_assets_to_keep = params["num_assets_to_keep"]
    min_signal_threshold = params["min_signal_threshold"]
    max_signal_threshold = params["max_signal_threshold"]

    df["rank"] = (
        df[df["volume_consistent"]]
        .groupby(["timestamp"])[momentum_factor]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    df["rank"].fillna(np.inf, inplace=True)
    df["signal_simple"] = (df["rank"] <= num_assets_to_keep) & (df["volume_consistent"])
    df["signal_quintile"] = (
        (df["rank"] <= num_assets_to_keep)
        & (df[quintile_factor] == 4)
        & (df["volume_consistent"])
    )
    # df["signal_ema"] = (
    #     (df["rank"] <= num_assets_to_keep)
    #     & (df[quintile_factor] == 4)
    #     & (df["5d_ema"] >= df["50d_ema"])
    #     & (df["volume_consistent"])
    # )
    df["position_simple"] = df.apply(
        lambda x: np.clip(
            (x[momentum_factor] - min_signal_threshold)
            / (max_signal_threshold - min_signal_threshold),
            0,
            1,
        )
        if x["signal_simple"]
        else 0,
        axis=1,
    )
    df["position_quintile"] = df.apply(
        lambda x: np.clip(
            (x[momentum_factor] - min_signal_threshold)
            / (max_signal_threshold - min_signal_threshold),
            0,
            1,
        )
        if x["signal_quintile"]
        else 0,
        axis=1,
    )
    # df["position_crossover"] = df.apply(
    #     lambda x: np.clip(
    #         (x[momentum_factor] - min_signal_threshold)
    #         / (max_signal_threshold - min_signal_threshold),
    #         0,
    #         1,
    #     )
    #     if x["signal_ema"]
    #     else 0,
    #     axis=1,
    # )

    position = f"position_{params['type']}"
    df["total_position_weight"] = df.groupby(["timestamp"])[position].transform("sum")
    df["scaled_position"] = df.apply(
        lambda x: x[position] / max(x["total_position_weight"], 1), axis=1
    )
    df["total_num_positions"] = df.groupby("timestamp")["scaled_position"].transform(
        lambda x: (x > 0).sum()
    )
    return df


def generate_benchmark(df: pd.DataFrame):
    # Benchmark is 100% BTC
    df_benchmark = df.copy()
    df_benchmark.loc[df_benchmark["ticker"] == "BTC/USD", "scaled_position"] = 1.0
    df_benchmark.loc[df_benchmark["ticker"] != "BTC/USD", "scaled_position"] = 0.0
    return df_benchmark


def nonempty_positions(
    df_positions: pd.DataFrame, tickers_to_keep: Optional[List[str]] = None
) -> pd.DataFrame:
    if tickers_to_keep is not None:
        tickers_to_keep_lst = tickers_to_keep.copy()
    else:
        tickers_to_keep_lst = []
    res = df_positions.sort_values(
        by=["timestamp", "rank"], ascending=[False, True]
    ).loc[
        (df_positions["scaled_position"] > 0.0)
        | (df_positions["ticker"].isin(tickers_to_keep_lst))
    ]
    return res[
        [
            "timestamp",
            "ticker",
            "30d_log_returns",
            "rank",
            "volume_consistent",
            "30d_num_days_volume_above_5M",
            "scaled_position",
        ]
    ]
