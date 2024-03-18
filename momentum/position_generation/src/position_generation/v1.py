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
position_types = ["simple", "quintile", "crossover"]


def generate_positions_v1(df: pd.DataFrame, params: dict):
    # Ensure that no duplicate rows exist for (ticker, timestamp) combination
    assert not df.duplicated(subset=["ticker", "timestamp"], keep=False).any()

    momentum_factor = params["momentum_factor"]
    quintile_factor = momentum_factor.split("_")[0] + "_log_quintile"
    num_assets_to_keep = params["num_assets_to_keep"]
    min_signal_threshold = params["min_signal_threshold"]
    max_signal_threshold = params["max_signal_threshold"]

    # Work around NAs in other columns
    cols_to_keep = [
        "ticker",
        "timestamp",
        momentum_factor,
        "volume_consistent",
        quintile_factor,
    ]
    df_operate = df[cols_to_keep].dropna()

    df["rank"] = (
        df_operate[df_operate["volume_consistent"]]
        .groupby(["timestamp"])[momentum_factor]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    df["rank"] = df["rank"].fillna(np.inf)
    df["signal_simple"] = (df["rank"] <= num_assets_to_keep) & (df["volume_consistent"])
    df["signal_quintile"] = (
        (df["rank"] <= num_assets_to_keep)
        & (df[quintile_factor] == 4)
        & (df["volume_consistent"])
    )
    # df["signal_ema"] = (
    #     (df["rank"] <= num_assets_to_keep)
    #     # & (df[quintile_factor] == 4)
    #     & (df["9d_ema"] >= df["26d_ema"])
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
    df["total_num_positions_long"] = df.groupby("timestamp")[
        "scaled_position"
    ].transform(lambda x: (x > 0).sum())
    return df
