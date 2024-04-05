import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List

from data.constants import TIMESTAMP_COL, TICKER_COL
from core.constants import POSITION_COL


CRYPTO_MOMO_DEFAULT_PARAMS = {
    "momentum_factor": "30d_log_returns",
    "num_assets_to_keep": int(1e6),
    "min_signal_threshold": 0.05,
    "max_signal_threshold": 0.15,
    "type": "simple",
    "rebalancing_freq": "1d",
}
position_types = ["simple", "decile", "crossover"]


def generate_positions_v1(df: pd.DataFrame, params: dict):
    # Ensure that no duplicate rows exist for (ticker, timestamp) combination
    assert not df.duplicated(subset=[TICKER_COL, TIMESTAMP_COL], keep=False).any()

    momentum_factor = params["momentum_factor"]
    decile_factor = momentum_factor.split("_")[0] + "_log_returns_decile"
    num_assets_to_keep = params["num_assets_to_keep"]
    min_signal_threshold = params["min_signal_threshold"]
    max_signal_threshold = params["max_signal_threshold"]

    # Work around NAs in other columns
    cols_to_keep = [
        TICKER_COL,
        TIMESTAMP_COL,
        momentum_factor,
        "volume_consistent",
        decile_factor,
    ]
    df_operate = df[cols_to_keep].dropna()

    df["rank"] = (
        df_operate[df_operate["volume_consistent"]]
        .groupby([TIMESTAMP_COL])[momentum_factor]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    df["rank"] = df["rank"].fillna(np.inf)
    df["signal_simple"] = (df["rank"] <= num_assets_to_keep) & (df["volume_consistent"])
    df["signal_decile"] = (
        (df["rank"] <= num_assets_to_keep)
        & (df[decile_factor] >= 8)
        & (df["volume_consistent"])
    )
    # df["signal_ema"] = (
    #     (df["rank"] <= num_assets_to_keep)
    #     # & (df[decile_factor] >= 8)
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
    df["position_decile"] = df.apply(
        lambda x: np.clip(
            (x[momentum_factor] - min_signal_threshold)
            / (max_signal_threshold - min_signal_threshold),
            0,
            1,
        )
        if x["signal_decile"]
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
    df["total_position_weight"] = df.groupby([TIMESTAMP_COL])[position].transform("sum")
    df[POSITION_COL] = df.apply(
        lambda x: x[position] / max(x["total_position_weight"], 1), axis=1
    )
    df["total_num_positions_long"] = df.groupby(TIMESTAMP_COL)[POSITION_COL].transform(
        lambda x: (x > 0).sum()
    )
    return df
