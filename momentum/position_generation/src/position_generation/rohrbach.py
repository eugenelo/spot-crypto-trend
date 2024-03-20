import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List


def generate_positions_rohrbach(df: pd.DataFrame, params: dict):
    # Ensure that no duplicate rows exist for (ticker, timestamp) combination
    assert not df.duplicated(subset=["ticker", "timestamp"], keep=False).any()
    df = df.copy()

    if params["response_fn"] == "exponential":
        trend_signal = "trend_signal"  # range [-1, 1]
    elif params["response_fn"] == "sigmoid":
        trend_signal = "trend_signal_sigmoid"  # range [-1, 1]
    else:
        raise ValueError(f"Invalid response function: {params['response_fn']}")

    # Generate volatility forecasts
    vol_short = "price_30d_sd"
    vol_medium = "price_182d_sd"
    vol_long = "price_365d_sd"
    vol_forecast = "vol_forecast"
    df.loc[
        (~df[vol_long].isna()) & (~df[vol_medium].isna()) & (~df[vol_short].isna()),
        vol_forecast,
    ] = (
        0.4 * df[vol_short] + 0.4 * df[vol_medium] + 0.2 * df[vol_long]
    )
    df.loc[
        (df[vol_long].isna()) & (~df[vol_medium].isna()) & (~df[vol_short].isna()),
        vol_forecast,
    ] = (
        0.5 * df[vol_short] + 0.5 * df[vol_medium]
    )
    df.loc[
        (df[vol_long].isna()) & (df[vol_medium].isna()) & (~df[vol_short].isna()),
        vol_forecast,
    ] = df[vol_short]
    df.loc[
        (df[vol_long].isna()) & (df[vol_medium].isna()) & (df[vol_short].isna()),
        vol_forecast,
    ] = np.nan

    # Trade signal
    scaled_signal_col = trend_signal + "_scaled"
    df[scaled_signal_col] = df[trend_signal]

    # Position sizing
    df["total_nonzero_positions"] = df.groupby("timestamp")[
        scaled_signal_col
    ].transform(lambda x: (x != 0.0).sum())
    vol_target = params[
        "vol_target"
    ]  # Annualized volatility target for entire portfolio
    if vol_target is not None:
        # Volatility targeting
        df["vol_scaling"] = (vol_target / df["total_nonzero_positions"]) / df[
            "vol_forecast"
        ]
        df["scaled_position"] = (
            df[scaled_signal_col].fillna(value=0.0) * df["vol_scaling"]
        )
    else:
        # Scale in proportion to total signal
        # This will scale positions up when signals are weak and scale positions down when signals are strong...
        df["total_weight"] = df.groupby("timestamp")[scaled_signal_col].transform("sum")
        df["scaled_position"] = (
            df[scaled_signal_col].fillna(value=0.0) / df["total_weight"]
        )

    # Leverage constraints :(
    df["scaled_position"] = np.clip(df["scaled_position"], -1, 1)

    # Metrics
    df["total_num_positions_long"] = df.groupby("timestamp")[
        "scaled_position"
    ].transform(lambda x: (x > 0).sum())
    df["total_num_positions_short"] = df.groupby("timestamp")[
        "scaled_position"
    ].transform(lambda x: (x < 0).sum())

    return df
