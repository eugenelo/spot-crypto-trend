from typing import Callable

import pandas as pd

from core.constants import POSITION_COL
from data.constants import TICKER_COL


def get_generate_benchmark_fn(params: dict) -> Callable:
    if params["generate_benchmark"] == "btc":
        generate_benchmark = generate_benchmark_btc
    else:
        raise ValueError(
            f"Unsupported 'generate_benchmark' argument: {params['generate_benchmark']}"
        )
    return generate_benchmark


def generate_benchmark_btc(df: pd.DataFrame) -> pd.DataFrame:
    # Benchmark is 100% BTC
    df_benchmark = df.copy()
    df_benchmark.loc[df_benchmark[TICKER_COL] == "BTC/USD", POSITION_COL] = 1.0
    df_benchmark.loc[df_benchmark[TICKER_COL] != "BTC/USD", POSITION_COL] = 0.0
    return df_benchmark
