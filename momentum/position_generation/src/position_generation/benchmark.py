import pandas as pd

from data.constants import TICKER_COL
from core.constants import POSITION_COL


def generate_benchmark_btc(df: pd.DataFrame) -> pd.DataFrame:
    # Benchmark is 100% BTC
    df_benchmark = df.copy()
    df_benchmark.loc[df_benchmark[TICKER_COL] == "BTC/USD", POSITION_COL] = 1.0
    df_benchmark.loc[df_benchmark[TICKER_COL] != "BTC/USD", POSITION_COL] = 0.0
    return df_benchmark
