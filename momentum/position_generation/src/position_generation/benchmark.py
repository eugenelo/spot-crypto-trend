import pandas as pd


def generate_benchmark_btc(df: pd.DataFrame) -> pd.DataFrame:
    # Benchmark is 100% BTC
    df_benchmark = df.copy()
    df_benchmark.loc[df_benchmark["ticker"] == "BTC/USD", "scaled_position"] = 1.0
    df_benchmark.loc[df_benchmark["ticker"] != "BTC/USD", "scaled_position"] = 0.0
    return df_benchmark
