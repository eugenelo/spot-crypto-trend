import pandas as pd

from signal_generation.common import sort_dataframe
from signal_generation.constants import SignalType
from signal_generation.future_returns import create_future_return_signals
from signal_generation.historical_returns import create_historical_return_signals
from signal_generation.rohrbach import create_rohrbach_signals
from signal_generation.volume import create_volume_signals


def create_trading_signals(
    df_ohlc: pd.DataFrame, periods_per_day: int, signal_type: SignalType
) -> pd.DataFrame:
    df = sort_dataframe(df_ohlc)

    df = create_historical_return_signals(df, periods_per_day=periods_per_day)
    df = create_volume_signals(df, periods_per_day=periods_per_day)
    if signal_type == SignalType.HistoricalReturns:
        return df
    elif signal_type == SignalType.Rohrbach:
        df = create_rohrbach_signals(df, periods_per_day=periods_per_day)
    else:
        raise ValueError("Invalid signal type!")
    return df


def create_analysis_signals(
    df_ohlc: pd.DataFrame, periods_per_day: int
) -> pd.DataFrame:
    df = sort_dataframe(df_ohlc)

    df = create_historical_return_signals(df, periods_per_day=periods_per_day)
    df = create_future_return_signals(df, periods_per_day=periods_per_day)
    return df
