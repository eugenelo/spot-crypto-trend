import numpy as np
import pandas as pd

from data.constants import DATETIME_COL, TICKER_COL


def sort_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Sort dataframe for use in computing returns, rolling statistics, etc.

    Args:
        df (pd.DataFrame): DataFrame containing [TICKER_COL, DATETIME_COL] columns

    Returns:
        pd.DataFrame: Sorted dataframe
    """
    return df.sort_values(by=[TICKER_COL, DATETIME_COL], ascending=True)


def returns(df: pd.DataFrame, column: str, periods: int) -> pd.Series:
    """Compute returns over periods

    Args:
        df (pd.DataFrame): DataFrame containing [TICKER_COL, column] columns
        column (str): Column over which to compute returns
        periods (int): Number of periods over which to compute

    Returns:
        pd.Series: Returns per ticker over periods
    """
    return df.groupby(TICKER_COL)[column].pct_change(periods=periods, fill_method=None)


def log_returns(df: pd.DataFrame, column: str, periods: int) -> pd.Series:
    """Compute log returns over periods

    Args:
        df (pd.DataFrame): DataFrame containing [TICKER_COL, column] columns
        column (str): Column over which to compute log returns
        periods (int): Number of periods over which to compute

    Returns:
        pd.Series: Log returns per ticker over periods
    """
    return df.groupby(TICKER_COL)[column].transform(
        lambda x: np.log(x / x.shift(periods=periods))
    )


def future_returns(df: pd.DataFrame, column: str, periods: int) -> pd.Series:
    """Compute future returns over periods

    Args:
        df (pd.DataFrame): DataFrame containing [TICKER_COL, column] columns
        column (str): Column over which to compute future returns
        periods (int): Number of periods over which to compute

    Returns:
        pd.Series: Future returns per ticker over periods
    """
    return returns(df, column, periods).shift(-periods)


def future_log_returns(df: pd.DataFrame, column: str, periods: int) -> pd.Series:
    """Compute future log returns over periods

    Args:
        df (pd.DataFrame): DataFrame containing [TICKER_COL, column] columns
        column (str): Column over which to compute future log returns
        periods (int): Number of periods over which to compute

    Returns:
        pd.Series: Future log returns per ticker over periods
    """
    return log_returns(df, column, periods).shift(-periods)


def ema(df: pd.DataFrame, column: str, periods: int) -> pd.Series:
    """Compute exponential moving average over periods

    Args:
        df (pd.DataFrame): DataFrame containing [TICKER_COL, column] columns
        column (str): Column over which to compute EMA
        periods (int): Number of periods over which to compute

    Returns:
        pd.Series: EMA of column per ticker over periods
    """
    return (
        df.groupby(TICKER_COL)[column]
        .ewm(span=periods, adjust=False, ignore_na=False)
        .mean()
        .reset_index(0, drop=True)
    )


def ema_daily(
    df: pd.DataFrame, column: str, days: int, periods_per_day: int
) -> pd.Series:
    """Compute daily exponential moving average for data which may be higher frequency

    Args:
        df (pd.DataFrame): DataFrame containing [TICKER_COL, column] columns
        column (str): Column over which to compute EMA
        days (int): Number of days over which to compute
        periods_per_day (int): Number of periods per day

    Returns:
        pd.Series: EMA of column per ticker over days
    """
    return (
        df.groupby(TICKER_COL)[column]
        .apply(
            lambda x: x[::periods_per_day]
            .ewm(span=days, adjust=False, ignore_na=False)
            .mean()
        )
        .reset_index(0, drop=True)
    )


def volatility(df: pd.DataFrame, column: str, periods: int) -> pd.Series:
    """Compute volatility (standard deviation) over periods

    Args:
        df (pd.DataFrame): DataFrame containing [TICKER_COL, column] columns
        column (str): Column over which to compute volatility
        periods (int): Number of periods over which to compute

    Returns:
        pd.Series: Volatility of column per ticker over periods
    """
    return (
        df.groupby(TICKER_COL)[column].rolling(periods).std().reset_index(0, drop=True)
    )


def future_volatility(df: pd.DataFrame, column: str, periods: int) -> pd.Series:
    """Compute future volatility over periods

    Args:
        df (pd.DataFrame): DataFrame containing [TICKER_COL, column] columns
        column (str): Column over which to compute future volatility
        periods (int): Number of periods over which to compute

    Returns:
        pd.Series: Future volatility per ticker over periods
    """
    return volatility(df, column, periods).shift(-(periods - 1))


def volatility_ema(df: pd.DataFrame, column: str, periods: int) -> pd.Series:
    """Compute volatility (standard deviation) using an EWMA over periods

    Args:
        df (pd.DataFrame): DataFrame containing [TICKER_COL, column] columns
        column (str): Column over which to compute volatility
        periods (int): Number of periods over which to compute

    Returns:
        pd.Series: Volatility of column per ticker over periods
    """
    return (
        df.groupby(TICKER_COL)[column]
        .ewm(span=periods, adjust=True, ignore_na=False)
        .std()
        .reset_index(0, drop=True)
    )


def rolling_sum(df: pd.DataFrame, column: str, periods: int) -> pd.Series:
    """Compute rolling sum of column over periods

    Args:
        df (pd.DataFrame): DataFrame containing [TICKER_COL, column] columns
        column (str): Column over which to compute rolling sum
        periods (int): Number of periods over which to compute

    Returns:
        pd.Series: Rolling sum of column per ticker over periods
    """
    return (
        df.groupby(TICKER_COL)[column].rolling(periods).sum().reset_index(0, drop=True)
    )


def bins(
    df: pd.DataFrame, column: str, num_bins: int, duplicates: str = "raise"
) -> pd.Series:
    """Bin data into num_bins

    Args:
        df (pd.DataFrame): DataFrame containing [TICKER_COL, column] columns
        column (str): Column over which to bin data
        num_bins (int): Number of bins
        duplicates (str): Policy for dealing with duplicates

    Returns:
        pd.Series: Bins of column
    """
    return pd.qcut(df[column], num_bins, labels=False, duplicates=duplicates)


def cross_sectional_abs_ema(
    df: pd.DataFrame, column: str, days: int, periods_per_day: int
) -> pd.Series:
    df_tmp = df.copy()

    # Calculate absolute value average for position sizing
    abs_col = "abs_" + column
    df_tmp[abs_col] = np.abs(df_tmp[column])
    abs_cross_section_mean_col = abs_col + "_mean"
    df_tmp[abs_cross_section_mean_col] = df_tmp.groupby(DATETIME_COL)[
        abs_col
    ].transform("mean")

    periods = days * periods_per_day
    return (
        df_tmp.groupby(TICKER_COL)[abs_cross_section_mean_col]
        .ewm(span=periods, adjust=True, ignore_na=False)
        .mean()
        .reset_index(0, drop=True)
    )


def apply_mask(
    df: pd.DataFrame, mask, target_col: str, fill_value=np.nan
) -> pd.DataFrame:
    df.loc[mask, target_col] = fill_value
    return df
