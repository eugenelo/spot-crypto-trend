import pandas as pd
import numpy as np


def sort_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Sort dataframe for use in computing returns, rolling statistics, etc.

    Args:
        df (pd.DataFrame): DataFrame containing ["ticker", "timestamp"] columns

    Returns:
        pd.DataFrame: Sorted dataframe
    """
    return df.sort_values(by=["ticker", "timestamp"], ascending=True)


def returns(df: pd.DataFrame, column: str, periods: int) -> pd.Series:
    """Compute returns over periods

    Args:
        df (pd.DataFrame): DataFrame containing ["ticker", column] columns
        column (str): Column over which to compute returns
        periods (int): Number of periods over which to compute

    Returns:
        pd.Series: Returns per ticker over periods
    """
    return df.groupby("ticker")[column].pct_change(periods=periods)


def log_returns(df: pd.DataFrame, column: str, periods: int) -> pd.Series:
    """Compute log returns over periods

    Args:
        df (pd.DataFrame): DataFrame containing ["ticker", column] columns
        column (str): Column over which to compute log returns
        periods (int): Number of periods over which to compute

    Returns:
        pd.Series: Log returns per ticker over periods
    """
    return df.groupby("ticker")[column].transform(
        lambda x: np.log(x / x.shift(periods=periods))
    )


def future_returns(df: pd.DataFrame, column: str, periods: int) -> pd.Series:
    """Compute future returns over periods

    Args:
        df (pd.DataFrame): DataFrame containing ["ticker", column] columns
        column (str): Column over which to compute future returns
        periods (int): Number of periods over which to compute

    Returns:
        pd.Series: Future returns per ticker over periods
    """
    return returns(df, column, periods).shift(-periods)


def future_log_returns(df: pd.DataFrame, column: str, periods: int) -> pd.Series:
    """Compute future log returns over periods

    Args:
        df (pd.DataFrame): DataFrame containing ["ticker", column] columns
        column (str): Column over which to compute future log returns
        periods (int): Number of periods over which to compute

    Returns:
        pd.Series: Future log returns per ticker over periods
    """
    return log_returns(df, column, periods).shift(-periods)


def ema(df: pd.DataFrame, column: str, periods: int) -> pd.Series:
    """Compute exponential moving average over periods

    Args:
        df (pd.DataFrame): DataFrame containing ["ticker", column] columns
        column (str): Column over which to compute EMA
        periods (int): Number of periods over which to compute

    Returns:
        pd.Series: EMA of column per ticker over periods
    """
    return (
        df.groupby("ticker")[column]
        .ewm(span=periods, adjust=False, ignore_na=False)
        .mean()
        .reset_index(0, drop=True)
    )


def ema_daily(
    df: pd.DataFrame, column: str, days: int, periods_per_day: int
) -> pd.Series:
    """Compute daily exponential moving average for data which may be higher frequency

    Args:
        df (pd.DataFrame): DataFrame containing ["ticker", column] columns
        column (str): Column over which to compute EMA
        days (int): Number of days over which to compute
        periods_per_day (int): Number of periods per day

    Returns:
        pd.Series: EMA of column per ticker over days
    """
    return (
        df.groupby("ticker")[column]
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
        df (pd.DataFrame): DataFrame containing ["ticker", column] columns
        column (str): Column over which to compute volatility
        periods (int): Number of periods over which to compute

    Returns:
        pd.Series: Volatility of column per ticker over periods
    """
    return df.groupby("ticker")[column].rolling(periods).std().reset_index(0, drop=True)


def future_volatility(df: pd.DataFrame, column: str, periods: int) -> pd.Series:
    """Compute future volatility over periods

    Args:
        df (pd.DataFrame): DataFrame containing ["ticker", column] columns
        column (str): Column over which to compute future volatility
        periods (int): Number of periods over which to compute

    Returns:
        pd.Series: Future volatility per ticker over periods
    """
    return volatility(df, column, periods).shift(-(periods - 1))


def volatility_ema(df: pd.DataFrame, column: str, periods: int) -> pd.Series:
    """Compute volatility (standard deviation) using an EWMA over periods

    Args:
        df (pd.DataFrame): DataFrame containing ["ticker", column] columns
        column (str): Column over which to compute volatility
        periods (int): Number of periods over which to compute

    Returns:
        pd.Series: Volatility of column per ticker over periods
    """
    return (
        df.groupby("ticker")[column]
        .ewm(span=periods, adjust=True, ignore_na=False)
        .std()
        .reset_index(0, drop=True)
    )


def rolling_sum(df: pd.DataFrame, column: str, periods: int) -> pd.Series:
    """Compute rolling sum of column over periods

    Args:
        df (pd.DataFrame): DataFrame containing ["ticker", column] columns
        column (str): Column over which to compute rolling sum
        periods (int): Number of periods over which to compute

    Returns:
        pd.Series: Rolling sum of column per ticker over periods
    """
    return df.groupby("ticker")[column].rolling(periods).sum().reset_index(0, drop=True)


def bins(df: pd.DataFrame, column: str, num_bins: int) -> pd.Series:
    """Bin data into num_bins

    Args:
        df (pd.DataFrame): DataFrame containing ["ticker", column] columns
        column (str): Column over which to bin data
        num_bins (int): Number of bins

    Returns:
        pd.Series: Bins of column
    """
    return pd.qcut(df[column], num_bins, labels=False)


def cross_sectional_abs_ema(
    df: pd.DataFrame, column: str, days: int, periods_per_day: int
) -> pd.Series:
    df_tmp = df.copy()

    # Calculate absolute value average for position sizing
    abs_col = "abs_" + column
    df_tmp[abs_col] = np.abs(df_tmp[column])
    abs_cross_section_mean_col = abs_col + "_mean"
    df_tmp[abs_cross_section_mean_col] = df_tmp.groupby("timestamp")[abs_col].transform(
        "mean"
    )

    periods = days * periods_per_day
    return (
        df_tmp[abs_cross_section_mean_col]
        .ewm(span=periods, adjust=True, ignore_na=False)
        .mean()
    )


def apply_mask(
    df: pd.DataFrame, target_column: str, mask, fill_value=np.nan
) -> pd.DataFrame:
    df = df.copy()
    df.loc[mask, target_column] = fill_value
    return df
