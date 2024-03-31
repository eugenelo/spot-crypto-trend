import pandas as pd

from enum import Enum


def kraken_maker_fees(rolling_30d_volume: float):
    # Define your fee structure based on rolling_30d_volume
    if rolling_30d_volume <= 10000:
        return 0.0025
    elif rolling_30d_volume <= 50000:
        return 0.0020
    elif rolling_30d_volume <= 100000:
        return 0.0014
    elif rolling_30d_volume <= 250000:
        return 0.0012
    elif rolling_30d_volume <= 500000:
        return 0.0010
    elif rolling_30d_volume <= 1000000:
        return 0.0008
    elif rolling_30d_volume <= 2500000:
        return 0.0006
    elif rolling_30d_volume <= 5000000:
        return 0.0004
    elif rolling_30d_volume <= 10000000:
        return 0.0002
    else:
        return 0.0


def kraken_taker_fees(rolling_30d_volume: float):
    # Define your fee structure based on rolling_30d_volume
    if rolling_30d_volume <= 10000:
        return 0.0040
    elif rolling_30d_volume <= 50000:
        return 0.0035
    elif rolling_30d_volume <= 100000:
        return 0.0024
    elif rolling_30d_volume <= 250000:
        return 0.0022
    elif rolling_30d_volume <= 500000:
        return 0.0020
    elif rolling_30d_volume <= 1000000:
        return 0.0018
    elif rolling_30d_volume <= 2500000:
        return 0.0016
    elif rolling_30d_volume <= 5000000:
        return 0.0014
    elif rolling_30d_volume <= 10000000:
        return 0.0012
    else:
        return 0.0010


class FeeType(Enum):
    MAKER = "MAKER"
    TAKER = "TAKER"


def compute_fees(rolling_30d_volume: pd.Series, fee_type: FeeType):
    fee_fn = kraken_maker_fees if fee_type == FeeType.MAKER else kraken_taker_fees
    fees = rolling_30d_volume.apply(fee_fn).rename("Fees [%]")
    return fees
