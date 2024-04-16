import math

import pandas as pd


def rebal_freq_supported(rebalancing_freq: str) -> bool:
    try:
        offset = pd.tseries.frequencies.to_offset(rebalancing_freq)
        # Accessing nanos will fail for non-fixed frequencies
        offset.nanos
        return True
    except ValueError:
        return False


def get_segment_mask(periods_per_day: int, rebalancing_freq: str) -> int:
    offset = pd.tseries.frequencies.to_offset(rebalancing_freq)
    offset_sec = offset.nanos * 1e-9
    # Number of seconds in a day is 60 * 60 * 24 = 86400
    num_days_per_rebal_period = offset_sec / 86400
    # Rebalance every (num_days_per_rebal_period * periods_per_day) ticks
    # segment_mask must be a whole number
    segment_mask = num_days_per_rebal_period * periods_per_day
    assert math.isclose(
        segment_mask, round(segment_mask)
    ), f"segment_mask {segment_mask} is not a whole number!"
    return int(segment_mask)
