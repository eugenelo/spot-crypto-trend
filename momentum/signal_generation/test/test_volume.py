import unittest

import pandas as pd

from core.constants import AVG_DOLLAR_VOLUME_COL, VOLUME_FILTER_COL
from data.constants import DATETIME_COL, TICKER_COL
from signal_generation.common import sort_dataframe
from signal_generation.volume import create_volume_filter_mask


class TestVolume(unittest.TestCase):
    def setup(self):
        volume_A = [1, 11, 15, 9, 8, 9, 5, 15, 19, 21, 29, 23, 22, 21, 19, 18]
        volume_B = [9, 9, 9, 9, 9, 9, 9, 9, 19, 19, 19, 19, 19, 19, 19, 19]
        volume = {
            TICKER_COL: ["A"] * len(volume_A) + ["B"] * len(volume_B),
            AVG_DOLLAR_VOLUME_COL: volume_A + volume_B,
        }
        volume_df = pd.DataFrame.from_dict(volume)
        start_date = "2020-01-01"
        dti = pd.date_range(start_date, periods=len(volume_A), freq="1d")
        volume_df.loc[volume_df[TICKER_COL] == "A", DATETIME_COL] = dti
        volume_df.loc[volume_df[TICKER_COL] == "B", DATETIME_COL] = dti
        return sort_dataframe(volume_df), start_date

    def test_create_volume_filter_mask(self):
        df, start_date = self.setup()

        min_daily_volume = 10
        max_daily_volume = 25
        df = create_volume_filter_mask(
            df, min_daily_volume=min_daily_volume, max_daily_volume=max_daily_volume
        )
        expected_volume_filter_A = pd.Series(
            [
                True,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                False,
            ],
            name=VOLUME_FILTER_COL,
        )
        volume_filter_A = df.loc[df[TICKER_COL] == "A"][VOLUME_FILTER_COL].astype(bool)
        self.assertTrue(expected_volume_filter_A.equals(volume_filter_A))
        volume_B = df.loc[df[TICKER_COL] == "B"][VOLUME_FILTER_COL].astype(bool)
        volume_B.index = pd.RangeIndex(len(volume_B.index))
        expected_volume_out_B = pd.Series(
            [True] * 8 + [False] * 8, name=VOLUME_FILTER_COL
        )
        self.assertTrue(expected_volume_out_B.equals(volume_B))


if __name__ == "__main__":
    unittest.main()
