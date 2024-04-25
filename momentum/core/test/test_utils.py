import unittest

import pandas as pd

from core.utils import apply_hysteresis, get_periods_per_day
from data.constants import DATETIME_COL, PRICE_COL, TICKER_COL


class TestSimulationUtils(unittest.TestCase):
    def setup(self):
        prices = {
            "A": [1, 10.1, 15, 9, 8, 9, 11, 5, 8, 9, 7, 12, 15, 8, 9],
            "B": [8] * 15,
        }
        close = pd.DataFrame.from_dict(prices)
        start_date = "2020-01-01"
        dti = pd.date_range(start_date, periods=len(close), freq="1d")
        close.index = dti
        return close, start_date

    def test_apply_hysteresis(self):
        df, start_date = self.setup()
        df = df.reset_index().rename(columns={"index": DATETIME_COL})
        # Transform dataframe from wide to long
        df = pd.melt(
            df,
            id_vars=[DATETIME_COL],
            value_vars=["A", "B"],
            var_name=TICKER_COL,
            value_name=PRICE_COL,
        )
        df = df.sort_values(by=[TICKER_COL, DATETIME_COL], ascending=True)

        entry_threshold = 10
        exit_threshold = 7
        output_col = "hysteresis"
        df = apply_hysteresis(
            df,
            group_col=TICKER_COL,
            value_col=PRICE_COL,
            output_col=output_col,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
        )

        expected_hysteresis_out_A = pd.Series(
            [
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
            ],
            name=output_col,
        )
        hysteresis_A = df.loc[df[TICKER_COL] == "A"][output_col].astype(bool)
        self.assertTrue(expected_hysteresis_out_A.equals(hysteresis_A))
        hysteresis_B = df.loc[df[TICKER_COL] == "B"][output_col].astype(bool)
        hysteresis_B.index = pd.RangeIndex(len(hysteresis_B.index))
        expected_hysteresis_out_B = pd.Series(
            [False] * len(hysteresis_B), name=output_col
        )
        self.assertTrue(expected_hysteresis_out_B.equals(hysteresis_B))

    def test_get_periods_per_day(self):
        # Hourly
        hourly_series = pd.date_range(
            "2020-01-01", periods=4 * 24, freq="1h"
        ).to_series()
        self.assertEqual(24, get_periods_per_day(hourly_series))

        # 4 hours
        four_hour_series = pd.date_range(
            "2020-01-01", periods=4 * 6, freq="4h"
        ).to_series()
        self.assertEqual(6, get_periods_per_day(four_hour_series))

        # Daily
        daily_series = pd.date_range("2020-01-01", periods=4, freq="1d").to_series()
        self.assertEqual(1, get_periods_per_day(daily_series))


if __name__ == "__main__":
    unittest.main()
