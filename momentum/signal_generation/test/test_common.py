import unittest
from datetime import timedelta

import numpy as np
import pandas as pd

from data.constants import CLOSE_COL, DATETIME_COL, TICKER_COL
from signal_generation.common import (
    bins,
    ema,
    ema_daily,
    future_log_returns,
    future_returns,
    future_volatility,
    log_returns,
    returns,
    rolling_sum,
    sort_dataframe,
    volatility,
)


class TestSimulationUtils(unittest.TestCase):
    def setup(self):
        prices_A = [1, 1.5, 1.2, 0.5, 0.01]
        prices_B = [1, 2, 4, 8, 16]
        prices = {TICKER_COL: ["A"] * 5 + ["B"] * 5, CLOSE_COL: prices_A + prices_B}
        close = pd.DataFrame.from_dict(prices)
        start_date = "2020-01-01"
        dti = pd.date_range(start_date, periods=len(prices_A), freq="1d")
        close.loc[close[TICKER_COL] == "A", DATETIME_COL] = dti
        close.loc[close[TICKER_COL] == "B", DATETIME_COL] = dti
        return sort_dataframe(close), start_date

    def test_sort_dataframe(self):
        df_prices, start_date = self.setup()
        # Shuffle the dataframe rows
        df_prices = df_prices.sample(frac=1)
        df_prices = sort_dataframe(df_prices)
        for i in range(5):
            self.assertEqual("A", df_prices.iloc[i][TICKER_COL])
            self.assertEqual(
                pd.to_datetime(start_date) + timedelta(days=i),
                df_prices.iloc[i][DATETIME_COL],
            )
        for i in range(5, 10):
            offset = 5
            self.assertEqual("B", df_prices.iloc[i][TICKER_COL])
            self.assertEqual(
                pd.to_datetime(start_date) + timedelta(days=(i - offset)),
                df_prices.iloc[i][DATETIME_COL],
            )

    def test_returns(self):
        df_prices, start_date = self.setup()
        A_mask = df_prices[TICKER_COL] == "A"
        B_mask = df_prices[TICKER_COL] == "B"
        prices_A = df_prices.loc[A_mask][CLOSE_COL]

        # 1 period returns
        df_prices["returns"] = returns(df_prices, column=CLOSE_COL, periods=1)
        returns_A = df_prices.loc[A_mask]["returns"]
        for i in range(5):
            if i == 0:
                self.assertTrue(returns_A.isna().iloc[i])
            else:
                expected_return = (
                    prices_A.iloc[i] - prices_A.iloc[i - 1]
                ) / prices_A.iloc[i - 1]
                self.assertAlmostEqual(expected_return, returns_A.iloc[i])
        returns_B = df_prices.loc[B_mask]["returns"]
        for i in range(5):
            if i == 0:
                self.assertTrue(returns_B.isna().iloc[i])
            else:
                expected_return = 1.0  # 2x return
                self.assertAlmostEqual(expected_return, returns_B.iloc[i])

        # 2 period returns
        df_prices["2d_returns"] = returns(df_prices, column=CLOSE_COL, periods=2)
        returns_2d_A = df_prices.loc[A_mask]["2d_returns"]
        for i in range(5):
            if i <= 1:
                self.assertTrue(returns_2d_A.isna().iloc[i])
            else:
                expected_return = (
                    prices_A.iloc[i] - prices_A.iloc[i - 2]
                ) / prices_A.iloc[i - 2]
                self.assertAlmostEqual(expected_return, returns_2d_A.iloc[i])
        returns_2d_B = df_prices.loc[B_mask]["2d_returns"]
        for i in range(5):
            if i <= 1:
                self.assertTrue(returns_2d_B.isna().iloc[i])
            else:
                expected_return = 3.0  # 4x return
                self.assertAlmostEqual(expected_return, returns_2d_B.iloc[i])

    def test_log_returns(self):
        df_prices, start_date = self.setup()
        A_mask = df_prices[TICKER_COL] == "A"
        B_mask = df_prices[TICKER_COL] == "B"
        prices_A = df_prices.loc[A_mask][CLOSE_COL]

        # 1 period returns
        df_prices["log_returns"] = log_returns(df_prices, column=CLOSE_COL, periods=1)
        log_returns_A = df_prices.loc[A_mask]["log_returns"]
        for i in range(5):
            if i == 0:
                self.assertTrue(log_returns_A.isna().iloc[i])
            else:
                expected_return = np.log(prices_A.iloc[i] / prices_A.iloc[i - 1])
                self.assertAlmostEqual(expected_return, log_returns_A.iloc[i])
        log_returns_B = df_prices.loc[B_mask]["log_returns"]
        for i in range(5):
            if i == 0:
                self.assertTrue(log_returns_B.isna().iloc[i])
            else:
                expected_return = np.log(2.0)  # 2x return
                self.assertAlmostEqual(expected_return, log_returns_B.iloc[i])

        # 2 period returns
        df_prices["2d_log_returns"] = log_returns(
            df_prices, column=CLOSE_COL, periods=2
        )
        log_returns_2d_A = df_prices.loc[A_mask]["2d_log_returns"]
        for i in range(5):
            if i <= 1:
                self.assertTrue(log_returns_2d_A.isna().iloc[i])
            else:
                expected_return = np.log(prices_A.iloc[i] / prices_A.iloc[i - 2])
                self.assertAlmostEqual(expected_return, log_returns_2d_A.iloc[i])
        log_returns_2d_B = df_prices.loc[B_mask]["2d_log_returns"]
        for i in range(5):
            if i <= 1:
                self.assertTrue(log_returns_2d_B.isna().iloc[i])
            else:
                expected_return = np.log(4.0)  # 4x return
                self.assertAlmostEqual(expected_return, log_returns_2d_B.iloc[i])

    def test_future_returns(self):
        df_prices, start_date = self.setup()

        # 1 period future returns
        df_prices["returns"] = returns(df_prices, column=CLOSE_COL, periods=1)
        df_prices["next_1d_returns"] = future_returns(
            df_prices, column=CLOSE_COL, periods=1
        )
        for ticker in ["A", "B"]:
            df_ticker = df_prices.loc[df_prices[TICKER_COL] == ticker]
            s_returns = df_ticker["returns"]
            s_next_1d_returns = df_ticker["next_1d_returns"]
            for i in range(5):
                if i == 4:
                    self.assertTrue(s_next_1d_returns.isna().iloc[i])
                else:
                    self.assertEqual(s_returns.iloc[i + 1], s_next_1d_returns.iloc[i])

        # 2 period future returns
        df_prices["2d_returns"] = returns(df_prices, column=CLOSE_COL, periods=2)
        df_prices["next_2d_returns"] = future_returns(
            df_prices, column=CLOSE_COL, periods=2
        )
        for ticker in ["A", "B"]:
            df_ticker = df_prices.loc[df_prices[TICKER_COL] == ticker]
            s_returns = df_ticker["2d_returns"]
            s_next_2d_returns = df_ticker["next_2d_returns"]
            for i in range(5):
                if i >= 3:
                    self.assertTrue(s_next_2d_returns.isna().iloc[i])
                else:
                    self.assertEqual(s_returns.iloc[i + 2], s_next_2d_returns.iloc[i])

    def test_future_log_returns(self):
        df_prices, start_date = self.setup()

        # 1 period future log_returns
        df_prices["log_returns"] = log_returns(df_prices, column=CLOSE_COL, periods=1)
        df_prices["next_1d_log_returns"] = future_log_returns(
            df_prices, column=CLOSE_COL, periods=1
        )
        for ticker in ["A", "B"]:
            df_ticker = df_prices.loc[df_prices[TICKER_COL] == ticker]
            s_log_returns = df_ticker["log_returns"]
            s_next_1d_log_returns = df_ticker["next_1d_log_returns"]
            for i in range(5):
                if i == 4:
                    self.assertTrue(s_next_1d_log_returns.isna().iloc[i])
                else:
                    self.assertEqual(
                        s_log_returns.iloc[i + 1], s_next_1d_log_returns.iloc[i]
                    )

        # 2 period future log_returns
        df_prices["2d_log_returns"] = log_returns(
            df_prices, column=CLOSE_COL, periods=2
        )
        df_prices["next_2d_log_returns"] = future_log_returns(
            df_prices, column=CLOSE_COL, periods=2
        )
        for ticker in ["A", "B"]:
            df_ticker = df_prices.loc[df_prices[TICKER_COL] == ticker]
            s_log_returns = df_ticker["2d_log_returns"]
            s_next_2d_log_returns = df_ticker["next_2d_log_returns"]
            for i in range(5):
                if i >= 3:
                    self.assertTrue(s_next_2d_log_returns.isna().iloc[i])
                else:
                    self.assertEqual(
                        s_log_returns.iloc[i + 2], s_next_2d_log_returns.iloc[i]
                    )

    def test_ema(self):
        df_prices, start_date = self.setup()

        # 3 period EMA
        num_periods = 4
        df_prices["4d_ema"] = ema(df_prices, column=CLOSE_COL, periods=num_periods)
        for ticker in ["A", "B"]:
            df_ticker = df_prices.loc[df_prices[TICKER_COL] == ticker]
            prices = df_ticker[CLOSE_COL]
            s_4d_ema = df_ticker["4d_ema"]

            expected_4d_ema = [prices.iloc[0]]
            alpha = 2 / (num_periods + 1)
            for i in range(1, 5):
                new_ema_val = prices.iloc[i] * alpha + expected_4d_ema[-1] * (1 - alpha)
                expected_4d_ema.append(new_ema_val)
            for i in range(len(s_4d_ema)):
                self.assertAlmostEqual(expected_4d_ema[i], s_4d_ema.iloc[i])

    def test_ema_daily(self):
        df = pd.DataFrame.from_dict(
            {
                TICKER_COL: ["A"] * 7,
                CLOSE_COL: [1, 2, 1, 2, 3, 4, 5],
            }
        )
        df_every_other = df[::2]

        # 5 day EMA on every other data
        num_days = 5
        true_5d_ema_every_other = ema(
            df_every_other, column=CLOSE_COL, periods=num_days
        )
        expected_5d_ema = [1, 1, 5 / 3, 25 / 9]
        self.assertEqual(expected_5d_ema, list(true_5d_ema_every_other))
        # Comparable 5 day EMA on original data...backfilled?
        backfill_5d_ema = ema_daily(
            df, column=CLOSE_COL, days=num_days, periods_per_day=2
        )
        self.assertEqual(expected_5d_ema, list(backfill_5d_ema))

    def test_volatility(self):
        df_prices, start_date = self.setup()

        # 3 period stddev
        df_prices["3d_vol"] = volatility(df_prices, CLOSE_COL, periods=3)
        for ticker in ["A", "B"]:
            df_ticker = df_prices.loc[df_prices[TICKER_COL] == ticker]
            prices = df_ticker[CLOSE_COL]
            s_3d_vol = df_ticker["3d_vol"]
            expected_3d_vol = [
                np.nan,
                np.nan,
                np.std(prices[0:3], ddof=1),
                np.std(prices[1:4], ddof=1),
                np.std(prices[2:5], ddof=1),
            ]
            for i in range(len(expected_3d_vol)):
                if np.isnan(expected_3d_vol[i]):
                    self.assertTrue(np.isnan(s_3d_vol.iloc[i]))
                else:
                    self.assertAlmostEqual(expected_3d_vol[i], s_3d_vol.iloc[i])

    def test_future_volatility(self):
        df_prices, start_date = self.setup()

        # 3 period stddev
        df_prices["next_3d_vol"] = future_volatility(
            df_prices, column=CLOSE_COL, periods=3
        )
        for ticker in ["A", "B"]:
            df_ticker = df_prices.loc[df_prices[TICKER_COL] == ticker]
            prices = df_ticker[CLOSE_COL]
            s_3d_future_vol = df_ticker["next_3d_vol"]
            expected_3d_future_vol = [
                np.std(prices[0:3], ddof=1),
                np.std(prices[1:4], ddof=1),
                np.std(prices[2:5], ddof=1),
                np.nan,
                np.nan,
            ]
            for i in range(len(expected_3d_future_vol)):
                if np.isnan(expected_3d_future_vol[i]):
                    self.assertTrue(np.isnan(s_3d_future_vol.iloc[i]))
                else:
                    self.assertAlmostEqual(
                        expected_3d_future_vol[i], s_3d_future_vol.iloc[i]
                    )

    def test_rolling_sum(self):
        df_prices, start_date = self.setup()

        # 3 period rolling sum
        df_prices["3d_sum_close"] = rolling_sum(df_prices, CLOSE_COL, periods=3)
        for ticker in ["A", "B"]:
            df_ticker = df_prices.loc[df_prices[TICKER_COL] == ticker]
            prices = df_ticker[CLOSE_COL]
            s_3d_sum_close = df_ticker["3d_sum_close"]
            expected_3d_sum = [
                np.nan,
                np.nan,
                np.sum(prices[0:3]),
                np.sum(prices[1:4]),
                np.sum(prices[2:5]),
            ]
            for i in range(len(expected_3d_sum)):
                if np.isnan(expected_3d_sum[i]):
                    self.assertTrue(np.isnan(s_3d_sum_close.iloc[i]))
                else:
                    self.assertAlmostEqual(expected_3d_sum[i], s_3d_sum_close.iloc[i])

    def test_bins(self):
        df = pd.DataFrame.from_dict({"A": list(range(30))})
        # Shuffle order
        df = df.sample(frac=1)
        # 5 bins
        df["quintile"] = bins(df, column="A", num_bins=5)
        for i in range(30):
            expected_quintile = int(i / 6)
            self.assertEqual(
                expected_quintile, df.loc[df["A"] == i]["quintile"].iloc[0]
            )
        # 3 bins
        df["3tile"] = bins(df, column="A", num_bins=3)
        for i in range(30):
            expected_3tile = int(i / 10)
            self.assertEqual(expected_3tile, df.loc[df["A"] == i]["3tile"].iloc[0])


if __name__ == "__main__":
    unittest.main()
