import unittest
import pandas as pd
import numpy as np
from datetime import timedelta

import vectorbt as vbt

from simulation.backtest import simulate


class TestBacktestSingleAsset(unittest.TestCase):
    def setup(self):
        # Price doubles every day
        prices = {
            "A": [1, 2, 4, 8, 16],
        }
        close = pd.DataFrame.from_dict(prices)
        start_date = "2020-01-01"
        dti = pd.date_range(start_date, periods=len(close), freq="1d")
        close.index = dti
        return close, start_date

    def test_simulate_single_asset(self):
        close, start_date = self.setup()
        volume = pd.DataFrame.from_dict(
            {
                "A": [np.inf] * 5,
            }
        )
        volume.index = close.index

        # Buy at start, hold position
        initial_capital = 100.0
        volume_max_size = 1.0
        rebalancing_buffer = 0.0
        size = pd.DataFrame.from_dict(
            {
                "A": [1.0] * 5,
            }
        )
        size.index = close.index
        pf = simulate(
            price=close,
            positions=size,
            volume=volume,
            volume_max_size=volume_max_size,
            rebalancing_buffer=rebalancing_buffer,
            initial_capital=initial_capital,
        )
        # Entry Trades
        entry_trades = pf.entry_trades.records_readable.sort_values(
            by="Entry Timestamp"
        )
        self.assertEqual(1, entry_trades.shape[0])
        self.assertAlmostEqual(100.0, entry_trades["Size"].iloc[0])
        self.assertEqual("Open", entry_trades["Status"].iloc[0])
        self.assertEqual(
            pd.to_datetime(start_date), entry_trades["Entry Timestamp"].iloc[0]
        )
        # Exit Trades
        exit_trades = pf.exit_trades.records_readable.sort_values(by="Exit Timestamp")
        exit_trades = exit_trades.loc[exit_trades["Status"] == "Closed"]
        self.assertEqual(0, exit_trades.shape[0])
        # Cash
        expected_cash = pd.Series([0, 0, 0, 0, 0])
        self.assertTrue(np.allclose(expected_cash, pf.cash()))
        # Portfolio Value
        self.assertAlmostEqual(initial_capital * close["A"].iloc[-1], pf.final_value())

    def test_simulate_volume_constraints_single_asset(self):
        close, start_date = self.setup()
        volume = pd.DataFrame.from_dict(
            {
                "A": [100, 100, 100, 100, np.inf],
            }
        )
        volume.index = close.index

        # Capacity constrained until the last day
        initial_capital = 100.0
        volume_max_size = 0.01
        rebalancing_buffer = 0.0
        size = pd.DataFrame.from_dict(
            {
                "A": [1.0] * 5,
            }
        )
        size.index = close.index
        pf = simulate(
            price=close,
            positions=size,
            volume=volume,
            volume_max_size=volume_max_size,
            rebalancing_buffer=rebalancing_buffer,
            initial_capital=initial_capital,
        )
        # Entry Trades
        entry_trades = pf.entry_trades.records_readable.sort_values(
            by="Entry Timestamp"
        )
        self.assertEqual(5, entry_trades.shape[0])
        for i in range(4):
            self.assertAlmostEqual(1.0, entry_trades["Size"].iloc[i])
        self.assertAlmostEqual(5.3125, entry_trades["Size"].iloc[4])
        # Exit Trades
        exit_trades = pf.exit_trades.records_readable.sort_values(by="Exit Timestamp")
        exit_trades = exit_trades.loc[exit_trades["Status"] == "Closed"]
        self.assertEqual(0, exit_trades.shape[0])
        # Cash
        expected_cash = pd.Series([99, 97, 93, 85, 0])
        self.assertTrue(np.allclose(expected_cash, pf.cash()))
        # Portfolio Value
        self.assertAlmostEqual(9.3125 * close["A"].iloc[-1], pf.final_value())

    def test_simulate_rebalancing_buffer_single_asset(self):
        close, start_date = self.setup()
        volume = pd.DataFrame.from_dict(
            {
                "A": [np.inf] * 5,
            }
        )
        volume.index = close.index

        # Don't breach rebalance buffer until 4th & 5th days
        initial_capital = 100.0
        volume_max_size = 1.0
        rebalancing_buffer = 0.4
        size = pd.DataFrame.from_dict(
            {
                "A": [0.1, 0.2, 0.3, 0.9, -0.4],
            }
        )
        size.index = close.index
        pf = simulate(
            price=close,
            positions=size,
            volume=volume,
            volume_max_size=volume_max_size,
            rebalancing_buffer=rebalancing_buffer,
            initial_capital=initial_capital,
        )

        # Entry Trades
        entry_trades = pf.entry_trades.records_readable.sort_values(
            by="Entry Timestamp"
        )
        self.assertEqual(1, entry_trades.shape[0])
        self.assertAlmostEqual(
            (size["A"].iloc[3] - rebalancing_buffer)
            * initial_capital
            / close["A"].iloc[3],
            entry_trades["Size"].iloc[0],
        )
        self.assertEqual("Closed", entry_trades["Status"].iloc[0])
        self.assertEqual(
            pd.to_datetime(start_date) + timedelta(days=3),
            entry_trades["Entry Timestamp"].iloc[0],
        )
        # Exit Trades
        exit_trades = pf.exit_trades.records_readable.sort_values(by="Exit Timestamp")
        exit_trades = exit_trades.loc[exit_trades["Status"] == "Closed"]
        self.assertEqual(1, exit_trades.shape[0])
        # Cash
        expected_cash = pd.Series([100, 100, 100, 50, 150])
        self.assertTrue(np.allclose(expected_cash, pf.cash()))
        # Portfolio Value
        self.assertAlmostEqual(150.0, pf.final_value())


class TestBacktestMultiAsset(unittest.TestCase):
    def setup(self):
        prices = {
            "A": [1, 1, 1, 1, 1],
            "B": [1, 2, 3, 4, 5],
            "C": [1, 2, 4, 8, 16],
        }
        close = pd.DataFrame.from_dict(prices)
        start_date = "2020-01-01"
        dti = pd.date_range(start_date, periods=len(close), freq="1d")
        close.index = dti
        return close, start_date

    def test_simulate_multi_asset(self):
        close, start_date = self.setup()
        volume = pd.DataFrame.from_dict(
            {
                "A": [np.inf] * 5,
                "B": [np.inf] * 5,
                "C": [np.inf] * 5,
            }
        )
        volume.index = close.index

        # Buy at start, hold position
        initial_capital = 100.0
        volume_max_size = 1.0
        rebalancing_buffer = 0.0
        # fmt: off
        size = pd.DataFrame.from_dict(
            {
                "A": [1/3, 1/5, 1/8, 1/13, 1/22],
                "B": [1/3, 2/5, 3/8, 4/13, 5/22],
                "C": [1/3, 2/5, 4/8, 8/13, 16/22],
            }
        )
        # fmt: on
        size.index = close.index
        pf = simulate(
            price=close,
            positions=size,
            volume=volume,
            volume_max_size=volume_max_size,
            rebalancing_buffer=rebalancing_buffer,
            initial_capital=initial_capital,
        )

        # Entry Trades
        entry_trades = pf.entry_trades.records_readable.sort_values(
            by="Entry Timestamp"
        )
        self.assertEqual(3, entry_trades.shape[0])
        for i in range(3):
            self.assertAlmostEqual(initial_capital / 3, entry_trades["Size"].iloc[i])
            self.assertEqual("Open", entry_trades["Status"].iloc[i])
            self.assertEqual(
                pd.to_datetime(start_date), entry_trades["Entry Timestamp"].iloc[i]
            )
        # Exit Trades
        exit_trades = pf.exit_trades.records_readable.sort_values(by="Exit Timestamp")
        exit_trades = exit_trades.loc[exit_trades["Status"] == "Closed"]
        self.assertEqual(0, exit_trades.shape[0])
        # Portfolio Value
        self.assertAlmostEqual(100 / 3 + 500 / 3 + 1600 / 3, pf.final_value())

    def test_simulate_volume_constraints_multi_asset(self):
        close, start_date = self.setup()
        volume = pd.DataFrame.from_dict(
            {
                "A": [np.inf] * 5,  # Not constrained
                "B": [100, 100, 100, 100, np.inf],  # Constrained until last day
                "C": [100, np.inf, np.inf, np.inf, np.inf],  # Constrained on first day
            }
        )
        volume.index = close.index

        # Buy at start, hold position
        initial_capital = 100.0
        volume_max_size = 0.01
        rebalancing_buffer = 0.0
        # fmt: off
        size = pd.DataFrame.from_dict(
            {
                "A": [0.2, 0, 0, 0, 0],
                "B": [0.5, 0, 0, 0, 1.0],
                "C": [0.4, 0.4, 0.571428, 0.727272, 0],
            }
        )
        # fmt: on
        size.index = close.index
        pf = simulate(
            price=close,
            positions=size,
            volume=volume,
            volume_max_size=volume_max_size,
            rebalancing_buffer=rebalancing_buffer,
            initial_capital=initial_capital,
        )

        # Entry Trades
        entry_trades = pf.entry_trades.records_readable.sort_values(
            by="Entry Timestamp"
        )
        self.assertEqual(5, entry_trades.shape[0])
        # Traded A on first day
        self.assertEqual("A", entry_trades["Column"].iloc[0])
        self.assertAlmostEqual(
            size["A"].iloc[0] * initial_capital, entry_trades["Size"].iloc[0]
        )
        self.assertEqual(
            pd.to_datetime(start_date), entry_trades["Entry Timestamp"].iloc[0]
        )
        # Traded B on first and last days
        for i in (1, 4):
            self.assertEqual("B", entry_trades["Column"].iloc[i])
            if i == 1:
                self.assertAlmostEqual(1.0, entry_trades["Size"].iloc[i])
            else:
                self.assertAlmostEqual(77.52, entry_trades["Size"].iloc[i], places=2)
        # Traded C on first 2 days
        for i in (2, 3):
            self.assertEqual("C", entry_trades["Column"].iloc[i])
            if i == 2:
                self.assertAlmostEqual(1.0, entry_trades["Size"].iloc[i])
            else:
                self.assertAlmostEqual(19.4, entry_trades["Size"].iloc[i], places=2)
        # # Exit Trades
        exit_trades = pf.exit_trades.records_readable.sort_values(by="Exit Timestamp")
        exit_trades = exit_trades.loc[exit_trades["Status"] == "Closed"]
        self.assertEqual(3, exit_trades.shape[0])
        # Exited A on second day
        self.assertEqual("A", exit_trades["Column"].iloc[0])
        self.assertAlmostEqual(20.0, exit_trades["Size"].iloc[0])
        # Exited B on second day (1 share)
        self.assertEqual("B", exit_trades["Column"].iloc[1])
        self.assertAlmostEqual(1.0, exit_trades["Size"].iloc[1])
        # Exited C on last day (1 + 19.4 shares)
        self.assertEqual("C", exit_trades["Column"].iloc[2])
        self.assertAlmostEqual(20.4, exit_trades["Size"].iloc[2])
        # Cash
        expected_cash = pd.Series([78, 61.2, 61.2, 61.2, 0.0])
        self.assertTrue(np.allclose(expected_cash, pf.cash()))
        # Portfolio Value
        self.assertAlmostEqual(387.6, pf.final_value())

    def test_simulate_rebalancing_buffer_multi_asset(self):
        close, start_date = self.setup()
        volume = pd.DataFrame.from_dict(
            {
                "A": [np.inf] * 5,
                "B": [np.inf] * 5,
                "C": [np.inf] * 5,
            }
        )
        volume.index = close.index

        # Buy at start, hold position
        initial_capital = 100.0
        volume_max_size = 1.0
        rebalancing_buffer = 0.3
        # fmt: off
        size = pd.DataFrame.from_dict(
            {
                "A": [0.2, 0.35, 0.35, 0.35, 0.35],
                "B": [0.2, 0.2, 0.2, 0.2, 0.34],
                "C": [0.2, 0.1, 0.2, 0.1, 0.31],
            }
        )
        # fmt: on
        size.index = close.index
        pf = simulate(
            price=close,
            positions=size,
            volume=volume,
            volume_max_size=volume_max_size,
            rebalancing_buffer=rebalancing_buffer,
            initial_capital=initial_capital,
        )

        # Entry Trades
        entry_trades = pf.entry_trades.records_readable.sort_values(
            by="Entry Timestamp"
        )
        print(entry_trades)
        self.assertEqual(3, entry_trades.shape[0])
        # Traded A on second day
        self.assertEqual("A", entry_trades["Column"].iloc[0])
        self.assertAlmostEqual(
            (size["A"].iloc[1] - rebalancing_buffer)
            * initial_capital
            / close["A"].iloc[1],
            entry_trades["Size"].iloc[0],
        )
        self.assertEqual(
            pd.to_datetime(start_date) + timedelta(days=1),
            entry_trades["Entry Timestamp"].iloc[0],
        )
        # Traded B on last day
        self.assertEqual("B", entry_trades["Column"].iloc[1])
        self.assertAlmostEqual(
            (size["B"].iloc[4] - rebalancing_buffer)
            * initial_capital
            / close["B"].iloc[4],
            entry_trades["Size"].iloc[1],
        )
        self.assertEqual(
            pd.to_datetime(start_date) + timedelta(days=4),
            entry_trades["Entry Timestamp"].iloc[1],
        )
        # Traded C on last day
        self.assertEqual("C", entry_trades["Column"].iloc[2])
        self.assertAlmostEqual(
            (size["C"].iloc[4] - rebalancing_buffer)
            * initial_capital
            / close["C"].iloc[4],
            entry_trades["Size"].iloc[2],
        )
        self.assertEqual(
            pd.to_datetime(start_date) + timedelta(days=4),
            entry_trades["Entry Timestamp"].iloc[2],
        )
        # Exit Trades
        exit_trades = pf.exit_trades.records_readable.sort_values(by="Exit Timestamp")
        exit_trades = exit_trades.loc[exit_trades["Status"] == "Closed"]
        self.assertEqual(0, exit_trades.shape[0])
        # Cash
        expected_cash = pd.Series([100, 95, 95, 95, 90])
        self.assertTrue(np.allclose(expected_cash, pf.cash()))
        # Portfolio Value
        self.assertAlmostEqual(initial_capital, pf.final_value())


if __name__ == "__main__":
    unittest.main()
